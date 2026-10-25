import copy
import ctypes as ct
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from .geometry_scene import projected_cells
from .geometry_cuda import GeometryKernel
from .geometry_integration import TrialTrace,GeometryWorldModelSandbox


class GeometryPrimitives(unittest.TestCase):
    def test_projected_negative_cells_and_multiple_floors(self):
        triangles=np.array([[[-.2,0,-.2],[.2,0,-.2],[-.2,0,.2]],[[-.2,2,-.2],[.2,2,-.2],[-.2,2,.2]]],np.float64)
        x,z,y=projected_cells(triangles,.1)
        self.assertIn((-2,-2,0.),set(zip(x,z,y)));self.assertIn((-2,-2,2.),set(zip(x,z,y)))
        self.assertNotIn((2,2,0.),set(zip(x,z,y)))

    def test_gpu_nearest_ray_has_two_sided_hits_and_misses(self):
        triangles=np.array([[[0,-2,-2],[0,2,-2],[0,-2,2]],[[0,2,-2],[0,2,2],[0,-2,2]]],np.float32)
        packed=np.concatenate([triangles[:,0],triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]],1)
        arrays=dict(keys=np.empty((0,2),np.int32),targets=np.empty((0,63,3)),triangles=packed,bounds=np.array([[-.001,-2.001,-2.001,.001,2.001,2.001]],np.float32),left=np.array([-1],np.int32),right=np.array([-1],np.int32),first=np.array([0],np.int32),count=np.array([2],np.int32),indices=np.arange(2,dtype=np.int32))
        m=GeometryKernel(arrays,.1,'cuda:1');origins=m.tensor([[-1,0,0],[1,0,0],[-1,3,0]],np.float32);directions=m.tensor([[1,0,0],[-1,0,0],[1,0,0]],np.float32);limits=m.tensor([10,10,10],np.float32);out=torch.empty(3,device=m.device)
        m.module.launch('test_rays',3,[origins,directions,limits,ct.c_int(3),out,*m.mesh]);np.testing.assert_allclose(out.cpu().numpy(),[1,1,10],rtol=0,atol=1.e-6)

    def test_trial_adapter_discards_all_result_labels_and_calls_stage1_once(self):
        rows=[dict(timestamp=t,robot_pose=[0,0,0,0],camera_origin=[0,1,0],camera_rotation=np.eye(3).tolist(),static_updates=['MUST_NOT_PASS'])for t in [0.,.048]]
        trial=TrialTrace.from_frames(rows);self.assertFalse(trial.trajectory.flags.writeable)
        class Stage1:
            def __init__(self):self.calls=0;self.last_context=SimpleNamespace(partial_graph={'current':'only'})
            def predict(self,*args,**kw):self.calls+=1;return 'frozen_people'
        class Stage2:
            def predict_segments(self,**kw):
                self.kw=kw;return [dict(future_partial_graph='hypothetical')]
        first,second=Stage1(),Stage2();sandbox=GeometryWorldModelSandbox(first,second);result=sandbox.predict_decision(None,{},executed_action='left',trials=[trial]);self.assertEqual(first.calls,1);self.assertEqual(set(second.kw),{'partial_graph','trajectories','camera_trajectories'});self.assertEqual(set(second.kw['camera_trajectories'][0]),{'origins','rotations'});self.assertEqual(result['hypothetical_graphs'],['hypothetical'])


class CompleteGeometryAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .diagnose import source_rows,member
        from .geometry_api import GeometryPredictor
        from eventwm_graph.stage2_segment.benchmark import inputs
        cls.item=inputs('multi')[0];rows,_=source_rows('dev');row=next(r for r in rows if r['root_id']==cls.item['root_id']);legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz');ep=member(row,'privileged_initialization.json.gz')['episode'];cls.graph=cls.item['graph'];cls.paths=[];cls.cameras=[]
        for p in cls.item['paths'][:6]:
            cls.paths.append(np.asarray(p['trajectory'],np.float64));obs=[legal['history'][-1]]+[nodes[i]['observation']for i in p['node_ids']];cls.cameras.append(dict(origins=np.array([o['camera_origin']for o in obs]),rotations=np.array([o['camera_rotation']for o in obs])))
        cls.model=GeometryPredictor(ep['scene_id'],ep['info']['graph_navmesh'],cls.graph['resolution'],legal['sensor'],device='cuda:1')

    @classmethod
    def tearDownClass(cls):cls.model.close()

    def test_six_distinct_real_paths_single_batch_full_future_and_immutable_input(self):
        from eventwm_graph.stage2_segment.update import apply_graph_delta
        m=self.model;before=copy.deepcopy(self.graph);batch=m.predict_segments(partial_graph=self.graph,trajectories=self.paths,camera_trajectories=self.cameras)
        for p,c,b in zip(self.paths,self.cameras,batch):
            a=m.predict(partial_graph=self.graph,trajectory=p,camera_origins=c['origins'],camera_rotations=c['rotations']);np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_array_equal(a['heights'],b['heights']);np.testing.assert_array_equal(a['probabilities'],b['probabilities']);self.assertEqual(b['future_partial_graph'].to_snapshot(),apply_graph_delta(self.graph,b));self.assertTrue(b['provenance']['hypothetical'])
        self.assertEqual(self.graph,before)

    def test_start_only_and_stationary_later_observation(self):
        m=self.model;p=self.paths[0][:1];c=self.cameras[0];g=copy.deepcopy(self.graph);g['cells']=[];g['global_nodes']={};g['global_edges']=[]
        first=m.predict(partial_graph=g,trajectory=p,camera_origins=c['origins'][:1],camera_rotations=c['rotations'][:1]);self.assertFalse(first['static_updates'])
        q=np.repeat(p,2,axis=0);q[1,0]+=.048;second=m.predict(partial_graph=g,trajectory=q,camera_origins=np.repeat(c['origins'][:1],2,axis=0),camera_rotations=np.repeat(c['rotations'][:1],2,axis=0));self.assertGreater(len(second['static_updates']),0)

    def test_metadata_invalidation_and_legacy_extension_fallback(self):
        from eventwm_graph.stage2_segment.update import apply_graph_delta
        m=self.model;g=copy.deepcopy(self.graph);before=m.prepare_graph(g);g['cells'][0]['last_seen']-=.1;self.assertIsNot(before,m.prepare_graph(g));g['cells'][0]['extension']={'keep':[1,2]};c=self.cameras[0];delta=m.predict(partial_graph=g,trajectory=self.paths[0],camera_origins=c['origins'],camera_rotations=c['rotations']);self.assertEqual(delta['future_format'],'snapshot');self.assertEqual(delta['future_partial_graph'],apply_graph_delta(g,delta))


if __name__=='__main__':unittest.main()
