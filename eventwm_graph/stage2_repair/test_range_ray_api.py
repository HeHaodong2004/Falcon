import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .diagnose import OUT
from .range_benchmark import workloads
from .range_ray_api import RayRangePredictor
from .range_ray_model import RayEvidenceNet
from .legal_camera import current_extrinsic
from .legal_camera_features import parameters,cell_features
from eventwm_graph.stage2_segment.fused import transform
from eventwm_graph.stage2_segment.update import apply_graph_delta


class RayPublicAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(7341)
        model=RayEvidenceNet();model.initialize_spatial(OUT/'training/support_pretrained256_control/selected.pt')
        torch.nn.init.normal_(model.head[-1].weight,std=.03)
        checkpoint=OUT/'training/range_ray_api_test.pt'
        torch.save(dict(schema='legal_range_ray_1',config=dict(base_checkpoint=str(OUT/'training/range_evidence64/selected.pt')),model=model.state_dict(),threshold=.2,decision_rule='argmax'),checkpoint)
        cls.model=RayRangePredictor(checkpoint,device='cuda:0');cls.ex,cls.packet,cls.tracks=workloads()[0]
        cls.paths=[np.asarray(p['trajectory'],np.float64)for p in cls.ex['paths'][:6]]

    def call(self,paths,graph=None):return self.model.predict_segments(partial_graph=self.ex['graph']if graph is None else graph,trajectories=paths,range_observations=self.packet,track_history=self.tracks)

    def test_single_batch_future_and_input_immutability(self):
        before=copy.deepcopy(self.ex['graph']);single=[self.call([p])[0]for p in self.paths]
        self.model.clear_cache();batch=self.call(self.paths)
        for a,b in zip(single,batch):
            np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_array_equal(a['states'],b['states'])
            np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=4.e-6,rtol=4.e-5)
            self.assertEqual(b['future_partial_graph'].to_snapshot(),apply_graph_delta(self.ex['graph'],b))
        self.assertEqual(before,self.ex['graph'])

    def test_independent_camera_assembly_matches_public_forward(self):
        model=self.model;p=self.paths[0];d=self.call([p])[0];g=model.prepare_graph(self.ex['graph'])
        context=model.feature_class(self.ex['graph'],p,g);x,_=transform(context,d['candidate_cells'],cuda=model.cuda,device_output=True)
        translation,rotation=current_extrinsic(p,self.packet['origins'][-1],self.packet['rotations'][-1]);params,_=parameters(p,translation,rotation,g.resolution)
        extra=np.zeros((len(x),16),np.float32);extra[:,8:]=cell_features(d['candidate_cells'],p,g.resolution,x.cpu().numpy(),params)
        canvas,low=model.canvas.build(g.keys,g.values,g.heights,p[0,1:4],g.resolution,self.packet['ranges'],self.packet['origins'],self.packet['rotations'],self.packet['timestamps'],self.tracks)
        def matrix(yaw):
            c,s=np.cos(yaw),np.sin(yaw);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
        local=matrix(p[0,4]).T@(self.packet['origins'][-1]-p[0,1:4]);selected=[1,len(p)//2,len(p)-1]
        lines=np.array([(p[i,1:4]+matrix(p[i,4])@local)[[0,2]]/g.resolution-low for i in selected],np.float32)
        actual_extra=model.camera_kernel.compute(d['candidate_cells'],x,p,self.packet['origins'][-1],self.packet['rotations'][-1],g.resolution)
        np.testing.assert_allclose(actual_extra.cpu().numpy(),extra,atol=2.e-7,rtol=1.e-6)
        with torch.inference_mode(),torch.cuda.amp.autocast():
            encoded=model.model.encode(torch.from_numpy(canvas.astype(np.float16))[None].to(model.device))
            expected,_=model.model.probabilities(x,torch.from_numpy(extra).to(model.device),torch.from_numpy(d['candidate_cells']-low).to(model.device),torch.zeros(len(x),dtype=torch.long,device=model.device),encoded,torch.from_numpy(lines)[None].expand(len(x),-1,-1).to(model.device))
        np.testing.assert_allclose(d['probabilities'],expected.float().cpu().numpy(),atol=4.e-6,rtol=4.e-5)

    def test_start_only_stationary_later_and_no_teacher_file_reads(self):
        p=self.paths[0];start=self.call([p[:1]])[0]
        np.testing.assert_array_equal(start['states'],0);np.testing.assert_array_equal(start['probabilities'][:,0],1.)
        stationary=np.repeat(p[:1],2,axis=0);stationary[1,0]+=.25
        with patch('numpy.load',side_effect=AssertionError('No runtime teacher/target caches')):
            later=self.call([stationary])[0]
        self.assertTrue(np.any(later['probabilities'][:,0]<1))

    def test_fully_known_candidate_domain_returns_empty_increment(self):
        p=self.paths[0];first=self.call([p])[0];graph=copy.deepcopy(self.ex['graph'])
        for key in first['candidate_cells']:
            graph['cells'].append(dict(cell=key.tolist(),state=1,height=float(p[0,2]),first_seen=graph['timestamp'],last_seen=graph['timestamp'],source='synthetic_fully_known_regression'))
        result=self.call([p],graph=graph)[0]
        self.assertEqual(len(result['candidate_cells']),0);self.assertEqual(len(result['static_updates']),0)
        self.assertEqual(len(result['future_partial_graph'].cell_columns['cell']),len(graph['cells']))


if __name__=='__main__':unittest.main()
