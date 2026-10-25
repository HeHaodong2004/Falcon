"""Independent Reviewer A regressions: fitting-only data and legal synthetic cases.

Run through scripts/stage2_quality_run.py unittest discover -s tests/stage2_quality
-p test_scientific_review.py -v. No model fitting or dev/protected payload access.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest
import numpy as np
import torch
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.stage2.features import SceneFeatures
from eventwm_graph.stage2.data import branches,sampling
from eventwm_graph.stage2_quality.features import Features,PreparedGraph,opportunity
from eventwm_graph.stage2_quality.data import sample
from eventwm_graph.stage2_quality.train import masked
from eventwm_graph.stage2_quality.model import GraphDeltaNet
from eventwm_graph.stage2_quality.common import OUT,ORIGINAL,MODES


def make_graph(cells=None,resolution=.25,height=0.):
    g=PartialGraph(resolution,height)
    if cells is None:cells=[([0,0],1,0.)]
    updates=[dict(cell=k,state=state,height=h,timestamp=1.,source='measured') for k,state,h in cells]
    g.update(dict(timestamp=1.,robot_pose=[0.,height,0.,0.],static_updates=updates))
    return g.snapshot()


def full_map_boxes(graph,keys):
    """Direct sparse-set counts, independent from integral-image implementation."""
    free={tuple(c['cell']) for c in graph['cells'] if c['state']==1}
    blocked={tuple(c['cell']) for c in graph['cells'] if c['state']==2}
    output=[]
    for x,z in keys:
        values=[]
        for radius in (1,4,12):
            support={(a,b) for a in range(x-radius,x+radius+1) for b in range(z-radius,z+radius+1)}
            f=len(support&free)/len(support);b=len(support&blocked)/len(support)
            values.extend((f,b,1-f-b))
        for radius in (4,12):
            for sx,sz in ((-1,-1),(-1,1),(1,-1),(1,1)):
                xa=x-radius if sx<0 else x;xb=x+1 if sx<0 else x+radius+1
                za=z-radius if sz<0 else z;zb=z+1 if sz<0 else z+radius+1
                support={(a,b) for a in range(xa,xb) for b in range(za,zb)}
                values.extend((len(support&free)/len(support),len(support&blocked)/len(support)))
        output.append(values)
    return np.asarray(output,np.float32)


class IndependentScientificRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ORIGINAL/'output/l2_two_stage_20261023/final_collection_20261024/manifest.json'
        cls.manifest_path=path;cls.manifest=json.loads(path.read_text());cls.rows=cls.manifest['rows']
        # Deterministic fitting-only cases, all siblings from three Roots.
        cls.examples=[ex for index in (0,len(cls.rows)//2,len(cls.rows)-1) for ex in branches(cls.rows[index])]

    def test_intermediate_rotation_alias_is_resolved(self):
        g=make_graph();p=np.zeros((9,5),np.float64);p[:,0]=np.arange(1.,10.)
        keys=np.array([[-20,0],[20,0],[0,20]],np.int32)
        legacy=SceneFeatures(g,p);xy=(keys.astype(np.float32)+.5)*.25
        _,indices=legacy.tree.query(xy,k=4,workers=1)
        omitted=next(j for j in range(1,8) if j not in set(indices.ravel()))
        altered=p.copy();altered[omitted,4]=math.pi
        old_a=legacy.transform(keys)[0];old_b=SceneFeatures(g,altered).transform(keys)[0]
        np.testing.assert_array_equal(old_a,old_b)
        for mode in ('trajectory','both'):
            a=Features(g,p).transform(keys,mode)[0];b=Features(g,altered).transform(keys,mode)[0]
            self.assertFalse(np.array_equal(a,b),mode)
            self.assertGreater(float(b[0,48]),float(a[0,48])+1.9)  # maximum within-range facing
            np.testing.assert_array_equal(a[:,:16],0.)

    def test_sparse_map_alias_is_resolved(self):
        common=[([-1,0],1,0.),([1,0],2,0.)]
        a=make_graph(common);b=make_graph(common+[([0,2],2,0.)]);p=np.array([[1.,0.,0.,0.,0.],[2.,0.,0.,0.,.2]])
        keys=np.array([[0,0]],np.int32)
        np.testing.assert_array_equal(SceneFeatures(a,p).transform(keys)[0],SceneFeatures(b,p).transform(keys)[0])
        for mode in ('map','both'):
            aa=Features(a,p).transform(keys,mode)[0];bb=Features(b,p).transform(keys,mode)[0]
            self.assertFalse(np.array_equal(aa[:,67:92],bb[:,67:92]),mode)

    def test_map_fractions_preserve_unknown_and_global_legal_context(self):
        cells=[([-5,-2],1,-.2),([0,0],1,0.),([3,2],2,.2),([9,12],2,.1),([-13,3],1,.1)]
        graph=make_graph(cells);original=copy.deepcopy(graph);keys=np.array([[0,1],[-14,3],[20,20]],np.int32)
        p=PreparedGraph(graph);actual=p.map_features(keys);expected=full_map_boxes(graph,keys)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=6e-8)
        for off in (0,3,6):np.testing.assert_allclose(actual[:,off:off+3].sum(1),1.,rtol=0,atol=6e-8)
        self.assertEqual(graph,original)
        empty=PreparedGraph(make_graph([])).map_features(keys)
        np.testing.assert_array_equal(empty[:,[0,1,3,4,6,7]],0.)
        np.testing.assert_array_equal(empty[:,[2,5,8]],1.)

    def test_factual_candidate_domains_and_common_features_match_legacy(self):
        for ex in self.examples:
            old=SceneFeatures(ex['graph'],ex['trajectory']);reference=old.candidates()
            old_x,old_h=old.transform(reference)
            for mode in MODES:
                new=Features(ex['graph'],ex['trajectory']);keys=new.candidates(mode)
                np.testing.assert_array_equal(keys,reference,err_msg=mode)
                # All cells verifies both common columns and baseline47, not just sampled rows.
                x,h=new.transform(keys,mode)
                np.testing.assert_array_equal(h,old_h,err_msg=mode)
                np.testing.assert_array_equal(x[:,16:47],old_x[:,16:47],err_msg=mode)
                if mode in ('baseline','map'):np.testing.assert_array_equal(x[:,:47],old_x,err_msg=mode)

    def test_float32_candidate_boundary_matches_reference(self):
        cases=[(.25,0.,0.),(.2,1024.125,-200.25),(.1,-11.000001,.0000003),(.037,19.125,8.75)]
        for resolution,x,z in cases:
            graph=make_graph([],resolution)
            p=np.array([[1.,x,0.,z,0.],[2.,x+.125,0.,z-.03125,1.2]])
            old=SceneFeatures(graph,p);want=old.candidates();new=Features(graph,p)
            np.testing.assert_array_equal(new.candidates('trajectory'),want)

    def test_actual_radius_boundary_float32_subtraction_regression(self):
        path=Path(__file__).resolve().parents[2]/'docs/l2/stage2_quality_runtime_20261024/reviews/reviewer_a/candidate_boundary_failure.json'
        case=json.loads(path.read_text());trajectory=np.asarray(case['trajectory'],np.float64)
        graph=make_graph([],case['resolution']);graph['timestamp']=float(trajectory[0,0])
        keys=np.asarray(case['bad_keys'],np.int32)
        keep,_=opportunity(keys,trajectory,case['resolution'])
        self.assertTrue(np.all(keep),'Actual originally eligible cell just below11m must remain included')
        reference=SceneFeatures(graph,trajectory).candidates()
        for mode in ('trajectory','both'):
            np.testing.assert_array_equal(Features(graph,trajectory).candidates(mode),reference)

    def test_training_sample_weights_targets_and_height_match_frozen_logic(self):
        for ex in self.examples:
            old,old_meta=sampling(ex)
            new,new_meta=sample(ex,Features(ex['graph'],ex['trajectory']))
            if old is None:self.assertIsNone(new);continue
            np.testing.assert_array_equal(new['x'][:,:47],old['x'])
            for field in ('y','height','weight'):np.testing.assert_array_equal(new[field],old[field],err_msg=field)
            self.assertEqual(new_meta['candidate_cells'],old_meta['candidate_cells'])
            self.assertEqual(new_meta['out_of_domain_positive'],old_meta['out_of_domain_positive'])
            self.assertEqual(new_meta['queries_in_root'],ex['queries_in_root'])
            self.assertAlmostEqual(float(new['weight'].sum()),1/ex['queries_in_root'],places=6)

    def test_trajectory_clock_end_duration_and_full_path_features(self):
        g=make_graph();p=np.array([[1.,0.,0.,0.,0.],[1.1,0.,0.,0.,math.pi/2],[3.,0.,0.,0.,math.pi]])
        f=Features(g,p);keys=np.array([[5,0]],np.int32);_,x=opportunity(keys,p,.25)
        self.assertAlmostEqual(float(x[0,16]),.5,places=6)  # total rotation / 2pi
        self.assertAlmostEqual(float(x[0,18]),1.,places=6)  # all elapsed time stationary
        self.assertEqual(f.duration,2.)
        for bad in (p[::-1],p[[0,0]],p[1:],p*np.nan):
            with self.assertRaises(ValueError):Features(g,bad)
        # Initial-only pose is legal, contributes no future-exposure counts.
        _,zero=opportunity(keys,p[:1],.25)
        np.testing.assert_array_equal(zero[:,6:12],0.)
        self.assertEqual(float(zero[0,19]),0.)

    def test_fixed_masks_and_same_network_capacity(self):
        original=np.arange(3*92,dtype=np.float32).reshape(3,92);before=original.copy()
        for mode in MODES:
            x=masked(original,mode);np.testing.assert_array_equal(x[:,16:47],original[:,16:47])
            if mode in ('trajectory','both'):np.testing.assert_array_equal(x[:,:16],0.)
            else:np.testing.assert_array_equal(x[:,47:67],0.)
            if mode in ('baseline','trajectory'):np.testing.assert_array_equal(x[:,67:92],0.)
        np.testing.assert_array_equal(before,original)
        net=GraphDeltaNet();self.assertEqual(sum(p.numel() for p in net.parameters()),6164)
        logits,height=net(torch.zeros(3,92));self.assertEqual(tuple(logits.shape),(3,3));self.assertEqual(tuple(height.shape),(3,))
        np.testing.assert_allclose(torch.softmax(logits,-1).detach().numpy().sum(1),1.,atol=1e-7)

    def test_fitting_only_scene_holdout_protocol_is_fixed(self):
        path=OUT/'protocol.json';protocol=json.loads(path.read_text())
        self.assertEqual(protocol['source_manifest_sha256'],hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())
        by_root={r['root_id']:r for r in self.rows};fit=set(protocol['pilot_fit_roots']);hold=set(protocol['pilot_holdout_roots'])
        self.assertFalse(fit&hold);self.assertEqual(len(fit),4096)
        fit_scenes={by_root[k]['scene_id'] for k in fit};hold_scenes={by_root[k]['scene_id'] for k in hold}
        self.assertFalse(fit_scenes&hold_scenes);self.assertEqual(hold_scenes,set(protocol['pilot_holdout_scenes']))
        self.assertEqual(len(hold_scenes),80);self.assertLessEqual(len(hold),320)
        for k in fit|hold:self.assertEqual(by_root[k]['split'],'train');self.assertEqual(by_root[k]['source_split'],'train')
        self.assertFalse(protocol['original_dev_tuning']);self.assertEqual(protocol['full_data_fits_max'],1)
        self.assertEqual(protocol['experiments'],list(MODES))


if __name__=='__main__':unittest.main()
