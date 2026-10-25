"""Independent checks for bounded execution-only final refinements."""
import copy,json,math,unittest
from collections.abc import Mapping
import numpy as np
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.stage2.api import graph_identity as reference_identity,apply_graph_delta as reference_apply
from eventwm_graph.stage2.common import SCHEMA
from eventwm_graph.stage2_quality.geometry import graph_identity
from eventwm_graph.stage2_quality.features import opportunity,inside_descriptors,candidate_mask
from eventwm_graph.stage2_quality.update import apply_graph_delta
from eventwm_graph.stage2_quality.common import BASE


def assert_tree(case,a,b):
    if isinstance(a,np.ndarray):
        case.assertIsInstance(b,np.ndarray);np.testing.assert_array_equal(a,b);case.assertEqual(a.dtype,b.dtype)
    elif isinstance(a,Mapping):
        case.assertEqual(set(a),set(b))
        for key in a:assert_tree(case,a[key],b[key])
    elif isinstance(a,(tuple,list)):
        case.assertEqual(type(a),type(b));case.assertEqual(len(a),len(b))
        for aa,bb in zip(a,b):assert_tree(case,aa,bb)
    else:case.assertEqual(a,b)


class FinalRuntimeRefinementReview(unittest.TestCase):
    def test_fast_graph_identity_exact_digest(self):
        rng=np.random.default_rng(825)
        for count in (0,1,2,10,100,512):
            for repeat in range(10):
                keys=rng.integers(-100000,100000,(count,2),dtype=np.int64)
                cells=[dict(cell=key.tolist(),state=int(rng.integers(1,3)),height=float(rng.normal())) for key in keys]
                graph=dict(cells=cells,timestamp=2.3,revision=repeat,resolution=.1016884229464652,height=-.01,clearance=.25)
                self.assertEqual(reference_identity(graph),graph_identity(graph))
                rng.shuffle(cells);self.assertEqual(reference_identity(graph),graph_identity(graph))
        for keys in ([[2**54,0],[2**54+1,0],[-2**54,1]],[[2**63,0],[0,0]],[[.5,1],[0,1]],[[-0.,0.],[0.,0.]]):
            graph=dict(cells=[dict(cell=k,state=1,height=(-0. if i%2 else 0.)) for i,k in enumerate(keys)],timestamp=1.,revision=0,resolution=.25,height=0.,clearance=.25)
            self.assertEqual(reference_identity(graph),graph_identity(graph))
        graph=dict(cells=[dict(cell=[0,0],state=s,height=h) for s,h in [(2,1.),(1,2.),(1,-0.),(1,0.)]],timestamp=1.,revision=2,resolution=.25,height=-0.,clearance=.25)
        self.assertEqual(reference_identity(graph),graph_identity(graph))

    def test_split_descriptors_preserve_all20_values(self):
        path=BASE/'docs/l2/stage2_quality_runtime_20261024/reviews/reviewer_a/candidate_boundary_failure.json'
        case=json.loads(path.read_text());cases=[]
        key=np.asarray(case['bad_keys'][0],np.int32)
        keys=np.asarray([key+[x,z] for x in range(-8,9) for z in range(-8,9)],np.int32)
        cases.append((keys,np.asarray(case['trajectory'],np.float64),case['resolution']))
        rng=np.random.default_rng(826)
        for i in range(20):
            r=(.02,.1,.25,1.)[i%4];p=np.zeros((1+i%9,5),np.float64);p[:,0]=np.arange(len(p))*.17+1
            p[:,[1,3]]=rng.uniform(-3,3,(len(p),2));p[:,4]=rng.uniform(-math.pi,math.pi,len(p))
            keys=rng.integers(-int(15/r),int(15/r)+1,(800,2),dtype=np.int32);cases.append((keys,p,r))
        # Full precision positions differ while float32 support positions tie.
        # Put a candidate at exactly10m so sub-float32 translation changes
        # observed-opportunity eligibility and must not be silently cached away.
        resolution=.1;focus=np.array([[100,0],[50,50],[-50,0]],np.int32)
        center=(focus[0].astype(np.float32)+.5)*resolution
        repeated=np.zeros((11,5),np.float64);repeated[:,0]=1.+np.cumsum(np.r_[0.,np.linspace(.013,.19,10)])
        repeated[:,1]=float(center[0])-10.+np.array([0.,1e-10,1e-10,-1e-10,0.,0.,.3,.3+1e-10,.3,0.,0.])
        repeated[:,3]=float(center[1]);repeated[:,4]=np.linspace(-math.pi,math.pi,11)
        cases.append((focus,repeated,resolution))
        stationary=repeated.copy();stationary[:,1:4]=repeated[0,1:4]
        stationary[:,4]=np.array([0.,.2,1.,2.,3.,-3.,-2.,-1.,.1,.7,0.])
        cases.append((focus,stationary,resolution))
        shared=0
        for keys,path,resolution in cases:
            keep,reference=opportunity(keys,path,resolution)
            xy=path[:,[1,3]].astype(np.float32)
            np.testing.assert_array_equal(candidate_mask(keys,xy,resolution),keep)
            np.testing.assert_array_equal(candidate_mask(keys,np.unique(xy,axis=0),resolution),keep)
            accepted,actual=inside_descriptors(keys[keep],path,resolution)
            self.assertTrue(np.all(accepted));np.testing.assert_array_equal(reference[keep],actual);shared+=int(keep.sum())
        self.assertGreater(shared,5000)

    def test_snapshot_transfer_preserves_extension_isolation(self):
        p=PartialGraph(1.,0.);p.update(dict(timestamp=1.,robot_pose=[0.,0.,0.,0.],static_updates=[dict(cell=[x,0],state=1,height=0.,timestamp=1.,source='measured') for x in (0,1)]));graph=p.snapshot()
        shared=dict(array=np.array([1.,2.],np.float32),nested=[dict(items=[3,4])])
        for cell in graph['cells']:cell['extension']=shared
        before=copy.deepcopy(graph)
        delta=dict(schema=SCHEMA,predicted=True,base_graph_identity=reference_identity(graph),timestamp=2.,robot_pose=[.8,0.,0.,.2],checkpoint_sha256='synthetic',static_updates=[dict(cell=[2,0],state=1,height=0.,timestamp=2.,source='predicted_stage2_research')])
        a=reference_apply(graph,delta);b=apply_graph_delta(graph,delta);assert_tree(self,a,b);assert_tree(self,graph,before)
        old=[c for c in b['cells'] if 'extension' in c]
        self.assertIsNot(old[0]['extension'],old[1]['extension'])
        self.assertFalse(np.shares_memory(old[0]['extension']['array'],shared['array']))
        old[0]['extension']['array'][0]=-99.;old[0]['extension']['nested'][0]['items'][0]=-99
        np.testing.assert_array_equal(shared['array'],[1.,2.]);self.assertEqual(shared['nested'][0]['items'],[3,4]);np.testing.assert_array_equal(old[1]['extension']['array'],[1.,2.])
        self.assertEqual(old[1]['extension']['nested'][0]['items'],[3,4]);assert_tree(self,graph,before)


if __name__=='__main__':unittest.main()
