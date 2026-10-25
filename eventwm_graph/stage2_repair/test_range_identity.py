import copy
import unittest
from unittest.mock import patch
import numpy as np
from .range_identity import identity
from .legal_camera_features import parameters as scalar,cell_features
from .camera_vectorized import parameters as vectorized


class InputIdentity(unittest.TestCase):
    def test_every_effective_geometry_input_invalidates_cache(self):
        arrays=[np.ones((2,4,4),np.float32),np.zeros((2,3)),np.repeat(np.eye(3)[None],2,0),np.array([1.,2.])];tracks=[[dict(position=[0,0,0],valid=True)],[]]
        def key(a=arrays,t=tracks,r=.1,stride=1):return identity(*a,t,r,stride)
        first=key()
        for i in range(4):
            changed=[x.copy()for x in arrays];changed[i].flat[0]+=.01;self.assertNotEqual(first,key(changed))
        changed=copy.deepcopy(tracks);changed[0][0]['position'][0]+=.01;self.assertNotEqual(first,key(t=changed));self.assertNotEqual(first,key(r=.2));self.assertNotEqual(first,key(stride=2))
        unused=copy.deepcopy(tracks);unused[0][0]['confidence']=.2;self.assertEqual(first,key(t=unused))

    def test_fallback_handles_zero_tracks(self):
        from . import range_identity
        with patch.object(range_identity,'LIB',None):
            value=identity(np.ones((1,2,2),np.float32),np.zeros((1,3)),np.eye(3)[None],np.zeros(1),[[]],.1,1)
        self.assertTrue(value.startswith('sha256_v1:'))

    def test_vectorized_planes_and_actual_feature_values_match(self):
        rng=np.random.default_rng(389);p=np.column_stack([np.arange(50)*.048,rng.normal(size=(50,3)),rng.uniform(-3,3,50)]);p[:,2]=0.;translation=np.array([.5,.7,.01]);rotation=np.array([[0,0,-1],[0,1,0],[1,0,0]],float);a,s=scalar(p,translation,rotation,.1016884210526);b,t=vectorized(p,translation,rotation,.1016884210526);np.testing.assert_allclose(a,b,rtol=1.e-14,atol=1.e-14);keys=rng.integers(-50,50,(5000,2),dtype=np.int32);x=np.zeros((len(keys),88),np.float32);np.testing.assert_allclose(cell_features(keys,p,.1016884210526,x,a),cell_features(keys,p,.1016884210526,x,b),rtol=1.e-6,atol=1.e-7)


if __name__=='__main__':unittest.main()
