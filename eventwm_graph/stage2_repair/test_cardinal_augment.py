import copy
import unittest
import numpy as np
from .cardinal_augment import matrix,cell_keys,item_transform
from .legal_camera import base_rotations,proxy_cameras
from eventwm_graph.stage2_quality.features import candidate_mask


class CardinalAugmentation(unittest.TestCase):
    def test_negative_cell_identity_and_domain_invariance(self):
        rng=np.random.default_rng(8241);keys=rng.integers(-10000,10000,(20000,2),dtype=np.int32);xy=rng.uniform(-10,10,(17,2)).astype(np.float32);r=.1016884229464652
        for turns in range(4):
            transform=matrix(turns)[np.ix_([0,2],[0,2])];new=cell_keys(keys,turns);centres=(keys.astype(np.float64)+.5)*r;np.testing.assert_array_equal(new,np.floor((centres@transform.T)/r).astype(np.int32));np.testing.assert_array_equal(cell_keys(new,4-turns),keys);np.testing.assert_array_equal(candidate_mask(keys,xy,r),candidate_mask(new,(xy@transform.T).astype(np.float32),r))

    def test_inputs_targets_camera_rig_and_original_item_are_preserved(self):
        p=np.array([[0.,-.2,.1,.3,.72],[1.,.3,.1,.5,1.21]]);origin=np.array([.1,1.3,.4]);rotation=base_rotations(np.array([.83]))[0];origins,rotations=proxy_cameras(p,origin,rotation)
        item=dict(keys=np.array([[-1,-1],[0,1]],np.int32),ego=p[0,1:4],windows=[dict(path=p,target_keys=np.array([[2,-4]],np.int32),target_states=np.array([2],np.uint8))],packet=dict(ranges=np.ones((1,2,2),np.float32),origins=origin[None],rotations=rotation[None],timestamps=np.array([0.])),tracks=[[dict(position=[.1,0.,.2],estimated_velocity=[1.,0.,.5])]])
        before=copy.deepcopy(item)
        for turns in [1,2,3]:
            result=item_transform(item,turns);r=matrix(turns);new_o,new_r=proxy_cameras(result['windows'][0]['path'],result['packet']['origins'][-1],result['packet']['rotations'][-1]);np.testing.assert_allclose(new_o,origins@r.T,atol=1.e-12);np.testing.assert_allclose(new_r,r@rotations,atol=1.e-12);np.testing.assert_array_equal(result['windows'][0]['target_states'],item['windows'][0]['target_states']);np.testing.assert_array_equal(result['packet']['ranges'],item['packet']['ranges']);self.assertFalse(set(map(tuple,result['keys']))&set(map(tuple,result['windows'][0]['target_keys'])))
        np.testing.assert_array_equal(item['keys'],before['keys']);np.testing.assert_array_equal(item['windows'][0]['path'],before['windows'][0]['path']);self.assertEqual(item['tracks'],before['tracks'])


if __name__=='__main__':unittest.main()
