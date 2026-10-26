"""Every input used by the shared map canvas belongs in its cache identity."""
from collections import OrderedDict
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from .range_spatial_api import SpatialRangePredictor
from .range_device_canvas import DeviceCanvasSpatialPredictor


class SpatialCachePose(unittest.TestCase):
    def test_start_position_and_height_are_not_cached_from_another_input(self):
        for cls,device_canvas in [(SpatialRangePredictor,False),(DeviceCanvasSpatialPredictor,True)]:
            calls=[]
            def build(keys,states,heights,ego,*rest):
                calls.append(np.array(ego));a=np.full((14,2,2),ego[0]+ego[1],np.float16);return(torch.from_numpy(a)if device_canvas else a),np.array([0,0])
            mesh=object();graph=SimpleNamespace(identity='same-legal-map',keys=np.empty((0,2),np.int32),values=np.empty(0,np.uint8),heights=np.empty(0,np.float32),resolution=.1)
            model=SimpleNamespace(mesh_cache=OrderedDict([('same-range-history',mesh)]),active_mesh=mesh,active_graph=graph,spatial_cache=OrderedDict(),canvas=SimpleNamespace(build=build),model=SimpleNamespace(encode=lambda x:x),device=torch.device('cpu'),current_range_observations=dict(ranges=np.zeros((1,2,2),np.float32),origins=np.zeros((1,3)),rotations=np.eye(3)[None],timestamps=np.array([0.])),current_track_history=[[]])
            def request(x,y):return[(np.empty((0,2),np.int32),None,np.array([[0.,x,y,0.,0.]]))]
            a,_=cls._encoded_map(model,request(0.,0.));b,_=cls._encoded_map(model,request(.25,0.));c,_=cls._encoded_map(model,request(.25,.1));again,_=cls._encoded_map(model,request(.25,.1))
            self.assertEqual(len(calls),3);self.assertFalse(torch.equal(a,b));self.assertFalse(torch.equal(b,c));self.assertTrue(torch.equal(c,again))


if __name__=='__main__':unittest.main()
