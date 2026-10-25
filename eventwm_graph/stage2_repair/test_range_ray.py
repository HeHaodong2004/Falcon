import unittest
import numpy as np
import torch
from .range_ray_model import RayEvidenceNet
from .range_ray_data import camera_lines


class RayContextTest(unittest.TestCase):
    def test_world_axes_bilinear_sampling_and_gradient(self):
        x,z=torch.meshgrid(torch.arange(192),torch.arange(192),indexing='ij')
        field=(x+100*z).float()[None,None].expand(1,8,-1,-1).clone().requires_grad_()
        indices=torch.tensor([[100,120],[140,150]])
        origins=torch.tensor([[[40.,50.],[60.,70.],[80.,90.]]]*2)
        result=RayEvidenceNet.pool_rays(field,indices,torch.zeros(2,dtype=torch.long),origins)
        expected=[]
        for endpoint,lines in zip(indices.numpy()+.5,origins.numpy()):
            values=[]
            for begin in lines:
                points=begin+(endpoint-begin)*np.arange(1,17)[:,None]/17
                v=(points[:,0]/2-.5)+100*(points[:,1]/2-.5)
                values.extend([np.full(8,v.mean()),np.full(8,v.max())])
            expected.append(np.concatenate(values))
        np.testing.assert_allclose(result.detach().numpy(),expected,atol=.001,rtol=1.e-6)
        result.sum().backward();self.assertGreater(float(field.grad.abs().sum()),0.)

    def test_supplied_pose_camera_lines_include_rotation_and_translation(self):
        p=np.array([[0.,0.,0.,0.,0.],[1.,1.,0.,2.,np.pi/2],[2.,2.,0.,3.,np.pi]])
        translation=np.array([.2,1.2,.1]);actual=camera_lines(p,translation,.1,np.array([-5,-7]))
        from .legal_camera import base_rotations
        expected=(p[[1,1,2],1:4]+base_rotations(p[[1,1,2],4])@translation)[:,[0,2]]/.1-[-5,-7]
        np.testing.assert_allclose(actual,expected,atol=1.e-5)

    def test_no_future_truth_or_measured_mesh_feature_is_consumed(self):
        model=RayEvidenceNet();torch.manual_seed(31)
        torch.nn.init.normal_(model.head[-1].weight,std=.05)
        torch.nn.init.normal_(model.head[-1].bias,std=.05)
        encoded=(torch.randn(1,24,192,192),torch.randn(1,16),torch.randn(1,8,192,192))
        x=torch.randn(9,88);extra=torch.randn(9,16);indices=torch.full((9,2),100);ids=torch.zeros(9,dtype=torch.long);origins=torch.full((9,3,2),80.)
        with torch.inference_mode():a=model.raw(x,extra,indices,ids,encoded,origins)
        extra[:,:8]+=10000
        with torch.inference_mode():b=model.raw(x,extra,indices,ids,encoded,origins)
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
        extra[:,8]+=10
        with torch.inference_mode():c=model.raw(x,extra,indices,ids,encoded,origins)
        self.assertGreater(float((c[0]-a[0]).abs().max()),1.e-6)


if __name__=='__main__':unittest.main()
