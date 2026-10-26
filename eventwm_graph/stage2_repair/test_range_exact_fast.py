import ctypes as ct
import unittest
import numpy as np
import torch
from .range_exact_fast import FastExactKernel
from .diagnose import OUT
from .range_benchmark import workloads
from .range_radix_api import RadixExactSpatialPredictor


class ExactFrustumTest(unittest.TestCase):
    def test_random_and_numerical_boundary_cases(self):
        torch.set_num_threads(1);kernel=FastExactKernel('cuda:1');rng=np.random.default_rng(3917)
        cameras=[];points=[];ids=[];resolution=.1016884229464652;radius=.25+resolution/np.sqrt(2)
        for i,yaw in enumerate([0.,np.pi/2,np.pi,-np.pi/2,.147,.92]):
            c,s=np.cos(yaw),np.sin(yaw);r=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            origin=np.array([1000.*i,1.2,-700.*i]);cameras.append(np.r_[origin,r.ravel(),0.])
            p=rng.uniform([-12.,-.2,-12.],[12.,.2,12.],(20000,3));p[:,[0,2]]+=origin[[0,2]]
            points.append(p);ids.extend([i]*len(p))
        # Touch a halo/prism side plane and a sphere boundary, with tiny
        # perturbations. Large translations exercise cancellation bounds.
        for translation in [np.array([0.,0.,0.]),np.array([1.e6,0.,-1.e6])]:
            i=len(cameras);origin=translation+[0.,1.2,0.];cameras.append(np.r_[origin,np.eye(3).ravel(),0.])
            base=[np.array([2.-radius*np.sqrt(2),0.,-2.]),np.array([0.,0.,-(np.sqrt(100-1.175**2)-radius)])]
            p=[]
            for centre in base:
                for delta in [0.,-1.e-12,1.e-12,-1.e-9,1.e-9,-1.e-6,1.e-6]:p.append(centre+translation+np.array([delta,0.,delta]))
            points.append(np.array(p));ids.extend([i]*len(p))
        def tensor(a,dtype):return torch.from_numpy(np.ascontiguousarray(a,dtype=dtype)).to('cuda:1')
        cameras=tensor(cameras,np.float64);points=tensor(np.concatenate(points),np.float64);ids=tensor(ids,np.int32)
        planes=torch.empty((len(cameras),20),device='cuda:1',dtype=torch.float64);offsets=torch.empty((63,3),device='cuda:1',dtype=torch.float64)
        kernel.module.launch('prepare_certificate',len(cameras),[cameras,ct.c_int(len(cameras)),ct.c_double(resolution),planes,offsets])
        result=torch.empty((len(points),2),device='cuda:1',dtype=torch.uint8)
        kernel.module.launch('test_certificate',len(points),[points,ids,ct.c_int(len(points)),cameras,planes,offsets,ct.c_double(resolution),result])
        values=result.cpu().numpy();np.testing.assert_array_equal(values[:,0],values[:,1])

    def test_real_complete_api_preserves_states_probabilities_and_heights(self):
        model=RadixExactSpatialPredictor(OUT/'training/support_pretrained256_control/selected.pt',device='cuda:1')
        model.config['observed_obstacle_schema']='capped_2m_complete';reference=model.range_kernel;fast=FastExactKernel('cuda:1')
        for index in [0,20,31]:
            ex,packet,tracks=workloads()[index];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
            def call():return model.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
            model.range_kernel=reference;before=call();model.range_kernel=fast;after=call()
            for a,b in zip(before,after):
                np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_array_equal(a['heights'],b['heights'])
                np.testing.assert_allclose(a['probabilities'],b['probabilities'],rtol=3.e-5,atol=3.e-6)
                np.testing.assert_array_equal(a['future_partial_graph'].cell_columns['cell'],b['future_partial_graph'].cell_columns['cell'])


if __name__=='__main__':unittest.main()
