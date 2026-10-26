import ctypes as ct
import unittest
import numpy as np
import torch
from .range_normalization import NormalizedRangeKernel


class RoundedNormalization(unittest.TestCase):
    def test_million_random_vectors_and_rounding_boundaries(self):
        torch.set_num_threads(1);rng=np.random.default_rng(91815);kernel=NormalizedRangeKernel('cuda:1');v=rng.uniform(-10.,10.,(1000000,3));scale=10.**rng.uniform(-7,1,20000);extra=rng.normal(size=(20000,3));extra=extra/np.linalg.norm(extra,axis=1)[:,None]*scale[:,None]
        # Float midpoints in normalized x plus length-.015 cancellation.
        f=rng.uniform(-.99,.99,20000).astype(np.float32);mid=(f.astype(np.float64)+np.nextafter(f,np.float32(np.inf)).astype(np.float64))*.5
        boundary=np.column_stack([mid,np.zeros(len(mid)),np.sqrt(1-mid**2)]);boundary*=rng.choice([.015,1.,10.],len(mid))[:,None];v=np.concatenate([v,extra,boundary]);vectors=torch.from_numpy(v).to('cuda:1');old=torch.empty((len(v),4),device='cuda:1');new=torch.empty_like(old)
        kernel.module.launch('test_normalization',len(v),[vectors,ct.c_int(len(v)),old,new]);np.testing.assert_array_equal(old.cpu().numpy(),new.cpu().numpy())


if __name__=='__main__':unittest.main()
