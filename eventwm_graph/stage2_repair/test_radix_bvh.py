"""Independent ray intersections and real complete-API regression checks."""
import ctypes as ct
import unittest
import numpy as np
import torch
from .radix_bvh import build_arrays
from .geometry_cuda import GeometryKernel


def reference_rays(triangles,origins,directions,limit):
    a=triangles[:,0].astype(np.float64);e=triangles[:,1]-a;f=triangles[:,2]-a
    result=[]
    for origin,direction in zip(origins,directions):
        p=np.cross(direction,f);det=(e*p).sum(1)
        good=np.abs(det)>=1.e-20
        inv=np.divide(1.,det,out=np.zeros_like(det),where=good)
        delta=origin-a;u=(delta*p).sum(1)*inv;q=np.cross(delta,e)
        v=(direction*q).sum(1)*inv;h=(f*q).sum(1)*inv
        good&=(u>=-.0001)&(u<=1.0001)&(v>=-.0001)&(u+v<=1.0001)&(h>=0)&(h<limit)
        result.append(h[good].min()if good.any()else limit)
    return np.array(result)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class RadixTest(unittest.TestCase):
    def test_topology_bounds_and_independent_intersections(self):
        torch.set_num_threads(1);rng=np.random.default_rng(7721)
        for n in [1,12,13,73,512]:
            vertices=rng.normal(size=(n,3,3)).astype(np.float32)
            if n==512:
                # Coincident centroids force the unique leaf-index suffix.
                vertices[:300]-=vertices[:300].mean(1,keepdims=True)
            packed=np.concatenate([vertices[:,0],vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]],1)
            arrays=build_arrays(torch.from_numpy(packed).to('cuda:1'))
            kernel=GeometryKernel(arrays,.1,'cuda:1')
            host={k:v.cpu().numpy()for k,v in arrays.items()}
            queue=[0];visited=[]
            while queue:
                node=queue.pop();self.assertNotIn(node,visited);visited.append(node)
                if host['count'][node]==0:
                    a,b=host['left'][node],host['right'][node];queue.extend([a,b])
                    for child in [a,b]:
                        self.assertTrue(np.all(host['bounds'][node,:3]<=host['bounds'][child,:3]))
                        self.assertTrue(np.all(host['bounds'][node,3:]>=host['bounds'][child,3:]))
            self.assertEqual(len(visited),len(host['bounds']))
            self.assertEqual(int(host['count'].sum()),n)
            origins=rng.normal(size=(800,3)).astype(np.float32)*2
            directions=rng.normal(size=(800,3)).astype(np.float32)
            directions/=np.linalg.norm(directions,axis=1)[:,None]
            output=torch.empty(len(origins),device='cuda:1')
            kernel.module.launch('test_rays',len(origins),[kernel.tensor(origins),kernel.tensor(directions),kernel.tensor(np.full(len(origins),10.,np.float32)),ct.c_int(len(origins)),output,*kernel.mesh])
            # Compare the packed F32 vertices actually consumed by CUDA.
            t=host['triangles'];actual=np.stack([t[:,:3],t[:,:3]+t[:,3:6],t[:,:3]+t[:,6:9]],1)
            np.testing.assert_allclose(output.cpu().numpy(),reference_rays(actual,origins,directions,10.),atol=3.e-5,rtol=3.e-5)

    def test_complete_api_cold_warm_and_single_batch(self):
        from .diagnose import OUT
        from .range_benchmark import workloads
        from .range_fast_proxy import FastRangePredictor
        from .range_radix_api import RadixRangePredictor
        checkpoint=OUT/'training/range_evidence64/selected.pt'
        old=FastRangePredictor(checkpoint,device='cuda:1')
        new=RadixRangePredictor(checkpoint,device='cuda:1')
        for index in [0,20]:
            ex,packet,tracks=workloads()[index]
            paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
            kw=dict(partial_graph=ex['graph'],range_observations=packet,track_history=tracks)
            baseline=old.predict_segments(trajectories=paths,**kw)
            cold=new.predict_segments(trajectories=paths,**kw)
            warm=new.predict_segments(trajectories=paths,**kw)
            for path,a,b,c in zip(paths,baseline,cold,warm):
                single=new.predict(trajectory=path,**kw)
                for result in [b,c,single]:
                    np.testing.assert_array_equal(a['candidate_cells'],result['candidate_cells'])
                    np.testing.assert_array_equal(a['states'],result['states'])
                    np.testing.assert_allclose(a['probabilities'],result['probabilities'],atol=3.e-6,rtol=3.e-5)
                    np.testing.assert_array_equal(a['future_partial_graph'].cell_columns['cell'],result['future_partial_graph'].cell_columns['cell'])


if __name__=='__main__':unittest.main()
