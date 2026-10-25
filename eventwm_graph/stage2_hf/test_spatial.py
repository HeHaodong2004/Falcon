import unittest
import numpy as np
import torch
from .packed import load_arrays,get_segment
from .spatial import build,SpatialNet,spatial_probabilities
from .common import OUT,read
from eventwm_graph.stage2_quality.features import Features

class SpatialCorrectness(unittest.TestCase):
 def test_full_candidate_domain_and_targets(self):
  row=read(OUT/'packed/pilot_fit/manifest.json')['rows'][0];a=load_arrays(row)
  for i in [0,len(row['segments'])//2,len(row['segments'])-1]:
   k,s,h,p,tk,ts,th,geometry=get_segment(a,i);data=build(k,s,h,p,geometry,tk,ts,th)
   graph=dict(schema='causal_partial_cspace_graph_1',resolution=geometry[0],height=geometry[1],timestamp=p[0,0],revision=0,clearance=geometry[2],cells=[dict(cell=q.tolist(),state=int(v),height=float(z))for q,v,z in zip(k,s,h)])
   ref=Features(graph,p).candidates('both');np.testing.assert_array_equal(ref,data['keys']);ci=ref-data['lo'];self.assertEqual(int(data['candidate'].sum()),len(ref));self.assertEqual(int(np.count_nonzero(data['labels'])),len(tk));self.assertTrue(np.isfinite(data['query']).all())
 def test_cuda_query_matches_cpu_covariates(self):
  from .spatial_cuda import QueryKernel,path_arrays
  row=read(OUT/'packed/pilot_fit/manifest.json')['rows'][1];a=load_arrays(row);v=get_segment(a,0);d=build(*v[:4],v[-1],*v[4:7]);device='cuda:1';kernel=QueryKernel(device)
  inputs=path_arrays([d['path']],[d['lo']],[d['shape']],[d['resolution']]);t={k:torch.from_numpy(x).to(device)for k,x in inputs.items()};out=kernel(torch.from_numpy(d['maps'][None]).to(device),t['path'],t['pre'],t['globals'],t['spec']).cpu().numpy()[0]
  np.testing.assert_allclose(out,d['query'],atol=2e-6,rtol=0)
 def test_probabilities_and_shared_map(self):
  torch.manual_seed(1);m=SpatialNet(4).eval();a=torch.randn(1,6,32,40);q=torch.randn(2,32,32,40);f=torch.randn(2,3,64,80)
  with torch.no_grad():
   together=m(a.expand(2,-1,-1,-1),q,f,encoded=m.encode_map(a).expand(2,-1,-1,-1));separate=torch.cat([m(a,q[i:i+1],f[i:i+1])for i in range(2)])
  torch.testing.assert_close(together,separate,rtol=2e-5,atol=2e-6);p=spatial_probabilities(together);torch.testing.assert_close(p.sum(1),torch.ones_like(p[:,0]));self.assertTrue(bool((p>=0).all()))
if __name__=='__main__':unittest.main()
