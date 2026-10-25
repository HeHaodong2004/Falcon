import unittest
import numpy as np
import torch
from .common import OUT,read
from .time_features import moment_matrix,basis_numpy,basis_torch
from .time_model import TimeNet
from .packed import load_arrays
from eventwm_graph.stage2.data import source_rows
from eventwm_graph.stage2_segment.data import replay

class CausalTimeFeatures(unittest.TestCase):
 def test_dense_timestamp_replay_matches_reference(self):
  old=read(OUT/'temporal/pilot_fit_limit32/manifest.json');fast=read(OUT/'temporal/pilot_fit_limit64/manifest.json');lookup={r['root_id']:r for r in fast['rows']}
  for row in old['rows']:
   with np.load(row['path'],allow_pickle=False)as a,np.load(lookup[row['root_id']]['path'],allow_pickle=False)as b:
    np.testing.assert_array_equal(a['first'],b['first']);np.testing.assert_array_equal(a['last'],b['last'])
  rows,_=source_rows('train');ids={r['root_id']for r in old['rows'][:8]};source=[r for r in rows if r['root_id']in ids]
  for row in source:
   meta=read(OUT/'packed/pilot_fit'/(row['root_id']+'.json'));a=load_arrays(meta);mapping={e['start_node']:e['graph_index']for e in meta['segments']};seen=set()
   with np.load(lookup[row['root_id']]['path'],allow_pickle=False)as z:first=z['first'];last=z['last']
   for ex in replay(row,'mixed'):
    if'audit'in ex or ex['start_node']in seen:continue
    seen.add(ex['start_node']);gi=mapping[ex['start_node']];sl=slice(*a['goffsets'][gi:gi+2]);cells=ex['graph']['cells'];np.testing.assert_array_equal(a['gkeys'][sl],np.array([c['cell']for c in cells]));np.testing.assert_array_equal(first[sl],[c['first_seen']for c in cells]);np.testing.assert_array_equal(last[sl],[c['last_seen']for c in cells])
 def test_clock_shift_and_future_time_guard(self):
  keys=np.array([[1,0],[0,1],[-1,1]],np.int32);s=np.array([1,2,2],np.uint8);last=np.array([1.,1.048,1.096]);a=moment_matrix(keys,s,last,1.096,[0.,0.],.1);b=moment_matrix(keys,s,last+10000,10001.096,[0.,0.],.1);np.testing.assert_allclose(a,b,atol=1e-6,rtol=0)
  with self.assertRaises(ValueError):moment_matrix(keys,s,last+1,1.096,[0.,0.],.1)
 def test_neutral_prior_and_fused_time_projection(self):
  torch.manual_seed(1);m=TimeNet().eval();m.initialize();x=torch.randn(100,88);xy=x[:,20:22].numpy();bn=basis_numpy(xy);bt=basis_torch(x);np.testing.assert_allclose(bt.numpy(),bn,atol=2e-6,rtol=1e-6);matrix=torch.randn(7,32)*.1;extra=bt@matrix
  with torch.no_grad():
   p,_=m.probabilities(x,extra);old=m.prior(x)[:,:3].softmax(-1);torch.testing.assert_close(p,old,rtol=1e-5,atol=1e-6)
   m.correction[4].weight.uniform_(-.01,.01);a,_=m.probabilities(x,extra);b,_=m.probabilities(x,basis=bt,projection=matrix@m.correction[0].weight[:,88:].T);torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
if __name__=='__main__':unittest.main()
