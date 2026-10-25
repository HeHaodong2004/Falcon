import unittest
import numpy as np
import torch
from eventwm_graph.stage2_hf.spatial import SpatialNet
from .diagnose import OUT,read,save


class SpatialContext(unittest.TestCase):
    def test_added_context_is_initially_neutral(self):
        torch.manual_seed(1);a=SpatialNet(2).eval();b=SpatialNet(2,[2,4,8]).eval();b.load_state_dict(a.state_dict(),strict=False)
        maps=torch.randn(1,6,32,32);query=torch.randn(1,32,32,32);fine=torch.randn(1,3,64,64)
        with torch.no_grad():torch.testing.assert_close(a(maps,query,fine),b(maps,query,fine),rtol=0,atol=0)

    def test_actual_learned_map_receptive_fields(self):
        torch.set_num_threads(1);rows=[]
        for context in [None,[2,4,8]]:
            m=SpatialNet(1,context).double()
            with torch.no_grad():
                for p in m.parameters():p.fill_(.02)
            for phase in [128,129,130,131]:
                maps=torch.ones(1,6,128,128,dtype=torch.float64,requires_grad=True);query=torch.ones(1,32,128,128,dtype=torch.float64);fine=torch.ones(1,3,256,256,dtype=torch.float64)
                g=torch.autograd.grad(m(maps,query,fine)[0,0,phase,phase],maps)[0].abs().sum((0,1));at=torch.nonzero(g>0);span=(at.max(0).values-at.min(0).values+1).tolist();rows.append(dict(context=context,phase=phase,coarse_span=span,original_cell_span=[2*v for v in span]))
                self.assertEqual(span,[24,24]if context is None else[80,80])
        resolution=read(OUT/'trace.json')['examples'][0]['resolution'];save(OUT/'receptive_field.json',dict(actual_class='eventwm_graph.stage2_hf.spatial.SpatialNet',resolution_m=resolution,original_span_m=48*resolution,context_span_m=160*resolution,rows=rows,
            scope='Structural learned convolutional dependencies. EDT and line probes supply additional nonlocal summaries; this is not a trained effective receptive-field estimate.'))


if __name__=='__main__':unittest.main()
