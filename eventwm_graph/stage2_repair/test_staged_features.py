"""Staging order must preserve exact GPU results and the CPU reference."""
import subprocess
import types
from pathlib import Path
import unittest
import numpy as np
import torch
from .diagnose import OUT
from .range_benchmark import workloads
from .range_exact_api import ExactFastSpatialPredictor
from .global_features import GlobalFeatures
from .features import transform_batch
from eventwm_graph.stage2_segment.fused import transform


class StagedFeatureParity(unittest.TestCase):
    def test_old_launch_order_and_independent_cpu_extraction(self):
        torch.set_num_threads(1);root=Path(__file__).resolve().parents[2]
        code=subprocess.check_output(['git','show','39cde293:eventwm_graph/stage2_repair/features.py'],cwd=root,text=True);old=types.ModuleType('eventwm_graph.stage2_repair.frozen_feature_reference');exec(compile(code,'git:39cde293/features.py','exec'),old.__dict__)
        m=ExactFastSpatialPredictor(OUT/'training/support_pretrained256_control/selected.pt',device='cuda:1')
        for index in [0,18,20,30,31]:
            ex,packet,tracks=workloads()[index];prepared=m.prepare_graph(ex['graph']);paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]];contexts=[GlobalFeatures(ex['graph'],p,prepared)for p in paths];keysets=[]
            for context in contexts:
                k=context.candidates();keysets.append(k[np.linspace(0,len(k)-1,min(512,len(k)),dtype=int)])
            a,ah,_=old.transform_batch(m.cuda,contexts,keysets);b,bh,_=transform_batch(m.cuda,contexts,keysets)
            for context,keys,x,y,h,hh in zip(contexts,keysets,a,b,ah,bh):
                np.testing.assert_array_equal(x.cpu().numpy(),y.cpu().numpy());np.testing.assert_array_equal(h,hh);cpu,height=transform(context,keys);np.testing.assert_allclose(y.cpu().numpy(),cpu,atol=3.e-6,rtol=3.e-5);np.testing.assert_array_equal(h,height)


if __name__=='__main__':unittest.main()
