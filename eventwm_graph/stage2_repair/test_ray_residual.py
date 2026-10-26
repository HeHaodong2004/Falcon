import os
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from .diagnose import OUT
from .ray_residual_model import RayResidualEvidenceNet
from .ray_residual_api import RayResidualPredictor
from .device_pipeline import ColumnPreparedPipelinePredictor
from .range_online_data import inventory,OnlineRangeGroups
from .range_online_train import make_batch
from .range_online_features import LegalFeatureBuilder
from .range_benchmark import workloads


class RayResidual(unittest.TestCase):
    def test_zero_residual_preserves_actual_complete_predictions(self):
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');cp=OUT/'training/range_online4096/selected_epoch02_argmax.pt';saved=torch.load(cp,map_location='cpu');model=RayResidualEvidenceNet();model.initialize_spatial(saved['model']);saved=dict(saved,schema='legal_range_ray_residual_1',model=model.state_dict())
        with tempfile.TemporaryDirectory(dir=OUT)as folder:
            path=Path(folder)/'zero.pt';torch.save(saved,path);old=ColumnPreparedPipelinePredictor(cp,device=device);new=RayResidualPredictor(path,device=device)
            for index in [0,18,20,30,31]:
                ex,packet,tracks=workloads()[index];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:6]];results=[]
                for m in [old,new]:results.append(m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks))
                for a,b in zip(*results):
                    for key in ['candidate_cells','states','heights']:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
                    np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=3.e-6,rtol=3.e-5);np.testing.assert_array_equal(a['future_partial_graph'].cell_columns['cell'],b['future_partial_graph'].cell_columns['cell'])

    def test_real_training_branch_receives_gradients(self):
        torch.set_num_threads(1);device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');model=RayResidualEvidenceNet().to(device);model.initialize_spatial(torch.load(OUT/'training/range_online4096/selected_epoch02_argmax.pt',map_location='cpu')['model']);builder=LegalFeatureBuilder(model,device,device_canvas=True,device_edt=True);items=iter(OnlineRangeGroups(inventory(4)));batch,_=make_batch([next(items),next(items)],builder,device,ray_residual=True);optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.001)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():encoded=model.encode(batch['maps']);r,b,h=model.raw(batch['x'],batch['extra'],batch['indices'],batch['mapids'],encoded,batch['ray_origins'])
            loss=torch.nn.functional.binary_cross_entropy_with_logits(r.float(),(batch['y']>0).float());loss.backward();self.assertGreater(float(model.ray_head[-1].weight.grad.norm()),0.);optimizer.step()
        self.assertGreater(float(model.ray_map.weight.grad.norm()),0.);self.assertTrue(torch.isfinite(h).all())

    def test_nonzero_branch_single_batch_fallback_and_camera_identity(self):
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');saved=torch.load(OUT/'training/range_online4096/selected_epoch02_argmax.pt',map_location='cpu');model=RayResidualEvidenceNet();model.initialize_spatial(saved['model']);torch.nn.init.normal_(model.ray_head[-1].weight,std=.001);saved=dict(saved,schema='legal_range_ray_residual_1',model=model.state_dict())
        with tempfile.TemporaryDirectory(dir=OUT)as folder:
            path=Path(folder)/'nonzero.pt';torch.save(saved,path);m=RayResidualPredictor(path,device=device);ex,packet,tracks=workloads()[18];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:4]];paths[0]=paths[0][:1];paths[1]=paths[1][:2]
            def call(ps):return m.predict_segments(partial_graph=ex['graph'],trajectories=ps,range_observations=packet,track_history=tracks)
            batch=call(paths);self.assertFalse(batch[0]['states'].any());self.assertEqual(call([]),[])
            for p,b in zip(paths,batch):
                a=call([p])[0];np.testing.assert_array_equal(a['candidate_cells'],b['candidate_cells']);np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=7.e-4,rtol=1.e-3)
            eligible=m._eligible;m._eligible=lambda *args:False
            try:reference=call(paths)
            finally:m._eligible=eligible
            for a,b in zip(reference,batch):np.testing.assert_array_equal(a['states'],b['states']);np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=7.e-4,rtol=1.e-3)
            low=np.array([-57,-123]);counts=[1]*len(paths);got=m._origins(paths,counts,low).cpu().numpy();offset=packet['origins'][-1]-paths[0][0,1:4];expected=[]
            for p in paths:
                take=[1 if len(p)>1 else 0,len(p)//2,len(p)-1];angle=p[take,4]-p[0,4];c,s=np.cos(angle),np.sin(angle);x=p[take,1]+c*offset[0]+s*offset[2];z=p[take,3]-s*offset[0]+c*offset[2];expected.append(np.column_stack([x,z])/m.active_graph.resolution-low)
            np.testing.assert_allclose(got,np.asarray(expected,np.float32),rtol=0,atol=1.e-5)


if __name__=='__main__':unittest.main()
