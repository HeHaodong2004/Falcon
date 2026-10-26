import ctypes as ct
import os
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from .diagnose import OUT
from .cuda_module import CudaModule
from .initial_visibility import SOURCE,TrainingInitialKernel
from .initial_model import InitialVisibilityNet
from .initial_api import InitialVisibilityPredictor
from .device_pipeline import ColumnPreparedPipelinePredictor
from .range_benchmark import workloads
from .range_online_features import LegalFeatureBuilder
from .range_online_data import inventory,OnlineRangeGroups
from .range_online_train import make_batch


class InitialVisibility(unittest.TestCase):
    def test_independent_row_query_mapping(self):
        device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');frame=torch.tensor([[0,1,-1,0,1,-1,0],[1,-1,0,1,-1,0,1],[-1,0,1,-1,0,1,-1]],dtype=torch.int32,device=device);initial=torch.tensor([2,0],dtype=torch.int32,device=device);queries=torch.tensor([0,1,1,0,1,0],dtype=torch.int32,device=device);cells=torch.tensor([0,2,4,6,0,3],dtype=torch.int32,device=device);out=torch.empty((6,3),device=device);module=CudaModule(SOURCE,device);module.launch('initial_code',6,[frame,ct.c_int(7),cells,queries,initial,ct.c_int(6),out]);f=frame.cpu().numpy();ii=initial.cpu().numpy();q=queries.cpu().numpy();c=cells.cpu().numpy();expected=[]
        for query,cell in zip(q,c):value=int(f[ii[query],cell]);expected.append([value==0,value>0,value<0])
        np.testing.assert_array_equal(out.cpu().numpy(),np.asarray(expected,np.float32));kernel=TrainingInitialKernel(device);self.assertEqual(kernel.compute(None,[(np.empty((0,2),np.int32),None,None)],None,None,None)[0].shape,(0,19))

    def test_zero_embedding_preserves_complete_real_outputs(self):
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');cp=OUT/'training/range_online4096/selected_epoch02_argmax.pt';saved=torch.load(cp,map_location='cpu');model=InitialVisibilityNet();model.initialize_spatial(saved['model']);saved=dict(saved,schema='legal_range_initial_visibility_1',model=model.state_dict())
        with tempfile.TemporaryDirectory(dir=OUT)as folder:
            path=Path(folder)/'zero.pt';torch.save(saved,path);old=ColumnPreparedPipelinePredictor(cp,device=device);new=InitialVisibilityPredictor(path,device=device)
            for i in [0,18,20,30,31]:
                ex,packet,tracks=workloads()[i];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:6]];outputs=[]
                for m in [old,new]:outputs.append(m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks))
                for a,b in zip(*outputs):
                    for key in ['candidate_cells','states','heights']:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
                    np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=3.e-6,rtol=3.e-5);np.testing.assert_array_equal(a['future_partial_graph'].cell_columns['cell'],b['future_partial_graph'].cell_columns['cell'])

    def test_real_training_and_public_feature_parity(self):
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');cp=OUT/'training/range_online4096/selected_epoch02_argmax.pt';saved=torch.load(cp,map_location='cpu');model=InitialVisibilityNet();model.initialize_spatial(saved['model']);saved=dict(saved,schema='legal_range_initial_visibility_1',model=model.state_dict())
        with tempfile.TemporaryDirectory(dir=OUT)as folder:
            path=Path(folder)/'zero.pt';torch.save(saved,path);api=InitialVisibilityPredictor(path,device=device);builder=LegalFeatureBuilder(api.model,device,device_canvas=True,device_edt=True,initial_visibility=True)
            for index in [0,18,30]:
                ex,packet,tracks=workloads()[index];paths=[np.asarray(e['trajectory'],np.float64)for e in ex['paths'][:4]];captured=[];original=api.range_kernel.transform_evidence
                def capture(values):
                    out=original(values);captured.append(out.cpu().numpy());return out
                api.range_kernel.transform_evidence=capture
                try:outputs=api.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
                finally:api.range_kernel.transform_evidence=original
                graph=ex['graph'];p=api.prepare_graph(graph);prepared,contexts=builder.prepare(keys=p.keys,states=p.values,heights=p.heights,geometry=[p.resolution,p.height,graph['clearance']],ego=paths[0][0,1:4],now=graph['timestamp'],paths=paths);xs,extras,_,_,_=builder.build(prepared=prepared,contexts=contexts,keysets=[d['candidate_cells']for d in outputs],known_keys=p.keys,known_states=p.values,known_heights=p.heights,ego=paths[0][0,1:4],packet=packet,tracks=tracks);np.testing.assert_array_equal(np.concatenate([v.cpu().numpy()for v in extras]),captured[0])

    def test_new_embedding_receives_real_gradient(self):
        torch.set_num_threads(1);device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');model=InitialVisibilityNet().to(device);model.initialize_spatial(torch.load(OUT/'training/range_online4096/selected_epoch02_argmax.pt',map_location='cpu')['model']);builder=LegalFeatureBuilder(model,device,device_canvas=True,device_edt=True,initial_visibility=True);iterator=iter(OnlineRangeGroups(inventory(4)));batch,_=make_batch([next(iterator),next(iterator)],builder,device);self.assertEqual(batch['extra'].shape[1],19);torch.testing.assert_close(batch['extra'][:,16:].sum(1),torch.ones(len(batch['extra']),device=device),rtol=0,atol=0)
        with torch.cuda.amp.autocast():encoded=model.encode(batch['maps']);r,b,_=model.raw(batch['x'],batch['extra'],batch['indices'],batch['mapids'],encoded)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(r.float(),(batch['y']>0).float());loss.backward();self.assertGreater(float(model.initial_projection.weight.grad.norm()),0.)


if __name__=='__main__':unittest.main()
