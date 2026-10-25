"""Real-coordinate comparison of streaming training and public API features."""
import unittest
import numpy as np
import torch
from .diagnose import OUT
from .range_radix_api import RadixExactSpatialPredictor
from .range_online_features import LegalFeatureBuilder
from .range_benchmark import workloads


class OnlineFeatureParity(unittest.TestCase):
    def test_real_candidates_features_canvas_and_training_gradients(self):
        torch.set_num_threads(1);model=RadixExactSpatialPredictor(OUT/'training/support_pretrained256_control/selected.pt',device='cuda:0')
        model.config['observed_obstacle_schema']='capped_2m_complete'
        builder=LegalFeatureBuilder(model.model,'cuda:0')
        for group in [0,20]:
            ex,packet,tracks=workloads()[group];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
            captured={};original_network=model._network;original_compute=model.range_kernel.compute
            def network(xs,details):
                captured['x']=[x.cpu().numpy()for x in xs]
                return original_network(xs,details)
            def compute(*args,**kwargs):
                values=original_compute(*args,**kwargs);captured['extra']=[v.cpu().numpy()for v in values];return values
            model._network=network;model.range_kernel.compute=compute
            try:outputs=model.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
            finally:model._network=original_network;model.range_kernel.compute=original_compute
            public=model.prepare_graph(ex['graph']);geometry=np.array([public.resolution,public.height,ex['graph']['clearance']])
            prepared,contexts=builder.prepare(keys=public.keys,states=public.values,heights=public.heights,geometry=geometry,ego=paths[0][0,1:4],now=paths[0][0,0],paths=paths)
            keysets=[];chosen=[]
            for context,out in zip(contexts,outputs):
                keys=context.candidates('both');np.testing.assert_array_equal(keys,out['candidate_cells'])
                take=np.linspace(0,len(keys)-1,512).astype(int);chosen.append(take);keysets.append(keys[take])
            xs,extras,canvas,low,_=builder.build(prepared=prepared,contexts=contexts,keysets=keysets,known_keys=public.keys,known_states=public.values,known_heights=public.heights,ego=paths[0][0,1:4],packet=packet,tracks=tracks)
            for i,take in enumerate(chosen):
                np.testing.assert_allclose(xs[i].cpu().numpy(),captured['x'][i][take],rtol=0,atol=0)
                np.testing.assert_allclose(extras[i].cpu().numpy(),captured['extra'][i][take],rtol=3.e-5,atol=3.e-6)
            reference,reference_low=model.canvas.build(public.keys,public.values,public.heights,paths[0][0,1:4],public.resolution,packet['ranges'],packet['origins'],packet['rotations'],packet['timestamps'],tracks)
            np.testing.assert_array_equal(low,reference_low);np.testing.assert_array_equal(canvas,reference.astype(np.float16))
            # Features built for optimization must be ordinary no-grad tensors,
            # rather than inference-only tensors that cannot enter backward.
            model.model.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                encoded=model.model.encode(torch.from_numpy(canvas.astype(np.float32))[None].to(model.device))
                r,b,h=model.model.raw(xs[0],extras[0],torch.from_numpy(keysets[0]-low).to(model.device),torch.zeros(len(xs[0]),dtype=torch.long,device=model.device),encoded)
            (r.float().square().mean()+b.float().square().mean()).backward()
            self.assertGreater(float(model.model.e0.net[0].weight.grad.norm()),0.)


if __name__=='__main__':unittest.main()
