import os
import unittest
import torch
from torch.nn import functional as F
from .diagnose import OUT
from .range_spatial import SpatialEvidenceNet
from .range_online_features import LegalFeatureBuilder
from .range_online_data import inventory,OnlineRangeGroups
from .range_online_train import make_batch


class JointRangeHead(unittest.TestCase):
    def test_real_gradient_updates_existing_range_head_and_preserves_height(self):
        torch.set_num_threads(1);device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');model=SpatialEvidenceNet().to(device);model.load_state_dict(torch.load(OUT/'training/range_online4096/selected_epoch02_argmax.pt',map_location='cpu')['model']);builder=LegalFeatureBuilder(model,device,device_canvas=True,device_edt=True);iterator=iter(OnlineRangeGroups(inventory(4)));batch,_=make_batch([next(iterator),next(iterator)],builder,device)
        self.assertFalse(any(p.requires_grad for p in model.point.parameters()))
        def forward():
            with torch.cuda.amp.autocast():return model.raw(batch['x'],batch['extra'],batch['indices'],batch['mapids'],model.encode(batch['maps']))
        with torch.no_grad():before=forward()
        model.set_range_head_training(True)
        with torch.no_grad():same=forward()
        for a,b in zip(before,same):torch.testing.assert_close(a,b,rtol=0,atol=0)
        old=model.point.head[0].weight.detach().clone();optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0002)
        r,b,h=forward();y=batch['y'];truth=(y>0).float();loss=((F.binary_cross_entropy_with_logits(r.float(),truth,pos_weight=r.new_tensor(8.),reduction='none')+8*truth*F.binary_cross_entropy_with_logits(b.float(),(y==2).float(),reduction='none'))*batch['weight']).sum();loss.backward()
        self.assertGreater(float(model.point.head[0].weight.grad.norm()),0.)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.point.point.parameters()))
        optim.step();self.assertGreater(float((old-model.point.head[0].weight).abs().max()),0.)
        with torch.no_grad():after=forward()
        torch.testing.assert_close(before[2],after[2],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
