import os
import unittest
import numpy as np
import torch
from .diagnose import OUT
from .range_online_data import inventory,OnlineRangeGroups
from .range_online_train import make_batch
from .range_online_features import LegalFeatureBuilder
from .range_spatial import SpatialEvidenceNet


class OnlineDeviceCanvas(unittest.TestCase):
    def test_real_training_batch_targets_weights_and_inputs(self):
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;device=os.environ.get('FALCON_TEST_DEVICE','cuda:0');model=SpatialEvidenceNet().to(device);model.load_state_dict(torch.load(OUT/'training/range_online4096/selected_epoch02_argmax.pt',map_location='cpu')['model']);builders=[LegalFeatureBuilder(model,device,device_canvas=b,device_edt=b)for b in [False,True]];items=iter(OnlineRangeGroups(inventory(4)))
        for _ in range(8):
            pair=[next(items),next(items)];old,old_counts=make_batch(pair,builders[0],device);new,new_counts=make_batch(pair,builders[1],device);self.assertEqual(old_counts,new_counts)
            for name in old:np.testing.assert_array_equal(old[name].cpu().numpy(),new[name].cpu().numpy(),err_msg=name)
            with torch.cuda.amp.autocast():
                old_encoded=model.encode(old['maps']);new_encoded=model.encode(new['maps'])
            for a,b in zip(old_encoded,new_encoded):torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
