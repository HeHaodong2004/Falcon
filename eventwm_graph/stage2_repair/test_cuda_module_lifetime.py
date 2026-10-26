import os
import unittest
from concurrent.futures import ThreadPoolExecutor
import torch
from .cuda_module import CudaModule

SOURCE='extern "C" __global__ void lifetime_fill(float* out){out[threadIdx.x]=7.f;}'


class CudaModuleLifetime(unittest.TestCase):
    def test_one_exact_code_instance_and_unmodified_kernel_result(self):
        device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');first=CudaModule(SOURCE,device);out=torch.empty(32,device=device)
        for _ in range(256):
            other=CudaModule(SOURCE,device);self.assertIsNot(first,other);self.assertEqual(first.module.value,other.module.value);self.assertIs(first.functions,other.functions);other.launch('lifetime_fill',32,[out])
        torch.testing.assert_close(out,torch.full_like(out,7.),rtol=0,atol=0)
        changed=CudaModule(SOURCE.replace('7.f','9.f'),device);self.assertNotEqual(first.module.value,changed.module.value);changed.launch('lifetime_fill',32,[out]);torch.testing.assert_close(out,torch.full_like(out,9.),rtol=0,atol=0)

    def test_threaded_initialization_and_device_identity(self):
        device=os.environ.get('FALCON_TEST_DEVICE','cuda:1')
        with ThreadPoolExecutor(max_workers=4)as pool:modules=list(pool.map(lambda _:CudaModule(SOURCE,device),range(16)))
        self.assertTrue(all(m.module.value==modules[0].module.value for m in modules))
        if torch.cuda.device_count()>1:self.assertNotEqual(CudaModule(SOURCE,'cuda:0').module.value,CudaModule(SOURCE,'cuda:1').module.value)

    def test_mutable_predictor_wrappers_are_independent(self):
        class Stateful(CudaModule):
            def __init__(self,device,tag=None):super().__init__(SOURCE,device);self.active=tag
        device=os.environ.get('FALCON_TEST_DEVICE','cuda:1');a=Stateful(device,tag=object());tag=a.active;b=Stateful(device,tag=object());self.assertIsNot(a,b);self.assertIs(a.active,tag);self.assertIsNot(a.active,b.active);self.assertEqual(a.module.value,b.module.value)


if __name__=='__main__':unittest.main()
