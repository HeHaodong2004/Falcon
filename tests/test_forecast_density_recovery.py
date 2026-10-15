import tempfile
from pathlib import Path
import unittest
import torch

from eventforecastwm.agent_scale_calibration import pack_supported
from eventforecastwm.performance_v2.density_cache import materialize


class Guard:
    def check(self):pass


class TestDensityRecovery(unittest.TestCase):
    def test_interrupted_build_is_exact_and_committed_corruption_fails(self):
        ids=['a','b','c'];generator=torch.Generator().manual_seed(31)
        batches={}
        for i,rid in enumerate(ids):
            n=i+1;d=torch.randn(2,6,n,7,generator=generator,dtype=torch.float64)
            c=torch.full((2,n),24.,dtype=torch.float64);c[0,-1]=0
            x=torch.randn(2,n,8,generator=generator,dtype=torch.float64)
            w=torch.randn(2,6,generator=generator,dtype=torch.float64).softmax(-1)
            keep=c.sum(-1)>0;batches[rid]=tuple(t[keep] for t in (d,c,x,w))
        calls=[]
        def build(selected):
            calls.extend(selected)
            if selected==['b'] and calls==['a','b']:raise RuntimeError('injected interruption')
            return [batches[r] for r in selected],[r for r in selected for _ in batches[r][0]]
        with tempfile.TemporaryDirectory() as directory:
            identity={'checkpoint':'frozen','seed':31}
            with self.assertRaisesRegex(RuntimeError,'injected'):
                materialize(directory,identity,ids,build,Guard(),1)
            tensors,rows=materialize(directory,identity,ids,build,Guard(),1)
            self.assertEqual(calls,['a','b','b','c'])
            self.assertEqual(rows,['a','b','b','c','c'])
            for actual,expected in zip(tensors,pack_supported(list(batches.values()))):
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            # Fitting sees exactly the original row order and objective gradient.
            def gradient(values):
                d,c,x,w=values;p=torch.zeros(8,7,dtype=torch.float64,requires_grad=True)
                lp=(x@p).log_softmax(-1)[:,None]
                component=((d+lp).logsumexp(-1)*(c>0)[:,None]).sum(-1)
                loss=-((component+w.log()).logsumexp(-1)/c.sum(-1)).mean()
                return torch.autograd.grad(loss,p)[0]
            torch.testing.assert_close(gradient(tensors),gradient(pack_supported(list(batches.values()))),atol=0,rtol=0)
            part=Path(directory)/'density_parts/000000.pt';part.write_bytes(part.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'bytes changed'):
                materialize(directory,identity,ids,build,Guard(),1)


if __name__=='__main__':unittest.main()
