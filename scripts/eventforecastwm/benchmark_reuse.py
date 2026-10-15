"""Incrementally reuse the model-independent training occupancy prior.

Identities include ordered fitting roots, the immutable physical manifest,
target-construction source and prior implementation. No raw data are copied.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import torch
from eventbeliefwm.data import sha256
from eventforecastwm import benchmark,benchmark_cached
from eventforecastwm.guard import Guard


class CooperativeGuard(Guard):
    def check(self,force=False):
        if (self.directory/'STOP_RESEARCH').exists():raise RuntimeError('Research stopped: explicit benchmark stop')
        return super().check(force)


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--prior-cache',required=True)
    a,remaining=p.parse_known_args();manifest=remaining[remaining.index('--manifest')+1]
    original=benchmark.anonymous_prior
    def prior(roots,device,guard=None):
        if str(device)!='cpu':raise ValueError('Shared prior cache currently pins CPU numerical order')
        from eventbeliefwm import experiment
        identity=dict(roots=[r['root_id'] for r in roots],manifest_sha256=sha256(manifest),torch=torch.__version__,
            target_source_sha256=sha256(experiment.__file__),
            prior_source_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest())
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        directory=Path(a.prior_cache);directory.mkdir(parents=True,exist_ok=True);path=directory/(digest+'.pt')
        if path.exists():
            value=torch.load(path,map_location='cpu')
            if value['identity']!=identity:raise ValueError('Anonymous prior identity changed')
            return value['prior'].to(device),value['counts'].to(device)
        value,counts=original(roots,device,guard)
        torch.save(dict(identity=identity,prior=value.cpu(),counts=counts.cpu()),path.with_suffix('.tmp'))
        os.replace(path.with_suffix('.tmp'),path);path.chmod(0o444)
        return value,counts
    benchmark.anonymous_prior=prior;benchmark.Guard=CooperativeGuard
    benchmark.EXPERIMENT_SOURCES['benchmark_reuse.py']=sha256(__file__)
    sys.argv=[sys.argv[0],*remaining];benchmark_cached.main()


if __name__=='__main__':main()
