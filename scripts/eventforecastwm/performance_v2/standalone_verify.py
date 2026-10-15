"""Bounded standalone inference with an enforced filesystem input boundary.

Only the named checkpoint, legal history/sensor/static map, Python source and
installed runtime can be read. Project datasets, caches and arbitrary home or
temporary files are rejected. This complements tensor/input leakage tests.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

import numpy as np

from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark_reuse import CooperativeGuard


def permitted(path, inputs, output, source, runtime):
    path=Path(path).resolve()
    if path in inputs or path==output or output in path.parents:return True
    if source in path.parents and path.suffix in ('.py','.pyc','.so'):return True
    if runtime==path or runtime in path.parents:return True
    # Runtime libraries, device handles and OS capability/configuration files.
    return any(root==path or root in path.parents for root in
               map(Path,('/usr','/lib','/lib64','/proc','/sys','/dev','/etc')))


def main():
    parser=argparse.ArgumentParser()
    for key in ('checkpoint','history','run'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--sensor');parser.add_argument('--static-map')
    parser.add_argument('--device',default='cpu');parser.add_argument('--cpu-preparation',action='store_true');args=parser.parse_args()
    out=Path(args.run).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed standalone proof is immutable')
    CooperativeGuard(out).check(True)
    named={key:str(Path(getattr(args,key)).resolve()) for key in ('checkpoint','history','sensor','static_map')
           if getattr(args,key,None)}
    source=Path(__file__).resolve().parents[2];runtime=Path(sys.prefix).resolve()
    sources={str(p.relative_to(source)):sha256(p) for package in ('eventforecastwm','eventbeliefwm','eventquerywm','collector')
             for p in (source/package).rglob('*.py')}
    identity=dict(inputs={k:dict(path=v,sha256=sha256(v)) for k,v in named.items()},
                  device=args.device,cpu_preparation=args.cpu_preparation,seed=7,horizon=3.,samples=6,
                  source_sha256=sha256(__file__),sources=sources)
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:
        raise ValueError('Standalone verification identity changed')
    save_json(out/'identity.json',identity)
    allowed={Path(v) for v in named.values()};opened=set();rejected=[]
    def audit(event,values):
        if event!='open' or not values or not isinstance(values[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(values[0])).resolve()
        if not permitted(path,allowed,out,source,runtime):
            rejected.append(str(path));raise PermissionError('Standalone input boundary rejected: '+str(path))
        opened.add(str(path))
    sys.addaudithook(audit)
    attempt=out/f'attempt_{time.time_ns()}';attempt.mkdir()
    started=time.perf_counter();files=[]
    for index in range(2):
        path=attempt/f'forecast_{index}.npz'
        sys.argv=['eventforecastwm.inference','--checkpoint',named['checkpoint'],'--history',named['history'],
            '--output',str(path),'--device',args.device,'--horizon','3','--samples','6','--seed','7']
        for key in ('sensor','static_map'):
            if key in named:sys.argv+=['--'+key.replace('_','-'),named[key]]
        if args.cpu_preparation:sys.argv.append('--cpu-preparation')
        runpy.run_module('eventforecastwm.inference',run_name='__main__');files.append(path)
    with np.load(files[0],allow_pickle=False) as first,np.load(files[1],allow_pickle=False) as second:
        if set(first.files)!=set(second.files):raise AssertionError('Output schema changed between identical calls')
        for key in first.files:
            if not np.array_equal(first[key],second[key]):raise AssertionError('Nondeterministic legal inference: '+key)
            if np.issubdtype(first[key].dtype,np.number) and not np.isfinite(first[key]).all():
                raise FloatingPointError('Nonfinite standalone result: '+key)
        shapes={key:list(first[key].shape) for key in first.files}
    if rejected:raise AssertionError('Inference attempted unauthorized input access')
    if any(sha256(row['path'])!=row['sha256'] for row in identity['inputs'].values()):
        raise ValueError('Standalone input bytes changed')
    if any(sha256(source/name)!=value for name,value in sources.items()):
        raise ValueError('Inference source changed during standalone verification')
    save_json(out/'result.json',dict(status='standalone_verification_complete',identity=identity,
        identical_seed_outputs_equal=True,all_numeric_outputs_finite=True,
        rejected_accesses=rejected,opened_files=sorted(opened),output_shapes=shapes,
        outputs=[dict(path=str(p),sha256=sha256(p)) for p in files],runtime_s=time.perf_counter()-started,
        boundary='Python-audited file opens: named legal inputs/checkpoint, source, installed runtime and OS files only; project targets/caches denied',
        limitation='File access boundary complements, and does not replace, causal tensor/API tests'))


if __name__=='__main__':main()
