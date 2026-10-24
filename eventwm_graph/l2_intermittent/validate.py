"""Retain previous synthetic regression/oracle comparisons in NEW output only."""
import json
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from .common import OUT, BASE, sha, dump


def run(tag='validation'):
    sys.path.insert(0,str(BASE/'scripts'))
    import l2_partial_structural_validation as preserved
    if Path(tag).name!=tag:raise ValueError('validation tag must be one output name')
    directory=OUT/tag
    directory.mkdir(exist_ok=False);start=time.perf_counter()
    optimized_paths=[BASE/'eventwm_graph/l2_intermittent'/name for name in
        ('model.py','dynamics.py','neural.py','baselines.py','observations.py',
         'fast_kernels.py','sampling_kernel.py','random_kernel.py')]
    optimized_paths+=list((BASE/'tests/l2_intermittent').glob('*.py'))
    optimized={str(p.relative_to(BASE)):sha(p) for p in optimized_paths}
    inherited={str(p.relative_to(BASE)):sha(p) for folder in ('l2_partial','l2_120k') for p in (BASE/'eventwm_graph'/folder).glob('*.py')}
    commands=preserved.test_commands(directory/'inherited.json')
    for folder in ('l2_decision_ready','l2_120k','l2_intermittent'):
        args=[sys.executable,'-m','unittest','discover','-s','tests/'+folder,'-v'];result=subprocess.run(args,cwd=BASE,text=True,capture_output=True)
        path=directory/(folder+'.log');path.write_text(result.stdout+result.stderr)
        count=re.findall(r'Ran (\d+) tests?',result.stdout+result.stderr)
        commands.append(dict(command=args,returncode=result.returncode,tests=int(count[-1]) if count else 0,log=str(path.relative_to(BASE)),log_sha256=sha(path)))
        if result.returncode:raise RuntimeError('test failure: '+folder)
    oracles=preserved.oracles()
    result=dict(commands=commands,total_tests=sum(c['tests'] or 0 for c in commands),preserved_oracles=oracles,
        inherited_sources=inherited,inherited_sources_unchanged=all(sha(BASE/n)==v for n,v in inherited.items()),seconds=time.perf_counter()-start,
        optimized_sources=optimized,optimized_sources_unchanged=all(sha(BASE/n)==v for n,v in optimized.items()),
        scope='synthetic regressions, previously preserved numerical oracles; no navigation experiment, protected data or production writes')
    dump(directory/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k in ('total_tests','inherited_sources_unchanged','seconds')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',default='validation');run(p.parse_args().tag)
