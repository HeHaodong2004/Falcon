"""Execute the unchanged forecasting suite and preserve source/test identities."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from eventbeliefwm.data import sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark_reuse import CooperativeGuard


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    out=Path(args.run);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed test receipt is immutable')
    CooperativeGuard(out).check(True);root=Path(__file__).resolve().parents[3]
    sources={str(p.relative_to(root)):sha256(p) for folder in ('scripts','tests')
             for p in (root/folder).rglob('*.py') if '__pycache__' not in p.parts}
    command=[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_forecast*.py','-v']
    started=time.time()
    with (out/'tests.log').open('w') as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Forecast suite failed; see '+str(out/'tests.log'))
    if any(sha256(root/name)!=digest for name,digest in sources.items()):raise ValueError('Sources changed during suite')
    log=(out/'tests.log').read_text();counts=re.findall(r'^Ran (\d+) tests? in ',log,re.MULTILINE)
    if len(counts)!=1:raise ValueError('Missing or ambiguous unittest completion count')
    save_json(out/'result.json',dict(status='forecast_tests_complete',passed=True,tests_run=int(counts[0]),
        command=command,returncode=result.returncode,runtime_s=time.time()-started,
        log_sha256=sha256(out/'tests.log'),sources=sources,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))


if __name__=='__main__':main()
