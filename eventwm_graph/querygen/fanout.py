"""Run a per-episode study module over the records of an evaluation run.

  python scripts/querygen_run.py eventwm_graph.querygen.fanout diversity RUN_DIR --out DIR [--workers 6] [--verdict PASS]
  python scripts/querygen_run.py eventwm_graph.querygen.fanout sandbox_study RUN_DIR --out DIR
  python scripts/querygen_run.py eventwm_graph.querygen.fanout l3_interface RUN_DIR --out DIR

Each record whose verdict matches is passed to ``<module> run RECORD --out DIR``
in its own niced subprocess (records with an existing output are skipped);
the module's ``summarize`` is run at the end.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('module', choices=['diversity', 'sandbox_study', 'l3_interface'])
    ap.add_argument('run_dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--verdict', default='PASS')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/'logs').mkdir(exist_ok=True)
    recs = []
    for p in sorted(Path(a.run_dir).glob('*.json')):
        if p.name in ('run.json', 'summary.json', 'report.json'):
            continue
        r = json.loads(p.read_text())
        if a.verdict and r.get('verdict') != a.verdict:
            continue
        if (out/p.name).exists():
            continue
        recs.append(p)
    if a.limit:
        recs = recs[:a.limit]
    runner = str(Path(__file__).resolve().parents[2]/'scripts/querygen_run.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES', '1'))
    procs = []
    while recs or procs:
        while recs and len(procs) < a.workers:
            p = recs.pop(0)
            f = open(out/'logs'/f'{p.stem}.log', 'w')
            cmd = ['nice', '-n', '10', sys.executable, runner, f'eventwm_graph.querygen.{a.module}', 'run', str(p),
                   '--out', str(out)]
            procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env), f))
        time.sleep(2.)
        still = []
        for pr, f in procs:
            if pr.poll() is None:
                still.append((pr, f))
            else:
                f.close()
        procs = still
    cmd = [sys.executable, runner, f'eventwm_graph.querygen.{a.module}', 'summarize', '--out', str(out)]
    subprocess.run(cmd, env=env)


if __name__ == '__main__':
    main()
