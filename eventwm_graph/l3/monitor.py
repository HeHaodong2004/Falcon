"""Training-run monitor: throughput, learner state, training-episode outcomes and the greedy dev evaluation by version.

    python scripts/l3_run.py eventwm_graph.l3.monitor --run output/runs/R [--window 3600] [--json]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def actor_logs(run, prefix='actor'):
    rows = []
    for p in Path(run).glob('%s_*.log' % prefix):
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def summarize(run, window=3600.):
    run = Path(run)
    now = time.time()
    out = dict(run=str(run))
    tr = [r for r in actor_logs(run) if 'decisions' in r]
    errs = [r for r in actor_logs(run) if 'error' in r]
    out['errors'] = len(errs)
    if errs:
        out['last_error'] = errs[-1]['error'][:300]
    if tr:
        t0 = min(r['t'] for r in tr)
        out['elapsed_h'] = round((now-t0)/3600, 2)
        out['episodes'] = len(tr)
        out['decisions'] = int(sum(r['decisions'] for r in tr))
        rec = [r for r in tr if r['t'] > now-window]
        out['window_decisions_per_s'] = round(sum(r['decisions'] for r in rec)/window, 3)
        by = {}
        for r in rec:
            cfg = r['name'].split('__')[1]
            by.setdefault(cfg, []).append(r)
        out['window_by_config'] = {c: dict(n=len(v), sr=round(float(np.mean([x['outcome'] == 'success' for x in v])), 3),
                                           coll=round(float(np.mean([x['outcome'] == 'human_collision' for x in v])), 3),
                                           timeout=round(float(np.mean([x['outcome'] == 'timeout' for x in v])), 3))
                                   for c, v in sorted(by.items())}
    lp = run/'learner.jsonl'
    if lp.exists():
        lines = lp.read_text().splitlines()
        if lines:
            last = json.loads(lines[-1])
            out['learner'] = {k: last[k] for k in last if k != 'recent'}
    ev = actor_logs(run, 'eval_spool')
    ev = [r for r in ev if 'decisions' in r]
    if ev:
        byv = {}
        for r in ev:
            v = r.get('version', -1)
            cfg = r['name'].split('__')[1]
            byv.setdefault(v, {}).setdefault(cfg, []).append(r['outcome'] == 'success')
        out['eval_by_version'] = {v: {c: '%d/%d' % (sum(x), len(x)) for c, x in sorted(d.items())}
                                  for v, d in sorted(byv.items())[-6:]}
    fl = run/'fleet.jsonl'
    if fl.exists():
        lines = fl.read_text().splitlines()[-10:]
        out['fleet_recent'] = [json.loads(x).get('decisions_per_s') for x in lines]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--window', type=float, default=3600.)
    a = ap.parse_args()
    print(json.dumps(summarize(a.run, a.window), indent=1))


if __name__ == '__main__':
    main()
