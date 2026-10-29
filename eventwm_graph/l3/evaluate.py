"""Layer 3 evaluation: a selector (learned policy or baseline) on a suite x configurations, official metrics.

    # run (fleet in sequential mode; greedy policy: tau = eps = 0)
    python scripts/l3_run.py eventwm_graph.l3.evaluate run --out output/eval/X --suite dev --configs orca-native-k10,dwa-native-k10 \
        --selector policy --policy output/runs/R/policy.pt --workers 24 --servers 3
    # aggregate (per configuration; paired comparison against another evaluation directory)
    python scripts/l3_run.py eventwm_graph.l3.evaluate summarize --out output/eval/X [--paired output/eval/Y]
    # official verification: re-execute every recorded decision sequence with the unchanged habitat.Env.step
    python scripts/l3_run.py eventwm_graph.l3.evaluate replay --out output/eval/X --procs 16

Metrics (official Falcon definitions): SR (official success: STOP within 0.2 m geodesic, <= 500 steps, episode ends at
the first human contact), SPL, PSC (share of steps with every human >= 1 m), human-collision rate; also timeouts,
STOP failures, steps and decisions per episode.  Every episode of the suite is in the denominator.
"""
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_results(out):
    from .actor import read_episode
    res = []
    for p in sorted(Path(out, 'spool').glob('*.ep')):
        d = read_episode(p)
        m = dict(d['metrics'])
        m['file'] = p.name
        m['selector'] = d.get('selector')
        m['spec'] = d['spec']
        m['labels'] = [r[a] for r, a in zip(d['rows']['labels'], d['rows']['actions'])]
        m['n_default'] = int(sum(a == b for a, b in zip(d['rows']['actions'], d['rows']['default'])))
        res.append(m)
    return res


def ci95(x):
    x = np.asarray(x, np.float64)
    if len(x) < 2:
        return 0.
    return float(1.96*x.std(ddof=1)/math.sqrt(len(x)))


def aggregate(rows):
    out = {}
    by = {}
    for r in rows:
        by.setdefault(r['config'], []).append(r)
    for c, v in sorted(by.items()):
        s = np.asarray([x['success'] for x in v])
        out[c] = dict(n=len(v), sr=float(s.mean()), sr_ci=ci95(s), spl=float(np.mean([x['spl'] for x in v])),
                      psc=float(np.mean([x['psc'] for x in v])),
                      h_coll=float(np.mean([x['outcome'] == 'human_collision' for x in v])),
                      timeout=float(np.mean([x['outcome'] == 'timeout' for x in v])),
                      stop_fail=float(np.mean([x['outcome'] == 'stop_fail' for x in v])),
                      truncated=int(sum(bool(x.get('truncated')) for x in v)),
                      steps=float(np.mean([x['steps'] for x in v])),
                      decisions=float(np.mean([x['decisions'] for x in v])),
                      non_default_share=float(1-np.sum([x['n_default'] for x in v])/max(1, np.sum([x['decisions'] for x in v]))),
                      spl_success=float(np.mean([x['spl'] for x in v if x['success']])) if s.any() else None)
    return out


def paired(rows, other):
    """Per configuration: episodes in both runs; success difference and discordant pairs (McNemar)."""
    A = {(r['key'], r['config']): r for r in rows}
    B = {(r['key'], r['config']): r for r in other}
    out = {}
    for c in sorted({k[1] for k in A}):
        keys = [k for k in A if k[1] == c and k in B]
        if not keys:
            continue
        a = np.asarray([A[k]['success'] for k in keys]); b = np.asarray([B[k]['success'] for k in keys])
        win = int(((a == 1) & (b == 0)).sum()); loss = int(((a == 0) & (b == 1)).sum())
        from math import comb
        n = win+loss
        p = sum(comb(n, i) for i in range(0, min(win, loss)+1))*2/2**n if n else 1.
        out[c] = dict(n=len(keys), sr_a=float(a.mean()), sr_b=float(b.mean()), diff=float(a.mean()-b.mean()),
                      a_only=win, b_only=loss, mcnemar_p=min(1., p),
                      spl_a=float(np.mean([A[k]['spl'] for k in keys])), spl_b=float(np.mean([B[k]['spl'] for k in keys])))
    return out


def run(a):
    from . import suites
    from .env import ConfigSpec
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    specs = suites.load(a.suite) if a.suite in ('train', 'train_querygen', 'dev', 'test', 'test_full') else None
    if specs is None:
        from eventwm_graph.querygen import suites as qs
        specs = qs.load(a.suite)
    if a.limit:
        specs = specs[:a.limit]
    configs = a.configs.split(',')
    jobs = [[s.asdict(), c] for s in specs for c in configs]
    (out/'jobs.json').write_text(json.dumps(jobs))
    if a.policy:
        import shutil
        shutil.copy(a.policy, out/'policy.pt')
    actor = '--sequential --jobs %s --selector %s --tau 0 --eps 0 --configs %s --suite %s' % (
        shlex.quote(str(out/'jobs.json')), a.selector, configs[0], a.suite if a.suite != 'test_full' else 'test')
    if a.selector == 'oracle':
        actor += ' --oracle-map %s' % shlex.quote(a.oracle_map)
    if a.blind:
        actor += ' --blind'
    cmd = [sys.executable, str(ROOT/'scripts/l3_run.py'), 'eventwm_graph.l3.fleet', '--run', str(out), '--servers',
           str(a.servers), '--server-devices', a.server_devices, '--workers', str(a.workers), '--sequential',
           '--actor-args', actor, '--tag', a.tag or out.name]
    if a.nice:
        cmd += ['--nice', str(a.nice)]
    subprocess.run(cmd, cwd=str(ROOT), check=False)
    rows = load_results(out)
    summ = dict(suite=a.suite, configs=configs, selector=a.selector, policy=a.policy, episodes=len(rows),
                expected=len(jobs), by_config=aggregate(rows))
    (out/'summary.json').write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ['by_config'], indent=1))


def _query(x):
    from eventwm_graph.querygen.executor import Query
    return Query(x['kind'], via=tuple(tuple(v) for v in x['via']), heading=x['heading'], duration=x['duration'],
                 speed=x['speed'], label=x['label'], meta=x['meta'] or {})


def replay_one(item):
    """Official replay of one recorded episode: identical actions and the official metrics."""
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.episode import replay
    from eventwm_graph.querygen.dynamics import FAMILIES
    from .env import ConfigSpec
    m = item
    cfg = ConfigSpec.parse(m['config'])
    decs = [dict(query=_query(d), cont=d['cont'], horizon=d['horizon']) for d in m['decision_list']]
    rp = replay(EpisodeSpec(**m['spec']), cfg.planner, FAMILIES[cfg.dynamics], decs, lean=True)
    om = rp['metrics']
    return dict(key=m['key'], config=m['config'], branch_outcome=m['outcome'], branch_success=m['success'],
                official_success=om.get('success'), official_spl=om.get('spl'), official_psc=om.get('psc'),
                official_human_collision=om.get('human_collision'), steps=rp['steps'], branch_steps=m['steps'])


def replay_all(a):
    from multiprocessing import get_context
    rows = load_results(a.out)
    items = [r for r in rows if 'decision_list' in r]
    with get_context('spawn').Pool(a.procs, initializer=_init_worker) as pool:
        res = pool.map(replay_one, items, chunksize=1)
    agree = sum(int(r['official_success'] == r['branch_success']) for r in res)
    s = dict(episodes=len(res), success_agreement=agree, by_config={})
    by = {}
    for r in res:
        by.setdefault(r['config'], []).append(r)
    for c, v in by.items():
        s['by_config'][c] = dict(n=len(v), official_sr=float(np.mean([x['official_success'] for x in v])),
                                 branch_sr=float(np.mean([x['branch_success'] for x in v])),
                                 official_spl=float(np.mean([x['official_spl'] for x in v])),
                                 official_psc=float(np.mean([x['official_psc'] for x in v])),
                                 official_h_coll=float(np.mean([x['official_human_collision'] for x in v])))
    Path(a.out, 'official_replay.json').write_text(json.dumps(dict(summary=s, episodes=res), indent=1))
    print(json.dumps(s, indent=1))


def _init_worker():
    import types
    if 'eventwm_graph' not in sys.modules:
        ns = types.ModuleType('eventwm_graph'); ns.__path__ = [str(ROOT/'eventwm_graph')]; sys.modules['eventwm_graph'] = ns
    os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'summarize', 'replay'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--configs', default='orca-native-k10')
    ap.add_argument('--selector', default='policy')
    ap.add_argument('--policy', default=None)
    ap.add_argument('--oracle-map', default=None)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--servers', type=int, default=3)
    ap.add_argument('--server-devices', default='0,0,1')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--nice', type=int, default=0)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--paired', default=None)
    ap.add_argument('--blind', action='store_true')
    ap.add_argument('--procs', type=int, default=16)
    a = ap.parse_args()
    if a.mode == 'run':
        run(a)
    elif a.mode == 'summarize':
        rows = load_results(a.out)
        s = dict(episodes=len(rows), by_config=aggregate(rows))
        if a.paired:
            s['paired'] = paired(rows, load_results(a.paired))
        print(json.dumps(s, indent=1))
    else:
        replay_all(a)


if __name__ == '__main__':
    main()
