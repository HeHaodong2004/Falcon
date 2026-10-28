"""Summarise a suite run against witness/certificate feasibility (v9 reporting).

  python scripts/querygen_run.py tests.querygen.summarize_v9 RUN_DIR --witness DIR [DIR ...] [--cert DIR ...]

Feasible = a verified witness exists (any of the witness dirs) or a replay-verified PASS of any listed run
(an Oracle solution is itself a witness); certified infeasible = a complete
certificate in any cert dir (start escape / progress / goal region); otherwise unresolved.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np


def witness_ok(dirs, name):
    for d in dirs:
        f = Path(d)/f'{name}.json'
        if f.exists():
            w = json.loads(f.read_text())
            if w.get('found') and w.get('verified', {}).get('ok'):
                return True
    return False


def certified_infeasible(dirs, name):
    for d in dirs:
        for f in (Path(d)/f'{name}.json', Path(d)/f'progress_exact_{name}.json', Path(d)/f'goal_{name}.json'):
            if f.exists():
                c = json.loads(f.read_text())
                if c.get('certified_infeasible') or (c.get('feasible_goal') is False and c.get('goal_points', 0) > 0):
                    return True
                # start-escape certificates written before the explicit flag existed
                if c.get('feasible_escape') is False and c.get('search_complete') and not c.get('depth_limited'):
                    return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--witness', nargs='+', default=[])
    ap.add_argument('--cert', nargs='+', default=[])
    ap.add_argument('--proof-runs', nargs='+', default=[],
                    help='run dirs whose replay-verified PASS records also prove feasibility (e.g. all final runs)')
    a = ap.parse_args()
    proven = set()
    for d in a.proof_runs:
        for f in glob.glob(f'{d}/*.json'):
            if f.endswith('run.json'):
                continue
            r = json.loads(Path(f).read_text())
            if r.get('verdict') == 'PASS':
                proven.add(r['key'].replace('/', '_'))
    rows = []
    for f in sorted(glob.glob(f'{a.run}/*.json')):
        if f.endswith('run.json'):
            continue
        r = json.loads(Path(f).read_text())
        if 'verdict' not in r:
            continue
        name = r['key'].replace('/', '_')
        rows.append(dict(key=r['key'], name=name, verdict=r['verdict'],
                         feasible=witness_ok(a.witness, name) or name in proven or r['verdict'] == 'PASS',
                         infeasible=certified_infeasible(a.cert, name), rec=r))
    n = len(rows)
    feas = [x for x in rows if x['feasible']]
    inf = [x for x in rows if x['infeasible'] and not x['feasible']]
    unres = [x for x in rows if not x['feasible'] and not x['infeasible']]
    ok = [x for x in rows if x['verdict'] == 'PASS']
    okf = [x for x in feas if x['verdict'] == 'PASS']
    print(f'{a.run}: {n} episodes; strict PASS {len(ok)}/{n}; feasible (witness) {len(okf)}/{len(feas)} '
          f'({100*len(okf)/max(1,len(feas)):.1f}%); certified infeasible {len(inf)}; unresolved {len(unres)} '
          f'(PASS among unresolved {sum(1 for x in unres if x["verdict"] == "PASS")})')
    fails = [x['key'] for x in feas if x['verdict'] != 'PASS']
    print('  feasible failures:', fails)
    if unres:
        print('  unresolved:', [(x['key'], x['verdict']) for x in unres])
    if ok:
        spl = [x['rec']['replay']['metrics'].get('spl', np.nan) for x in ok]
        clr = [x['rec']['replay']['min_rect_distance'] for x in ok]
        edges = [x['rec']['oracle']['stats']['children'] for x in ok]
        secs = [x['rec']['oracle']['seconds'] for x in ok]
        steps = [x['rec']['replay']['steps'] for x in ok]
        gen = [g for x in ok for g in x['rec']['oracle']['stats'].get('gen_ms', [])]
        cached = [x['rec']['oracle']['runtime_perf'].get('cached_steps', 0)/max(1, x['rec']['oracle']['runtime_perf']['steps'])
                  for x in ok]
        print(f'  solved: SPL mean {np.nanmean(spl):.3f}; steps P50 {np.median(steps):.0f}; min human clearance '
              f'min {min(clr):.3f} m / P50 {np.median([c for c in clr if c != float("inf")] or [np.inf]):.3f} m')
        print(f'  Oracle: edges P50 {np.median(edges):.0f} P95 {np.percentile(edges, 95):.0f}; seconds P50 {np.median(secs):.0f} '
              f'P95 {np.percentile(secs, 95):.0f}; generator ms P50 {np.median(gen) if gen else float("nan"):.0f} '
              f'P95 {np.percentile(gen, 95) if gen else float("nan"):.0f}; cached-step share {np.mean(cached):.2f}')


if __name__ == '__main__':
    main()
