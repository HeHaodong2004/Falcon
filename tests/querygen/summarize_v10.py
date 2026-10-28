"""Summarise v10 runs against witness/certificate feasibility (v10 reporting: adds SR vs edge budget, portfolio
member wins and Layer-3 proxies to the v9 summary).

  python tests/querygen/summarize_v10.py RUN_DIR --witness DIR [DIR ...] [--cert DIR ...] [--proof-runs DIR ...]

Feasible = a verified witness (any witness dir) or a replay-verified PASS of any proof run (an Oracle solution is
itself a witness); certified infeasible = a complete certificate (start escape / exact-gait progress / goal
region); otherwise unresolved.  Layer-3 proxies are computed over solved episodes: decisions per episode,
default-query share (goal / continue / stop decisions) and distinct query types per solution.
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from summarize_v9 import witness_ok, certified_infeasible

BUDGETS = (500, 1000, 2000, 3000, 6000, 9000, 12000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--witness', nargs='+', default=[])
    ap.add_argument('--cert', nargs='+', default=[])
    ap.add_argument('--proof-runs', nargs='+', default=[])
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    proven = set()
    for d in a.proof_runs:
        for f in glob.glob(f'{d}/*.json'):
            if f.endswith(('run.json', 'summary.json')):
                continue
            r = json.loads(Path(f).read_text())
            if r.get('verdict') == 'PASS':
                proven.add(r['key'].replace('/', '_'))
    rows = []
    for f in sorted(glob.glob(f'{a.run}/*.json')):
        if f.endswith(('run.json', 'summary.json')):
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
    out = dict(run=a.run, episodes=n, strict_pass=len(ok), feasible=len(feas), feasible_pass=len(okf),
               certified_infeasible=[x['key'] for x in inf], unresolved=[(x['key'], x['verdict']) for x in unres],
               feasible_failures=[(x['key'], x['verdict']) for x in feas if x['verdict'] != 'PASS'])
    print(f'{a.run}: {n} episodes; strict PASS {len(ok)}/{n}; feasible {len(okf)}/{len(feas)} '
          f'({100*len(okf)/max(1, len(feas)):.1f}%); certified infeasible {len(inf)}; unresolved {len(unres)} '
          f'(PASS among unresolved {sum(1 for x in unres if x["verdict"] == "PASS")})')
    print('  feasible failures:', out['feasible_failures'])
    if unres:
        print('  unresolved:', out['unresolved'])
    if inf:
        print('  certified infeasible:', out['certified_infeasible'])
    sr_b = {b: sum(1 for x in okf if x['rec']['oracle']['stats']['children'] <= b) for b in BUDGETS}
    out['sr_vs_budget'] = sr_b
    print('  feasible SR vs edge budget: ' + ', '.join(f'{b}: {v}/{len(feas)}' for b, v in sr_b.items()))
    if ok:
        spl = [x['rec']['replay']['metrics'].get('spl', np.nan) for x in ok]
        clr = [x['rec']['replay']['min_rect_distance'] for x in ok]
        edges = [x['rec']['oracle']['stats']['children'] for x in ok]
        secs = [x['rec']['oracle']['seconds'] for x in ok]
        steps = [x['rec']['replay']['steps'] for x in ok]
        gen = [g for x in ok for g in x['rec']['oracle']['stats'].get('gen_ms', [])]
        cached = [x['rec']['oracle']['runtime_perf'].get('cached_steps', 0)/max(1, x['rec']['oracle']['runtime_perf']['steps'])
                  for x in ok]
        fin = [c for c in clr if c != float('inf')] or [np.inf]
        out.update(spl=float(np.nanmean(spl)), steps_p50=float(np.median(steps)), clearance_min=float(min(clr)),
                   clearance_p50=float(np.median(fin)), edges_p50=float(np.median(edges)),
                   edges_p95=float(np.percentile(edges, 95)), seconds_p50=float(np.median(secs)),
                   seconds_p95=float(np.percentile(secs, 95)),
                   gen_ms_p50=float(np.median(gen)) if gen else None, gen_ms_p95=float(np.percentile(gen, 95)) if gen else None,
                   cached_share=float(np.mean(cached)))
        print(f'  solved: SPL mean {out["spl"]:.3f}; steps P50 {out["steps_p50"]:.0f}; min human clearance '
              f'min {out["clearance_min"]:.3f} m / P50 {out["clearance_p50"]:.3f} m')
        print(f'  Oracle: edges P50 {out["edges_p50"]:.0f} P95 {out["edges_p95"]:.0f}; seconds P50 {out["seconds_p50"]:.0f} '
              f'P95 {out["seconds_p95"]:.0f}; generator ms P50 {out["gen_ms_p50"] or float("nan"):.0f} '
              f'P95 {out["gen_ms_p95"] or float("nan"):.0f}; cached-step share {out["cached_share"]:.2f}')
        wins = {}
        for x in ok:
            w = x['rec']['oracle']['stats'].get('portfolio_winner')
            wins[w] = wins.get(w, 0)+1
        out['portfolio_wins'] = {str(k): v for k, v in wins.items()}
        print('  portfolio member wins:', out['portfolio_wins'])
        dec = [len(x['rec']['decisions']) for x in ok]
        labs = [[re.sub(r'\d+$', '', d['label']) for d in x['rec']['decisions']] for x in ok]
        dflt = [sum(l in ('goal', 'continue', 'stop') for l in L)/len(L) for L in labs]
        types_ = [len(set(L)) for L in labs]
        out.update(decisions_p50=float(np.median(dec)), decisions_p90=float(np.percentile(dec, 90)),
                   default_share=float(np.mean(dflt)), distinct_types_p50=float(np.median(types_)))
        print(f'  Layer-3 proxies: decisions/episode P50 {out["decisions_p50"]:.0f} P90 {out["decisions_p90"]:.0f}; '
              f'default-query share mean {out["default_share"]:.2f}; distinct query types P50 {out["distinct_types_p50"]:.0f}')
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
