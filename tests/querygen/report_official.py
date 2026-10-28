"""Aggregate the official-rules evaluation into one JSON (report + visualization input).

  python tests/querygen/report_official.py --out output/runs/official/report.json
      [--oracle-root output/runs/official/base] [--floor ARM_DIR ...] [--minint DIR] [--readiness DIR]

Capability: per configuration and suite, official SR over every episode not proven natively infeasible
(unresolved and unfinished episodes count as failures), plus official SPL / PSC / H-Coll / steps / Oracle cost on
solved episodes.  Floor: the no-selection default policy per arm.  Learnability: minimum non-default decisions per
solved episode; readiness study summary if present.
"""
import argparse
import glob
import json
import os
import statistics as S
from collections import Counter, defaultdict

SUITES = {'dev': 'docs/querygen/suites.json', 'unseen': 'docs/querygen/suites_unseen.json',
          'unseen2': 'docs/querygen/suites_unseen2.json', 'unseen3': 'docs/querygen/suites_unseen3.json'}


def suite_keys(name):
    return [f"{s['dataset']}/{s['split']}/{s['scene']}/{s['episode_id']}" for s in json.load(open(SUITES[name]))[name]]


def native_infeasible():
    out = set()
    for f in glob.glob('output/runs/native_cert/*/native_*.json'):
        c = json.load(open(f))
        if c.get('certified_infeasible') and (c.get('official_verification') or {}).get('ok'):
            out.add(c['key'])
    return out


def runs(d):
    rows = {}
    for f in glob.glob(d+'/*.json'):
        if f.endswith(('run.json', 'summary.json')):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if 'verdict' in r:
            rows[r['key']] = r
    return rows


def path_quality(actions, traj):
    """Route-quality measures from the replayed trace: turn-direction reversals per 100 steps (an l->r or r->l switch
    with no forward step in between, holds excluded: a native hold is an l/r pair), in-place turn share, and
    backtracking (summed increase of the geodesic distance to the goal, metres)."""
    rev, last_turn, i = 0, None, 0
    a = actions or ''
    while i < len(a):
        c = a[i]
        if c == 'f':
            last_turn = None
        elif c in 'lr':
            hold = i+1 < len(a) and a[i+1] in 'lr' and a[i+1] != c   # an alternating pair = a hold, not a reversal
            if last_turn is not None and c != last_turn and not hold:
                rev += 1
            last_turn = c
        i += 1
    n = max(1, len(a))
    geo = [row[3] for row in (traj or []) if row[3] is not None]
    back = sum(max(0., geo[k+1]-geo[k]) for k in range(len(geo)-1))
    return dict(reversals_per_100=100.*rev/n, turn_share=sum(c in 'lr' for c in a)/n, backtrack_m=back)


def capability(root, tag=''):
    inf = native_infeasible()
    out = {}
    for cfg in ('c1k10', 'c1k6', 'c2k10'):
        for suite in SUITES:
            d = f'{root}/{suite}_{cfg}'
            if not os.path.isdir(d):
                continue
            R = runs(d)
            denom = [k for k in suite_keys(suite) if k not in inf]
            ok = [k for k in denom if k in R and R[k]['verdict'] == 'PASS']
            done = [k for k in denom if k in R]
            met = [R[k]['replay']['metrics'] for k in ok]
            pq = [path_quality(R[k]['replay'].get('trace', {}).get('actions'), R[k]['replay'].get('trace', {}).get('trajectory'))
                  for k in ok]
            edges = [R[k]['oracle']['stats']['children'] for k in ok]
            out[f'{suite}_{cfg}{tag}'] = dict(
                episodes=len(suite_keys(suite)), proven_infeasible=len(suite_keys(suite))-len(denom), denominator=len(denom),
                finished=len(done), passed=len(ok), sr=len(ok)/len(denom) if denom else None,
                sr_finished=len(ok)/len(done) if done else None,
                failures=sorted(k.split('/')[2] for k in done if R[k]['verdict'] != 'PASS'),
                spl=S.mean(m.get('spl', 0.) for m in met) if met else None,
                psc=S.mean(m.get('psc', 0.) for m in met) if met else None,
                hcoll_among_solved=sum(m.get('human_collision', 0.) for m in met),
                steps_p50=S.median(R[k]['replay']['steps'] for k in ok) if ok else None,
                reversals_per_100=S.mean(x['reversals_per_100'] for x in pq) if pq else None,
                turn_share=S.mean(x['turn_share'] for x in pq) if pq else None,
                backtrack_m=S.mean(x['backtrack_m'] for x in pq) if pq else None,
                edges_p50=S.median(edges) if edges else None,
                edges_p95=sorted(edges)[int(0.95*(len(edges)-1))] if edges else None)
    return out


def floor(arm_dirs):
    out = {}
    for d in arm_dirs:
        arm = os.path.basename(d.rstrip('/'))
        per = defaultdict(Counter)
        for f in glob.glob(d+'/*/*.json'):
            if f.endswith(('run.json', 'summary.json')):
                continue
            try:
                r = json.load(open(f))
            except Exception:
                continue
            per[r['planner']][r['branch']['outcome']] += 1
        out[arm] = {p: dict(c, n=sum(c.values()), sr=c['success']/max(1, sum(c.values()))) for p, c in per.items()}
    return out


def minint(d):
    rows = []
    for f in glob.glob(d+'/*/*.json'):
        if f.endswith(('run.json', 'summary.json')):
            continue
        r = json.load(open(f))
        if r.get('verdict') != 'PASS':
            continue
        dec = r['decisions']
        crit = [x['label'] for x in dec if not (x['cont'] or x['label'] == 'goal' or x['kind'] == 'stop')]
        rows.append(dict(key=r['key'], decisions=len(dec), critical=len(crit), labels=crit))
    hist = Counter(min(x['critical'], 6) for x in rows)
    return dict(episodes=len(rows), critical_hist={str(k): hist.get(k, 0) for k in range(7)},
                critical_median=S.median(x['critical'] for x in rows) if rows else None,
                decisions_median=S.median(x['decisions'] for x in rows) if rows else None,
                label_counts=dict(Counter(l.rstrip('0123456789') for x in rows for l in x['labels']).most_common()),
                rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--oracle-root', default='output/runs/official/base')
    ap.add_argument('--oracle-extra', nargs='*', default=['output/runs/official/test_v10:@test'],
                    help='ROOT:TAG -- further Oracle runs (e.g. a foundation variant); keys get the tag suffix')
    ap.add_argument('--floor', nargs='*', default=['output/runs/official/exec', 'output/runs/official/exec_ub',
                                                   'output/runs/official/exec_ube'])
    ap.add_argument('--minint', default='output/runs/official/minint_v11')
    ap.add_argument('--minint-ref', default='output/runs/official/minint', help='earlier foundation (v10) reference')
    ap.add_argument('--readiness', default='')
    a = ap.parse_args()
    cap = capability(a.oracle_root)
    for item in a.oracle_extra:
        root, tag = item.rsplit(':', 1)
        if os.path.isdir(root):
            cap.update(capability(root, tag))
    rep = dict(capability=cap, floor=floor(a.floor), minint=minint(a.minint), minint_ref=minint(a.minint_ref),
               native_infeasible=sorted(native_infeasible()))
    if a.readiness and os.path.exists(os.path.join(a.readiness, 'summary.json')):
        rep['readiness'] = json.load(open(os.path.join(a.readiness, 'summary.json')))
    with open(a.out, 'w') as fh:
        json.dump(rep, fh, indent=1)
    for k, v in rep['capability'].items():
        print(f"{k:15} SR {v['passed']}/{v['denominator']} (finished {v['finished']}) failures {v['failures']} "
              f"SPL {v['spl'] if v['spl'] is None else round(v['spl'], 3)} "
              f"rev/100 {v['reversals_per_100'] if v['reversals_per_100'] is None else round(v['reversals_per_100'], 1)} "
              f"backtrack {v['backtrack_m'] if v['backtrack_m'] is None else round(v['backtrack_m'], 2)} m")
    for arm, per in rep['floor'].items():
        print(arm, {p: f"{c['success'] if 'success' in c else 0}/{c['n']}" for p, c in per.items()})
    m = rep['minint']
    print('critical decisions', m['critical_hist'], 'median', m['critical_median'], 'episodes', m['episodes'])


if __name__ == '__main__':
    main()
