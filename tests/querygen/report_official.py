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
          'unseen2': 'docs/querygen/suites_unseen2.json'}


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


def capability(root):
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
            edges = [R[k]['oracle']['stats']['children'] for k in ok]
            out[f'{suite}_{cfg}'] = dict(
                episodes=len(suite_keys(suite)), proven_infeasible=len(suite_keys(suite))-len(denom), denominator=len(denom),
                finished=len(done), passed=len(ok), sr=len(ok)/len(denom) if denom else None,
                sr_finished=len(ok)/len(done) if done else None,
                failures=sorted(k.split('/')[2] for k in done if R[k]['verdict'] != 'PASS'),
                spl=S.mean(m.get('spl', 0.) for m in met) if met else None,
                psc=S.mean(m.get('psc', 0.) for m in met) if met else None,
                hcoll_among_solved=sum(m.get('human_collision', 0.) for m in met),
                steps_p50=S.median(R[k]['replay']['steps'] for k in ok) if ok else None,
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
    ap.add_argument('--floor', nargs='*', default=['output/runs/official/exec', 'output/runs/official/exec_ub',
                                                   'output/runs/official/exec_ube'])
    ap.add_argument('--minint', default='output/runs/official/minint')
    ap.add_argument('--readiness', default='')
    a = ap.parse_args()
    rep = dict(capability=capability(a.oracle_root), floor=floor(a.floor), minint=minint(a.minint),
               native_infeasible=sorted(native_infeasible()))
    if a.readiness and os.path.exists(os.path.join(a.readiness, 'summary.json')):
        rep['readiness'] = json.load(open(os.path.join(a.readiness, 'summary.json')))
    with open(a.out, 'w') as fh:
        json.dump(rep, fh, indent=1)
    for k, v in rep['capability'].items():
        print(f"{k:15} SR {v['passed']}/{v['denominator']} (finished {v['finished']}) failures {v['failures']} "
              f"SPL {v['spl'] if v['spl'] is None else round(v['spl'], 3)}")
    for arm, per in rep['floor'].items():
        print(arm, {p: f"{c['success'] if 'success' in c else 0}/{c['n']}" for p, c in per.items()})
    m = rep['minint']
    print('critical decisions', m['critical_hist'], 'median', m['critical_median'], 'episodes', m['episodes'])


if __name__ == '__main__':
    main()
