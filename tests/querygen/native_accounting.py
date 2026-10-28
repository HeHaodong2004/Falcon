"""Corrected feasibility accounting (user, 2026-10-28): denominator = every episode not PROVEN impossible in the
original Falcon simulator (native actions, navmesh geometry, official success / human-collision rules).

  confirmed feasible : a replay-verified witness (declared or native rules) or a replay-verified PASS of any run
  native-proven infeasible : a certificate valid under the native rules (none exist yet; declared-rule certificates
                             -- static contact, body proxy, spawn-escape window -- do NOT count)
  unresolved : everything else (counted in the denominator)

  python tests/querygen/native_accounting.py SUITE RUN_DIR [RUN_DIR ...] [--proof-runs ...]
Headline SR = strict PASS / (episodes - native-proven infeasible).
"""
import argparse
import glob
import json
import os

SUITE_FILES = {'dev': 'docs/querygen/suites.json', 'unseen': 'docs/querygen/suites_unseen.json',
               'unseen2': 'docs/querygen/suites_unseen2.json', 'unseen3': 'docs/querygen/suites_unseen3.json'}
def native_infeasible():
    """Keys with an exact native-rule infeasibility certificate (native_cert.py) whose closed action sequences
    were all reproduced as official human collisions in the official environment."""
    out = set()
    for f in glob.glob('output/runs/native_cert/*/native_*.json'):
        c = json.load(open(f))
        if c.get('certified_infeasible') and (c.get('official_verification') or {}).get('ok'):
            out.add(c['key'])
    return out


NATIVE_INFEASIBLE = native_infeasible()


def witnesses():
    out = {}
    for f in glob.glob('output/runs/witness*/*.json')+glob.glob('output/runs/witness*/*/*.json'):
        try:
            w = json.load(open(f))
        except Exception:
            continue
        v = w.get('verified') or {}
        if w.get('found') and (v.get('ok') or v.get('native_ok')):
            out[os.path.basename(f)[:-5]] = w.get('rules', 'declared')
    return out


def runs(dirs):
    rows = {}
    for d in dirs:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('suite')
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--proof-runs', nargs='*', default=[])
    a = ap.parse_args()
    keys = [f"{s['dataset']}/{s['split']}/{s['scene']}/{s['episode_id']}"
            for s in json.load(open(SUITE_FILES[a.suite]))[a.suite]]
    W = witnesses()
    proof = set()
    for d in a.proof_runs+a.runs:
        for k, r in runs([d]).items():
            if r['verdict'] == 'PASS':
                proof.add(k)
    cat = {}
    for k in keys:
        n = k.replace('/', '_')
        cat[k] = 'infeasible' if k in NATIVE_INFEASIBLE else 'feasible' if (n in W or k in proof) else 'unresolved'
    print(f"{a.suite}: {len(keys)} episodes; confirmed feasible {sum(v == 'feasible' for v in cat.values())}, "
          f"unresolved {sum(v == 'unresolved' for v in cat.values())}, native-proven infeasible "
          f"{sum(v == 'infeasible' for v in cat.values())}")
    print('  unresolved:', sorted(k.split('/')[2] for k, v in cat.items() if v == 'unresolved'))
    denom = [k for k in keys if cat[k] != 'infeasible']
    for d in a.runs:
        R = runs([d])
        done = [k for k in denom if k in R]
        p = [k for k in done if R[k]['verdict'] == 'PASS']
        missing = len(denom)-len(done)
        fails = sorted(k.split('/')[2]+('' if cat[k] == 'feasible' else '(unres)') for k in done if R[k]['verdict'] != 'PASS')
        print(f"  {d}: headline SR {len(p)}/{len(denom)} = {100*len(p)/len(denom):.1f}%"
              + (f"  [{missing} not run/finished: counted as failures]" if missing else '')
              + f"; SR over confirmed-feasible {sum(1 for k in p if cat[k] == 'feasible')}/"
                f"{sum(1 for k in denom if cat[k] == 'feasible')}")
        print('     failures:', fails)


if __name__ == '__main__':
    main()
