"""Virtual Oracle portfolios from single-search runs (exact up to +-1 edge per member).

Portfolio members (``OracleConfig.portfolio``) are independent searches that share only exact caches, so
with shares s_i a portfolio first succeeds at total edges ~ min_i e_i * sum(s) / s_i, where e_i is member
i's edges-to-solution when run alone (member runs capped at E edges score budgets up to E * sum(s) / s_i).
Given single-search runs of every candidate member (same generator, planner, dynamics; one directory per
member), this prints SR over feasible episodes at several budgets for every single member and every
equal-share combination (``--shares`` adds unequal 2-member splits).

  python tests/querygen/virtual_portfolio.py --cfg c1k10 RUN_DIR_A RUN_DIR_B [...]
RUN_DIR_X/{dev,unseen}_<cfg>/*.json as written by scripts/querygen_dev_v10.sh.
"""
import argparse
import glob
import itertools
import json

DEV_INFEASIBLE = {'yr17PDCnDDW', '7Ukhou1GxYi', 'Dd4bFSTQ8gi', '5cdEh9F2hJL'}
UNSEEN_INFEASIBLE = {'sT4fr6TAbpF', 'D8bT1ambLFc', 'FgswoxWb3uN', 's19Uyn7AWwv', 'JeFG25nYj2p', 'VLzqgDo317F',
                     'gDDiZeyaVc2'}
UNRESOLVED = {'kdFEfVoT1WE', 'nMeXfQU4PMS'}     # counted as feasible only if some run solves them


def load(d):
    out = {}
    for f in glob.glob(d+'/*.json'):
        if f.endswith('run.json') or f.endswith('summary.json'):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if 'oracle' in r and 'verdict' in r:
            out[r['key']] = r
    return out


def edges(r):
    return r['oracle']['stats']['children'] if r is not None and r['verdict'] == 'PASS' else None


def share_sets(m, spec):
    out = [(1.,)*m]
    if m == 2 and spec:
        for x in spec.split(','):
            u, v = (float(t) for t in x.split(':'))
            out += [(u, v), (v, u)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--budgets', default='1000,3000,6000,12000')
    ap.add_argument('--max-members', type=int, default=3)
    ap.add_argument('--complete', action='store_true', help='only episodes finished in every run')
    ap.add_argument('--shares', default='', help='also unequal shares for 2-member portfolios, e.g. "2:1,3:1"')
    a = ap.parse_args()
    budgets = [int(b) for b in a.budgets.split(',')]
    for suite, inf in (('dev', DEV_INFEASIBLE), ('unseen', UNSEEN_INFEASIBLE)):
        R = [load(f'{d}/{suite}_{a.cfg}') for d in a.runs]
        keys = set().union(*[set(x) for x in R])
        keys = {k for k in keys if k.split('/')[2] not in inf}
        if a.complete:
            keys = {k for k in keys if all(k in x for x in R)}
        solved_any = {k for k in keys if any(edges(x.get(k)) is not None for x in R)}
        feas = sorted(k for k in keys if k.split('/')[2] not in UNRESOLVED or k in solved_any)
        missing = [sum(k not in x for k in feas) for x in R]
        print(f'== {suite}_{a.cfg}: {len(feas)} feasible episodes (missing per run: {missing})')
        hdr = 'members'.ljust(44)+''.join(f'{b:>8}' for b in budgets)
        print(hdr)
        rows = []
        for m in range(1, min(a.max_members, len(R))+1):
            for combo in itertools.combinations(range(len(R)), m):
                for shares in share_sets(m, a.shares):
                    sr = []
                    for b in budgets:
                        n = 0
                        for k in feas:
                            cost = [edges(R[i].get(k))*sum(shares)/s for i, s in zip(combo, shares)
                                    if edges(R[i].get(k)) is not None]
                            if cost and min(cost) <= b:
                                n += 1
                        sr.append(n)
                    name = '+'.join(a.runs[i].rstrip('/').split('/')[-1] for i in combo)
                    if m > 1 and len(set(shares)) > 1:
                        name += ' '+':'.join(f'{x:g}' for x in shares)
                    rows.append((name, sr))
                    print(name[:43].ljust(44)+''.join(f'{x:>8}' for x in sr))
        fails = {}
        for k in feas:
            es = [edges(x.get(k)) for x in R]
            if any(e is None for e in es):
                fails[k.split('/')[2]] = ['-' if k not in x else (e if e is not None else 'F') for x, e in zip(R, es)]
        print('   not solved by every run:', json.dumps(fails))


if __name__ == '__main__':
    main()
