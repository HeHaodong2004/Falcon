"""Latency tables and the speed rule from bracketed benchmarks (scripts/stage1_latency_bracket.sh).

A bracket TAG holds TAG_legacy_a, TAG_<candidate>..., TAG_legacy_b under output/stage1_hwm/benchmark.
The speed rule (fixed before the final comparison): a candidate must be >=20% faster than legacy
at P50 and at P95 on every online cohort, against the mean of the two bracketing legacy runs,
in every bracket in which it was measured.

Run: python scripts/stage1_hwm_run.py eventwm_graph.stage1_hwm.latency_table TAG [TAG ...]
"""
import argparse
import json
from pathlib import Path
import numpy as np

BENCH = Path(__file__).resolve().parents[2] / 'output/stage1_hwm/benchmark'
COHORTS = ('representative', 'dense', 'longest')
MARGIN = .20


def runs(tag):
    out = {}
    for d in sorted(BENCH.glob(tag + '_*')):
        if (d / 'summary.json').exists():out[d.name[len(tag) + 1:]] = json.loads((d / 'summary.json').read_text())
    return out


def bracket(tag):
    r = runs(tag);legacy = [r.pop(k) for k in ('legacy_a', 'legacy_b') if k in r]
    assert len(legacy) == 2, 'bracket %s needs legacy_a and legacy_b' % tag
    ref = {c:{q:float(np.mean([l['online'][c][q] for l in legacy])) for q in ('p50', 'p95')} for c in COHORTS}
    rows = {}
    for name, s in r.items():
        change = {c:{q:s['online'][c][q] / ref[c][q] - 1 for q in ('p50', 'p95')} for c in COHORTS}
        rows[name] = dict(change=change, passes=all(v <= -MARGIN for c in change.values() for v in c.values()))
    return dict(legacy=legacy, candidates=r, reference=ref, rule=rows,
                load=[l['load_average'][0] for l in legacy] + [s['load_average'][0] for s in r.values()])


def fmt(s, c):
    o = s['online'][c];return '%.2f / %.2f / %.1f' % (o['p50'], o['p95'], o['p99'])


def main():
    p = argparse.ArgumentParser();p.add_argument('tags', nargs='+');a = p.parse_args()
    verdict = {}
    for tag in a.tags:
        b = bracket(tag);names = list(b['candidates'])
        print('\n## bracket %s (1-min load %.1f-%.1f); P50 / P95 / P99 ms' % (tag, min(b['load']), max(b['load'])))
        print('| cohort | legacy (before) | legacy (after) | ' + ' | '.join(names) + ' |')
        print('|---' * (3 + len(names)) + '|')
        for c in COHORTS:
            print('| %s | %s | %s | %s |' % (c, fmt(b['legacy'][0], c), fmt(b['legacy'][1], c), ' | '.join(fmt(b['candidates'][n], c) for n in names)))
        st = lambda s:'%.0f / %.0f' % (s['startup']['representative']['p50'], s['startup']['representative']['p95'])
        print('| startup P50 / P95 | %s | %s | %s |' % (st(b['legacy'][0]), st(b['legacy'][1]), ' | '.join(st(b['candidates'][n]) for n in names)))
        for n in names:
            ch = b['rule'][n]['change']
            print('%s vs bracket legacy mean: %s -> %s' % (n, ', '.join('%s P50 %+.1f%% P95 %+.1f%%' % (c, 100 * ch[c]['p50'], 100 * ch[c]['p95']) for c in COHORTS),
                                                         'PASS' if b['rule'][n]['passes'] else 'FAIL'))
            verdict.setdefault(n, []).append(b['rule'][n]['passes'])
    print('\nspeed rule over all brackets: ' + ', '.join('%s %s' % (n, 'PASS' if all(v) else 'FAIL') for n, v in verdict.items()))


if __name__ == '__main__':
    main()
