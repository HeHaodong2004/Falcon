"""Learning curves of training runs (text tables / JSON): training-episode outcomes per configuration in bins of
online decisions, and the greedy dev evaluation (eval actors) per bin of policy versions.

    python scripts/l3_run.py eventwm_graph.l3.curves --runs output/runs/R1[,output/runs/B1] [--bin 20000] [--json OUT]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .monitor import actor_logs


def training_curve(run, bin_decisions=20000):
    rows = sorted([r for r in actor_logs(run) if 'decisions' in r], key=lambda r: r['t'])
    out, acc, cum = [], {}, 0
    edge = bin_decisions
    t0 = rows[0]['t'] if rows else 0
    for r in rows:
        cum += r['decisions']
        cfg = r['name'].split('__')[1]
        acc.setdefault(cfg, []).append(r['outcome'])
        if cum >= edge:
            out.append(dict(decisions=cum, hours=round((r['t']-t0)/3600, 2),
                            by_config={c: dict(n=len(v), sr=round(float(np.mean([x == 'success' for x in v])), 3),
                                               coll=round(float(np.mean([x == 'human_collision' for x in v])), 3))
                                       for c, v in sorted(acc.items())}))
            acc = {}
            edge += bin_decisions
    if acc:
        out.append(dict(decisions=cum, partial=True, by_config={c: dict(n=len(v), sr=round(float(np.mean([x == 'success' for x in v])), 3),
                                                                        coll=round(float(np.mean([x == 'human_collision' for x in v])), 3))
                                                              for c, v in sorted(acc.items())}))
    return out


def eval_curve(run, bin_versions=20):
    rows = [r for r in actor_logs(run, 'eval_spool') if 'decisions' in r]
    by = {}
    for r in rows:
        b = int(r.get('version', 0))//bin_versions
        cfg = r['name'].split('__')[1]
        by.setdefault(b, {}).setdefault(cfg, []).append(r['outcome'])
    return [dict(versions='%d-%d' % (b*bin_versions, (b+1)*bin_versions-1),
                 by_config={c: dict(n=len(v), sr=round(float(np.mean([x == 'success' for x in v])), 3),
                                    coll=round(float(np.mean([x == 'human_collision' for x in v])), 3))
                            for c, v in sorted(d.items())}) for b, d in sorted(by.items())]


def table(curve, key='sr'):
    cfgs = sorted({c for row in curve for c in row['by_config']})
    head = '| bin | '+' | '.join(cfgs)+' |'
    lines = [head, '|'+'---|'*(len(cfgs)+1)]
    for row in curve:
        lab = row.get('versions') or ('%dk dec (%.1f h)' % (row['decisions']//1000, row.get('hours', 0)))
        cells = []
        for c in cfgs:
            v = row['by_config'].get(c)
            cells.append('%.2f (%d)' % (v[key], v['n']) if v else '')
        lines.append('| %s | %s |' % (lab, ' | '.join(cells)))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--bin', type=int, default=20000)
    ap.add_argument('--vbin', type=int, default=20)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    res = {}
    for run in a.runs.split(','):
        tc = training_curve(run, a.bin)
        ec = eval_curve(run, a.vbin)
        res[run] = dict(training=tc, eval=ec)
        print('##', run, '-- training episodes (exploration), SR (n)')
        print(table(tc))
        if ec:
            print('##', run, '-- greedy dev evaluation by policy version, SR (n)')
            print(table(ec))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
