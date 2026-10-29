"""Failure analysis of evaluated episodes (an evaluation directory's spool or a run's eval_spool).

For every failed episode, per configuration:
  * human collision: did the last decision's chosen candidate carry a predicted risk (Stage 1 P(collision) over the
    interval / the 3 s extension), was a predicted-safer candidate available, was the choice the default one,
    how many people were tracked;
  * timeout: stuck (net displacement over the last 100 steps < 1 m) vs slow; share of wait / rotate / manoeuvre
    choices; legal route cost-to-go at the end;
  * STOP failure: count.

    python scripts/l3_run.py eventwm_graph.l3.failures --dir output/eval/X/spool [--dir2 ...]
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from . import compiler as C


def analyse(spool):
    from .actor import read_episode
    I = {c: C.CAND.index(c) for c in C.CAND}
    E = {c: C.EGO.index(c) for c in C.EGO}
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in sorted(Path(spool).glob('*.ep')):
        d = read_episode(p)
        m, r = d['metrics'], d['rows']
        cfg = d['config']
        o = m['outcome']
        out[cfg]['outcome'].append(o)
        if not r.get('packed'):
            continue
        a = r['actions'][-1]
        g = r['packed'][-1]
        n = int(g['cand_mask'].sum())
        cand = g['cand'][:n].astype(np.float32)
        labels = [r['labels'][t][r['actions'][t]] for t in range(len(r['actions']))]
        if o == 'human_collision':
            out[cfg]['coll_pcol'].append(float(cand[a, I['pcol']]))
            out[cfg]['coll_xpcol'].append(float(cand[a, I['x_pcol']]))
            safer = cand[:, I['pcol']]+cand[:, I['x_pcol']] < cand[a, I['pcol']]+cand[a, I['x_pcol']]-0.2
            out[cfg]['coll_safer_available'].append(float(safer.any()))
            out[cfg]['coll_default'].append(float(a == r['default'][-1]))
            out[cfg]['coll_tracks'].append(float(g['ego'][E['n_tracks']])*6)
            out[cfg]['coll_step'].append(m['steps'])
        elif o == 'timeout':
            ego = g['ego'].astype(np.float32)
            out[cfg]['to_stuck'].append(float(ego[E['disp_96']]*6 < 1.0))
            out[cfg]['to_ctg'].append(float(ego[E['ctg']]*20))
            wr = np.mean([any(l.startswith(x) for x in ('wait', 'rot', 'turn', 'creep', 'look', 'kturn')) for l in labels])
            out[cfg]['to_wait_rotate_share'].append(float(wr))
    res = {}
    for cfg, v in sorted(out.items()):
        oc = collections.Counter(v['outcome'])
        row = dict(n=len(v['outcome']), outcomes=dict(oc))
        for k, x in v.items():
            if k != 'outcome' and x:
                row[k] = round(float(np.mean(x)), 3)
        res[cfg] = row
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    a = ap.parse_args()
    print(json.dumps(analyse(a.dir), indent=1))


if __name__ == '__main__':
    main()
