"""Episode curriculum for training actors: spend simulation on episodes whose outcome is still uncertain.

Per training episode (all configurations pooled) the outcomes seen so far -- earlier runs' actor logs as a prior plus
the current run's own logs (re-read every ``refresh_s``) -- give a smoothed success rate s; the sampling weight is
0.25 + 3 s (1 - s) (1.0 at s = 0.5, 0.25 for episodes that always / never succeed), 0.8 for unseen episodes, and 0.03
for episodes that never succeeded and ended in a collision within 3 steps of the start (spawn collisions: no decision
can matter).  Training data only; evaluation suites are never weighted.

    python scripts/l3_run.py eventwm_graph.l3.curriculum --runs output/runs/R1,...,output/runs/R5 --out output/curriculum_90.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def key_of(name):
    """Actor episode name -> (flattened episode key, configuration)."""
    parts = name.split('__')
    return parts[0], parts[1] if len(parts) > 1 else ''


def scan_logs(run_dirs, table=None):
    table = {} if table is None else table
    for d in run_dirs:
        for p in Path(d).glob('actor_*.log'):
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if 'outcome' not in r or 'name' not in r:
                    continue
                k, _ = key_of(r['name'])
                n, s, e = table.get(k, (0, 0, 0))
                table[k] = (n+1, s+int(r['outcome'] == 'success'),
                            e+int(r['outcome'] == 'human_collision' and int(r.get('steps', 99)) <= 3))
    return table


def weight(entry):
    if entry is None:
        return 0.8
    n, s, e = entry
    if s == 0 and e >= 1:
        return 0.03            # a collision within 3 steps of the start: a spawn collision, no decision can matter
    p = (s+0.5)/(n+1.)
    return 0.25+3.*p*(1.-p)


def cached_tapes():
    """Flattened episode keys whose human tape is in the shared tape cache (a miss records the tape: 30-70 s)."""
    import os
    d = Path(os.environ.get('QG_TAPE_CACHE', Path(__file__).resolve().parents[2]/'output/sandbox_cache/tapes'))
    out = set()
    if d.exists():
        for p in d.iterdir():
            k = p.name.rsplit('_s', 1)[0] if '_s' in p.name else None
            if k:
                out.add(k)
    return out


class EpisodeWeights:
    """Sampling weights for a list of EpisodeSpecs (``prior``: JSON table path; ``run``: the current run, re-read).
    ``miss``: weight factor of episodes whose human tape is not cached yet (recording costs about one episode)."""

    def __init__(self, specs, prior=None, run=None, refresh_s=600., miss=0.3):
        self.specs = specs
        self.prior = json.loads(Path(prior).read_text()) if prior else {}
        self.run = run
        self.refresh_s = refresh_s
        self.miss = float(miss)
        self.t = 0.
        self.w = None
        self.refresh(force=True)

    def refresh(self, force=False):
        if not force and time.time()-self.t < self.refresh_s:
            return False
        table = {k: tuple(v) for k, v in self.prior.items()}
        if self.run is not None:
            table = scan_logs([self.run], table)
        cached = cached_tapes() if self.miss < 1 else set()
        keys = [s.key.replace('/', '_') for s in self.specs]
        self.w = np.asarray([weight(table.get(k))*(1. if (self.miss >= 1 or k in cached) else self.miss)
                             for k in keys], np.float64)
        self.t = time.time()
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    table = scan_logs(a.runs.split(','))
    Path(a.out).write_text(json.dumps({k: list(v) for k, v in sorted(table.items())}))
    n = np.asarray([v[0] for v in table.values()])
    s = np.asarray([v[1] for v in table.values()])
    e = np.asarray([v[2] for v in table.values()])
    w = np.asarray([weight(v) for v in table.values()])
    print(json.dumps(dict(episodes=len(table), attempts=int(n.sum()), mean_attempts=float(n.mean()),
                          always=int(((s == n) & (n >= 2)).sum()), never=int(((s == 0) & (n >= 2)).sum()),
                          spawn=int(((s == 0) & (n >= 2) & (e >= n/2)).sum()), mean_weight=float(w.mean()))))


if __name__ == '__main__':
    main()
