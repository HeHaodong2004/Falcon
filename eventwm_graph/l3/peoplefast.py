"""Fast tracked-people dataset (legal inputs, privileged labels): no Layer 2, no GPU.

Episodes of a run's spool are replayed in querygen's exact branch simulator with the legal sensor (the robot's own
recorded decisions), and every ``--every`` steps every tracked person's legal track history and true future are
recorded (``peopledata.track_histories``) -- the training set of the tracked-people forecaster (``peoplefc.py``).

    L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.peoplefast --run output/runs/R10 --out output/peoplefast/R10 \
        [--shard i --shards n --every 8]
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='output/runs/R10')
    ap.add_argument('--out', default='output/peoplefast/R10')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--every', type=int, default=8)
    ap.add_argument('--limit', type=int, default=100000)
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.querygen import native
    native._install()
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.sandbox.episode import generator
    from .actor import read_episode
    from .env import ConfigSpec
    from .cfonline import _query
    from .peopledata import track_histories
    from . import sensorcache
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.run, 'spool').glob('*.ep'))
    files = [f for f in files if zlib.crc32(f.name.encode()) % a.shards == a.shard][:a.limit]
    for f in files:
        target = out/(f.stem+'.npz')
        if target.exists():
            continue
        t0 = time.perf_counter()
        try:
            d = read_episode(f)
            m = d['metrics']
            if 'decision_list' not in m:
                continue
            cfg = ConfigSpec.parse(d['config'])
            generator(cfg.gen, cfg.dynamics)
            rt = EpisodeRuntime(EpisodeSpec(**d['spec']), cfg.planner, FAMILIES[cfg.dynamics], strict=False)
            rt.sim.check_static = False
            sensorcache.attach(rt)
            tape = np.asarray(rt.tape.positions, np.float64)
            H, V, A, F = [], [], [], []
            st = rt.root
            try:
                for dec in m['decision_list']:
                    if st.terminal:
                        break
                    # step the recorded decision in chunks of ``every`` steps, sampling at each chunk start
                    remaining = int(dec['horizon'])
                    first = True
                    while remaining > 0 and not st.terminal:
                        i0 = int(st.index)
                        idx = np.minimum(np.arange(i0, i0+65), len(tape)-1)
                        item = dict(legal=st.legal, truth_people=tape[idx][:, :, [0, 2]].astype(np.float32))
                        for h in track_histories(item):
                            H.append(h['hist']); V.append(h['vel']); A.append(h['age']); F.append(h['fut'])
                        n = min(a.every, remaining)
                        cont = (not first) or bool(dec['cont'])
                        if cont and (st.execution is None or st.execution.status != 'running'):
                            break
                        st = rt.run(st, None if cont else _query(dec), horizon=n, cont=cont, geodesic=False)
                        remaining -= n
                        first = False
                        if st.execution is not None and st.execution.status != 'running':
                            break
            finally:
                try:
                    sensorcache.save(rt)
                except Exception:
                    pass
                rt.close()
            np.savez_compressed(target, t_hist=np.asarray(H, np.float32).reshape(-1, 10, 3), t_vel=np.asarray(V, np.float32).reshape(-1, 2),
                                t_age=np.asarray(A, np.float32), t_fut=np.asarray(F, np.float32).reshape(-1, 3, 2))
            print(json.dumps(dict(episode=f.name, instances=len(H), seconds=round(time.perf_counter()-t0, 1))), flush=True)
        except Exception as exc:
            print(json.dumps(dict(episode=f.name, error=repr(exc))), flush=True)


if __name__ == '__main__':
    main()
