"""Complete-call latency of the online human world model on the legacy workloads.

Mirrors the legacy Stage 1 runtime protocol (reviewer_c_runtime_large.py):
representative = first 200 development Roots in manifest order; dense = top 30
of all 2,110 by (legacy positive episodes, static updates, graph cells), frozen
before any timing. Startup = one complete raw-history call per Root (timed
separately). Each timed online call passes the PRE-arrival state plus the next
actual native frame: the call ingests that frame (map/visibility/track update),
builds features, runs the network, samples 40 worlds and returns 64 times.
The pre-arrival state is restored between repeats outside the timer.

Also measures K-candidate downstream reuse: one shared bank, risk features for
K executed-length robot candidates (vectorised distance/event reduction).
"""
import argparse
import copy
import gc
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch

from .extract import DEV_MANIFEST, DEV_ROOTS, read, selected_branch
from .legal_state import positives
from .api import HumanWorldModel


def inventory():
    m = json.loads(DEV_MANIFEST.read_text());rows = [r for r in m['rows'] if r['split'] == 'dev'];meta = []
    for i, r in enumerate(rows):
        legal = read(DEV_ROOTS / r['root_id'] / 'legal_root.json.gz');h = legal['history']
        keys = {str(t['track_id']) for ob in h for t in positives(ob)}
        meta.append(dict(index=i, root_id=r['root_id'], positive_episodes=len(keys),
                         static_updates=sum(len(ob['static_updates']) for ob in h), cells=len(legal['partial_graph']['cells'])))
    representative = meta[:200]
    dense = sorted(meta, key=lambda r:(r['positive_episodes'], r['static_updates'], r['cells']), reverse=True)[:30]
    return representative, dense


def stats(x):
    x = np.asarray(x)
    return dict(n=len(x), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)), max=float(x.max()))


def risk_features(positions, present, robots, radius=1.):
    """Shared-bank reduction for K robot candidates [K,T,2] -> per-candidate P(social), E[min distance]."""
    d = np.linalg.norm(positions[None] - robots[:, None, :, None], axis=-1)    # [K,W,T,N]
    d = np.where(present[None, :, None], d, np.inf).min(-1)                    # [K,W,T]
    return (d < radius).any(-1).mean(-1), np.minimum(d.min(-1), 5.).mean(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint');p.add_argument('name');p.add_argument('--device', default='cpu')
    p.add_argument('--warm', type=int, default=5);p.add_argument('--worlds', type=int, default=40)
    p.add_argument('--candidates', type=int, nargs='*', default=[1, 6, 16])
    a = p.parse_args()
    torch.set_num_threads(1)
    out = Path(__file__).resolve().parents[2] / 'output/stage1_hwm/benchmark' / a.name;out.mkdir(parents=True, exist_ok=False)
    rep, dense = inventory()
    t0 = time.perf_counter();model = HumanWorldModel(a.checkpoint, a.device);load_ms = (time.perf_counter() - t0) * 1000
    rows = [];startup = [];cand_rows = []
    cohorts = [('representative', rep), ('dense', dense)]
    for cohort, roots in cohorts:
        for meta in roots:
            folder = DEV_ROOTS / meta['root_id']
            legal = read(folder / 'legal_root.json.gz');_, branch = selected_branch(folder)
            history = list(legal['history']);graph = legal['partial_graph']
            tick = time.perf_counter()
            model.predict(history[:-1], history[-1], graph, worlds=a.worlds)
            startup.append(dict(cohort=cohort, root_id=meta['root_id'], ms=(time.perf_counter() - tick) * 1000, frames=len(history)))
            if not branch:continue
            ob = branch[0]['observation'];new_graph = dict(resolution=graph['resolution'], timestamp=ob['timestamp'])
            saved = copy.deepcopy(model.streams['default'])
            for rep_i in range(a.warm):
                model.streams['default'] = copy.deepcopy(saved)
                tick = time.perf_counter()
                bank = model.predict(history, ob, new_graph, worlds=a.worlds)
                ms = (time.perf_counter() - tick) * 1000
                rows.append(dict(cohort=cohort, root_id=meta['root_id'], repeat=rep_i, ms=ms, people=int(bank.positions.shape[2]),
                                 **{k: v for k, v in model.last_profile.items()}))
            # K-candidate reuse of the one shared bank (synthetic straight candidates fanned around the robot).
            T = bank.positions.shape[1];x0 = np.asarray(ob['robot_pose'])[[0, 2]];yaw = float(ob['robot_pose'][3])
            for K in a.candidates:
                ang = yaw + np.linspace(-1., 1., K);speed = .25 / .048 * .2
                robots = x0 + bank.times[None, :, None] * speed * np.stack([np.sin(ang), np.cos(ang)], -1)[:, None]
                tick = time.perf_counter();risk_features(bank.positions, bank.present, robots)
                cand_rows.append(dict(cohort=cohort, K=K, ms=(time.perf_counter() - tick) * 1000))
            gc.collect()
    summary = dict(checkpoint=a.checkpoint, checkpoint_sha256=model.checkpoint_sha256, device=a.device, worlds=a.worlds, times=64, horizon=3.,
                   threads=torch.get_num_threads(), load_ms=load_ms, platform=platform.processor() or platform.machine(),
                   load_average=os.getloadavg(),
                   online={c: stats([r['ms'] for r in rows if r['cohort'] == c]) for c in ('representative', 'dense')},
                   online_profile={c: {k: stats([r[k] for r in rows if r['cohort'] == c]) for k in ('state_ms', 'sample_ms', 'output_ms')} for c in ('representative', 'dense')},
                   startup={c: stats([r['ms'] for r in startup if r['cohort'] == c]) for c in ('representative', 'dense')},
                   candidates={'%s_K%d' % (c, K): stats([r['ms'] for r in cand_rows if r['cohort'] == c and r['K'] == K]) for c in ('representative', 'dense') for K in a.candidates},
                   scope='Complete online call: one new actual frame ingested from the pre-arrival state, features, network, 40-world sampling, 64-time world output. Excludes JSON decoding. Startup (full raw-history replay) reported separately.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    np.save(out / 'rows.npy', np.asarray(dict(online=rows, startup=startup, candidates=cand_rows), dtype=object), allow_pickle=True)
    print(json.dumps(dict(online=summary['online'], startup=summary['startup']), indent=1))


if __name__ == '__main__':
    main()
