"""Reveal-model calibration (analysis; LEGAL data only): how faithfully does an imagined map stream reproduce the TIMING
of the legal sensor's certificates?

The legal map of a banked decision (``statebank.py``) holds every cell the robot certified up to t with its first
certification time (``first_seen``), and the legal state holds the robot's own pose history.  A reveal model
(``perceive.Discovery`` -- 5 m frustum --, ``perceive.ObserverReveal`` with given distances) is replayed along those
poses from an empty map with the map at t as its truth; every certified cell gets the model's reveal time, compared
with the real one.  Per cell type: share revealed within +-0.25 / 0.5 / 1 s of the real certificate, early / late by
more than 0.5 s, never revealed.  One state per episode (its last banked decision: the longest history).

    python scripts/l3_run.py eventwm_graph.l3.revealfit --bank output/statebank/orca_train \
        --models frustum,frustum_mount,observer,observer:free_min=1.0 [--limit 40]
"""
import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np


def _empty_like(g, pose):
    from eventwm_graph.querygen.legal import GridMap
    return GridMap(g.res, center=(float(pose[0]), float(pose[2])))


def _model(name, work, truth):
    from . import perceive
    if name in ('frustum', 'frustum_mount'):
        return perceive.Discovery(work, truth)
    if name.startswith('observer'):
        kw = {}
        for kv in filter(None, (name.split(':', 1)[1] if ':' in name else '').split(';')):
            k, v = kv.split('=')
            kw[k] = float(v)
        return perceive.ObserverReveal(work, truth, **kw)
    raise ValueError(name)


def replay(legal, name):
    """Model reveal time of every certified cell of ``legal.map`` (NaN: never), its real time and state."""
    from . import perceive
    truth = legal.map
    P = np.asarray(legal.poses, np.float64)
    T = np.asarray(legal.times, np.float64)
    work = _empty_like(truth, P[0])
    rv = _model(name, work, truth)
    old = perceive.CAM_MOUNT
    perceive.CAM_MOUNT = name == 'frustum_mount'
    try:
        for p, t in zip(P, T):
            rv.reveal(p, float(t))
    finally:
        perceive.CAM_MOUNT = old
    ii, jj = np.nonzero((truth.state != 0) & np.isfinite(truth.first_seen))
    keys = np.stack([ii+truth.ox, jj+truth.oz], 1)
    wi, wj = keys[:, 0]-work.ox, keys[:, 1]-work.oz
    ok = (wi >= 0) & (wi < work.state.shape[0]) & (wj >= 0) & (wj < work.state.shape[1])
    mt = np.full(len(keys), np.nan)
    mt[ok] = work.first_seen[wi[ok], wj[ok]]
    rt = truth.first_seen[ii, jj].astype(np.float64)
    # distance of the real certificate from the camera (pose at the certification time, measured mount)
    k = np.clip(np.searchsorted(T, rt-1e-4), 0, len(T)-1)
    c, s = np.cos(P[k, 3]), np.sin(P[k, 3])
    cx, cz = P[k, 0]+perceive.CAM_FWD*c, P[k, 2]-perceive.CAM_FWD*s
    dist = np.hypot((ii+truth.ox+.5)*truth.res-cx, (jj+truth.oz+.5)*truth.res-cz)
    # BLOCKED cells on the band's free side (a certified FREE cell within 2 cells; FREE certificates keep ~0.33 m from
    # a wall, the band reaches 0.25 m: one UNKNOWN ring between them): the cells the planner meets first
    from scipy import ndimage
    nb = ndimage.binary_dilation(truth.state == 1, structure=np.ones((3, 3), bool), iterations=2)
    st = truth.state[ii, jj].astype(np.int64)
    st = np.where((st == 2) & nb[ii, jj], 3, st)
    return mt, rt, st, dist


def summarize(rows, name):
    mt = np.concatenate([r[0] for r in rows]); rt = np.concatenate([r[1] for r in rows])
    st = np.concatenate([r[2] for r in rows]); dist = np.concatenate([r[3] for r in rows])
    out = {}
    for s, lab in (((2, 3), 'BLOCKED'), ((3,), 'B-front'), ((1,), 'FREE')):
        for dlab, dm in (('all', np.ones(len(st), bool)), ('<3m', dist < 3.), ('>=3m', dist >= 3.)):
            m = np.isin(st, s) & dm
            if not m.any():
                continue
            d = mt[m]-rt[m]
            never = ~np.isfinite(d)
            dd = np.where(never, np.inf, d)
            out['%s %s' % (lab, dlab)] = dict(
                n=int(m.sum()), within025=float((np.abs(dd) <= 0.25).mean()), within05=float((np.abs(dd) <= 0.5).mean()),
                within1=float((np.abs(dd) <= 1.0).mean()), early=float((dd < -0.5).mean()),
                late=float(((dd > 0.5) & ~never).mean()), never=float(never.mean()),
                median_abs=float(np.median(np.abs(dd))))
    print('== %s' % name)
    for k, v in out.items():
        print('   %-14s n %6d  |dt|<=0.25 s %.3f  <=0.5 s %.3f  <=1 s %.3f  early>0.5 s %.3f  late>0.5 s %.3f  never %.3f'
              '  median |dt| %.2f s' % (k, v['n'], v['within025'], v['within05'], v['within1'], v['early'], v['late'],
                                         v['never'], v['median_abs']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='output/statebank/orca_train')
    ap.add_argument('--models', default='frustum,frustum_mount,observer')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    last = {}
    for p in sorted(glob.glob(os.path.join(a.bank, '*.pkl'))):
        ep = Path(p).name.split('__d')[0]
        last[ep] = p                                   # sorted: the last decision of each episode
    paths = sorted(last.values())
    if a.limit:
        paths = paths[:a.limit]
    items = [pickle.load(open(p, 'rb'))['legal'] for p in paths]
    print('%s: %d episodes (last banked decision each)' % (a.bank, len(items)))
    res = {}
    for name in a.models.split(','):
        res[name] = summarize([replay(legal, name) for legal in items], name)
    if a.out:
        import json
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(bank=a.bank, episodes=len(items), models=res), indent=1))


if __name__ == '__main__':
    main()
