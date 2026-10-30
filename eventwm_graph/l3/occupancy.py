"""Near-field people occupancy for the closed-loop Sandbox (Layer 2 improvement; legal inputs, privileged supervision).

Stage 1's bank places the people the robot tracks reasonably (best-of-worlds 0.10 m at 0.77 s) but its near-field
occupancy is weakly informative where it matters (``sbeval.py`` calib / AUC: ~0.64 for untracked people, Brier no
better than a constant at >= 1.5 s), and most occupied points near the robot belong to people it does not track.
This model predicts P(a person within ``RADIUS`` of point x at t + h) for points around the robot from LEGAL inputs
only:

* Stage 1: its occupancy probability at the point (radius 0.5 / 1.0 m) and its hidden-slot share;
* the legal map: the point's state, unknown and blocked shares within 1 / 2 m, distance to the nearest unknown cell;
* the information state: view age of the point (from the robot's own pose history), visible now (frustum + line of
  sight), seconds since anyone was visible, number of people seen so far;
* people: distance to the constant-velocity prediction of every tracked person at t + h, to their current positions,
  and to recently LOST tracks from the legal track log (last seen up to 10 s ago; their last position and a capped
  constant-velocity extrapolation), with ages;
* geometry: distance / bearing of the point from the robot, the horizon h.

Labels (training only): the recorded true people (privileged supervision).  LightGBM binary objective.

    python scripts/l3_run.py eventwm_graph.l3.occupancy fit --train output/statebank/orca_train \
        --eval output/statebank/orca_dev --save output/occupancy/v1
"""
import argparse
import glob
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np

RADIUS = 0.5
HORIZONS = (16, 32, 62)
EXTENT, SPACING = 3.0, 0.5
NAMES = ['s1_p05', 's1_p10', 's1_hidden', 'map_state_unknown', 'map_state_free', 'map_state_blocked', 'unk_1m',
         'unk_2m', 'blk_1m', 'dist_unknown', 'view_age', 'visible_now', 't_since_seen', 'n_seen', 'd_cv_track',
         'd_now_track', 'n_tracks', 'd_lost_last', 'd_lost_cv', 'lost_age', 'n_lost', 'r', 'bear_cos', 'bear_sin', 'h']


def grid(extent=EXTENT, spacing=SPACING):
    g = np.arange(-extent, extent+1e-9, spacing)
    gx, gz = np.meshgrid(g, g)
    keep = np.hypot(gx, gz) <= extent
    return gx[keep], gz[keep]


def _lost_tracks(legal, max_age=10.0):
    """Latest record of every person in the legal track log that is not currently a valid track (age 0-10 s)."""
    latest = {}
    for _, trs in getattr(legal, 'track_log', []) or []:
        for t in trs:
            k = str(t.get('track_id'))
            if k not in latest or float(t['last_seen_time']) >= float(latest[k]['last_seen_time']):
                latest[k] = t
    now = float(legal.time)
    cur = {str(t['track_id']) for t in legal.tracks if now-float(t['last_seen_time']) <= 2.0}
    out = []
    for k, t in latest.items():
        age = now-float(t['last_seen_time'])
        if k in cur or age > max_age:
            continue
        p = np.asarray(t['position'], np.float64)
        v = np.asarray(t.get('estimated_velocity') or (0., 0., 0.), np.float64) if t.get('velocity_valid') else np.zeros(3)
        out.append((p[0], p[2], v[0], v[2], age))
    return out


def features(legal, bank, px, pz, h, view_age_fn=None):
    """Feature rows [len(px), len(NAMES)] for points (px, pz) at horizon h (steps)."""
    from eventwm_graph.querygen.control import human_estimates
    from eventwm_graph.querygen.legal import BLOCKED, UNKNOWN, FREE
    from .infofeat import _blocked_lookup, _los, VIEW_M
    n = len(px)
    pose = np.asarray(legal.pose, np.float64)
    F = np.zeros((n, len(NAMES)), np.float32)
    col = {k: i for i, k in enumerate(NAMES)}
    # Stage 1
    P = np.asarray(bank['positions'], np.float64)
    W = np.asarray(bank['weights'], np.float64); W = W/max(W.sum(), 1e-9)
    hp = min(int(h), P.shape[1])
    if P.shape[2] and hp >= 1:
        pr = P[:, hp-1]
        d = np.hypot(px[None, :, None]-pr[:, None, :, 0], pz[None, :, None]-pr[:, None, :, 1])
        d = np.where(np.isfinite(d), d, np.inf)
        F[:, col['s1_p05']] = (W[:, None]*(d < 0.5).any(2)).sum(0)
        F[:, col['s1_p10']] = (W[:, None]*(d < 1.0).any(2)).sum(0)
        hidden = np.asarray([not str(k).startswith('track:') for k in bank['track_keys']], bool)
        if hidden.any():
            F[:, col['s1_hidden']] = (W[:, None]*(d[:, :, hidden] < 1.0).any(2)).sum(0)
    # legal map (vectorised lookups; outside the grid = UNKNOWN)
    g = legal.map
    gs = g.state

    def state_at(x, z):
        i = np.floor(np.asarray(x)/g.res).astype(np.int64)-g.ox
        j = np.floor(np.asarray(z)/g.res).astype(np.int64)-g.oz
        ok = (i >= 0) & (i < gs.shape[0]) & (j >= 0) & (j < gs.shape[1])
        out = np.zeros(np.shape(i), np.uint8)
        out[ok] = gs[i[ok], j[ok]]
        return out
    st = state_at(px, pz)
    F[:, col['map_state_unknown']] = st == UNKNOWN
    F[:, col['map_state_free']] = st == FREE
    F[:, col['map_state_blocked']] = st == BLOCKED
    offs = [(dx, dz) for dx in np.arange(-2., 2.01, 0.5) for dz in np.arange(-2., 2.01, 0.5) if math.hypot(dx, dz) <= 2.0]
    unk1 = np.zeros(n); unk2 = np.zeros(n); blk1 = np.zeros(n); c1 = c2 = 0
    for dx, dz in offs:
        s = state_at(px+dx, pz+dz)
        r = math.hypot(dx, dz)
        unk2 += s == UNKNOWN; c2 += 1
        if r <= 1.0:
            unk1 += s == UNKNOWN; blk1 += s == BLOCKED; c1 += 1
    F[:, col['unk_1m']] = unk1/max(c1, 1); F[:, col['unk_2m']] = unk2/max(c2, 1); F[:, col['blk_1m']] = blk1/max(c1, 1)
    dist_u = np.full(n, 5.)
    for rr in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        ring = [(rr*math.cos(a), rr*math.sin(a)) for a in np.linspace(0, 2*math.pi, 16, endpoint=False)]
        hit = np.zeros(n, bool)
        for dx, dz in ring:
            hit |= state_at(px+dx, pz+dz) == UNKNOWN
        dist_u = np.where(hit & (dist_u > rr), rr, dist_u)
    F[:, col['dist_unknown']] = dist_u
    # information state
    blocked = _blocked_lookup(g)
    if view_age_fn is not None:
        F[:, col['view_age']] = view_age_fn(px, pz)
    dx, dz = px-pose[0], pz-pose[2]
    fw = dx*math.cos(pose[3])-dz*math.sin(pose[3])
    lf = -dx*math.sin(pose[3])-dz*math.cos(pose[3])
    ang = np.arctan2(lf, fw)
    r = np.hypot(dx, dz)
    inview = (np.abs(ang) <= math.radians(45)) & (r <= VIEW_M)
    if inview.any():
        idx = np.flatnonzero(inview)
        clear = _los(blocked, np.full(len(idx), pose[0]), np.full(len(idx), pose[2]), px[idx], pz[idx])
        vis = np.zeros(n, bool); vis[idx[clear]] = True
        F[:, col['visible_now']] = vis
    last_vis, ids = None, set()
    for tm, trs in getattr(legal, 'track_log', []) or []:
        for t in trs:
            ids.add(str(t.get('track_id')))
            if t.get('visible_now'):
                last_vis = tm
    F[:, col['t_since_seen']] = min(float(legal.time)-last_vis, 20.) if last_vis is not None else 20.
    F[:, col['n_seen']] = len(ids)
    # people
    est = human_estimates(legal.tracks, legal.time)
    dcv = np.full(n, 10.); dnow = np.full(n, 10.)
    for e in est:
        x, z = e.at(h*0.048); x0, z0 = e.position
        dcv = np.minimum(dcv, np.hypot(px-x, pz+z))
        dnow = np.minimum(dnow, np.hypot(px-x0, pz+z0))
    F[:, col['d_cv_track']] = dcv; F[:, col['d_now_track']] = dnow; F[:, col['n_tracks']] = len(est)
    lost = _lost_tracks(legal)
    dl = np.full(n, 10.); dlc = np.full(n, 10.); la = np.full(n, 10.)
    for (x, z, vx, vz, age) in lost:
        t_ex = min(age+h*0.048, 4.0)
        d1 = np.hypot(px-x, pz-z)
        d2 = np.hypot(px-(x+vx*t_ex), pz-(z+vz*t_ex))
        closer = np.minimum(d1, d2) < np.minimum(dl, dlc)
        la = np.where(closer, age, la)
        dl = np.minimum(dl, d1); dlc = np.minimum(dlc, d2)
    F[:, col['d_lost_last']] = dl; F[:, col['d_lost_cv']] = dlc; F[:, col['lost_age']] = la; F[:, col['n_lost']] = len(lost)
    # geometry
    F[:, col['r']] = r; F[:, col['bear_cos']] = np.cos(ang); F[:, col['bear_sin']] = np.sin(ang); F[:, col['h']] = h*0.048
    return F


def state_rows(item, horizons=HORIZONS):
    from .infofeat import view_age, _blocked_lookup
    legal = item['legal']
    pose = np.asarray(legal.pose, np.float64)
    gx, gz = grid()
    px, pz = pose[0]+gx, pose[2]+gz
    blocked = _blocked_lookup(legal.map)
    va = view_age(legal, px, pz, blocked)
    truth = np.asarray(item['truth_people'], np.float64)
    X, Y = [], []
    for h in horizons:
        F = features(legal, item['bank'], px, pz, h, view_age_fn=lambda a, b: va)
        hp = min(h, len(truth)-1)
        tp = truth[hp]; tp = tp[np.isfinite(tp).all(1)]
        y = (np.hypot(px[:, None]-tp[None, :, 0], pz[:, None]-tp[None, :, 1]) < RADIUS).any(1) if len(tp) else np.zeros(len(px), bool)
        X.append(F); Y.append(y)
    return np.concatenate(X), np.concatenate(Y)


def dataset(bank_dir, cache=None):
    if cache and Path(cache).exists():
        d = np.load(cache)
        return d['X'], d['Y'], d['G']
    X, Y, G = [], [], []
    for i, p in enumerate(sorted(glob.glob(os.path.join(bank_dir, '*.pkl')))):
        item = pickle.load(open(p, 'rb'))
        x, y = state_rows(item)
        X.append(x); Y.append(y); G.append(np.full(len(y), i))
    X, Y, G = np.concatenate(X), np.concatenate(Y).astype(np.int8), np.concatenate(G)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, Y=Y, G=G)
    return X, Y, G


def auc(s, y):
    y = np.asarray(y, bool)
    if not y.any() or y.all():
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    sv = np.asarray(s)[order]
    i = 0
    while i < len(sv):
        j = i
        while j+1 < len(sv) and sv[j+1] == sv[i]:
            j += 1
        ranks[order[i:j+1]] = (i+j+2)/2.
        i = j+1
    return float((ranks[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['fit'])
    ap.add_argument('--train', default='output/statebank/orca_train')
    ap.add_argument('--eval', default='output/statebank/orca_dev')
    ap.add_argument('--save', default=None)
    ap.add_argument('--rounds', type=int, default=400)
    a = ap.parse_args()
    import lightgbm as lgb
    Xt, Yt, _ = dataset(a.train, cache='output/occupancy/cache_%s.npz' % Path(a.train).name)
    Xe, Ye, _ = dataset(a.eval, cache='output/occupancy/cache_%s.npz' % Path(a.eval).name)
    print('train rows %d (occupied %.4f), eval rows %d (occupied %.4f)' % (len(Yt), Yt.mean(), len(Ye), Ye.mean()), flush=True)
    params = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0,
                  num_threads=int(os.environ.get('L3_LGB_THREADS', '4')))
    b = lgb.train(params, lgb.Dataset(Xt, Yt), a.rounds)
    pe = b.predict(Xe)
    col = {k: i for i, k in enumerate(NAMES)}
    s1 = Xe[:, col['s1_p05']]
    hh = Xe[:, col['h']]
    near = Xe[:, col['d_cv_track']] < 1.5
    for h in sorted(set(np.round(hh, 3))):
        m = np.isclose(hh, h)
        base = Ye[m].mean()
        print('%.2f s  AUC model %.3f vs Stage 1 %.3f | untracked area: model %.3f vs Stage 1 %.3f | Brier model %.5f '
              'Stage 1 %.5f constant %.5f' % (h, auc(pe[m], Ye[m]), auc(s1[m], Ye[m]), auc(pe[m & ~near], Ye[m & ~near]),
                                               auc(s1[m & ~near], Ye[m & ~near]), ((pe[m]-Ye[m])**2).mean(),
                                               ((s1[m]-Ye[m])**2).mean(), ((base-Ye[m])**2).mean()), flush=True)
    imp = b.feature_importance('gain')
    top = np.argsort(-imp)[:10]
    print('top features:', [(NAMES[i], int(imp[i])) for i in top])
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
        b.save_model(str(Path(a.save, 'model.txt')))
        Path(a.save, 'model.json').write_text(json.dumps(b.dump_model()))
        Path(a.save, 'meta.json').write_text(json.dumps(dict(keys=['occupancy'], names=NAMES, radius=RADIUS,
                                                             train=a.train)))


if __name__ == '__main__':
    main()
