"""Learned threat reflex for the shared executor (Layer 3 option; default off).

``L3_EVADE2`` (querygen/executor.py) scores each candidate native sequence by its clearance against the TRACKED people
predicted at constant velocity.  Privileged diagnostics on the TRAIN tuning episodes (31 Oct, ORCA floor) show that the
reflex mechanism is strong and its predictions are the weak part: with the TRUE next 1.5 s of every person the same
reflex cuts collisions 40% -> 13%, with the true future of the tracked people only 42% -> 29%, with legal constant
velocity 43% -> 33%.  This module learns -- from privileged LABELS only -- the probability that a candidate sequence
brings the robot rectangle within ``LABEL_R`` of a TRUE person centre within the horizon (anyone, tracked or not),
from LEGAL features of the step and the candidate:

* the candidate's own legal rollout (exact dynamics, as the rule): constant-velocity clearance to moving / standing
  tracked people and when it is reached, how much it turns / advances / holds, legal progress, displacement;
* the person it meets: speed, detection age, visibility, recent walking speed and pause time from the legal track log
  (Falcon people stop and go and restart at ~1.7 m/s), bearing and closing speed;
* what the robot has not seen: the share of non-wall cells around the robot (by sector) and around the candidate's end
  pose not observed in the last 2 s (people hidden there can reach the robot).

Data (``L3_EVADE_LOG=<dir>`` with ``L3_EVADE2=1``): the executor logs every candidate's features at sampled steps with
the TRUE clearance from the recorded human tape (labels only; ``collect`` rows).  Model (``fit``): LightGBM binary
classifier, scene-grouped folds.  Deployment (``L3_EVADE_RISK=<model dir>``): real legal states only (imagined states
keep the constant-velocity rule: their people move at constant velocity).

    PYTHONPATH=<lightgbm> python scripts/l3_run.py eventwm_graph.l3.reflex fit --data <dir>[,<dir>] --out <model dir>
"""
import argparse
import glob
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np

LABEL_R = 0.30            # m: true rectangle-to-centre distance counted as a contact-level threat (contacts: <= 0.39 m)
STALE_S = 2.0             # s: a cell not observed for this long may hide a person
HIST_S = 3.0              # s: legal track history read for walking / pause features
KIND = dict(own=0, hold=1, fwd=2, turnfwd=3, turn=4, waitplan=5, esc=6)
NAMES = ['n_trk3', 'n_trk6', 'n_vis', 'n_dist', 'n_speed',
         'stale_f', 'stale_l', 'stale_b', 'stale_r', 'turn_rad', 'stateful', 'level', 'cv_own',
         'c_cv', 'c_cvm', 'c_cvs', 'c_tmin', 'c_nf', 'c_nl', 'c_nr', 'c_nh', 'c_first', 'c_disp', 'c_dpsi',
         'c_prog', 'c_off', 'c_kind', 'c_stale_end',
         'p_speed', 'p_age', 'p_vis', 'p_dist0', 'p_cos', 'p_sin', 'p_close', 'p_walk', 'p_pause', 'p_nobs']
CAP = 3.0


def track_context(legal, ests):
    """Per current estimate (``control.human_estimates`` order): max legal speed seen in the last HIST_S s, seconds since
    its speed last was >= 0.2 m/s (0 when moving now), number of observations."""
    now = float(legal.time)
    hist = {}
    for tm, trs in reversed(getattr(legal, 'track_log', None) or []):
        if now-float(tm) > HIST_S:
            break
        for t in trs:
            if t.get('velocity_valid') and t.get('estimated_velocity') is not None:
                v = t['estimated_velocity']
                hist.setdefault(t['track_id'], []).append((float(tm), math.hypot(float(v[0]), float(v[2]))))
    out = []
    for e in ests:
        h = hist.get(e.track_id, [])
        walk = max((s for _, s in h), default=float('nan'))
        sp = float(np.hypot(*e.velocity))
        if sp >= 0.2 or not h:
            pause = 0.
        else:
            moving = [t for t, s in h if s >= 0.2]
            pause = now-max(moving) if moving else HIST_S
        out.append((walk, pause))
    return out


def stale_grid(legal, P0, radius=7.0):
    """(integral image of stale non-wall cells, integral image of non-wall cells, origin index, res) on a window."""
    from eventwm_graph.querygen.legal import BLOCKED
    g = legal.map
    x0, z0 = float(P0[0]), -float(P0[1])
    i0, j0 = g.index(x0, z0)
    r = int(math.ceil(radius/g.res))
    a0, a1 = max(0, i0-r), min(g.state.shape[0], i0+r+1)
    b0, b1 = max(0, j0-r), min(g.state.shape[1], j0+r+1)
    st = g.state[a0:a1, b0:b1]
    ls = g.last_seen[a0:a1, b0:b1]
    ok = st != BLOCKED
    stale = ok & (np.isnan(ls) | (ls < float(legal.time)-STALE_S))
    pad = lambda m: np.pad(np.cumsum(np.cumsum(m.astype(np.int32), 0), 1), ((1, 0), (1, 0)))
    return pad(stale), pad(ok), (a0, b0), g


def stale_share(sg, x, z, half_m):
    S, O, (a0, b0), g = sg
    i, j = g.index(x, z)
    i, j = i-a0, j-b0
    h = int(round(half_m/g.res))
    lo_i, hi_i = max(0, i-h), min(S.shape[0]-1, i+h+1)
    lo_j, hi_j = max(0, j-h), min(S.shape[1]-1, j+h+1)
    if hi_i <= lo_i or hi_j <= lo_j:
        return 1.
    s = S[hi_i, hi_j]-S[lo_i, hi_j]-S[hi_i, lo_j]+S[lo_i, lo_j]
    o = O[hi_i, hi_j]-O[lo_i, hi_j]-O[hi_i, lo_j]+O[lo_i, lo_j]
    return float(s)/o if o > 0 else 1.


def sector_stale(legal, P0, psi0, r_in=0.6, r_out=3.0):
    """Share of stale non-wall cells in a ring around the robot per 90-degree sector (front, left, back, right)."""
    from eventwm_graph.querygen.legal import BLOCKED
    g = legal.map
    x0, z0 = float(P0[0]), -float(P0[1])
    i0, j0 = g.index(x0, z0)
    r = int(math.ceil(r_out/g.res))
    a0, a1 = max(0, i0-r), min(g.state.shape[0], i0+r+1)
    b0, b1 = max(0, j0-r), min(g.state.shape[1], j0+r+1)
    ii, jj = np.mgrid[a0:a1, b0:b1]
    cx = (ii+g.ox+.5)*g.res-x0
    cz = (jj+g.oz+.5)*g.res-z0
    X, Y = cx, -cz                                                    # planar offsets
    d = np.hypot(X, Y)
    ring = (d >= r_in) & (d <= r_out)
    st = g.state[a0:a1, b0:b1]
    ls = g.last_seen[a0:a1, b0:b1]
    ok = ring & (st != BLOCKED)
    stale = ok & (np.isnan(ls) | (ls < float(legal.time)-STALE_S))
    ang = (np.arctan2(Y, X)-psi0+math.pi/4) % (2*math.pi)
    sec = (ang//(math.pi/2)).astype(int)                             # 0 front, 1 left, 2 back, 3 right
    out = []
    for k in range(4):
        m = ok & (sec == k)
        n = int(m.sum())
        out.append(float((stale & m).sum())/n if n else 1.)
    return out


def rect_dist(Q, Hp):
    """Rectangle-to-point distances: Q [S, H, 3] poses, Hp [H, N, 2] -> [S, H, N]."""
    from eventwm_graph.querygen.control import BODY_X, BODY_HALF_WIDTH
    c, s = np.cos(Q[..., 2])[..., None], np.sin(Q[..., 2])[..., None]
    d = Hp[None]-Q[..., None, :2]
    lx = d[..., 0]*c+d[..., 1]*s
    ly = -d[..., 0]*s+d[..., 1]*c
    dx = np.maximum(np.maximum(BODY_X[0]-lx, 0.), lx-BODY_X[1])
    dy = np.maximum(np.abs(ly)-BODY_HALF_WIDTH, 0.)
    return np.hypot(dx, dy)


def features(legal, dyn, level, P0, psi0, cands, kinds, trajs, ests, prog, off):
    """Legal features [S, len(NAMES)] (float32) of the candidates ``cands`` (action strings, ``trajs`` [S, H, 3] their
    rollouts) against the tracked people ``ests`` (HumanEstimates within range; constant-velocity predictions)."""
    S, H = trajs.shape[0], trajs.shape[1]
    X = np.full((S, len(NAMES)), np.nan, np.float32)
    col = {n: i for i, n in enumerate(NAMES)}
    T = np.arange(1, H+1)[:, None, None]*0.048
    if ests:
        pos = np.stack([e.position for e in ests])
        vel = np.stack([e.velocity for e in ests])
        Hp = pos[None]+vel[None]*T                                        # [H, N, 2]
        D = rect_dist(trajs, Hp)                                          # [S, H, N]
        speed = np.hypot(vel[:, 0], vel[:, 1])
        moving = speed*(H-1)*0.048 >= 0.3
        d0 = rect_dist(np.array([[[P0[0], P0[1], psi0]]]), pos[None])[0, 0]     # [N]
        ctx = np.asarray(track_context(legal, ests), np.float64).reshape(-1, 2)
        Dm = np.where(moving[None, None], D, np.inf)
        Ds = np.where(~moving[None, None], D, np.inf)
        cvm = Dm.min(axis=(1, 2)); cvs = Ds.min(axis=(1, 2))
        cv = np.minimum(cvm, cvs+(0.45-0.20))
        X[:, col['n_trk3']] = float((d0 < 3.).sum()); X[:, col['n_trk6']] = float((d0 < 6.).sum())
        X[:, col['n_vis']] = float(sum(e.visible for e in ests))
        k0 = int(np.argmin(d0))
        X[:, col['n_dist']] = min(float(d0[k0]), CAP); X[:, col['n_speed']] = float(speed[k0])
        X[:, col['c_cv']] = np.minimum(cv, CAP); X[:, col['c_cvm']] = np.minimum(cvm, CAP)
        X[:, col['c_cvs']] = np.minimum(cvs, CAP)
        X[:, col['cv_own']] = min(float(cv[0]), CAP)
        X[:, col['c_tmin']] = Dm.min(axis=2).argmin(axis=1)/float(H)
        # the person each candidate meets (closest moving one, else closest)
        who = np.where(np.isfinite(cvm), Dm.min(axis=1).argmin(axis=1), D.min(axis=1).argmin(axis=1))
        rel = pos[who]-P0[None]
        b = np.arctan2(rel[:, 1], rel[:, 0])-psi0
        dist = np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-6)
        nobs = {t.get('track_id'): t.get('num_observations') for t in legal.tracks}
        nob = np.asarray([nobs.get(e.track_id) if nobs.get(e.track_id) is not None else np.nan for e in ests], np.float64)
        X[:, col['p_speed']] = speed[who]; X[:, col['p_age']] = np.asarray([e.age for e in ests])[who]
        X[:, col['p_vis']] = np.asarray([float(e.visible) for e in ests])[who]
        X[:, col['p_dist0']] = np.minimum(d0[who], CAP); X[:, col['p_cos']] = np.cos(b); X[:, col['p_sin']] = np.sin(b)
        X[:, col['p_close']] = -(vel[who]*rel).sum(1)/dist
        X[:, col['p_walk']] = ctx[who, 0]; X[:, col['p_pause']] = ctx[who, 1]
        X[:, col['p_nobs']] = np.minimum(nob[who], 50.)
    else:
        X[:, col['n_trk3']] = 0.; X[:, col['n_trk6']] = 0.; X[:, col['n_vis']] = 0.
        for n in ('c_cv', 'c_cvm', 'c_cvs', 'cv_own'):
            X[:, col[n]] = CAP
    st = sector_stale(legal, P0, psi0)
    for n, v in zip(('stale_f', 'stale_l', 'stale_b', 'stale_r'), st):
        X[:, col[n]] = v
    X[:, col['turn_rad']] = dyn.turn_rad; X[:, col['stateful']] = float(dyn.stateful); X[:, col['level']] = float(level)
    M = np.full((S, H), 'h', dtype='<U1')
    for s_, c in enumerate(cands):
        n = min(len(c), H)
        if n:
            M[s_, :n] = c[:n]
    X[:, col['c_nf']] = (M == 'f').mean(1); X[:, col['c_nl']] = (M == 'l').mean(1)
    X[:, col['c_nr']] = (M == 'r').mean(1); X[:, col['c_nh']] = (M == 'h').mean(1)
    X[:, col['c_first']] = np.select([M[:, 0] == 'f', M[:, 0] == 'l', M[:, 0] == 'r'], [0., 1., 2.], 3.)
    end = trajs[:, -1]
    X[:, col['c_disp']] = np.hypot(end[:, 0]-P0[0], end[:, 1]-P0[1])
    dpsi = end[:, 2]-psi0
    X[:, col['c_dpsi']] = np.abs(np.arctan2(np.sin(dpsi), np.cos(dpsi)))
    X[:, col['c_prog']] = np.clip(np.asarray(prog, np.float64), -CAP, CAP)
    X[:, col['c_off']] = np.asarray(off, np.float64)/float(H)
    X[:, col['c_kind']] = np.asarray(kinds, np.float64)
    sg = stale_grid(legal, P0)
    X[:, col['c_stale_end']] = stale_share_many(sg, end[:, 0], -end[:, 1], 1.5)
    return X


def stale_share_many(sg, xs, zs, half_m):
    """``stale_share`` for arrays of world points."""
    S, O, (a0, b0), g = sg
    i = np.floor(np.asarray(xs)/g.res).astype(np.int64)-g.ox-a0
    j = np.floor(np.asarray(zs)/g.res).astype(np.int64)-g.oz-b0
    h = int(round(half_m/g.res))
    lo_i, hi_i = np.clip(i-h, 0, S.shape[0]-1), np.clip(i+h+1, 0, S.shape[0]-1)
    lo_j, hi_j = np.clip(j-h, 0, S.shape[1]-1), np.clip(j+h+1, 0, S.shape[1]-1)
    s = S[hi_i, hi_j]-S[lo_i, hi_j]-S[hi_i, lo_j]+S[lo_i, lo_j]
    o = O[hi_i, hi_j]-O[lo_i, hi_j]-O[hi_i, lo_j]+O[lo_i, lo_j]
    return np.where(o > 0, s/np.maximum(o, 1), 1.)


# ---------------------------------------------------------------------------------------------- data logging
_BUF = []
_SEQ = [0]


def log_rows(out_dir, meta, X, y, yclear):
    """Append one step's candidate rows (privileged labels: y = true clearance < LABEL_R; yclear = true clearance)."""
    _BUF.append(dict(meta=meta, X=X, y=y, clear=yclear))
    if sum(len(b['X']) for b in _BUF) >= 3000:
        flush_rows(out_dir)


def flush_rows(out_dir):
    if not _BUF:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    name = Path(out_dir, '%d_%05d.pkl' % (os.getpid(), _SEQ[0]))
    with open(str(name)+'.tmp', 'wb') as f:
        pickle.dump(list(_BUF), f, protocol=4)
    os.replace(str(name)+'.tmp', name)
    _SEQ[0] += 1
    del _BUF[:]


# ---------------------------------------------------------------------------------------------- model
class ReflexRisk:
    """A fitted model directory (``fit --out``): P(true clearance < LABEL_R) per candidate row."""

    def __init__(self, path):
        self.meta = json.loads(Path(path, 'meta.json').read_text())
        self.names = list(self.meta['names'])
        try:
            import lightgbm as lgb
            self.booster = lgb.Booster(model_file=str(Path(path, 'model.txt')))
            self.backend = 'lightgbm'
        except ImportError:
            self.trees = _VecTrees(json.loads(Path(path, 'model.json').read_text()))
            self.backend = 'numpy'
        self.idx = [NAMES.index(n) for n in self.names]

    def prob(self, X):
        X = np.asarray(X, np.float64)[:, self.idx]
        if self.backend == 'lightgbm':
            return np.asarray(self.booster.predict(X, num_threads=1), np.float64)
        return 1./(1.+np.exp(-self.trees.raw(X)))


class _VecTrees:
    """Vectorised numpy evaluator of a LightGBM binary model dump (numerical splits; a missing value goes to the default
    side of a NaN-type split and is treated as 0 otherwise): all trees advance together, one level per iteration."""

    def __init__(self, dump):
        feat, thr, left, right, dleft, nanmiss, leaf, roots = [], [], [], [], [], [], [], []
        for t in dump['tree_info']:
            def walk(n):
                i = len(feat)
                feat.append(-1); thr.append(0.); left.append(i); right.append(i); dleft.append(False)
                nanmiss.append(False); leaf.append(0.)
                if 'leaf_value' in n:
                    leaf[i] = float(n['leaf_value'])
                    return i
                if n['decision_type'] != '<=':
                    raise ValueError('categorical split in a reflex model')
                feat[i] = int(n['split_feature']); thr[i] = float(n['threshold'])
                dleft[i] = bool(n.get('default_left', False)); nanmiss[i] = str(n.get('missing_type')) == 'NaN'
                left[i] = walk(n['left_child']); right[i] = walk(n['right_child'])
                return i
            roots.append(walk(t['tree_structure']))
        self.feat = np.asarray(feat); self.thr = np.asarray(thr); self.left = np.asarray(left)
        self.right = np.asarray(right); self.dleft = np.asarray(dleft); self.nanmiss = np.asarray(nanmiss)
        self.leaf = np.asarray(leaf); self.roots = np.asarray(roots)

    def raw(self, X):
        X = np.asarray(X, np.float64)
        node = np.broadcast_to(self.roots[None], (len(X), len(self.roots))).copy()      # [R, T]
        rows = np.arange(len(X))[:, None]
        while True:
            f = self.feat[node]
            act = f >= 0
            if not act.any():
                break
            v = X[np.broadcast_to(rows, node.shape), np.maximum(f, 0)]
            nan = np.isnan(v)
            v = np.where(nan & ~self.nanmiss[node], 0., v)
            go = np.where(nan & self.nanmiss[node], self.dleft[node], v <= self.thr[node])
            node = np.where(act, np.where(go, self.left[node], self.right[node]), node)
        return self.leaf[node].sum(axis=1)


_MODEL = {}


def load(path):
    if path not in _MODEL:
        _MODEL[path] = ReflexRisk(path)
    return _MODEL[path]


# ---------------------------------------------------------------------------------------------- fit
def load_rows(dirs):
    Xs, ys, cl, grp, step, meta = [], [], [], [], [], []
    for d in str(dirs).split(','):
        for f in sorted(glob.glob(os.path.join(d, '*.pkl'))):
            for b in pickle.load(open(f, 'rb')):
                Xs.append(b['X']); ys.append(b['y']); cl.append(b['clear'])
                m = b['meta']
                grp.append(np.full(len(b['X']), hash(m['scene']) % (2**31)))
                step.append(np.full(len(b['X']), len(meta)))
                meta.append(m)
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(cl), np.concatenate(grp), np.concatenate(step),
            meta)


def roc_auc_score(y, s):
    """Rank AUC (ties averaged)."""
    y = np.asarray(y) > 0.5
    s = np.asarray(s, np.float64)
    order = np.argsort(s, kind='mergesort')
    r = np.empty(len(s)); r[order] = np.arange(1, len(s)+1)
    ss = s[order]
    i = 0
    while i < len(ss):                       # average ranks of ties
        j = i
        while j+1 < len(ss) and ss[j+1] == ss[i]:
            j += 1
        if j > i:
            r[order[i:j+1]] = (i+j+2)/2.
        i = j+1
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum()-n1*(n1+1)/2.)/max(n1*n0, 1))


def offline_rules(X, clear, step, p, R=0.45, cap=0.30, gain=0.10, thetas=(0.1, 0.2, 0.3, 0.5), pgains=(0.05, 0.1)):
    """Open-loop replay of the reflex rules on the logged steps (one action per step, true clearances as outcome):
    'threat' steps = the committed motion (candidate 0) truly comes within LABEL_R; a rule SAVES a threat step when it
    acts and its choice is truly clear, HARMS a safe step when it acts and its choice is truly not clear; ACT = share
    of steps where it deviates.  Constant-velocity rule (EVADE2) vs the learned probability (out-of-fold)."""
    c = {n: i for i, n in enumerate(NAMES)}
    order = np.argsort(step, kind='stable')
    bounds = np.flatnonzero(np.diff(step[order]))+1
    groups = [g for g in np.split(order, bounds) if X[g[0], c['c_kind']] == KIND['own']]

    def run(choose):
        thr = sav = safe = harm = act = 0
        for g in groups:
            tc = clear[g]
            k = choose(g)
            threat = tc[0] < LABEL_R
            thr += threat
            safe += not threat
            if k is not None and k != 0:
                act += 1
                sav += threat and tc[k] >= LABEL_R
                harm += (not threat) and tc[k] < LABEL_R
        return 'threat steps %d saved %.3f | safe steps %d harmed %.4f | acts %.3f' % (
            thr, sav/max(thr, 1), safe, harm/max(safe, 1), act/max(len(groups), 1))

    def cv_rule(g):
        cv = X[g, c['c_cv']]
        if cv[0] >= R:
            return None
        capped = np.minimum(cv, R+cap)
        pool = [i for i in range(len(g)) if capped[i] >= capped.max()-0.02]
        best = max(pool, key=lambda i: (X[g[i], c['c_prog']], -X[g[i], c['c_off']]))
        return best if capped[best] >= capped[0]+gain else None
    print('offline  CV rule (R %.2f):            %s' % (R, run(cv_rule)), flush=True)
    for th in thetas:
        for pg in pgains:
            def lr(g, th=th, pg=pg):
                pp = p[g]
                if pp[0] < th:
                    return None
                pool = [i for i in range(len(g)) if pp[i] <= pp.min()+0.03]
                best = max(pool, key=lambda i: (X[g[i], c['c_prog']], -X[g[i], c['c_off']]))
                return best if pp[best] <= pp[0]-pg else None
            print('offline  learned (trig %.2f gain %.2f): %s' % (th, pg, run(lr)), flush=True)


def fit_main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--rounds', type=int, default=300)
    ap.add_argument('--leaves', type=int, default=31)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args(argv)
    import lightgbm as lgb
    t0 = time.perf_counter()
    X, y, clear, grp, step, meta = load_rows(a.data)
    y = y.astype(np.float32)
    print('rows %d steps %d positive %.4f scenes %d' % (len(X), len(meta), y.mean(), len(np.unique(grp))), flush=True)
    params = dict(objective='binary', learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=a.threads)
    scenes = np.unique(grp)
    rng = np.random.default_rng(0)
    rng.shuffle(scenes)
    fold = {s: i % a.folds for i, s in enumerate(scenes)}
    fid = np.asarray([fold[g] for g in grp])
    oof = np.zeros(len(X))
    for k in range(a.folds):
        tr, va = fid != k, fid == k
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=NAMES), a.rounds)
        oof[va] = m.predict(X[va])
    auc = roc_auc_score(y, oof)
    # the legal rule's score for comparison: constant-velocity clearance (lower = riskier)
    auc_cv = roc_auc_score(y, -X[:, NAMES.index('c_cv')])
    # within-step ranking: AUC over candidate pairs of the same step with different labels
    print('pooled out-of-fold AUC %.4f (constant-velocity clearance %.4f)' % (auc, auc_cv), flush=True)
    within, within_cv, n = 0., 0., 0
    order = np.argsort(step, kind='stable')
    bounds = np.flatnonzero(np.diff(step[order]))+1
    for idx in np.split(order, bounds):
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        pos, neg = idx[yy == 1], idx[yy == 0]
        within += (oof[pos][:, None] > oof[neg][None]).mean()
        within_cv += (X[pos, NAMES.index('c_cv')][:, None] < X[neg, NAMES.index('c_cv')][None]).mean()
        n += 1
    print('within-step AUC %.4f (constant velocity %.4f) over %d mixed steps' % (within/max(n, 1), within_cv/max(n, 1), n),
          flush=True)
    offline_rules(X, clear, step, oof)
    m = lgb.train(params, lgb.Dataset(X, y, feature_name=NAMES), a.rounds)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m.save_model(str(out/'model.txt'))
    (out/'model.json').write_text(json.dumps(m.dump_model()))
    imp = sorted(zip(NAMES, m.feature_importance('gain')), key=lambda t: -t[1])
    (out/'meta.json').write_text(json.dumps(dict(names=NAMES, label_r=LABEL_R, rows=int(len(X)), steps=len(meta),
                                                 positive=float(y.mean()), oof_auc=float(auc), cv_auc=float(auc_cv),
                                                 within_auc=float(within/max(n, 1)),
                                                 within_cv_auc=float(within_cv/max(n, 1)), data=a.data,
                                                 params=params, rounds=a.rounds,
                                                 importance=[(k, float(v)) for k, v in imp]), indent=1))
    np.save(out/'oof.npy', oof.astype(np.float32))
    print('top features:', ', '.join('%s %.0f' % (k, v) for k, v in imp[:12]), flush=True)
    print('saved %s (%.0f s)' % (out, time.perf_counter()-t0), flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'fit':
        fit_main(sys.argv[2:])
    else:
        raise SystemExit('usage: reflex fit --data DIRS --out DIR')
