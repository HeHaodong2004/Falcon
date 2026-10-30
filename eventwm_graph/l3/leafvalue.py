"""Leaf-state features for a learned long-horizon value (Layer 3 option; analysis and selector).

Short-horizon imagination + learned value: a candidate is imagined (or, for analysis, executed exactly) for its
decision interval, and the state it reaches -- the LEAF -- is valued by a model trained on real outcomes.  The same
feature function describes an exact leaf (training data, diagnostics) and an imagined leaf (the legal selector), so
the value model sees the same inputs in both.

``legal_features`` uses only what a selector holds for an imagined leaf: the leaf pose and its recent history, the
legal map AT THE DECISION (and its route cost-to-go field), the people the robot perceives at the leaf (legal-format
tracks: the real tracker at an exact leaf, the camera model of the imagination at an imagined one), the executor's
state, the robot's speed level, the configuration and the episode clock.

``privileged_features`` (ANALYSIS / TRAINING ONLY) describes the true people at the leaf -- everyone, seen or not --
and their true positions 1, 2 and 3 s ahead, for a privileged critic.
"""
import math

import numpy as np

from eventwm_graph.querygen.dynamics import STEP_SECONDS

PLANNERS = ('orca', 'dwa', 'sfm')
N_PEOPLE = 3
FAMILY_KEYS = ('continue', 'goal', 'st_goal', 'rot', 'turn', 'scan', 'wait', 'stop', 'creep', 'kturn', 'other')

LEGAL_NAMES = (
    ['ctg', 'progress', 'ctg_rel', 'route_err', 'route_err_abs', 'local_min', 'goal_euclid']
    + ['free_%s' % d for d in ('f', 'fl', 'l', 'bl', 'b', 'br', 'r', 'fr')] + ['unknown_ahead']
    + ['p%d_%s' % (i, k) for i in range(N_PEOPLE)
       for k in ('d', 'bear_cos', 'bear_sin', 'closing', 'dca', 'tca', 'dstat', 'visible', 'age')]
    + ['n_1_5', 'n_3', 'n_5', 'min_dca', 'min_dstat', 'n_tracks']
    + ['speed', 'yaw_rate', 'level', 'still', 'escalated', 'running', 'failed', 'reason_unstick', 'reason_evade']
    + ['fam_%s' % f for f in FAMILY_KEYS]
    + ['clock', 'step_m', 'turn_rad', 'stateful', 'brake'] + ['planner_%s' % p for p in PLANNERS]
    + ['leaf_steps', 'terminal_collision', 'terminal_success', 'terminal_other'])

PRIV_NAMES = (['t%d_%s' % (i, k) for i in range(N_PEOPLE)
               for k in ('d', 'bear_cos', 'bear_sin', 'closing', 'dca', 'tca', 'dstat', 'in_fov')]
              + ['t_n_1_5', 't_n_3', 't_n_5', 't_min_dca', 't_min_dstat', 't_n_unseen_3']
              + ['f%d_%s_%d' % (i, c, s) for i in range(N_PEOPLE) for s in (1, 2, 3) for c in ('x', 'y')]
              + ['f_min_route_%d' % s for s in (1, 2, 3)])


def family(label, kind=None):
    lab = str(label or '')
    if lab == 'continue':
        return 'continue'
    if lab.startswith('st_goal'):
        return 'st_goal'
    if lab.startswith('goal'):
        return 'goal'
    if lab.startswith('scan') or lab == 'look_lost_human':
        return 'scan'
    if lab.startswith('rot'):
        return 'rot'
    if lab.startswith('turn'):
        return 'turn'
    for f in ('wait', 'stop', 'creep', 'kturn'):
        if lab.startswith(f) or kind == f:
            return f
    if kind == 'rotate':
        return 'rot'
    return 'other'


def _planar(pose):
    return np.array([float(pose[0]), -float(pose[2])]), float(pose[3])


def _robot_velocity(poses, times, k=4):
    """Planar velocity over the last ``k`` steps and yaw rate over the last step."""
    if len(poses) < 2:
        return np.zeros(2), 0.
    k = min(k, len(poses)-1)
    a, b = np.asarray(poses[-1-k], np.float64), np.asarray(poses[-1], np.float64)
    dt = max(float(times[-1])-float(times[-1-k]), 1e-6)
    v = np.array([(b[0]-a[0])/dt, -(b[2]-a[2])/dt])
    a1 = np.asarray(poses[-2], np.float64)
    dt1 = max(float(times[-1])-float(times[-2]), 1e-6)
    dyaw = math.atan2(math.sin(b[3]-a1[3]), math.cos(b[3]-a1[3]))/dt1
    return v, dyaw


def _person_block(P, psi, Vr, pos, vel, horizon=3.0):
    """Relative state of one person (planar ``pos`` / ``vel``) to the robot at P, heading psi, velocity Vr."""
    rel = pos-P
    d = float(np.hypot(*rel))
    bear = math.atan2(rel[1], rel[0])-psi
    vrel = vel-Vr
    closing = -float(rel @ vrel)/max(d, 1e-6)
    vv = float(vrel @ vrel)
    tca = min(max(-float(rel @ vrel)/vv, 0.), horizon) if vv > 1e-9 else 0.
    dca = float(np.hypot(*(rel+vrel*tca)))
    hv = float(vel @ vel)
    ts = min(max(-float(rel @ vel)/hv, 0.), horizon) if hv > 1e-9 else 0.
    dstat = float(np.hypot(*(rel+vel*ts)))
    return d, math.cos(bear), math.sin(bear), closing, dca, tca, dstat


def _route_direction(field, x, z, r=0.4, n=16):
    """Direction (world yaw convention: forward = (cos a, -sin a)) of steepest cost-to-go descent at (x, z)."""
    c0 = field.at(x, z)
    best, ba = math.inf, 0.
    for a in np.linspace(-math.pi, math.pi, n, endpoint=False):
        c = field.at(x+r*math.cos(a), z-r*math.sin(a))
        if c < best:
            best, ba = c, float(a)
    return ba, best >= c0-0.05


def _free_distances(gmap, x, z, yaw, max_m=3.0, step=0.1):
    """Distance to the first legally BLOCKED cell along 8 rays around the heading, and the unknown share ahead."""
    from eventwm_graph.querygen.legal import BLOCKED, UNKNOWN
    out = []
    unknown = 0
    n = int(max_m/step)
    for j, off in enumerate((0., 45., 90., 135., 180., -135., -90., -45.)):
        a = yaw+math.radians(off)
        ca, sa = math.cos(a), -math.sin(a)
        dist = max_m
        for i in range(1, n+1):
            s = gmap.state_at(x+ca*i*step, z+sa*i*step)
            if s == BLOCKED:
                dist = i*step
                break
            if j == 0 and s == UNKNOWN:
                unknown += 1
        out.append(dist)
    return out, unknown/n


def legal_features(pose, poses, times, now, gmap, field, goal, tracks, execution=None, level=0, dyn=None,
                   planner='orca', step_index=0, leaf_steps=0, terminal=None):
    """Legal leaf features (``LEGAL_NAMES`` order).  ``gmap`` / ``field``: the legal map at the decision and its
    route cost-to-go (compiler.LegalField); ``tracks``: legal-format tracks perceived at the leaf."""
    from eventwm_graph.querygen.control import human_estimates
    x, z, yaw = float(pose[0]), float(pose[2]), float(pose[3])
    P, psi = _planar(pose)
    Vr, yaw_rate = _robot_velocity(poses, times)
    ctg = float(field.at(x, z))
    start = float(field.start)
    rdir, local_min = _route_direction(field, x, z)
    err = math.atan2(math.sin(rdir-yaw), math.cos(rdir-yaw))
    f = [ctg, start-ctg, ctg/max(start, 1e-3), math.cos(err), abs(err), float(local_min),
         math.hypot(float(goal[0])-x, float(goal[2])-z)]
    free, unk = _free_distances(gmap, x, z, yaw)
    f += free+[unk]
    blocks = []
    for h in human_estimates(tracks, now):
        b = _person_block(P, psi, Vr, np.asarray(h.position, np.float64), np.asarray(h.velocity, np.float64))
        blocks.append(b+(float(h.visible), float(h.age)))
    blocks.sort(key=lambda b: b[0])
    for i in range(N_PEOPLE):
        f += list(blocks[i]) if i < len(blocks) else [10., 1., 0., 0., 10., 0., 10., 0., 0.]
    ds = [b[0] for b in blocks]
    f += [sum(d < 1.5 for d in ds), sum(d < 3. for d in ds), sum(d < 5. for d in ds),
          min([b[4] for b in blocks], default=10.), min([b[6] for b in blocks], default=10.), len(blocks)]
    ex = execution
    q = getattr(ex, 'query', None)
    nlev = max(len(getattr(dyn, 'levels', ()) or ()) - 1, 1) if dyn is not None else 1
    f += [float(np.hypot(*Vr)), yaw_rate, float(level)/nlev, min(float(getattr(ex, '_still', 0) or 0), 40.)/10.,
          float(bool(getattr(ex, 'escalated', False))), float(getattr(ex, 'status', '') == 'running'),
          float(getattr(ex, 'status', '') == 'failed'), float(getattr(ex, 'reason', '') == 'unstick'),
          float(getattr(ex, 'reason', '') == 'evade')]
    fam = family(getattr(q, 'label', None), getattr(q, 'kind', None)) if q is not None else 'other'
    f += [float(fam == k) for k in FAMILY_KEYS]
    f += [float(step_index)/500., float(getattr(dyn, 'step_m', 0.2)), float(getattr(dyn, 'turn_rad', 0.083)),
          float(bool(getattr(dyn, 'stateful', False))), float(getattr(dyn, 'brake', 0))]
    f += [float(planner == p) for p in PLANNERS]
    f += [float(leaf_steps)/16., float(terminal == 'human_collision'), float(terminal == 'success'),
          float(terminal is not None and terminal not in ('human_collision', 'success'))]
    return np.asarray(f, np.float32)


def privileged_features(pose, poses, times, tape_positions, index, fov=math.radians(90.)):
    """ANALYSIS ONLY: the true people at tape index ``index`` (everyone, seen or not) and their true positions 1, 2
    and 3 s ahead relative to the leaf pose (``PRIV_NAMES`` order)."""
    tape = np.asarray(tape_positions, np.float64)
    P, psi = _planar(pose)
    Vr, _ = _robot_velocity(poses, times)
    n = len(tape)
    i = min(int(index), n-1)
    cur = tape[i]
    blocks = []
    futs = []
    if len(cur):
        prev = tape[max(i-1, 1)] if i > 1 else cur
        nxt = tape[min(i+1, n-1)]
        dt = STEP_SECONDS*max(min(i+1, n-1)-max(i-1, 1), 1)
        for m in range(len(cur)):
            p = cur[m]
            if not np.isfinite(p).all():
                continue
            pos = np.array([p[0], -p[2]])
            v3 = (nxt[m]-prev[m])/dt if np.isfinite(nxt[m]).all() and np.isfinite(prev[m]).all() else np.zeros(3)
            vel = np.array([v3[0], -v3[2]])
            b = _person_block(P, psi, Vr, pos, vel)
            in_fov = float(abs(math.atan2(b[2], b[1])) <= fov/2 and b[0] <= 10.)
            blocks.append(b+(in_fov,))
            fut = []
            for s in (1, 2, 3):
                j = min(i+int(round(s/STEP_SECONDS)), n-1)
                q = tape[j][m]
                rel = np.array([q[0], -q[2]])-P if np.isfinite(q).all() else np.array([10., 10.])
                c, sn = math.cos(-psi), math.sin(-psi)
                fut.append((c*rel[0]-sn*rel[1], sn*rel[0]+c*rel[1]))
            futs.append((b[0], fut))
    order = np.argsort([b[0] for b in blocks]) if blocks else []
    f = []
    for k in range(N_PEOPLE):
        f += list(blocks[order[k]]) if k < len(blocks) else [10., 1., 0., 0., 10., 0., 10., 0.]
    ds = [b[0] for b in blocks]
    f += [sum(d < 1.5 for d in ds), sum(d < 3. for d in ds), sum(d < 5. for d in ds),
          min([b[4] for b in blocks], default=10.), min([b[6] for b in blocks], default=10.),
          sum(b[0] < 3. and not b[7] for b in blocks)]
    futs.sort(key=lambda t: t[0])
    for k in range(N_PEOPLE):
        fut = futs[k][1] if k < len(futs) else [(10., 10.)]*3
        for (fx, fy) in fut:
            f += [fx, fy]
    # true closest approach of anyone to the robot's straight-ahead corridor (1 m wide) within 1 / 2 / 3 s
    for s in (1, 2, 3):
        best = 10.
        for _, fut in futs:
            fx, fy = fut[s-1]
            if fx > -0.5:
                best = min(best, abs(fy) if fx < 3. else math.hypot(fx-3., fy))
        f.append(best)
    return np.asarray(f, np.float32)


# ------------------------------------------------------------------------------------------------ value model
class ValueModel:
    """A fitted leaf value (``leaffit.py --save``): P(success | leaf features).  LightGBM when importable, else a
    numpy evaluator of the dumped trees (``model.json``; identical predictions up to float rounding)."""

    def __init__(self, path):
        import json
        from pathlib import Path
        self.path = Path(path)
        self.meta = json.loads((self.path/'meta.json').read_text())
        self.keys = tuple(self.meta['keys'])
        self._lgb = None
        try:
            import lightgbm as lgb
            self._lgb = lgb.Booster(model_file=str(self.path/'model.txt'))
        except Exception:
            self.trees = [_flatten(t['tree_structure']) for t in json.loads((self.path/'model.json').read_text())['tree_info']]

    def predict(self, X):
        return 1./(1.+np.exp(-self.predict_raw(X)))

    def predict_raw(self, X):
        """Raw additive score (log-odds for a binary model, the ranking score for a lambdarank model)."""
        X = np.asarray(X, np.float64)
        if X.ndim == 1:
            X = X[None]
        if self._lgb is not None:
            return self._lgb.predict(X, raw_score=True)
        raw = np.zeros(len(X))
        for feat, thr, left, right, value, dleft in self.trees:
            node = np.zeros(len(X), np.int64)
            active = feat[node] >= 0
            while active.any():
                i = np.flatnonzero(active)
                f = feat[node[i]]
                x = X[i, f]
                go_left = np.where(np.isnan(x), dleft[node[i]], x <= thr[node[i]])
                node[i] = np.where(go_left, left[node[i]], right[node[i]])
                active = feat[node] >= 0
            raw += value[node]
        return raw


def _flatten(root):
    """LightGBM tree JSON -> arrays (feature, threshold, left, right, leaf value, default-left); leaves have feature -1."""
    feat, thr, left, right, value, dleft = [], [], [], [], [], []

    def add(n):
        k = len(feat)
        feat.append(-1); thr.append(0.); left.append(-1); right.append(-1); value.append(0.); dleft.append(True)
        if 'leaf_value' in n:
            value[k] = float(n['leaf_value'])
            return k
        feat[k] = int(n['split_feature']); thr[k] = float(n['threshold']); dleft[k] = bool(n.get('default_left', True))
        left[k] = add(n['left_child'])
        right[k] = add(n['right_child'])
        return k
    add(root)
    return (np.asarray(feat), np.asarray(thr), np.asarray(left), np.asarray(right), np.asarray(value), np.asarray(dleft))
