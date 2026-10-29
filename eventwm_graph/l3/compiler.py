"""Graph Compiler (Layer 2.5 -> Layer 3): one legal decision + its Sandbox imagination -> a consequence graph.

The compiled graph is a dense heterogeneous spatio-temporal graph in the ROBOT frame at decision time t:

  ego    [De]           the decision context (goal, legal route cost, time budget, recent motion, local legal map,
                        human situation incl. Stage 1's unseen-person mass, generator diagnostics)
  cand   [K, Dc]        one node per candidate Query: coarse intent + the IMAGINED consequences of executing it with
                        the robot's own planner and dynamics (motion, progress, legal-map/static risk, Stage 1 joint
                        human risk over the interval and a 3 s hold extension, Stage 2 hypothetical map increment)
  traj   [K, P, Dt]     P time-indexed waypoints of each imagined execution (pose, speed, legal map at the pose,
                        human proximity across the 40 Stage 1 worlds at that time)
  hum    [N, Dh]        tracked humans (legal tracks) with their Stage 1 future summary
  hfut   [N, P, Df]     each tracked human's Stage 1 future at the same P times (world mean, spread, presence)
  pair   [K, P, N, Dp]  traj(k, p) -- hfut(n, p) interaction edges at the same time (relative position, distance
                        statistics over the worlds)

Relations are implied by the tensor axes (cand-traj membership, traj temporal order along P, human-future
membership, same-time traj-hfut interaction, all-pairs cand-cand / ego-cand); ``model.py`` runs message passing on
them.  K and N vary per decision; arrays are stored unpadded and padded at batch time.

Agnosticism: nothing identifies the planner, the dynamics family or the robot.  Candidates enter only through what
their execution is imagined to DO (the same executor + planner + dynamics that will execute them; different robots
produce different consequences for the same Query) and through a coarse intent class that exists for every robot.
Execution-style modifiers of the generator (tight / close / thru copies, dynamics-specific subsets) are not encoded in
v1/v2; v3 encodes the Query itself (kind, family, execution flags) and the robot's information state (infofeat.py).

Legality: inputs are the legal state at t, the generator's legal candidates and diagnostics, and the Sandbox result
(hypothetical predictions from legal inputs).  Predicted map cells are used as features only and never merged into
any legal map.  No simulator, navmesh, geodesic, true future or Oracle quantity is read.
"""
import math
import os

import numpy as np

from eventwm_graph.querygen.legal import UNKNOWN, FREE, BLOCKED
from eventwm_graph.querygen.dynamics import STEP_SECONDS

SCHEMA = 'l3_graph_v1'
P = 8                         # waypoints per imagined execution (every 2 steps of the 16-step decision interval)
SAMPLE_STEPS = np.arange(1, P+1)*2
HOLD_STEPS = 62               # 3 s risk extension (Stage 1 horizon): the robot holds its imagined end pose
MAX_STEPS = 500
INTENTS = ('goal', 'stgoal', 'goto', 'wait', 'rotate', 'maneuver', 'stop', 'continue')
GOTO_PREFIX = ('pass', 'yield', 'sidestep', 'approach', 'advance', 'retreat', 'alt', 'frontier', 'goal_slow')
CROP_M = 9.0                  # planning-view half window around the robot for legal route costs (m)
UNKNOWN_COST = 1.25           # the default goal query's price of unobserved cells (generator goal_unknown_cost)
CTG_CAP = 40.


# ----------------------------------------------------------------------------- feature layout (names = columns)
EGO = (['goal_fwd', 'goal_left', 'goal_dist', 'goal_cos', 'goal_sin', 'ctg', 'ctg_ratio', 'route_ok',
        'time_used', 'time_left', 'speed', 'yaw_rate',
        'disp_16', 'disp_48', 'disp_96', 'prog_16', 'prog_48', 'prog_96', 'turn_frac_48', 'stall_48',
        'clear_here', 'bumps_near']
       + ['sec%d_%s' % (s, k) for s in range(8) for k in ('blk15', 'unk15', 'blk30', 'unk30')]
       + ['n_tracks', 'n_visible', 'near_human', 'near_closing', 'resid_near_1s', 'resid_near_3s', 'resid_mass',
          'hold_pcol', 'hold_psoc', 'hold_mind_q10', 'gen_stuck', 'gen_maneuver', 'gen_conflicts', 'gen_route',
          'n_cands'])
CAND = ([f'intent_{k}' for k in INTENTS]
        + ['continued', 'via_fwd', 'via_left', 'via_dist', 'duration']
        + ['steps', 'end_horizon', 'end_done', 'end_stop', 'end_failed', 'end_error', 'disp_fwd', 'disp_left',
           'dyaw_cos', 'dyaw_sin', 'path_len', 'mean_speed', 'frac_move', 'frac_turn', 'frac_hold']
        + ['goal_dist_end', 'progress', 'ctg_end', 'ctg_gain', 'at_goal', 'ctg_gain_per_step']
        + ['blocked_frac', 'unknown_frac', 'min_clear', 'mean_clear', 'bump_near']
        + ['pcol', 'psoc', 'soc_dur', 'soc_sev', 'mind_q10', 'mind_q50', 'mind_q90', 'p_nohum', 'tcol', 'nom_clear']
        + ['x_pcol', 'x_psoc', 'x_soc_dur', 'x_mind_q10', 'x_mind_q50', 'x_tcol']
        + ['m_cells', 'm_free', 'm_blk', 'm_free_goal', 'm_blk_goal', 'm_blk_ahead', 'm_hard_free', 'm_hard_blk'])
# v2 layout (L3_LAYOUT=v2): consequences of the imagined CONTINUATION after the interval (SandboxConfig.continuation):
# risk of interval + continuation (then the braking tail / hold) against every Stage 1 world, continuation progress
CAND_V2 = ['y_pcol', 'y_psoc', 'y_soc_dur', 'y_mind_q10', 'y_tcol', 'y_ctg_gain', 'y_disp', 'y_steps']
LAYOUT = os.environ.get('L3_LAYOUT', 'v1')
if LAYOUT == 'v2':
    CAND = CAND+CAND_V2
# v3 layout (L3_LAYOUT=v3): the query itself (kind, family, execution-style flags) and the robot's information state
# (camera view history, unseen-person mass by direction, lost tracks) with what each imagined execution would see
# (infofeat.py)
if LAYOUT in ('v3', 'v4'):
    from .infofeat import CAND_SEM, CAND_INFO, EGO_INFO, CAND_ROUTE
    CAND = CAND+CAND_SEM+CAND_INFO+(CAND_ROUTE if LAYOUT == 'v4' else [])
    EGO = EGO+EGO_INFO
TRAJ = ['t', 'active', 'fwd', 'left', 'hcos', 'hsin', 'speed', 'm_free', 'm_unk', 'm_blk', 'clear', 'ctg_gain',
        'p_soc', 'p_col', 'mind_q10', 'mind_q50', 'nom_clear']
HUM = ['fwd', 'left', 'vfwd', 'vleft', 'speed', 'dist', 'bcos', 'bsin', 'age', 'visible', 'closing',
       'f1_fwd', 'f1_left', 'f3_fwd', 'f3_left', 'f3_spread', 'presence', 'modeled']
HFUT = ['fwd', 'left', 'spread', 'presence', 'dist']
PAIR = ['dfwd', 'dleft', 'dist_mean', 'p_1m', 'p_col', 'dmin_q10']

# feature groups (for ablations): column names per group and node type
IMAGINATION_GROUPS = ('motion', 'progress', 'static', 'human', 'map')
GROUPS = {
    'intent': dict(cand=[c for c in CAND if c.startswith('intent_')]+['continued', 'via_fwd', 'via_left', 'via_dist', 'duration']),
    'motion': dict(cand=['steps', 'end_horizon', 'end_done', 'end_stop', 'end_failed', 'end_error', 'disp_fwd', 'disp_left',
                         'dyaw_cos', 'dyaw_sin', 'path_len', 'mean_speed', 'frac_move', 'frac_turn', 'frac_hold'],
                   traj=['t', 'active', 'fwd', 'left', 'hcos', 'hsin', 'speed']),
    'progress': dict(cand=['goal_dist_end', 'progress', 'ctg_end', 'ctg_gain', 'at_goal', 'ctg_gain_per_step'],
                     traj=['ctg_gain']),
    'static': dict(cand=['blocked_frac', 'unknown_frac', 'min_clear', 'mean_clear', 'bump_near'],
                   traj=['m_free', 'm_unk', 'm_blk', 'clear']),
    'human': dict(cand=['pcol', 'psoc', 'soc_dur', 'soc_sev', 'mind_q10', 'mind_q50', 'mind_q90', 'p_nohum', 'tcol',
                        'nom_clear', 'x_pcol', 'x_psoc', 'x_soc_dur', 'x_mind_q10', 'x_mind_q50', 'x_tcol'],
                  traj=['p_soc', 'p_col', 'mind_q10', 'mind_q50', 'nom_clear'],
                  ego=['resid_near_1s', 'resid_near_3s', 'resid_mass', 'hold_pcol', 'hold_psoc', 'hold_mind_q10']),
    'map': dict(cand=['m_cells', 'm_free', 'm_blk', 'm_free_goal', 'm_blk_goal', 'm_blk_ahead', 'm_hard_free', 'm_hard_blk']),
}
if LAYOUT == 'v2':
    GROUPS['continuation'] = dict(cand=list(CAND_V2))
    IMAGINATION_GROUPS = IMAGINATION_GROUPS+('continuation',)
if LAYOUT in ('v3', 'v4'):
    GROUPS['query'] = dict(cand=list(CAND_SEM))
    GROUPS['info'] = dict(cand=list(CAND_INFO), ego=list(EGO_INFO))
if LAYOUT == 'v4':
    GROUPS['route'] = dict(cand=list(CAND_ROUTE))


def column_index(names, cols):
    return [names.index(c) for c in cols]


# ----------------------------------------------------------------------------- geometry helpers
class Frame:
    """Robot frame at the decision pose: forward = (cos yaw, -sin yaw), left = (-sin yaw, -cos yaw) in world (x, z)."""

    def __init__(self, pose):
        self.x, self.z, self.yaw = float(pose[0]), float(pose[2]), float(pose[3])
        self.c, self.s = math.cos(self.yaw), math.sin(self.yaw)

    def rel(self, wx, wz):
        dx = np.asarray(wx, np.float64)-self.x
        dz = np.asarray(wz, np.float64)-self.z
        return dx*self.c-dz*self.s, -dx*self.s-dz*self.c

    def vrel(self, vx, vz):
        vx = np.asarray(vx, np.float64); vz = np.asarray(vz, np.float64)
        return vx*self.c-vz*self.s, -vx*self.s-vz*self.c

    def dyaw(self, yaw):
        d = np.asarray(yaw, np.float64)-self.yaw
        return np.arctan2(np.sin(d), np.cos(d))


def intent_of(label, kind, continued):
    if continued or kind == 'continue':
        return 'continue'
    if kind == 'stop':
        return 'stop'
    if kind in ('wait', 'wait_clear'):
        return 'wait'
    if kind in ('creep', 'kturn') or label.startswith(('man_', 'rot_', 'creep')):
        return 'maneuver'
    if kind == 'rotate':
        return 'rotate'
    if kind == 'stgoal' or label.startswith('st_goal'):
        return 'stgoal'
    if kind == 'goal' and not label.startswith(GOTO_PREFIX):
        return 'goal'
    return 'goto'


def _clip(x, lo, hi):
    return float(min(max(float(x), lo), hi))


def _wquant(values, weights, qs=(0.1, 0.5, 0.9)):
    """Weighted quantiles of finite values (inf -> cap 10 m); NaN-safe."""
    v = np.asarray(values, np.float64)
    w = np.asarray(weights, np.float64)
    v = np.where(np.isfinite(v), np.minimum(v, 10.), 10.)
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    tot = cw[-1] if len(cw) else 0.
    if tot <= 0:
        return [10.]*len(qs)
    return [float(v[min(np.searchsorted(cw, q*tot), len(v)-1)]) for q in qs]


class LegalField:
    """Legal route cost-to-go to the public goal on the CURRENT legal map (bounded window around the robot)."""

    def __init__(self, legal):
        from eventwm_graph.querygen.gridplan import PlanningView, INF
        g = legal.map
        pose = legal.pose
        robot = (float(pose[0]), float(pose[2]))
        goal = (float(legal.goal[0]), float(legal.goal[2]))
        cells = int(math.ceil(CROP_M/g.res))
        self.view = PlanningView(g, robot, goal, unknown_cost=UNKNOWN_COST, max_cells=cells)
        gi, gj = self.view.to_local(*goal)
        inside = self.view.inside(gi, gj)
        if inside:
            self.field = self.view.field([goal])
            self.extra = 0.
        else:
            # goal outside the window: seed the window border cells with their Euclidean distance to the goal
            # (optimistic boundary condition, the same idea as the generator's distant-goal handling)
            n0, n1 = self.view.state.shape
            border = [(i, j) for i in (0, n0-1) for j in range(n1)]+[(i, j) for j in (0, n1-1) for i in range(1, n0-1)]
            pts, dist = [], []
            for i, j in border[::2]:
                if self.view.blocked[i, j]:
                    continue
                x, z = self.view.to_world(i, j)
                pts.append((x, z)); dist.append(math.hypot(x-goal[0], z-goal[1]))
            self.field = _multi_field(self.view, pts, dist)
            self.extra = 0.
        self.INF = INF
        self.res = g.res
        c0 = self.at(robot[0], robot[1])
        self.start = c0

    def at(self, x, z):
        i, j = self.view.to_local(x, z)
        if not self.view.inside(i, j):
            return CTG_CAP
        v = float(self.field[i, j])
        if v >= self.INF/2:
            # robot / pose inside an inflated BLOCKED cell: cheapest passable neighbour within 2 cells
            best = self.INF
            for a in range(i-2, i+3):
                for b in range(j-2, j+3):
                    if self.view.inside(a, b):
                        best = min(best, float(self.field[a, b]))
            if best >= self.INF/2:
                return CTG_CAP
            v = best
        return min(v*self.res, CTG_CAP)

    def clearance(self, x, z):
        i, j = self.view.to_local(x, z)
        if not self.view.inside(i, j):
            return 2.
        return min(float(self.view.clearance[i, j]), 2.)


def _multi_field(view, pts, dist):
    from eventwm_graph.querygen.gridplan import _dijkstra
    si, sj, sd = [], [], []
    for (x, z), d in zip(pts, dist):
        i, j = view.to_local(x, z)
        si.append(i); sj.append(j); sd.append(d/view.res)
    if not si:
        return np.full(view.state.shape, 1e18)
    return _dijkstra(view.cost, np.asarray(si, np.int64), np.asarray(sj, np.int64), np.asarray(sd, np.float64),
                     view.blocked)


# ----------------------------------------------------------------------------- compiler
class GraphCompiler:
    """Stateless compiler.  ``compile(result, legal, queries, diag, steps)`` -> dict of float32 arrays + metadata."""

    schema = SCHEMA

    def __init__(self, cv_ablation=False, brake_tail=False):
        self.cv_ablation = bool(cv_ablation)
        # v1b: the 3 s risk extension starts with the Sandbox's imagined emergency stop (robots with momentum cannot
        # hold their imagined end pose instantly); identical to v1 for robots without speed state
        self.brake_tail = bool(brake_tail)
        if self.brake_tail:
            self.schema = SCHEMA+'b'
        if LAYOUT == 'v2':
            self.schema = 'l3_graph_v2'+('b' if self.brake_tail else '')
        if LAYOUT in ('v3', 'v4'):
            self.schema = 'l3_graph_%s' % LAYOUT+('b' if self.brake_tail else '')

    def compile(self, result, legal, queries, diag=None, steps=0, running=None):
        diag = diag or {}
        F = Frame(legal.pose)
        field = LegalField(legal)
        bank = result.humans
        cands = result.candidates
        K = len(cands)
        out = {}
        # ------------------------------------------------------- shared human futures (Stage 1 bank)
        W = np.asarray(bank.weights, np.float64)
        W = W/W.sum() if W.sum() > 0 else W
        keys = [str(k) for k in bank.track_keys]
        unseen = np.asarray(bank.unseen, bool)
        present = np.asarray(bank.present, bool)                       # [W, Nb]
        rel = np.arange(1, HOLD_STEPS+1)*STEP_SECONDS
        horizon = float(bank.times[-1])
        rel = rel[rel <= horizon+1e-9]
        HP = np.asarray(bank.sandbox_positions(rel), np.float64) if len(keys) else np.zeros((len(W), len(rel), 0, 2))
        # [W, T, Nb, 2] world (x, z), +inf where absent
        # tracked humans: legal tracks perceivable now
        tracks = [t for t in legal.tracks if t.get('valid', True)]
        slot = {k[len('track:'):]: n for n, k in enumerate(keys) if k.startswith('track:')}
        # ------------------------------------------------------- per candidate
        cand_rows, traj_rows, robots_ext, lengths_ext = [], [], [], []
        robots_cont, cand_v2_progress = [], []
        sample_xy = np.zeros((K, P, 2))                                  # world (x, z) at the sample steps
        sample_active = np.zeros((K, P), bool)
        for k, c in enumerate(cands):
            ro = c.rollout
            d = c.candidate
            q = queries[k] if k < len(queries) else None
            label = str(d.get('label') or '')
            kind = str(d.get('kind') or '')
            continued = bool(d.get('continued'))
            intent = intent_of(label, kind, continued)
            poses = np.asarray(ro.poses, np.float64)                     # [T+1, 4]
            T = len(poses)-1
            # descriptor
            via_f = via_l = via_d = 0.
            via = d.get('via') or []
            if via:
                vf, vl = F.rel(via[-1][0], via[-1][1])
                via_f, via_l = _clip(vf, -8, 8), _clip(vl, -8, 8)
                via_d = _clip(math.hypot(float(vf), float(vl)), 0, 10)
            dur = _clip((d.get('duration') or 0)/48., 0, 1)
            # motion
            fw, lf = F.rel(poses[:, 0], poses[:, 2])
            dy = F.dyaw(poses[:, 3])
            seg = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 2])) if T > 0 else np.zeros(0)
            dyaw_step = np.abs(np.angle(np.exp(1j*np.diff(poses[:, 3])))) if T > 0 else np.zeros(0)
            move = seg > 1e-4
            turn = (~move) & (dyaw_step > 1e-4)
            hold = (~move) & (~turn)
            ended = ro.ended_by
            # progress / legal route cost along the imagined path
            gx, gz = float(legal.goal[0]), float(legal.goal[2])
            ctg_end = field.at(poses[-1, 0], poses[-1, 2])
            ctg_gain = field.start-ctg_end
            gd_end = math.hypot(gx-poses[-1, 0], gz-poses[-1, 2])
            # static: legal map states and clearance along the path
            st = np.asarray([legal.map.state_at(p[0], p[2]) for p in poses])
            clear = np.asarray([field.clearance(p[0], p[2]) for p in poses])
            bumps = np.asarray(getattr(legal, 'bumps', ()) or np.zeros((0, 2)), np.float64).reshape(-1, 2)
            bump_near = 0.
            if len(bumps):
                dd = np.hypot(bumps[:, 0][None]-poses[:, 0][:, None], bumps[:, 1][None]-poses[:, 2][:, None])
                bump_near = float((dd.min(1) < 0.8).mean())
            # human risk over the interval (Layer 2 reductions)
            h = c.humans or {}
            mq = [float(x) if x is not None and np.isfinite(x) else 10. for x in (h.get('minimum_quantiles') or [10., 10., 10.])]
            fcm = np.asarray(h.get('first_collision_mass', np.zeros(1)), np.float64)
            tcol = float((fcm*np.arange(1, len(fcm)+1)).sum()/max(fcm.sum(), 1e-9))*STEP_SECONDS if fcm.sum() > 0 else 3.
            # Stage 2 hypothetical map increment
            m = c.map
            if m is not None and len(m['states']):
                cells = np.asarray(m['candidate_cells'], np.float64)
                pr = np.asarray(m['probabilities'], np.float64)
                res_ = legal.map.res
                cx = (cells[:, 0]+0.5)*res_; cz = (cells[:, 1]+0.5)*res_
                ex, ez = poses[-1, 0], poses[-1, 2]
                to_goal = np.array([gx-ex, gz-ez]); ng = np.linalg.norm(to_goal)
                vx, vz = cx-ex, cz-ez
                nv = np.hypot(vx, vz)+1e-9
                cosang = (vx*to_goal[0]+vz*to_goal[1])/(nv*max(ng, 1e-9)) if ng > 1e-6 else np.zeros_like(nv)
                goalward = cosang > math.cos(math.radians(45))
                ahead = goalward & (nv < 3.) & (cosang > math.cos(math.radians(20)))
                hard = np.asarray(m['states'])
                mfeat = [math.log1p(len(cells))/8., math.log1p(pr[:, 1].sum())/6., math.log1p(pr[:, 2].sum())/6.,
                         math.log1p(pr[goalward, 1].sum())/6., math.log1p(pr[goalward, 2].sum())/6.,
                         math.log1p(pr[ahead, 2].sum())/5., math.log1p(float((hard == 1).sum()))/6.,
                         math.log1p(float((hard == 2).sum()))/6.]
            else:
                mfeat = [0.]*8
            row = ([1. if intent == x else 0. for x in INTENTS]
                   + [float(continued), via_f/5., via_l/5., via_d/5., dur]
                   + [T/16., float(ended == 'horizon'), float(ended == 'done'), float(ended == 'stop'),
                      float(ended == 'failed'), float(ended == 'error'), _clip(fw[-1], -4, 4)/2., _clip(lf[-1], -4, 4)/2.,
                      math.cos(dy[-1]), math.sin(dy[-1]), _clip(ro.path_length, 0, 6)/3.,
                      _clip(ro.path_length/max(T*STEP_SECONDS, 1e-6), 0, 6)/4. if T else 0.,
                      float(move.mean()) if T else 0., float(turn.mean()) if T else 0., float(hold.mean()) if T else 1.]
                   + [_clip(gd_end, 0, 30)/10., _clip(ro.progress, -4, 4)/2., ctg_end/20., _clip(ctg_gain, -6, 6)/3.,
                      float(gd_end < 0.2), _clip(ctg_gain/max(T, 1), -0.4, 0.4)*4.]
                   + [float((st == BLOCKED).mean()), float((st == UNKNOWN).mean()), float(clear.min()), float(clear.mean()),
                      bump_near]
                   + [float(h.get('p_collision', 0.)), float(h.get('p_social', 0.)),
                      _clip(h.get('expected_social_duration', 0.), 0, 3), _clip(h.get('expected_social_severity', 0.), 0, 3),
                      mq[0]/5., mq[1]/5., mq[2]/5., float(h.get('p_no_humans', 1.)), tcol/3.,
                      _clip(ro.nominal_clearance if np.isfinite(ro.nominal_clearance) else 10., 0, 10)/5.]
                   + [0.]*6
                   + mfeat
                   + ([0.]*len(CAND_V2) if LAYOUT == 'v2' else [])
                   + ([0.]*(len(CAND_SEM)+len(CAND_INFO)) if LAYOUT in ('v3', 'v4') else [])
                   + ([0.]*len(CAND_ROUTE) if LAYOUT == 'v4' else []))
            cand_rows.append(row)
            # waypoints
            idx = np.minimum(SAMPLE_STEPS, T)
            sample_active[k] = SAMPLE_STEPS <= T
            sp = poses[idx]
            sample_xy[k] = sp[:, [0, 2]]
            spd = np.zeros(P)
            for p_, i in enumerate(idx):
                a = max(i-2, 0)
                spd[p_] = math.hypot(poses[i, 0]-poses[a, 0], poses[i, 2]-poses[a, 2])/max((i-a)*STEP_SECONDS, 1e-6) if i > a else 0.
            sw, sl = F.rel(sp[:, 0], sp[:, 2])
            sdy = F.dyaw(sp[:, 3])
            sst = st[idx]
            trows = []
            for p_ in range(P):
                i = idx[p_]
                trows.append([SAMPLE_STEPS[p_]/16., float(sample_active[k, p_]), _clip(sw[p_], -4, 4)/2., _clip(sl[p_], -4, 4)/2.,
                              math.cos(sdy[p_]), math.sin(sdy[p_]), _clip(spd[p_], 0, 6)/4.,
                              float(sst[p_] == FREE), float(sst[p_] == UNKNOWN), float(sst[p_] == BLOCKED), clear[i]/2.,
                              _clip(field.start-field.at(sp[p_, 0], sp[p_, 2]), -6, 6)/3.,
                              0., 0., 0., 0., 0.])
            traj_rows.append(trows)
            if LAYOUT == 'v2':
                cont = getattr(ro, 'cont', None)
                cont = np.zeros((0, 4)) if cont is None else np.asarray(cont, np.float64)
                cseq = np.concatenate([poses[1:], cont]) if len(cont) else poses[1:]
                ctail = getattr(ro, 'tail', None)
                cext = np.zeros((len(rel), 4))
                nn_ = min(len(cseq), len(rel))
                cext[:nn_] = cseq[:nn_]
                cext[nn_:] = cseq[nn_-1] if nn_ else poses[-1]
                robots_cont.append(cext)
                cend = cont[-1] if len(cont) else poses[-1]
                cand_v2_progress.append([_clip(ctg_end-field.at(cend[0], cend[2]), -6, 6)/3.,
                                         _clip(math.hypot(cend[0]-poses[-1, 0], cend[2]-poses[-1, 2]), 0, 6)/3.,
                                         len(cont)/32.])
            # hold extension: imagined poses then the end pose held for the remaining Stage 1 horizon
            ext = np.zeros((len(rel), 4))
            seq = poses[1:]
            tail = getattr(ro, 'tail', None) if self.brake_tail else None
            if tail is not None and len(tail):
                seq = np.concatenate([seq, tail])
            n_im = min(len(seq), len(rel))
            ext[:n_im] = seq[:n_im]
            ext[n_im:] = seq[n_im-1] if n_im else poses[-1]
            robots_ext.append(ext)
            lengths_ext.append(len(rel))
        cand = np.asarray(cand_rows, np.float32).reshape(K, len(CAND))
        traj = np.asarray(traj_rows, np.float32).reshape(K, P, len(TRAJ))
        info = None
        if LAYOUT in ('v3', 'v4'):
            from .infofeat import InfoState, query_semantics, route_features
            info = InfoState(legal, F, keys, unseen, HP, W)
            s0, i0 = CAND.index(CAND_SEM[0]), CAND.index(CAND_INFO[0])
            if LAYOUT == 'v4':
                r0 = CAND.index(CAND_ROUTE[0])
                # intention scored over the bank's own horizon (Stage 1: 3 s; the analysis-only true-future bank may be
                # longer, L3_ORACLE_BANK_STEPS)
                HPr = HP
                nb = int(getattr(bank, 'long_steps', 0) or 0)
                if nb > len(rel) and len(keys):
                    HPr = np.asarray(bank.sandbox_positions(np.arange(1, nb+1)*STEP_SECONDS), np.float64)
                vh = np.asarray(legal.poses[-24:], np.float64)
                sp = np.hypot(np.diff(vh[:, 0]), np.diff(vh[:, 2])) if len(vh) > 1 else np.zeros(0)
                step_m = float(np.median(sp[sp > 0.02])) if (sp > 0.02).sum() >= 3 else 0.15
            for k, c in enumerate(cands):
                q = queries[k] if k < len(queries) else None
                cont = bool(c.candidate.get('continued')) or (q is not None and q.kind == 'continue')
                cand[k, s0:s0+len(CAND_SEM)] = query_semantics(running if cont else q, F, continued=cont)
                cand[k, i0:i0+len(CAND_INFO)] = info.cand(np.asarray(c.rollout.poses, np.float64))
                if LAYOUT == 'v4':
                    ro = c.rollout
                    moving = np.hypot(np.diff(ro.poses[:, 0]), np.diff(ro.poses[:, 2])) if len(ro.poses) > 1 else np.zeros(0)
                    v = float(moving[moving > 0.02].mean()) if (moving > 0.02).sum() >= 3 else step_m
                    cand[k, r0:r0+len(CAND_ROUTE)] = route_features(info, ro, field, legal, HPr, W, keys, unseen, v)
        # ------------------------------------------------------- human-side statistics against all worlds
        Nb = HP.shape[2]
        if K and Nb and len(rel):
            from eventwm_graph.layer2.reduction import aggregate_joint
            robots = np.stack(robots_ext)                                   # [K, T, 4] (x, y, z, yaw)
            red = aggregate_joint(HP, robots, np.asarray(lengths_ext), W)
            fcm = red['first_collision_mass']
            tt = np.arange(1, fcm.shape[1]+1)*STEP_SECONDS
            for k in range(K):
                mq = [float(x) if x is not None and np.isfinite(x) else 10. for x in red['minimum_quantiles'][k]]
                tcol = float((fcm[k]*tt).sum()/max(fcm[k].sum(), 1e-9)) if fcm[k].sum() > 0 else 3.
                cand[k, CAND.index('x_pcol')] = red['p_collision'][k]
                cand[k, CAND.index('x_psoc')] = red['p_social'][k]
                cand[k, CAND.index('x_soc_dur')] = min(float(red['expected_social_duration'][k]), 3.)
                cand[k, CAND.index('x_mind_q10')] = mq[0]/5.
                cand[k, CAND.index('x_mind_q50')] = mq[1]/5.
                cand[k, CAND.index('x_tcol')] = tcol/3.
            # per-waypoint statistics: distances robot centre -> every bank person at the sample times
            ti = np.minimum(SAMPLE_STEPS-1, len(rel)-1)
            Hs = HP[:, ti]                                                  # [W, P, Nb, 2]
            R = sample_xy                                                   # [K, P, 2]
            dx = Hs[None, :, :, :, 0]-R[:, None, :, None, 0]
            dz = Hs[None, :, :, :, 1]-R[:, None, :, None, 1]
            dist = np.where(np.isfinite(dx) & np.isfinite(dz), np.hypot(dx, dz), np.inf)   # [K, W, P, Nb]
            dmin = dist.min(-1)                                             # [K, W, P]
            p_soc = np.einsum('kwp,w->kp', (dmin < 1.0).astype(np.float64), W)
            p_col = np.einsum('kwp,w->kp', (dmin < 0.55).astype(np.float64), W)
            for k in range(K):
                for p_ in range(P):
                    q10, q50 = _wquant(dmin[k, :, p_], W, (0.1, 0.5))
                    traj[k, p_, TRAJ.index('mind_q10')] = q10/5.
                    traj[k, p_, TRAJ.index('mind_q50')] = q50/5.
            traj[:, :, TRAJ.index('p_soc')] = p_soc
            traj[:, :, TRAJ.index('p_col')] = p_col
        else:
            dist = None
            for k in range(K):
                cand[k, CAND.index('x_mind_q10')] = cand[k, CAND.index('x_mind_q50')] = 2.
                cand[k, CAND.index('x_tcol')] = 1.
            traj[:, :, TRAJ.index('mind_q10')] = 2.
            traj[:, :, TRAJ.index('mind_q50')] = 2.
        if LAYOUT == 'v2' and K:
            for k in range(K):
                cand[k, CAND.index('y_ctg_gain')], cand[k, CAND.index('y_disp')], cand[k, CAND.index('y_steps')] = cand_v2_progress[k]
            if Nb and len(rel):
                from eventwm_graph.layer2.reduction import aggregate_joint
                red = aggregate_joint(HP, np.stack(robots_cont), np.asarray([len(rel)]*K), W)
                fcm = red['first_collision_mass']
                tt = np.arange(1, fcm.shape[1]+1)*STEP_SECONDS
                for k in range(K):
                    mq = [float(x) if x is not None and np.isfinite(x) else 10. for x in red['minimum_quantiles'][k]]
                    cand[k, CAND.index('y_pcol')] = red['p_collision'][k]
                    cand[k, CAND.index('y_psoc')] = red['p_social'][k]
                    cand[k, CAND.index('y_soc_dur')] = min(float(red['expected_social_duration'][k]), 3.)
                    cand[k, CAND.index('y_mind_q10')] = mq[0]/5.
                    cand[k, CAND.index('y_tcol')] = (float((fcm[k]*tt).sum()/max(fcm[k].sum(), 1e-9)) if fcm[k].sum() > 0 else 3.)/3.
            else:
                for k in range(K):
                    cand[k, CAND.index('y_mind_q10')] = 2.
                    cand[k, CAND.index('y_tcol')] = 1.
        # nominal clearance at the sample times (the people the imagined planner perceived)
        nom = result.nominal
        if K and nom.positions.shape[1]:
            npos = nom.positions[np.minimum(SAMPLE_STEPS, nom.steps)][:, :, [0, 2]]   # [P, M, 2]
            dd = np.hypot(npos[None, :, :, 0]-sample_xy[:, :, None, 0], npos[None, :, :, 1]-sample_xy[:, :, None, 1])
            traj[:, :, TRAJ.index('nom_clear')] = np.minimum(dd.min(-1), 10.)/5.
        else:
            traj[:, :, TRAJ.index('nom_clear')] = 2.
        # ------------------------------------------------------- tracked humans and their futures
        hum_rows, hfut_rows, pair = [], [], []
        hold_robot = np.repeat(np.asarray(legal.pose, np.float64)[None], max(len(rel), 1), 0)
        for t in tracks[:12]:
            pos = np.asarray(t['position'], np.float64)
            vel = np.asarray(t.get('estimated_velocity') or (0., 0., 0.), np.float64)
            hf, hl = F.rel(pos[0], pos[2])
            vf, vl = F.vrel(vel[0], vel[2])
            dist0 = math.hypot(float(hf), float(hl))
            closing = -(float(hf)*float(vf)+float(hl)*float(vl))/max(dist0, 1e-6)
            age = float(legal.time)-float(t.get('last_seen_time', legal.time))
            n = slot.get(str(t['track_id']))
            modeled = n is not None and not unseen[n]
            if modeled and len(rel):
                pw = present[:, n]
                ww = W*pw
                presence = float(ww.sum())
                Q = HP[:, :, n, :]                                           # [W, T, 2]
                ok = np.isfinite(Q).all(-1)                                  # [W, T]
                wt = ww[:, None]*ok
                den = np.maximum(wt.sum(0), 1e-9)
                Qz = np.where(ok[..., None], Q, 0.)
                mean = (wt[..., None]*Qz).sum(0)/den[:, None]                # [T, 2]
                var = (wt[..., None]*(Qz-mean[None])**2).sum(0)/den[:, None]
                spread = np.sqrt(var.sum(-1))
                # fall back to constant velocity where no world has the person
                cvp = np.stack([pos[0]+vel[0]*rel, pos[2]+vel[2]*rel], -1)
                mean = np.where((den > 1e-6)[:, None], mean, cvp)
            else:
                presence = 1.
                mean = np.stack([pos[0]+vel[0]*rel, pos[2]+vel[2]*rel], -1) if len(rel) else np.zeros((0, 2))
                spread = 0.2+0.3*rel if len(rel) else np.zeros(0)
                Q = None
            def at(sec):
                if not len(rel):
                    return (float(hf), float(hl))
                i = min(int(round(sec/STEP_SECONDS))-1, len(rel)-1)
                return F.rel(mean[i, 0], mean[i, 1])
            f1 = at(1.0); f3 = at(3.0)
            sp3 = float(spread[-1]) if len(rel) else 0.
            hum_rows.append([_clip(hf, -10, 10)/5., _clip(hl, -10, 10)/5., _clip(vf, -3, 3), _clip(vl, -3, 3),
                             _clip(math.hypot(float(vf), float(vl)), 0, 3), _clip(dist0, 0, 10)/5.,
                             float(hf)/max(dist0, 1e-6), float(hl)/max(dist0, 1e-6), _clip(age, 0, 3),
                             float(bool(t.get('visible_now'))), _clip(closing, -3, 3),
                             _clip(f1[0], -10, 10)/5., _clip(f1[1], -10, 10)/5., _clip(f3[0], -10, 10)/5., _clip(f3[1], -10, 10)/5.,
                             _clip(sp3, 0, 5)/2., presence, float(modeled)])
            ti = np.minimum(SAMPLE_STEPS-1, max(len(rel)-1, 0))
            frows = []
            for p_ in range(P):
                if len(rel):
                    mf, ml = F.rel(mean[ti[p_], 0], mean[ti[p_], 1])
                    s_ = float(spread[ti[p_]])
                else:
                    mf, ml, s_ = hf, hl, 0.
                frows.append([_clip(mf, -10, 10)/5., _clip(ml, -10, 10)/5., _clip(s_, 0, 5)/2., presence,
                              _clip(math.hypot(float(mf), float(ml)), 0, 10)/5.])
            hfut_rows.append(frows)
            # interaction edges with every candidate waypoint at the same time
            prow = np.zeros((K, P, len(PAIR)), np.float32)
            for k in range(K):
                for p_ in range(P):
                    rx, rz = sample_xy[k, p_]
                    mx, mz = (mean[ti[p_]] if len(rel) else (pos[0], pos[2]))
                    # relative position of the human mean in the waypoint's heading frame
                    yaw = float(np.asarray(cands[k].rollout.poses)[min(SAMPLE_STEPS[p_], len(cands[k].rollout.poses)-1), 3])
                    cc, ss = math.cos(yaw), math.sin(yaw)
                    ddx, ddz = mx-rx, mz-rz
                    prow[k, p_, 0] = _clip(ddx*cc-ddz*ss, -10, 10)/5.
                    prow[k, p_, 1] = _clip(-ddx*ss-ddz*cc, -10, 10)/5.
                    prow[k, p_, 2] = _clip(math.hypot(ddx, ddz), 0, 10)/5.
                    if Q is not None:
                        qd = Q[:, ti[p_]]
                        okw = np.isfinite(qd).all(-1)
                        dw = np.where(okw, np.hypot(qd[:, 0]-rx, qd[:, 1]-rz), np.inf)
                        prow[k, p_, 3] = float((W*(dw < 1.0)).sum())
                        prow[k, p_, 4] = float((W*(dw < 0.55)).sum())
                        prow[k, p_, 5] = _wquant(dw, W, (0.1,))[0]/5.
                    else:
                        dd_ = math.hypot(ddx, ddz)
                        prow[k, p_, 3] = float(dd_ < 1.0)
                        prow[k, p_, 4] = float(dd_ < 0.55)
                        prow[k, p_, 5] = _clip(dd_, 0, 10)/5.
            pair.append(prow)
        N = len(hum_rows)
        hum = np.asarray(hum_rows, np.float32).reshape(N, len(HUM))
        hfut = np.asarray(hfut_rows, np.float32).reshape(N, P, len(HFUT))
        pair = np.stack(pair, 2) if N else np.zeros((K, P, 0, len(PAIR)), np.float32)   # [K, P, N, Dp]
        # ------------------------------------------------------- ego
        out['ego'] = self._ego(legal, F, field, diag, steps, K, tracks, keys, unseen, HP, W, rel)
        if info is not None:
            out['ego'] = np.concatenate([out['ego'], info.ego()]).astype(np.float32)
        out.update(cand=cand, traj=traj, hum=hum, hfut=hfut, pair=pair.astype(np.float32))
        return out

    def _ego(self, legal, F, field, diag, steps, K, tracks, keys, unseen, HP, W, rel):
        gx, gz = float(legal.goal[0]), float(legal.goal[2])
        gf, gl = F.rel(gx, gz)
        gd = math.hypot(float(gf), float(gl))
        poses = np.asarray(legal.poses, np.float64)
        def disp(n):
            a = poses[max(len(poses)-1-n, 0)]
            return math.hypot(poses[-1, 0]-a[0], poses[-1, 2]-a[2])
        def prog(n):
            a = poses[max(len(poses)-1-n, 0)]
            return math.hypot(gx-a[0], gz-a[2])-math.hypot(gx-poses[-1, 0], gz-poses[-1, 2])
        tail = poses[-49:]
        seg = np.hypot(np.diff(tail[:, 0]), np.diff(tail[:, 2])) if len(tail) > 1 else np.zeros(1)
        dyaw = np.abs(np.angle(np.exp(1j*np.diff(tail[:, 3])))) if len(tail) > 1 else np.zeros(1)
        turn_frac = float(((seg < 1e-4) & (dyaw > 1e-4)).mean())
        stall = float(disp(48) < 0.3)
        v, w = legal.velocity(4) if hasattr(legal, 'velocity') else (np.zeros(2), 0.)
        speed = float(np.hypot(*v))
        # local legal map in 8 robot-frame sectors (1.5 m and 3 m)
        g = legal.map
        r_cells = int(math.ceil(3.0/g.res))
        ci, cj = g.index(F.x, F.z)
        ii, jj = np.mgrid[ci-r_cells:ci+r_cells+1, cj-r_cells:cj+r_cells+1]
        sub = np.zeros(ii.shape, np.uint8)
        a0, a1 = max(ci-r_cells, 0), min(ci+r_cells+1, g.state.shape[0])
        b0, b1 = max(cj-r_cells, 0), min(cj+r_cells+1, g.state.shape[1])
        if a1 > a0 and b1 > b0:
            sub[a0-(ci-r_cells):a1-(ci-r_cells), b0-(cj-r_cells):b1-(cj-r_cells)] = g.state[a0:a1, b0:b1]
        wx = (ii+g.ox+0.5)*g.res; wz = (jj+g.oz+0.5)*g.res
        fw, lf = F.rel(wx, wz)
        rr = np.hypot(fw, lf)
        sec = ((np.arctan2(lf, fw)+math.pi)/(2*math.pi)*8).astype(int) % 8
        secf = []
        for s in range(8):
            for rmax in (1.5, 3.0):
                msk = (sec == s) & (rr <= rmax) & (rr > 0.3)
                n = max(int(msk.sum()), 1)
                secf += [float((sub[msk] == BLOCKED).sum())/n, float((sub[msk] == UNKNOWN).sum())/n]
        # reorder to (blk15, unk15, blk30, unk30) per sector
        bumps = np.asarray(getattr(legal, 'bumps', ()) or np.zeros((0, 2)), np.float64).reshape(-1, 2)
        bumps_near = float((np.hypot(bumps[:, 0]-F.x, bumps[:, 1]-F.z) < 1.5).sum()) if len(bumps) else 0.
        # humans
        nt = len(tracks)
        nvis = sum(bool(t.get('visible_now')) for t in tracks)
        near, near_closing = 10., 0.
        for t in tracks:
            pos = np.asarray(t['position'], np.float64)
            vel = np.asarray(t.get('estimated_velocity') or (0., 0., 0.), np.float64)
            hf, hl = F.rel(pos[0], pos[2])
            d = math.hypot(float(hf), float(hl))
            if d < near:
                vf, vl = F.vrel(vel[0], vel[2])
                near, near_closing = d, -(float(hf)*float(vf)+float(hl)*float(vl))/max(d, 1e-6)
        # Stage 1's unseen people (residual slots) near the robot and the risk of holding the current pose
        resid1 = resid3 = mass = 0.
        hold_pcol = hold_psoc = 0.; hold_q10 = 10.
        if HP.shape[2] and len(rel):
            res_slots = np.asarray([k.startswith('residual:') for k in keys], bool)
            D = np.hypot(HP[..., 0]-F.x, HP[..., 1]-F.z)
            D = np.where(np.isfinite(D), D, np.inf)                          # [W, T, Nb]
            i1 = min(int(round(1.0/STEP_SECONDS))-1, len(rel)-1)
            if res_slots.any():
                resid1 = float((W[:, None]*(D[:, i1, res_slots] < 3.)).sum())
                resid3 = float((W[:, None]*(D[:, :, res_slots] < 3.).any(1)).sum())
                mass = float((W[:, None]*np.isfinite(HP[:, 0, res_slots, 0])).sum())
            dmin = D.min(-1)                                                 # [W, T]
            hold_psoc = float((W*(dmin < 1.).any(1)).sum())
            hold_pcol = float((W*(dmin < 0.55).any(1)).sum())
            hold_q10 = _wquant(dmin.min(1), W, (0.1,))[0]
        ctg = field.start
        row = ([_clip(gf, -20, 20)/10., _clip(gl, -20, 20)/10., _clip(gd, 0, 30)/10., float(gf)/max(gd, 1e-6),
                float(gl)/max(gd, 1e-6), ctg/20., _clip(ctg/max(gd, 0.2), 0, 5)/2., float(ctg < CTG_CAP),
                steps/MAX_STEPS, 1.-steps/MAX_STEPS, _clip(speed, 0, 6)/4., _clip(w, -3, 3),
                _clip(disp(16), 0, 4)/2., _clip(disp(48), 0, 10)/4., _clip(disp(96), 0, 20)/6.,
                _clip(prog(16), -4, 4)/2., _clip(prog(48), -10, 10)/4., _clip(prog(96), -20, 20)/6., turn_frac, stall,
                field.clearance(F.x, F.z)/2., min(bumps_near, 5.)/5.]
               + secf
               + [min(nt, 12)/6., min(nvis, 12)/6., near/5., _clip(near_closing, -3, 3), resid1, resid3, min(mass, 6.)/3.,
                  hold_pcol, hold_psoc, hold_q10/5., float(bool(diag.get('stuck'))), float(bool(diag.get('maneuver_mode'))),
                  min(float(diag.get('conflicts', 0) or 0), 5.)/5., _clip(diag.get('route_len', 0.) or 0., 0, 40)/20.,
                  K/10.])
        return np.asarray(row, np.float32)


class _NoBank:
    """An empty Stage 1 bank (no-imagination compile: nobody is predicted beyond constant velocity)."""
    weights = np.ones(1)
    times = np.asarray([0., 3.])
    track_keys = ()
    unseen = np.zeros(0, bool)
    present = np.zeros((1, 0), bool)

    @staticmethod
    def sandbox_positions(rel):
        return np.zeros((1, len(rel), 0, 2))


def compile_blind(legal, queries, diag=None, steps=0, running=None):
    """The NO-IMAGINATION input (ablation): the same graph layout built from the legal state and the candidates'
    query descriptors only -- no imagined execution, no Stage 1 / Stage 2 output.  Candidate nodes carry the coarse
    intent, the continued flag, the via-point geometry and the requested duration; waypoint and interaction tensors are
    zero; tracked humans keep their current state with constant-velocity extrapolations."""
    from types import SimpleNamespace
    comp = GraphCompiler()
    K = len(queries)
    F = Frame(legal.pose)
    field = LegalField(legal)
    cand = np.zeros((K, len(CAND)), np.float32)
    for k, q in enumerate(queries):
        continued = q.kind == 'continue'
        intent = intent_of(q.label, q.kind, continued)
        cand[k, CAND.index('intent_'+intent)] = 1.
        cand[k, CAND.index('continued')] = float(continued)
        if q.via:
            vf, vl = F.rel(q.via[-1][0], q.via[-1][1])
            cand[k, CAND.index('via_fwd')] = _clip(vf, -8, 8)/5.
            cand[k, CAND.index('via_left')] = _clip(vl, -8, 8)/5.
            cand[k, CAND.index('via_dist')] = _clip(math.hypot(float(vf), float(vl)), 0, 10)/5.
        cand[k, CAND.index('duration')] = _clip((q.duration or 0)/48., 0, 1)
        if LAYOUT in ('v3', 'v4'):
            from .infofeat import query_semantics
            s0 = CAND.index(CAND_SEM[0])
            cand[k, s0:s0+len(CAND_SEM)] = query_semantics(running if continued else q, F, continued=continued)
    tracks = [t for t in legal.tracks if t.get('valid', True)]
    rel = np.arange(1, HOLD_STEPS+1)*STEP_SECONDS
    hum_rows, hfut_rows = [], []
    for t in tracks[:12]:
        pos = np.asarray(t['position'], np.float64)
        vel = np.asarray(t.get('estimated_velocity') or (0., 0., 0.), np.float64)
        hf, hl = F.rel(pos[0], pos[2]); vf, vl = F.vrel(vel[0], vel[2])
        d0 = math.hypot(float(hf), float(hl))
        closing = -(float(hf)*float(vf)+float(hl)*float(vl))/max(d0, 1e-6)
        age = float(legal.time)-float(t.get('last_seen_time', legal.time))
        cv = lambda sec: F.rel(pos[0]+vel[0]*sec, pos[2]+vel[2]*sec)
        f1, f3 = cv(1.), cv(3.)
        hum_rows.append([_clip(hf, -10, 10)/5., _clip(hl, -10, 10)/5., _clip(vf, -3, 3), _clip(vl, -3, 3),
                         _clip(math.hypot(float(vf), float(vl)), 0, 3), _clip(d0, 0, 10)/5., float(hf)/max(d0, 1e-6),
                         float(hl)/max(d0, 1e-6), _clip(age, 0, 3), float(bool(t.get('visible_now'))), _clip(closing, -3, 3),
                         _clip(f1[0], -10, 10)/5., _clip(f1[1], -10, 10)/5., _clip(f3[0], -10, 10)/5., _clip(f3[1], -10, 10)/5.,
                         0., 1., 0.])
        frows = []
        for p_ in range(P):
            mf, ml = cv(SAMPLE_STEPS[p_]*STEP_SECONDS)
            frows.append([_clip(mf, -10, 10)/5., _clip(ml, -10, 10)/5., 0., 1., _clip(math.hypot(float(mf), float(ml)), 0, 10)/5.])
        hfut_rows.append(frows)
    N = len(hum_rows)
    ego = comp._ego(legal, F, field, diag or {}, steps, K, tracks, [], np.zeros(0, bool), np.zeros((1, 0, 0, 2)),
                    np.ones(1), rel)
    if LAYOUT in ('v3', 'v4'):
        from .infofeat import InfoState
        ego = np.concatenate([ego, InfoState(legal, F).ego()]).astype(np.float32)
    return dict(ego=ego, cand=cand, traj=np.zeros((K, P, len(TRAJ)), np.float32),
                hum=np.asarray(hum_rows, np.float32).reshape(N, len(HUM)),
                hfut=np.asarray(hfut_rows, np.float32).reshape(N, P, len(HFUT)),
                pair=np.zeros((K, P, N, len(PAIR)), np.float32))


GraphCompiler.compile_blind = staticmethod(compile_blind)


def compile_record(rec, compiler=None):
    """Compile one capture record (``capture.py``)."""
    compiler = compiler or GraphCompiler()
    return compiler.compile(rec['result'], rec['legal'], rec['queries'], rec.get('diag'), rec.get('steps', 0))
