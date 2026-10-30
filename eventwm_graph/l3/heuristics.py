"""Hand-written selectors over the compiled consequence graph (baselines; no learning).

``greedy_index``: the obvious imagination-based rule -- among candidates whose imagined execution is predicted safe
(Stage 1 collision probability over the interval and the 3 s hold extension below thresholds, no imagined pose on
legally BLOCKED cells), take the largest legal route-cost reduction; STOP only when the imagined end is at the goal;
if nothing is predicted safe, take the smallest predicted collision risk.
"""
import math

import numpy as np

from .compiler import CAND

_I = {c: CAND.index(c) for c in CAND}


def greedy_index(graph, p_max=0.15, x_max=0.3):
    c = graph['cand']
    n = len(c)
    pcol = c[:, _I['pcol']]
    xcol = c[:, _I['x_pcol']]
    gain = c[:, _I['ctg_gain']]
    stop = c[:, _I['intent_stop']] > 0.5
    at_goal = c[:, _I['at_goal']] > 0.5
    blocked = c[:, _I['blocked_frac']] > 0
    safe = (pcol <= p_max) & (xcol <= x_max) & ~blocked & ~(stop & ~at_goal)
    score = gain+0.5*at_goal*stop
    if safe.any():
        idx = np.flatnonzero(safe)
        return int(idx[np.argmax(score[idx])])
    risk = pcol+0.5*xcol
    return int(np.argmin(risk))


def progress_index(graph):
    """Pure progress: the largest legal route-cost reduction of the imagined interval (ties: the earlier candidate)."""
    c = graph['cand']
    stop = c[:, _I['intent_stop']] > 0.5
    at_goal = c[:, _I['at_goal']] > 0.5
    score = c[:, _I['ctg_gain']].astype(np.float64)+2.0*(stop & at_goal)-5.0*(stop & ~at_goal)
    return int(np.argmax(score))


_E = None


def escape_index(graph):
    """Progress, with a stall breaker: when the robot has been stalled (< 0.3 m over 48 steps) and no candidate
    shows route progress, take the non-continue candidate whose imagined interval moves the robot the most
    (route alternatives / retreats / exploration first, then any motion); otherwise the largest route progress."""
    global _E
    from .compiler import EGO
    if _E is None:
        _E = {c: EGO.index(c) for c in EGO}
    c = graph['cand']
    e = graph['ego']
    gain = c[:, _I['ctg_gain']].astype(np.float64)
    if e[_E['stall_48']] > 0.5 and gain.max() <= 0.02:
        disp = np.hypot(c[:, _I['disp_fwd']], c[:, _I['disp_left']]).astype(np.float64)
        cont = c[:, _I['intent_continue']] > 0.5
        stop = c[:, _I['intent_stop']] > 0.5
        score = np.where(cont | stop, -1., disp)
        if score.max() > 0.05:
            return int(np.argmax(score))
    return progress_index(graph)


def routesafe_index(graph, default, thr=0.15, ratio=0.5):
    """Default unless its imagined interval or its INTENTION (v4 route columns) carries a predicted contact risk above
    ``thr``; then the candidate with the most route progress among those whose risk is below ``ratio`` x the default's
    (never STOP away from the goal, never a scan), else the least risky one.  Needs L3_LAYOUT=v4."""
    c = graph['cand']
    n = len(c)
    risk = np.maximum(c[:, _I['pcol']], c[:, _I['r_pcol']]).astype(np.float64)
    d = int(default)
    if risk[d] < thr:
        return d
    stop = c[:, _I['intent_stop']] > 0.5
    at_goal = c[:, _I['at_goal']] > 0.5
    scan = c[:, _I['q_fam_scan']] > 0.5 if 'q_fam_scan' in _I else np.zeros(n, bool)
    ok = (risk < ratio*risk[d]) & ~(stop & ~at_goal) & ~scan
    if ok.any():
        idx = np.flatnonzero(ok)
        safe = idx[risk[idx] < thr]
        pool = safe if len(safe) else idx
        return int(pool[np.argmax(c[pool, _I['ctg_gain']]-0.5*risk[pool])])
    return d


def routesafe_progress_index(graph, default, thr=0.15, slack=0.1):
    """Like ``routesafe_index`` but never trades progress for safety: deviate from the default only to a candidate whose
    interval + intention risk is below ``thr`` and half the default's AND whose legal route progress over the interval
    is at least the default's minus ``slack`` (feature units: 0.1 = 0.3 m); otherwise keep the default."""
    c = graph['cand']
    n = len(c)
    risk = np.maximum(c[:, _I['pcol']], c[:, _I['r_pcol']]).astype(np.float64)
    d = int(default)
    if risk[d] < thr:
        return d
    stop = c[:, _I['intent_stop']] > 0.5
    at_goal = c[:, _I['at_goal']] > 0.5
    scan = c[:, _I['q_fam_scan']] > 0.5 if 'q_fam_scan' in _I else np.zeros(n, bool)
    gain = c[:, _I['ctg_gain']].astype(np.float64)
    ok = (risk < min(thr, 0.5*risk[d])) & ~(stop & ~at_goal) & ~scan & (gain >= gain[d]-slack)
    if ok.any():
        idx = np.flatnonzero(ok)
        return int(idx[np.argmax(gain[idx]-0.5*risk[idx])])
    return d


def routesafe_bank_index(graph, default, thr=0.15, slack=0.1):
    """``routesafe_progress_index`` with every risk taken from the compiler's bank-derived columns (waypoint P(contact)
    over the interval and the intention's P(contact)), so the same rule can run on Stage 1's bank or, for analysis, on
    the true human futures (L3_ORACLE_BANK)."""
    from .compiler import TRAJ
    c = graph['cand']
    n = len(c)
    ip = graph['traj'][:, :, TRAJ.index('p_col')].max(1).astype(np.float64) if n else np.zeros(0)
    risk = np.maximum(ip, c[:, _I['r_pcol']].astype(np.float64))
    d = int(default)
    if risk[d] < thr:
        return d
    stop = c[:, _I['intent_stop']] > 0.5
    at_goal = c[:, _I['at_goal']] > 0.5
    scan = c[:, _I['q_fam_scan']] > 0.5 if 'q_fam_scan' in _I else np.zeros(n, bool)
    gain = c[:, _I['ctg_gain']].astype(np.float64)
    ok = (risk < min(thr, 0.5*risk[d])) & ~(stop & ~at_goal) & ~scan & (gain >= gain[d]-slack)
    if ok.any():
        idx = np.flatnonzero(ok)
        return int(idx[np.argmax(gain[idx]-0.5*risk[idx])])
    return d


def imagined_rollout_index(result, legal, default, slack_p=0.05, slack_m=0.5, radius=0.75, step_s=0.048):
    """LEGAL rollout policy (no learning, no privileged input): every candidate's imagined interval + its imagined
    default continuation (SandboxConfig.continuation: keep the running query, else the goal query; the executor and
    planner in the loop against the nominal Stage 1 people) is scored against the 40 Stage 1 worlds -- the bank's 3 s,
    then each world's people continue at their last velocity -- for P(contact) (centre distance < ``radius``), and
    against the CURRENT legal map for route progress at the end of the imagined horizon.  The least risky candidates
    (within ``slack_p`` of the minimum) compete on progress; the default is kept unless another one is clearly better
    (more than ``slack_m`` metres of progress or lower risk beyond the slack)."""
    from .compiler import LegalField
    bank = result.humans
    W = np.asarray(bank.weights, np.float64)
    W = W/W.sum() if W.sum() > 0 else W
    field = LegalField(legal)
    seqs = []
    for c in result.candidates:
        ro = c.rollout
        poses = np.asarray(ro.poses, np.float64)
        cont = getattr(ro, 'cont', None)
        seq = np.concatenate([poses[1:], cont]) if cont is not None and len(cont) else poses[1:]
        if not len(seq):
            seq = poses[:1]
        seqs.append(seq)
    T = max(len(s) for s in seqs)
    keys = list(getattr(bank, 'track_keys', ()))
    pcol = np.zeros(len(seqs))
    if len(keys):
        horizon = max(float(bank.times[-1]), float(getattr(bank, 'long_steps', 0) or 0)*step_s)   # analysis bank: longer
        rel = np.arange(1, T+1)*step_s
        inside = rel[rel <= horizon+1e-9]
        HP = np.asarray(bank.sandbox_positions(inside), np.float64)              # [W, Tb, N, 2]
        if len(inside) < T:
            if HP.shape[1] >= 2:
                vel = (HP[:, -1]-HP[:, -2])/step_s
            else:
                vel = np.zeros_like(HP[:, -1])
            extra = np.arange(1, T-len(inside)+1)*step_s
            ext = HP[:, -1][:, None]+extra[None, :, None, None]*vel[:, None]
            HP = np.concatenate([HP, ext], 1)
        with np.errstate(invalid='ignore'):
            for k, seq in enumerate(seqs):
                n = len(seq)
                d = np.hypot(HP[:, :n, :, 0]-seq[None, :, None, 0], HP[:, :n, :, 1]-seq[None, :, None, 2])
                d = np.where(np.isfinite(d), d, np.inf)
                hit = (d.min(axis=(1, 2)) < radius)
                pcol[k] = float((W*hit).sum())
    prog = np.asarray([field.start-field.at(s[-1, 0], s[-1, 2]) for s in seqs])
    stop = np.asarray([bool(c.rollout.stop) for c in result.candidates])
    at_goal = np.asarray([math.hypot(s[-1, 0]-legal.goal[0], s[-1, 2]-legal.goal[2]) < 0.3 for s in seqs])
    bad_stop = stop & ~at_goal
    pc = np.where(bad_stop, 1.0, pcol)
    d = int(default)
    best_p = float(pc.min())
    pool = np.flatnonzero(pc <= best_p+slack_p)
    k = int(pool[np.argmax(prog[pool])])
    if pc[d] <= best_p+slack_p and prog[d] >= prog[k]-slack_m:
        return d
    return k


def _rect_clearance(poses, people):
    """Clearance [n, N] of the robot body rectangle at poses [n, 4] to people centres [n, N, 2] (world x, z), as
    ``sandbox.imagine.rectangle_clearance`` per step (vectorised; non-finite people give non-finite entries)."""
    from eventwm_graph.querygen.native import ROBOT_BOUNDS
    c, s = np.cos(poses[:, 3])[:, None], np.sin(poses[:, 3])[:, None]
    with np.errstate(invalid='ignore'):
        dx = people[..., 0]-poses[:, 0][:, None]
        dz = people[..., 1]-poses[:, 2][:, None]
        lx = dx*c-dz*s
        lz = dx*s+dz*c
        xmin, xmax, zmin, zmax = ROBOT_BOUNDS
        ex = np.maximum(np.maximum(xmin-lx, 0.), lx-xmax)
        ez = np.maximum(np.maximum(zmin-lz, 0.), lz-zmax)
        return np.hypot(ex, ez)


def _discovery_rollouts(sb, legal, d, humans, state, cont):
    """Legal map discovery (L3_MW_DISCOVER=1, with L3_MW_PERCEIVE=1): pass 1 imagines every candidate on the frozen
    legal map; Stage 2 predicts the cells each imagined path would reveal (one batched Layer 2 call); pass 2 re-imagines
    every candidate while those predicted cells are certified as they enter the imagined camera (perceive.Discovery),
    so the imagined planner re-plans on discovered geometry as the real one does."""
    from eventwm_graph.sandbox.imagine import imagine
    from eventwm_graph.sandbox.episode import default_query
    from eventwm_graph.layer2.state import trajectory_from_poses
    from .infofeat import _blocked_lookup
    from .perceive import Discovery, predicted_truth
    ros = sb.imagine_all(legal, d.candidates, humans)
    trajs = []
    for ro in ros:
        P = np.asarray(ro.poses, np.float64)
        c = getattr(ro, 'cont', None)
        if c is not None and len(c):
            P = np.concatenate([P, np.asarray(c, np.float64)])
        trajs.append(trajectory_from_poses(P, legal.time))
    try:
        pred = sb.l2.predict(state, trajs, materialize=False, worlds=sb.config.worlds, seed=sb.config.seed)
        deltas = [c.map for c in pred.candidates]
    except Exception as exc:             # no Stage 2 answer: keep the frozen-map rollouts
        print('mwroll discovery: Stage 2 failed (%s); frozen map' % (exc,), flush=True)
        return ros
    cq = default_query()
    out = []
    union = None
    if __import__('os').environ.get('L3_MW_DISCOVER') == 'union':
        # one predicted local map from every candidate's Stage 2 increment (perceive.predicted_truth_union)
        from .perceive import predicted_truth_union
        union = predicted_truth_union(legal.map, deltas)
    for c, ro, delta in zip(d.candidates, ros, deltas):
        ex, dyn = sb._execution(c)
        if isinstance(ex, Exception):
            out.append(ro)
            continue
        truth = union if union is not None else predicted_truth(legal.map, delta)
        work = legal.map.copy()
        hk = humans.fork(Discovery(work, truth), _blocked_lookup(truth))
        out.append(imagine(legal, ex, dyn, hk, sb.config.steps_for(c), level=int(c.level), gmap=work,
                           continuation=int(cont), cont_query=cq, planner=c.planner,
                           route=bool(getattr(sb.config, 'route', False))))
    return out


def _truth_rollouts(sb, legal, d, humans, truth, cont):
    """ANALYSIS ONLY (privileged, L3_MW_TRUTHMAPS): every candidate imagined with progressive map discovery from a
    truth map (the legal map at t completed with the cells of ``truth``, truthmaps.merged_truth)."""
    from eventwm_graph.sandbox.imagine import imagine
    from eventwm_graph.sandbox.episode import default_query
    from .infofeat import _blocked_lookup
    from .perceive import Discovery
    from .truthmaps import merged_truth
    full = merged_truth(legal.map, truth)
    blocked = _blocked_lookup(full)
    cq = default_query()
    out = []
    for c in d.candidates:
        ex, dyn = sb._execution(c)
        if isinstance(ex, Exception):
            from eventwm_graph.sandbox.imagine import failed_rollout
            out.append(failed_rollout(legal, '%s: %s' % (type(ex).__name__, ex)))
            continue
        work = legal.map.copy()
        hk = humans.fork(Discovery(work, full), blocked)
        out.append(imagine(legal, ex, dyn, hk, sb.config.steps_for(c), level=int(c.level), gmap=work,
                           continuation=int(cont), cont_query=cq, planner=c.planner,
                           route=bool(getattr(sb.config, 'route', False))))
    return out


def _leaf_rollouts(sb, legal, d, humans, cont):
    """Every candidate imagined for its interval + ``cont`` default steps (continuation fidelity: decision boundaries
    every 16 steps, goal-query inheritance and re-issue, as the real default policy), keeping the imagined leaf state
    for the learned value (``imagine(keep_leaf=True)``)."""
    from eventwm_graph.sandbox.imagine import imagine, frozen_map, failed_rollout
    from eventwm_graph.sandbox.episode import default_query
    from eventwm_graph.sandbox import accel
    cq = default_query()
    gmap = frozen_map(legal.map)
    cache = accel.FrozenMapCache() if sb.config.memoize else None
    out = []
    with accel.active(cache):
        for c in d.candidates:
            ex, dyn = sb._execution(c)
            if isinstance(ex, Exception):
                out.append(failed_rollout(legal, '%s: %s' % (type(ex).__name__, ex)))
                continue
            out.append(imagine(legal, ex, dyn, humans, sb.config.steps_for(c), level=int(c.level), gmap=gmap,
                               continuation=int(cont), cont_query=cq, planner=c.planner, cont_boundary=16,
                               cont_inherit=True, keep_leaf=True))
    return out


_VALUE = {}
MW_DEBUG = {}          # analysis (L3_MW_DEBUG=1): the last multiworld_index call's per-candidate scores


def _value_model(path):
    if path not in _VALUE:
        from .leafvalue import ValueModel
        _VALUE[path] = ValueModel(path)
    return _VALUE[path]


def multiworld_index(sb, legal, d, n_worlds=6, cont=60, radius=0.75, slack_p=0.1, slack_m=0.5, step_s=0.048,
                     state=None, truth=None):
    """LEGAL multi-world imagined rollout (no learning, no privileged input): each candidate is imagined once per
    sampled Stage 1 world -- the robot's own planner perceiving THAT world's tracked people, interval + ``cont`` steps of
    default continuation -- and checked for contact against that world's people (tracked and hidden); P(contact) is the
    world-weighted share of contact rollouts, progress the mean legal route gain at the end.  Unlike a single nominal
    world, the planner's reactions to each possible future are simulated (the timing that decides hit vs miss)."""
    from eventwm_graph.sandbox.imagine import NominalHumans
    from .compiler import LegalField
    bank = d.result.humans
    W = np.asarray(bank.weights, np.float64)
    W = W/W.sum() if W.sum() > 0 else np.full(len(W), 1./max(len(W), 1))
    worlds = np.argsort(-W)[:n_worlds] if len(np.unique(W)) > 1 else np.arange(min(n_worlds, len(W)))
    ww = W[worlds]/W[worlds].sum()
    field = LegalField(legal)
    K = len(d.candidates)
    steps = max(sb.config.steps_for(c) for c in d.candidates)
    T = steps+cont
    keys = list(getattr(bank, 'track_keys', ()))
    if len(keys):
        rel = np.arange(1, T+1)*step_s
        horizon = max(float(bank.times[-1]), float(getattr(bank, 'long_steps', 0) or 0)*step_s)   # analysis bank: longer
        inside = rel[rel <= horizon+1e-9]
        HP = np.asarray(bank.sandbox_positions(inside), np.float64)              # [W, Tb, N, 2]
        if len(inside) < T and HP.shape[1] >= 2:
            vel = (HP[:, -1]-HP[:, -2])/step_s
            extra = np.arange(1, T-len(inside)+1)*step_s
            HP = np.concatenate([HP, HP[:, -1][:, None]+extra[None, :, None, None]*vel[:, None]], 1)
    else:
        HP = None
    extra = getattr(bank, 'score_extra', None)
    if extra is not None and len(extra.cols) and extra.unseen.any():
        # ANALYSIS ONLY (env.SCORE_TRUTH): the true futures of the people not tracked at t join the contact scoring
        E = np.asarray(extra.sandbox_positions(np.arange(1, T+1)*step_s), np.float64)[:, :, extra.unseen]   # [1, T, Nu, 2]
        HP = np.repeat(E, len(W), 0) if HP is None else np.concatenate([HP, np.repeat(E, HP.shape[0], 0)], 2)
    pc = np.zeros(K)
    prog = np.zeros(K)
    oc = np.zeros(K)
    rule = __import__('os').environ.get('L3_MW_RULE', 'contact')       # 'contact' (default) | 'outcome'
    stop_bad = np.zeros(K, bool)
    dbg = __import__('os').environ.get('L3_MW_DEBUG') == '1'
    dmin = np.full(K, np.inf)        # analysis: min contact distance (centre or rect) per candidate over the worlds
    # contacts counted over the first L3_MW_CONTACT_STEPS imagined steps only (default: the whole rollout); with Stage 1
    # worlds people beyond the 3 s bank horizon are constant-velocity guesses -- the long continuation then judges
    # progress (deadlocks, timeouts) while contacts are judged where the worlds are predictions
    contact_steps = int(__import__('os').environ.get('L3_MW_CONTACT_STEPS', '0')) or 10**9
    # contact = robot body rectangle within L3_MW_RECT m of a person's centre (the native simulator's body-proxy
    # measure) instead of a centre distance < radius (default)
    rect = float(__import__('os').environ.get('L3_MW_RECT', '0'))
    # learned leaf value (L3_MW_VALUE=<model dir>, default off): an imagined contact in a world scores 0, a STOP at the
    # goal 1, otherwise the model's P(success) at the imagined leaf (leafvalue.legal_features of the leaf state)
    vpath = __import__('os').environ.get('L3_MW_VALUE')
    vm = _value_model(vpath) if vpath else None
    vs = np.zeros(K)
    xs_ctx = None
    if vm is not None and 'Xs' in vm.keys:
        from .leafvalue import legal_features
        from eventwm_graph.sandbox.imagine import dynamics as _dyn
        c0 = d.candidates[0]
        xs_ctx = legal_features(legal.pose, legal.poses[-8:], legal.times[-8:], legal.time, legal.map, field, legal.goal,
                                legal.tracks, c0.execution, c0.level, _dyn(c0.dynamics), c0.planner, int(d.steps or 0), 0,
                                None)
    old = sb.config.continuation
    sb.config.continuation = int(cont)
    try:
        for j, w in enumerate(worlds):
            if __import__('os').environ.get('L3_MW_PERCEIVE') == '1':
                # perception-faithful imagination: the planner sees what the imagined camera would (perceive.py)
                from .perceive import VisibleWorld
                humans = VisibleWorld(legal, bank, int(w), T)
            else:
                humans = NominalHumans(legal.tracks, legal.time, T, bank=bank, mode='world:%d' % int(w))
            if vm is not None:
                ros = _leaf_rollouts(sb, legal, d, humans, cont)
            elif truth is not None and __import__('os').environ.get('L3_MW_PERCEIVE') == '1':
                ros = _truth_rollouts(sb, legal, d, humans, truth, cont)
            elif (state is not None and __import__('os').environ.get('L3_MW_DISCOVER') in ('1', 'union')
                    and __import__('os').environ.get('L3_MW_PERCEIVE') == '1'):
                ros = _discovery_rollouts(sb, legal, d, humans, state, cont)
            else:
                ros = sb.imagine_all(legal, d.candidates, humans)
            for k, ro in enumerate(ros):
                poses = np.asarray(ro.poses, np.float64)
                c = getattr(ro, 'cont', None)
                seq = np.concatenate([poses[1:], c]) if c is not None and len(c) else poses[1:]
                if not len(seq):
                    seq = poses[:1]
                if ro.stop and math.hypot(seq[-1, 0]-legal.goal[0], seq[-1, 2]-legal.goal[2]) > 0.3:
                    stop_bad[k] = True
                if HP is not None:
                    n = min(len(seq), HP.shape[1], contact_steps)
                    if rect > 0:
                        dd = _rect_clearance(seq[:n], HP[w, :n])
                        hit = np.nanmin(np.where(np.isfinite(dd), dd, np.inf)) < rect if dd.size else False
                    else:
                        with np.errstate(invalid='ignore'):
                            dd = np.hypot(HP[w, :n, :, 0]-seq[:n, None, 0], HP[w, :n, :, 1]-seq[:n, None, 2])
                        hit = np.nanmin(np.where(np.isfinite(dd), dd, np.inf)) < radius if dd.size else False
                    if hit:
                        pc[k] += ww[j]
                    if dbg and dd.size:
                        dmin[k] = min(dmin[k], float(np.nanmin(np.where(np.isfinite(dd), dd, np.inf))))
                prog[k] += ww[j]*(field.start-field.at(seq[-1, 0], seq[-1, 2]))
                if vm is not None:
                    near = math.hypot(seq[-1, 0]-legal.goal[0], seq[-1, 2]-legal.goal[2]) <= 0.3
                    if HP is not None and hit:
                        val = 0.
                    elif ro.stop:
                        val = 1. if near else 0.
                    elif ro.leaf is None:
                        val = 0.
                    else:
                        from .leafvalue import legal_features
                        from eventwm_graph.sandbox.imagine import dynamics as _dyn
                        lf = ro.leaf
                        c0 = d.candidates[k]
                        x = legal_features(lf['poses'][-1], lf['poses'][-8:], lf['times'][-8:], lf['time'], legal.map, field,
                                           legal.goal, lf['tracks'], lf['execution'], lf['level'], _dyn(c0.dynamics),
                                           c0.planner, int(d.steps or 0)+int(lf['steps']), int(lf['steps']), None)
                        if xs_ctx is not None:
                            x = np.concatenate([x, xs_ctx])
                        val = float(vm.predict(x)[0])
                    vs[k] += ww[j]*val
                if rule == 'outcome':
                    # imagined outcome in the exact reference's order (rolloutpolicy.outcome_key): goal reached (fewer
                    # steps better) > still running (more route progress) > contact / STOP away from the goal (later
                    # better); category + a tie-break in [0, 0.5)
                    Tk = max(len(seq), 1)
                    near = math.hypot(seq[-1, 0]-legal.goal[0], seq[-1, 2]-legal.goal[2]) <= 0.3
                    if HP is not None and hit:
                        first = int(np.argmax(np.nanmin(np.where(np.isfinite(dd), dd, np.inf), axis=1) <
                                              (rect if rect > 0 else radius)))
                        val = 0.+0.5*first/max(n, 1)
                    elif ro.stop and not near:
                        val = 0.
                    elif ro.stop and near:
                        val = 3.+0.5*(1.-Tk/max(steps+cont, 1))
                    else:
                        g = field.start-field.at(seq[-1, 0], seq[-1, 2])
                        val = 2.+0.5*float(np.clip(g/max(field.start, 1e-6), 0., 0.999))
                    oc[k] += ww[j]*val
    finally:
        sb.config.continuation = old
    pc = np.where(stop_bad, 1., pc)
    if dbg:
        MW_DEBUG.clear()
        MW_DEBUG.update(pc=pc.tolist(), prog=prog.tolist(), oc=oc.tolist(), stop_bad=stop_bad.tolist(),
                        dmin=[x if math.isfinite(x) else None for x in dmin.tolist()], rule=rule)
    dflt = int(d.default)
    if vm is not None:
        best = int(np.argmax(vs))
        return dflt if vs[dflt] >= vs[best]-float(__import__('os').environ.get('L3_MW_EPS', '0.02')) else best
    if rule == 'outcome':
        best = int(np.argmax(oc))
        return dflt if oc[dflt] >= oc[best]-float(__import__('os').environ.get('L3_MW_EPS', '0.05')) else best
    # deviation margins (env, defaults = the original rule): the default is kept unless its contact probability is at
    # least L3_MW_MIN_PC and exceeds the best candidate's by more than L3_MW_SLACK_P
    slack_p = float(__import__('os').environ.get('L3_MW_SLACK_P', slack_p))
    if pc[dflt] < float(__import__('os').environ.get('L3_MW_MIN_PC', '0')):
        return dflt
    best_p = float(pc.min())
    pool = np.flatnonzero(pc <= best_p+slack_p)
    k = int(pool[np.argmax(prog[pool])])
    if pc[dflt] <= best_p+slack_p and prog[dflt] >= prog[k]-slack_m:
        return dflt
    return k


def closed_loop_index(sb, legal, d, window=32, worlds=8, body=0.3, eps=0.05, selection='cluster', state=None):
    """LEGAL selector on the closed-loop Sandbox (sandbox/closedloop.py): every candidate imagined in ``worlds``
    clustered Stage 1 worlds (camera model, continuation fidelity, frozen map), contacts scored against every world's
    people; value = P(no contact within ``window``) x (2 + legal progress share) + P(STOP at the goal), STOP away
    from the goal counted as contact.  The default is kept unless another candidate is better by ``eps``."""
    from eventwm_graph.sandbox.closedloop import imagine_closed_loop, ClosedLoopConfig
    from .compiler import LegalField
    res = imagine_closed_loop(sb, state, legal, d.candidates, d.result.humans,
                              ClosedLoopConfig(window=int(window), worlds=int(worlds), map_loop='off', selection=selection))
    field = LegalField(legal)
    start = max(float(field.start), 1e-6)
    vals = []
    for k in range(len(d.candidates)):
        pc = res.contact(k, window, body)
        ws = np.asarray([x.weight for x in res.worlds]); ws = ws/max(ws.sum(), 1e-9)
        prog, goal, bad = 0., 0., 0.
        for x, wt in zip(res.worlds, ws):
            ro = x.rollouts[k]
            seq = res._seq(ro)
            p = seq[min(window, len(seq))-1]
            prog += wt*(start-float(field.at(p[0], p[2])))/start
            near = math.hypot(seq[-1, 0]-legal.goal[0], seq[-1, 2]-legal.goal[2]) <= 0.3
            if ro.stop and len(seq) <= window:
                goal += wt*float(near); bad += wt*float(not near)
        safe = max(0., 1.-pc-bad)
        vals.append(safe*(2.+float(np.clip(prog, -1., 1.)))+goal)
    vals = np.asarray(vals)
    dflt = int(d.default)
    best = int(np.argmax(vals))
    return dflt if vals[dflt] >= vals[best]-eps else best
