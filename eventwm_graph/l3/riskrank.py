"""Learned listwise candidate ranker for the multi-world selector (Layer 3 option ``L3_MW_RISK``, default off; M6
approach B).  LEGAL inputs only; the exact outcomes of the TRAIN state bank are the training labels only.

The legal Sandbox (``heuristics.multiworld_index``, outcome rule, the legal constant-velocity people ``oraclebank.CVBank``)
foresees few of the contacts that really happen within 1.5-3 s: most involve people it does not track at t.  This
module learns, per decision state, which candidate is exactly best from LEGAL per-candidate features:

* the rule's own per-candidate values (outcome score ``oc``, contact share ``pc``, legal route gain ``prog``, min CV
  people distance ``dmin``, STOP away from the goal) at continuation 16 and 48 steps, absolute and relative to the
  default / best candidate of the state;
* the imagined robot trajectory (interval + 48 continuation steps): displacement, path length, rotation, static and
  in-place-turn shares, STOP / end reason, legal cost-to-go at 16 / 32 / 64 steps;
* the CV people along it (body-rectangle clearance / centre distance within 16 / 32 / 64 steps);
* the untracked-people hazard (``hazard.UntrackedHazard``, ``configs/l3_hazard_v2.json``: lost tracks, out-of-view prior,
  negative information from the robot's own camera history) along it: P(contact) within 16 / 32 / 64 steps, never-seen
  prior and remembered-track parts separately;
* the legal map along it (distance to the nearest legal wall / UNKNOWN cell);
* the candidate's query (label, kind, interval, default flag) and the legal context at t (tracked / lost / ever-seen
  people from the legal track list and track log, the robot's recent motion from its pose history, bumps, cost-to-go).

Never read: ``truth_people``, the tape, exact outcomes (``probe``), the navmesh, ``human_num``.

Model: LightGBM lambdarank per state on the exact outcome at 64 steps (``shortprobe._key_real``: contact / STOP away from
the goal 0 < running, graded by the exact legal cost-to-go gap to the state's best running candidate < goal reached).
Selector: the ranker's argmax; the default is kept unless the ranker's score margin over the default exceeds the
model's threshold (tuned on TRAIN by inner grouped cross-validation).

Selector option: ``L3_MW_RISK=<model dir>`` (``model.txt`` or a seed ensemble ``model_<k>.txt`` + ``meta.json``; frozen
model ``configs/l3_riskrank_m6b``) in ``heuristics.multiworld_index`` (selector ``mwroll``); the features are recomputed
with the model's training settings (one world of CV people, continuation 16 and 48, outcome rule, no Stage 2 discovery,
no hazard / contact-model / value options, L3_MW_GOAL_R unset); a compatible current call (L3_MW_BANK=cv, L3_MW_CONT 16
or 48, those options) is reused.  The continuation-16 values are replayed on the truncated continuation-48 rollouts
(``derive_outputs``: exact prefixes, bit-identical to a real call; ``L3_MW_RISK_DERIVE=0`` imagines them instead), so
with ``L3_MW_CONT=48`` the ranker adds no imagination at all (~0.2 s of features; otherwise one continuation-48
imagination).  The process-wide options the model
was trained with (SANDBOX_PHYSICS=core, SANDBOX_CONT_BOUNDARY=16, SANDBOX_CONT_INHERIT=1, L3_INHERIT=1) are checked
once (warning).  ``L3_MW_RISK_THR`` overrides the threshold; ``RISK_DEBUG`` holds the last decision's scores / latency.
With ``L3_MW_RISK`` unset nothing changes (``multiworld_index`` defaults bit-identical).

Offline (TRAIN only for fitting; dev evaluated once with the frozen model):

    L3_INHERIT=1 L3_MW_PERCEIVE=1 L3_MW_RULE=outcome L3_MW_EPS=0 SANDBOX_PHYSICS=core SANDBOX_CONT_BOUNDARY=16 \\
    SANDBOX_CONT_INHERIT=1 QG_FAST_SENSE=1 L3_MW_DISCOVER=0 CUDA_VISIBLE_DEVICES= SANDBOX_KEEP_CUDA_ENV=1 \\
        python scripts/l3_run.py eventwm_graph.l3.riskrank extract --bank output/statebank/orca_train_blind \\
        --out output/agents/m6_ranker/feat/orca_train_blind --shard 0 --shards 3          # (scratchpad crit_rule.py setup)
    python scripts/l3_run.py eventwm_graph.l3.riskfit cv ...     # grouped CV, threshold, frozen model (riskfit.py)
    python scripts/l3_run.py eventwm_graph.l3.riskfit evaluate ...
"""
import argparse
import contextlib
import glob
import json
import math
import os
import pickle
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

STEP = 0.048
CONTS = (16, 48)
WINDOWS = (16, 32, 64)
UNKNOWN, FREE, BLOCKED = 0, 1, 2
HAZARD_PARAMS = 'configs/l3_hazard_v2.json'
FAR = 10.0                       # stand-in distance when nobody / nothing exists (m)
RISK_DEBUG = {}                  # the last risk_select call: features, scores, choice, latency

# per-call options of multiworld_index fixed to the training values for the feature calls (offline benchmark =
# scratchpad hidden/crit_rule.py with people cv and L3_MW_DISCOVER=0)
CALL_ENV = dict(L3_MW_PERCEIVE='1', L3_MW_RULE='outcome', L3_MW_EPS='0', L3_MW_DISCOVER='0', L3_MW_DEBUG=None,
                L3_MW_RADIUS=None, L3_MW_GOAL_R=None, L3_MW_CONTACT_STEPS=None, L3_MW_RECT=None, L3_MW_VALUE=None,
                L3_MW_HAZARD=None, L3_MW_CONTACT_MODEL=None, L3_MW_TRUTH_WALLS=None, L3_CV_MAX_AGE=None)


def process_settings():
    """Process-wide (import-time) options the imagination depends on; stored with the model, checked at run time."""
    from eventwm_graph.sandbox import imagine as im
    from eventwm_graph.querygen.executor import QueryExecution
    from . import perceive
    return dict(CONT_BOUNDARY=int(im.CONT_BOUNDARY), CONT_INHERIT=bool(im.CONT_INHERIT), PHYSICS=str(im.PHYSICS),
                PHYSICS_LIVE=bool(im.PHYSICS_LIVE), CAM_MOUNT=bool(perceive.CAM_MOUNT),
                CAM_TRACKER=bool(perceive.CAM_TRACKER), INHERIT=bool(getattr(QueryExecution, 'INHERIT', False)))


@contextlib.contextmanager
def call_env():
    old = {k: os.environ.get(k) for k in CALL_ENV}
    try:
        for k, v in CALL_ENV.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def env_matches():
    for k, v in CALL_ENV.items():
        cur = os.environ.get(k)
        if k == 'L3_MW_DEBUG':
            continue
        if k == 'L3_MW_DISCOVER':
            if cur not in (None, '', '0'):
                return False
            continue
        if k == 'L3_MW_EPS':
            continue                     # the selection margin does not change the scores
        if cur != v:
            return False
    return True


def _seq(ro):
    """The imagined poses after 1..n steps (interval + continuation), as multiworld_index scores them."""
    poses = np.asarray(ro.poses, np.float64)
    c = getattr(ro, 'cont', None)
    seq = np.concatenate([poses[1:], np.asarray(c, np.float64)]) if c is not None and len(c) else poses[1:]
    if not len(seq):
        seq = poses[:1]
    return seq


def rule_outputs(sb, legal, d, cont, state=None):
    """multiworld_index (one world, the legal CV people of ``d.result.humans``) at continuation ``cont``: the rule's
    choice and its per-candidate scores / rollouts."""
    from .heuristics import multiworld_index
    o = {}
    with call_env():
        o['choice'] = int(multiworld_index(sb, legal, d, n_worlds=1, cont=int(cont), state=state, _risk=False, _out=o))
    return o


def derive_outputs(o, legal, d, cont, radius=0.75, goal_r=0.3):
    """``rule_outputs`` at a SHORTER continuation ``cont`` from a longer call ``o`` without imagining again: with one world
    of CV people the imagined rollouts at continuation c are exact prefixes of those at c' > c (interval + the first c
    continuation steps; verified on the 7850 TRAIN candidates) and the CV people are analytic, so multiworld_index's
    outcome-rule scoring (call_env settings: centre radius 0.75 m, goal radius 0.3 m, every step scored) is replayed on
    the truncated rollouts.  Checked bit-identical to a real call (scratchpad m6r_runtime.py)."""
    from .compiler import LegalField
    cont = int(cont)
    assert cont <= int(o['cont']) and len(o['ros']) == 1
    field = LegalField(legal)
    ros = o['ros'][0]
    K = len(ros)
    steps = int(o['steps'])
    T = steps+cont
    HP = None if o['HP'] is None else np.asarray(o['HP'], np.float64)[:, :T]
    w = int(o['worlds'][0])
    pc, prog, oc, dmin = np.zeros(K), np.zeros(K), np.zeros(K), np.full(K, np.inf)
    stop_bad = np.zeros(K, bool)
    trunc = []
    for k, ro in enumerate(ros):
        poses = np.asarray(ro.poses, np.float64)
        c = getattr(ro, 'cont', None)
        if c is not None and len(c):
            c = c[:cont]
        seq = np.concatenate([poses[1:], c]) if c is not None and len(c) else poses[1:]
        if not len(seq):
            seq = poses[:1]
        trunc.append(SimpleNamespace(poses=ro.poses, cont=c, stop=ro.stop, ended_by=ro.ended_by))
        if ro.stop and math.hypot(seq[-1, 0]-legal.goal[0], seq[-1, 2]-legal.goal[2]) > goal_r:
            stop_bad[k] = True
        hit = False
        if HP is not None:
            n = min(len(seq), HP.shape[1])
            with np.errstate(invalid='ignore'):
                dd = np.hypot(HP[w, :n, :, 0]-seq[:n, None, 0], HP[w, :n, :, 1]-seq[:n, None, 2])
            hit = np.nanmin(np.where(np.isfinite(dd), dd, np.inf)) < radius if dd.size else False
            if hit:
                pc[k] += 1.
            if dd.size:
                dmin[k] = min(dmin[k], float(np.nanmin(np.where(np.isfinite(dd), dd, np.inf))))
        prog[k] += 1.*(field.start-field.at(seq[-1, 0], seq[-1, 2]))
        Tk = max(len(seq), 1)
        near = math.hypot(seq[-1, 0]-legal.goal[0], seq[-1, 2]-legal.goal[2]) <= goal_r
        if HP is not None and hit:
            first = int(np.argmax(np.nanmin(np.where(np.isfinite(dd), dd, np.inf), axis=1) < radius))
            val = 0.+0.5*first/max(n, 1)
        elif ro.stop and not near:
            val = 0.
        elif ro.stop and near:
            val = 3.+0.5*(1.-Tk/max(steps+cont, 1))
        else:
            g = field.start-field.at(seq[-1, 0], seq[-1, 2])
            val = 2.+0.5*float(np.clip(g/max(field.start, 1e-6), 0., 0.999))
        oc[k] += 1.*val
    pc = np.where(stop_bad, 1., pc)
    dflt = int(d.default)
    best = int(np.argmax(oc))
    return dict(oc=oc, pc=pc, prog=prog, dmin=dmin, stop_bad=stop_bad, HP=HP, worlds=o['worlds'], ww=o['ww'], cont=cont,
                steps=steps, ros=[trunc], choice=dflt if oc[dflt] >= oc[best] else best, derived=True)


# ---------------------------------------------------------------------------------------------- legal features
def _crop(g, pose, crop_m=12.0):
    """Distance (m) to the nearest legal BLOCKED / UNKNOWN cell around the robot (cells outside the map = UNKNOWN)."""
    from scipy.ndimage import distance_transform_edt
    r = int(round(crop_m/g.res))
    ci, cj = g.index(pose[0], pose[2])
    S = np.full((2*r+1, 2*r+1), UNKNOWN, np.uint8)
    a0, a1 = max(ci-r, 0), min(ci+r+1, g.state.shape[0])
    b0, b1 = max(cj-r, 0), min(cj+r+1, g.state.shape[1])
    if a1 > a0 and b1 > b0:
        S[a0-(ci-r):a1-(ci-r), b0-(cj-r):b1-(cj-r)] = g.state[a0:a1, b0:b1]
    from scipy.ndimage import uniform_filter
    out = dict(i0=ci-r, j0=cj-r, n=2*r+1, res=float(g.res), ox=int(g.ox), oz=int(g.oz), S=S)
    box = 2*int(round(1.5/g.res))+1                       # +-1.5 m window
    for name, val in (('dB', BLOCKED), ('dU', UNKNOWN)):
        m = S == val
        out[name] = (distance_transform_edt(~m)*g.res).astype(np.float32) if m.any() else np.full(S.shape, FAR, np.float32)
        out['f'+name[1]] = uniform_filter(m.astype(np.float32), size=box, mode='constant',
                                          cval=1. if val == UNKNOWN else 0.)
    return out


def _lookup(C, xs, zs, name):
    i = np.floor(np.asarray(xs)/C['res']).astype(np.int64)-C['ox']-C['i0']
    j = np.floor(np.asarray(zs)/C['res']).astype(np.int64)-C['oz']-C['j0']
    ok = (i >= 0) & (i < C['n']) & (j >= 0) & (j < C['n'])
    v = np.full(len(i), {'S': UNKNOWN, 'fU': 1., 'fB': 0.}.get(name, FAR), np.float64)
    v[ok] = C[name][i[ok], j[ok]]
    return v


def context_features(legal, field, C, n_est):
    """Legal context at t (identical for every candidate of the state)."""
    t = float(legal.time)
    tracks = [x for x in legal.tracks if x.get('valid', True)]
    latest = {}
    for _, trs in getattr(legal, 'track_log', []) or []:
        for x in trs:
            if not x.get('valid', True):
                continue
            k = str(x.get('track_id'))
            if k not in latest or float(x['last_seen_time']) >= float(latest[k]['last_seen_time']):
                latest[k] = x
    ages = np.asarray([t-float(x['last_seen_time']) for x in latest.values()], np.float64)
    pose = np.asarray(legal.pose, np.float64)
    dists = np.asarray([math.hypot(float(x['position'][0])-pose[0], float(x['position'][2])-pose[2])
                        for x in latest.values()], np.float64)
    lost = (ages > 2.0) & (ages <= 10.0)
    P = np.asarray(legal.poses, np.float64).reshape(-1, 4)
    T = np.asarray(legal.times, np.float64)
    f = {}
    for h in (1.0, 3.0):
        k = T >= t-h-1e-9
        Q = P[k] if k.sum() >= 2 else P[-2:]
        f['ctx_disp_%gs' % h] = float(math.hypot(Q[-1, 0]-Q[0, 0], Q[-1, 2]-Q[0, 2])) if len(Q) >= 2 else 0.
        dy = np.abs(np.angle(np.exp(1j*np.diff(Q[:, 3])))) if len(Q) >= 2 else np.zeros(0)
        f['ctx_yaw_%gs' % h] = float(dy.sum())
        st = np.hypot(np.diff(Q[:, 0]), np.diff(Q[:, 2])) < 0.01 if len(Q) >= 2 else np.zeros(0, bool)
        f['ctx_static_%gs' % h] = float(st.mean()) if len(st) else 1.
    f['ctx_n_est'] = float(n_est)
    f['ctx_n_tracks'] = float(len(tracks))
    f['ctx_n_seen'] = float(len(latest))
    f['ctx_seen_rate'] = float(len(latest))/max(t-float(getattr(legal, 'start_time', 0.) or 0.), 5.)
    f['ctx_n_lost10'] = float(lost.sum())
    f['ctx_lost_near'] = float(np.min(dists[lost])) if lost.any() else FAR
    f['ctx_last_track_age'] = float(ages.min()) if len(ages) else 99.
    f['ctx_bumps'] = float(len(getattr(legal, 'bumps', []) or []))
    f['ctx_ctg0'] = float(field.start)
    f['ctx_goal_dist'] = float(math.hypot(legal.goal[0]-pose[0], legal.goal[2]-pose[2]))
    f['ctx_time'] = t
    S = C['S']
    r = int(round(3.0/C['res']))
    c = C['n']//2
    win = S[c-r:c+r+1, c-r:c+r+1]
    f['ctx_unknown_3m'] = float((win == UNKNOWN).mean())
    f['ctx_blocked_3m'] = float((win == BLOCKED).mean())
    return f


def _first(arr, n):
    return arr[:max(1, min(n, len(arr)))]


def features(legal, d, outs, hz=None, labels_vocab=None):
    """[K, F] LEGAL features of the decision's candidates and their names.  ``outs``: {cont: rule_outputs(...)} for
    CONTS (one world of CV people); ``hz``: a ``hazard.UntrackedHazard`` of the state (None: hazard features NaN)."""
    from .compiler import LegalField
    field = LegalField(legal)
    K = len(d.candidates)
    dflt = int(d.default)
    pose = np.asarray(legal.pose, np.float64)
    C = _crop(legal.map, pose)
    o48 = outs[max(CONTS)]
    HP = o48.get('HP')
    HPw = None if HP is None else np.asarray(HP, np.float64)[int(o48['worlds'][0])]     # [T, N, 2]
    n_est = 0 if HPw is None else HPw.shape[1]
    ctx = context_features(legal, field, C, n_est)
    ros48 = o48['ros'][0]
    rows = []
    for k in range(K):
        f = {}
        ro = ros48[k]
        seq = _seq(ro)
        n = len(seq)
        lab = str(d.labels[k]) if getattr(d, 'labels', None) is not None else ''
        q = getattr(d.candidates[k], 'query', None)
        kind = str(getattr(q, 'kind', 'continue')) if q is not None else 'continue'
        f['c_label'] = float(labels_vocab.get(lab, -1)) if labels_vocab is not None else -1.
        f['c_default'] = float(k == dflt)
        f['c_th'] = float(lab.endswith('_th'))
        f['c_tc'] = float(lab.endswith('_tc'))
        f['c_st'] = float(lab.startswith('st_'))
        f['c_continue'] = float(kind == 'continue')
        f['c_goal'] = float('goal' in lab)
        f['c_rot'] = float(lab.startswith('rot') or lab.startswith('turn') or 'scan' in lab)
        f['c_wait'] = float('wait' in lab)
        f['c_creep'] = float('creep' in lab)
        f['c_h'] = float(len(ro.poses)-1)
        f['c_n'] = float(n)
        f['c_stop'] = float(bool(ro.stop))
        f['c_ended'] = float({'horizon': 0, 'done': 1, 'stop': 2, 'failed': 3, 'error': 4}.get(str(ro.ended_by), 5))
        f['c_near_goal'] = float(math.hypot(seq[-1, 0]-legal.goal[0], seq[-1, 2]-legal.goal[2]) <= 0.3)
        f['c_goal_dist_end'] = float(math.hypot(seq[-1, 0]-legal.goal[0], seq[-1, 2]-legal.goal[2]))
        step = np.hypot(np.diff(np.concatenate([pose[None], seq])[:, 0]), np.diff(np.concatenate([pose[None], seq])[:, 2]))
        dyaw = np.abs(np.angle(np.exp(1j*np.diff(np.concatenate([pose[None], seq])[:, 3]))))
        for w in WINDOWS:
            s = _first(seq, w)
            f['c_disp%d' % w] = float(math.hypot(s[-1, 0]-pose[0], s[-1, 2]-pose[2]))
            f['c_ctg%d' % w] = float(field.at(s[-1, 0], s[-1, 2]))
            f['c_gain%d' % w] = float(field.start)-f['c_ctg%d' % w]
        for w in (32, 64):
            f['c_path%d' % w] = float(step[:w].sum())
            f['c_yawabs%d' % w] = float(dyaw[:w].sum())
            f['c_static%d' % w] = float((step[:w] < 0.005).mean()) if len(step[:w]) else 1.
            f['c_turnin%d' % w] = float(((step[:w] < 0.005) & (dyaw[:w] > 0.005)).mean()) if len(step[:w]) else 0.
        f['c_yawnet32'] = float(abs(math.atan2(math.sin(_first(seq, 32)[-1, 3]-pose[3]), math.cos(_first(seq, 32)[-1, 3]-pose[3]))))
        # the legal CV people along the imagined path
        if HPw is not None and n_est:
            from .heuristics import _rect_clearance
            m = min(n, HPw.shape[0])
            rc = _rect_clearance(seq[:m], HPw[:m])
            cd = np.hypot(HPw[:m, :, 0]-seq[:m, None, 0], HPw[:m, :, 1]-seq[:m, None, 2])
            rc = np.where(np.isfinite(rc), rc, FAR).min(1)
            cd = np.where(np.isfinite(cd), cd, FAR).min(1)
            for w in WINDOWS:
                f['p_rc%d' % w] = float(rc[:w].min()) if len(rc[:w]) else FAR
                f['p_cd%d' % w] = float(cd[:w].min()) if len(cd[:w]) else FAR
            f['p_ca_step'] = float(np.argmin(cd))/64. if len(cd) else 1.
        else:
            for w in WINDOWS:
                f['p_rc%d' % w] = FAR
                f['p_cd%d' % w] = FAR
            f['p_ca_step'] = 1.
        # untracked-people hazard along the imagined path
        if hz is not None:
            curves = hazard_curves(hz, seq[:64])
            for name, cv in curves.items():
                for w in WINDOWS:
                    f['h_%s%d' % (name, w)] = float(cv[min(w, len(cv))-1]) if len(cv) else 0.
        else:
            for name in ('all', 'prior', 'mem'):
                for w in WINDOWS:
                    f['h_%s%d' % (name, w)] = float('nan')
        # the legal map along the imagined path
        dB = _lookup(C, seq[:, 0], seq[:, 2], 'dB')
        dU = _lookup(C, seq[:, 0], seq[:, 2], 'dU')
        for w in (32, 64):
            f['m_wall%d' % w] = float(dB[:w].min())
            f['m_unk%d' % w] = float(dU[:w].min())
        f['m_unk_near64'] = float((dU[:64] < 1.0).mean())
        f['m_wall_mean64'] = float(dB[:64].mean())
        fU = _lookup(C, seq[:, 0], seq[:, 2], 'fU')
        fB = _lookup(C, seq[:, 0], seq[:, 2], 'fB')
        f['m_unkfrac32'] = float(fU[:32].mean())
        f['m_unkfrac64'] = float(fU[:64].mean())
        f['m_unkfrac_end'] = float(fU[min(64, len(fU))-1])
        f['m_blkfrac64'] = float(fB[:64].mean())
        f['m_blkfrac_max64'] = float(fB[:64].max())
        # the rule's own scores
        for c in CONTS:
            o = outs[c]
            dm = float(o['dmin'][k])
            f['r_oc%d' % c] = float(o['oc'][k])
            f['r_pc%d' % c] = float(o['pc'][k])
            f['r_prog%d' % c] = float(o['prog'][k])
            f['r_dmin%d' % c] = dm if math.isfinite(dm) else FAR
            f['r_stopbad%d' % c] = float(o['stop_bad'][k])
            f['r_cat%d' % c] = float(math.floor(o['oc'][k]))
            f['r_choice%d' % c] = float(k == int(o['choice']))
        f.update(ctx)
        rows.append(f)
    names = list(rows[0].keys())
    X = np.asarray([[r[nm] for nm in names] for r in rows], np.float64)
    # within-state relative features
    rel = ['r_oc16', 'r_oc48', 'r_prog16', 'r_prog48', 'c_gain16', 'c_gain32', 'c_gain64', 'h_all32', 'h_all64',
           'h_prior64', 'h_mem64', 'p_rc32', 'p_rc64', 'r_dmin16', 'r_dmin48', 'c_path64', 'c_static64', 'm_unk64',
           'm_wall64', 'm_unkfrac64', 'm_blkfrac64', 'c_yawabs64']
    extra, enames = [], []
    for nm in rel:
        v = X[:, names.index(nm)]
        if not np.isfinite(v).all():
            vv = np.where(np.isfinite(v), v, 0.)
        else:
            vv = v
        extra += [vv-vv[dflt], vv-vv.max(), vv-vv.min()]
        enames += ['d_%s' % nm, 'mx_%s' % nm, 'mn_%s' % nm]
        better = (vv[None, :] > vv[:, None]+1e-9).sum(1)          # candidates strictly above (ties share a rank)
        extra.append(better/max(K-1, 1)); enames.append('rk_%s' % nm)
    X = np.concatenate([X, np.stack(extra, 1)], 1)
    return X, names+enames


def hazard_curves(hz, seq):
    """P(contact by step k) of every untracked-people particle ('all'), the never-seen prior only ('prior') and the
    remembered (lost-track) particles only ('mem')."""
    out = {'all': hz.curve(seq)}
    src = np.asarray(hz.sub['src'])
    wl = hz.wl
    try:
        for name, mask in (('prior', src == 0), ('mem', src != 0)):
            hz.wl = np.where(mask, wl, 0.)
            out[name] = hz.curve(seq)
    finally:
        hz.wl = wl
    return out


def make_hazard(legal, params=None):
    from .hazard import UntrackedHazard, load_params
    p = params if isinstance(params, dict) else load_params(params or HAZARD_PARAMS)
    return UntrackedHazard(legal, p, steps=64)


# ---------------------------------------------------------------------------------------------- labels (TRAIN)
def exact_keys(item, w):
    from .shortprobe import _key_real
    return [tuple(_key_real(c, w)) for c in item['probe']['cands']]


def relevance(keys, bin_m=0.25, levels=4):
    """Graded relevance of the exact keys of one state: contact / STOP away from the goal 0; running 1 .. levels+1 by
    the exact cost-to-go gap to the state's best running candidate (``bin_m`` m bins); goal reached levels+2."""
    run = [k[1] for k in keys if k[0] == 2.]
    best = max(run) if run else 0.
    out = []
    for k in keys:
        if k[0] == 0.:
            out.append(0)
        elif k[0] == 3.:
            out.append(levels+2)
        else:
            gap = max(0., best-k[1])
            out.append(1+max(0, levels-int(gap/bin_m)))
    return out


# ---------------------------------------------------------------------------------------------- extraction
def extract(bank, out, shard=0, shards=1, wait_for=0, poll=60.):
    from eventwm_graph.querygen import native
    native._install()
    from eventwm_graph.sandbox.core import WorldModelSandbox, SandboxConfig
    from .sbeval import candidates_of, _NoLayer2
    from .oraclebank import CVBank
    from .hazard import load_params
    sb = WorldModelSandbox(_NoLayer2(), SandboxConfig(materialize=False))
    hp = load_params(HAZARD_PARAMS)
    Path(out).mkdir(parents=True, exist_ok=True)
    settings = process_settings()
    print('riskrank extract: %s -> %s shard %d/%d settings %s' % (bank, out, shard, shards, settings), flush=True)
    seen = set()
    while True:
        files = sorted(glob.glob(os.path.join(bank, '*.pkl')))
        todo = [f for f in files if zlib.crc32(os.path.basename(f).encode()) % shards == shard and f not in seen]
        for f in todo:
            seen.add(f)
            name = os.path.basename(f)
            dst = Path(out, name)
            if dst.exists():
                continue
            try:
                it = pickle.load(open(f, 'rb'))
            except Exception as exc:          # still being written
                seen.discard(f)
                continue
            legal = it['legal']
            cands = it['probe']['cands']
            if [c['label'] for c in cands] != list(it['labels']):
                print('skip (labels differ)', name, flush=True)
                continue
            t0 = time.perf_counter()
            d = SimpleNamespace(candidates=candidates_of(it), result=SimpleNamespace(humans=CVBank(legal)), steps=it['step'],
                                labels=it['labels'], default=it['default'])
            try:
                outs = {c: rule_outputs(sb, legal, d, c, state=None) for c in CONTS}
                t1 = time.perf_counter()
                hz = make_hazard(legal, hp)
                X, names = features(legal, d, outs, hz)
            except Exception as exc:
                import traceback
                print('error', name, repr(exc), traceback.format_exc()[-600:], flush=True)
                continue
            t2 = time.perf_counter()
            rec = dict(file=name, bank=str(bank), key=it['key'], config=it['config'], decision=int(it['decision']),
                       step=int(it['step']), default=int(it['default']), labels=list(it['labels']),
                       kinds=[c['kind'] for c in cands], X=X.astype(np.float32), names=names,
                       choice={c: int(outs[c]['choice']) for c in CONTS},
                       rule={c: {k: np.asarray(outs[c][k], np.float64) for k in ('oc', 'pc', 'prog', 'dmin', 'stop_bad')}
                             for c in CONTS},
                       seq={c: [_seq(r).astype(np.float32) for r in outs[c]['ros'][0]] for c in CONTS},
                       keys={w: exact_keys(it, w) for w in WINDOWS},
                       real=[dict(outcome=c['real']['outcome'], steps=int(c['real']['steps'])) for c in cands],
                       ctg0=float(it['probe'].get('ctg0', float('nan'))), n_est=int(len(d.result.humans.track_keys)),
                       seconds=dict(imagine=round(t1-t0, 3), features=round(t2-t1, 3), hazard_build=round(hz.build_ms/1000, 3)),
                       settings=settings)
            with open(str(dst)+'.tmp', 'wb') as fh:
                pickle.dump(rec, fh, protocol=4)
            os.replace(str(dst)+'.tmp', dst)
            print('%s K=%d imagine %.2f s features %.2f s' % (name, len(cands), t1-t0, t2-t1), flush=True)
        n_have = len(glob.glob(os.path.join(bank, '*.pkl')))
        if n_have >= wait_for:
            break
        time.sleep(poll)


# ---------------------------------------------------------------------------------------------- model
class RiskModel:
    def __init__(self, path):
        self.path = str(path)
        self.meta = json.loads(Path(path, 'meta.json').read_text())
        import lightgbm as lgb
        self.boosters = [lgb.Booster(model_file=str(Path(path, f))) for f in self.meta.get('models', ['model.txt'])]
        self.names = list(self.meta['names'])
        self.vocab = dict(self.meta['labels_vocab'])
        self.thr = float(self.meta['thr'])
        self.hazard_params = self.meta.get('hazard_params') or None
        self._hp = None

    def hp(self):
        if self._hp is None:
            from .hazard import load_params
            self._hp = dict(self.hazard_params) if isinstance(self.hazard_params, dict) else load_params(HAZARD_PARAMS)
        return self._hp

    def columns(self, X, names):
        idx = [names.index(n) for n in self.names]
        return X[:, idx]

    def score(self, X):
        """Ranker score per candidate (the mean over a seed ensemble's boosters)."""
        return np.asarray(np.mean([b.predict(X, num_threads=1) for b in self.boosters], 0), np.float64)


_MODELS = {}
_WARNED = set()


def select_from_scores(s, dflt, thr):
    best = int(np.argmax(s))
    return best if s[best]-s[dflt] > thr else int(dflt)


def risk_select(path, sb, legal, d, state=None, cur=None):
    """The ranker's choice for one decision (``heuristics.multiworld_index`` with ``L3_MW_RISK=<path>``)."""
    from .oraclebank import CVBank
    t0 = time.perf_counter()
    m = _MODELS.get(path)
    if m is None:
        m = _MODELS[path] = RiskModel(path)
        print('mwroll: risk ranker %s (threshold %.4f, %d features)' % (path, m.thr, len(m.names)), flush=True)
    ps = process_settings()
    want = m.meta.get('settings', {})
    bad = {k: (ps.get(k), v) for k, v in want.items() if ps.get(k) != v}
    if bad and path not in _WARNED:
        _WARNED.add(path)
        print('mwroll: risk ranker WARNING: process settings differ from training %s' % bad, flush=True)
    humans = d.result.humans
    if not isinstance(humans, CVBank):
        humans = CVBank(legal)
    dd = SimpleNamespace(candidates=d.candidates, result=SimpleNamespace(humans=humans), steps=getattr(d, 'steps', 0),
                         labels=d.labels, default=d.default)
    outs = {}
    reused = []
    if (cur is not None and cur.get('cont') in CONTS and isinstance(d.result.humans, CVBank) and env_matches()
            and len(cur.get('ros', ())) == 1):
        # the current call imagined with the training settings: reuse it (its choice recomputed with EPS 0)
        o = dict(cur)
        oc = np.asarray(o['oc'])
        best = int(np.argmax(oc))
        o['choice'] = int(d.default) if oc[int(d.default)] >= oc[best] else best
        outs[int(cur['cont'])] = o
        reused.append(int(cur['cont']))
    derived = []
    for c in sorted(CONTS, reverse=True):
        if c in outs:
            continue
        longer = [outs[x] for x in sorted(outs) if x > c and not outs[x].get('derived')]
        if longer and os.environ.get('L3_MW_RISK_DERIVE', '1') == '1':
            outs[c] = derive_outputs(longer[0], legal, dd, c)       # exact prefix: no second imagination
            derived.append(c)
        else:
            outs[c] = rule_outputs(sb, legal, dd, c, state=None)
    t1 = time.perf_counter()
    hz = make_hazard(legal, m.hp())
    X, names = features(legal, dd, outs, hz, labels_vocab=m.vocab)
    Xm = m.columns(X, names)
    s = m.score(Xm)
    thr = float(os.environ.get('L3_MW_RISK_THR', m.thr))
    a = select_from_scores(s, int(d.default), thr)
    RISK_DEBUG.clear()
    RISK_DEBUG.update(scores=s.tolist(), thr=thr, choice=a, default=int(d.default),
                      rule_choice={c: int(outs[c]['choice']) for c in CONTS},
                      ms=dict(imagine=1000*(t1-t0), total=1000*(time.perf_counter()-t0)), reused=reused,
                      derived=derived)
    return a


# ---------------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--bank', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--shard', type=int, default=0)
    e.add_argument('--shards', type=int, default=1)
    e.add_argument('--wait-for', type=int, default=0, help='keep polling the bank until it holds this many states')
    a = ap.parse_args()
    if a.cmd == 'extract':
        for k, v in dict(L3_INHERIT='1', L3_MW_PERCEIVE='1', L3_MW_RULE='outcome', L3_MW_EPS='0', SANDBOX_PHYSICS='core',
                         SANDBOX_CONT_BOUNDARY='16', SANDBOX_CONT_INHERIT='1', QG_FAST_SENSE='1').items():
            if os.environ.get(k) != v:
                raise SystemExit('extract: set %s=%s before start (process-wide option; scratchpad crit_rule.py setup)' % (k, v))
        extract(a.bank, a.out, a.shard, a.shards, a.wait_for)


if __name__ == '__main__':
    main()
