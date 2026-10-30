"""Untracked-people contact hazard for the Sandbox's contact scoring (Layer 3 option, default off; LEGAL inputs only).

The legal Sandbox imagines every candidate against the people it tracks at t (``oraclebank.CVBank``: legal tracks at
most 2 s old, constant velocity); nobody else exists in it.  The contacts it misses are mostly with people it does not
track at t -- beside / behind the robot outside the 90 deg camera, lost tracks, people behind corners
(``output/agents/bankimag``).  This module scores them as a HAZARD over each imagined robot trajectory, from weighted
path particles of hypothetical untracked people.  The particles are SCORED only: they are never given to the imagined
planner (the variants that inserted hypothetical people into the planner's perception scored below the CV-null).

* Remembered people: every legal track lost ``mem_min_age`` .. ``mem_max_age`` s ago (the legal track log: the last
  record of the track -- last-seen position and velocity), continued from where it was last seen along velocity
  hypotheses (stopped / slower / as seen / faster x turned by ``mem_turns``; a track without a usable velocity walks at
  ``mem_walk`` in any of ``mem_headings`` directions).  Existence ``mem_exist`` decays with the time since it was lost
  (``mem_tau``).  A hypothesis within ``mem_match`` m of a person tracked now is that person (re-acquired): dropped.
* Never-seen people: an out-of-view prior of ``prior_rate`` people per m^2 on the legal map's non-BLOCKED cells within
  ``prior_radius`` m of the robot (UNKNOWN cells x ``prior_unknown``), on a lattice of positions (``prior_spacing``) x
  velocities (``speeds`` x ``headings``; ``speed_weights``).
* Negative information (the robot's own camera history): a particle whose straight path would have put it inside the
  real jaw camera's view (90 deg, 10 m, legal-map line of sight; ``mount``) at a past pose is down-weighted by
  exp(-kappa x seconds in view) -- back-traced over the last ``view_window`` s for never-seen people (``view_kappa``),
  since the track was lost for remembered ones (``mem_kappa``).  Optionally (``view_recover`` > 0) the never-seen
  density at a point recovers as 1 - exp(-time since the point was last in view / ``view_recover``).  A particle in
  view NOW is removed, as is one overlapping the robot's body at t (no contact has happened).

Each particle walks straight at its velocity (stopping at legal-map walls) for ``steps`` steps of 0.048 s.  For an
imagined robot trajectory (poses after 1..n steps, the Sandbox's ``seq``) particle m hits when its centre comes within
``radius`` of the robot (``metric`` 'centre': centre distance; 'rect': body-rectangle clearance) at any step; the
hazard is P(contact within w) = 1 - prod_m (1 - w_m hit_m(w)).  With ``perceive`` < 1, a hit whose particle is inside
the imagined camera's view (same camera model, from the imagined pose) at least ``react_steps`` before the hit counts
x ``perceive`` (the real robot would see that person and its planner would react).

Legal by construction: the inputs are the legal map, the legal tracks and track log, and the robot's own pose history
at t (``legal_context``).  The tape, ``truth_people``, exact branch outcomes and the episode's ``human_num`` are never
read here; the privileged labels of the TRAIN bank are used by ``fit`` only (dev: evaluation).

Fitting (``fit``; output/agents/bankimag, ORCA, the legal CV imagination ``a``): the weight parameters are fitted on
the TRAIN states by the log-likelihood of a dense privileged label -- the legal imagined trajectory comes within 0.75 m
of a TRUE person absent from the CV imagination (what ``L3_MW_SCORE_TRUTH`` scores) -- over 32 and 64 steps; the
exact contacts the CV imagination misses are too few on TRAIN to fit on (16 untracked-person events within 1.5 s).
The selector threshold ``thr`` gives 5% false alarms on the TRAIN CV-safe candidates without an exact contact within 32
steps.  Fitted v2: ``configs/l3_hazard_v2.json``.

    python scripts/l3_run.py eventwm_graph.l3.hazard cache --data output/agents/bankimag/orca_train.npz \
        --out output/agents/hazard/ctx_orca_train.pkl                     # legal contexts of the banked states
    python scripts/l3_run.py eventwm_graph.l3.hazard tables --ctx output/agents/hazard/ctx_orca_train.pkl \
        --data output/agents/bankimag/orca_train.npz --out output/agents/hazard/tab_orca_train.pkl
    python scripts/l3_run.py eventwm_graph.l3.hazard fit --train output/agents/hazard/tab_orca_train.pkl \
        --out output/agents/hazard/v2                                     # TRAIN only
    python scripts/l3_run.py eventwm_graph.l3.hazard score --params output/agents/hazard/v2/params.json \
        --train output/agents/hazard/tab_orca_train.pkl --dev output/agents/hazard/tab_orca_dev.pkl
    python scripts/l3_run.py eventwm_graph.l3.hazard check --params ... --train <tables> --ctx <contexts>  # runtime == tables

Selector option (``heuristics.multiworld_index``): ``L3_MW_HAZARD=<params.json>`` (+ ``L3_MW_HAZARD_THR``, default the
JSON's ``thr``): a candidate whose hazard reaches the threshold within its imagined trajectory (the contact steps)
counts as contacting from the step it does (outcome rule: contact category; contact rule: P(contact)).
``L3_MW_HAZARD_LOG=<jsonl>``: one analysis line per decision.
"""
import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np

STEP = 0.048
HFOV = math.radians(90.)
RANGE = 10.
CAM_FWD, CAM_RIGHT, CAM_YAW = 0.52, 0.02, math.radians(-4.5)       # = perceive.py's jaw-camera mount model
UNKNOWN, FREE, BLOCKED = 0, 1, 2
WINDOWS = (16, 32, 64)
VIEW_WINDOWS = (1., 2., 3., 4., 6., 8.)         # never-seen back-trace windows tabulated for fitting (s)
R_CENTRE = (0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.3)     # contact radii tabulated for fitting (m)
R_RECT = (0.1, 0.2, 0.3, 0.4, 0.55)
PROXY_R = 0.75                                   # the selector's contact radius (multiworld_index ``radius``)

DEFAULTS = dict(
    version=1,
    steps=64,
    # contact
    metric='centre', radius=0.75,
    # never-seen people
    prior_rate=0.01, prior_radius=7.0, prior_spacing=0.5, prior_unknown=0.5, exclude_clear=0.1,
    speeds=[0.0, 0.6, 1.2, 1.7], speed_weights=[0.34, 0.16, 0.25, 0.25], headings=12,
    # negative information
    view_window=4.0, view_stride=3, view_kappa=5.0, mount=True, rays=90,
    view_recover=0., age_window=20.0,
    # remembered people
    mem_min_age=2.0, mem_max_age=10.0, mem_exist=0.5, mem_tau=4.0, mem_kappa=2.0, mem_match=0.75,
    mem_mult=[0.0, 0.5, 1.0, 1.5], mem_mult_weights=[0.3, 0.2, 0.4, 0.1],
    mem_turns=[0., 30., -30., 60., -60., 90., -90., 135., -135., 180.], mem_turn_sigma=45.,
    mem_still=0.3, mem_walk=1.2, mem_headings=12,
    # perceivable hits
    perceive=1.0, react_steps=10,
    # selector threshold (L3_MW_HAZARD_THR overrides)
    thr=0.5,
)


def load_params(path):
    p = dict(DEFAULTS)
    p.update(json.loads(Path(path).read_text()))
    return p


# ---------------------------------------------------------------------------------------------- legal inputs
def legal_context(legal, history_s=12.0, crop_m=20.0):
    """The legal inputs of the hazard at decision time t, as plain arrays: the robot pose and pose history (camera
    views; the last ``history_s`` s), the legal map around the robot (``crop_m``), the people tracked now
    (``control.human_estimates``: the CV people of the legal Sandbox) and every other legal track's last record."""
    from eventwm_graph.querygen.control import human_estimates
    t = float(legal.time)
    pose = np.asarray(legal.pose, np.float64).copy()
    times = np.asarray(legal.times, np.float64)
    poses = np.asarray(legal.poses, np.float64).reshape(-1, 4)
    keep = times >= t-history_s-1e-9
    g = legal.map
    ci, cj = g.index(pose[0], pose[2])
    r = int(math.ceil(crop_m/g.res))
    st = np.zeros((2*r+1, 2*r+1), np.uint8)
    a0, a1 = max(ci-r, 0), min(ci+r+1, g.state.shape[0])
    b0, b1 = max(cj-r, 0), min(cj+r+1, g.state.shape[1])
    if a1 > a0 and b1 > b0:
        st[a0-(ci-r):a1-(ci-r), b0-(cj-r):b1-(cj-r)] = g.state[a0:a1, b0:b1]
    latest = {}
    for _, trs in list(getattr(legal, 'track_log', None) or ())+[(t, legal.tracks)]:
        for tr in trs:
            k = str(tr.get('track_id'))
            ls = float(tr.get('last_seen_time', -1e9))
            if k not in latest or ls >= latest[k][0]:
                latest[k] = (ls, tr)
    est = human_estimates(legal.tracks, t)
    cur_ids = {str(h.track_id) for h in est}
    cur = np.asarray([[h.position[0], -h.position[1], h.velocity[0], -h.velocity[1]] for h in est],
                     np.float64).reshape(-1, 4)                                     # world x, z, vx, vz at t
    lost = []
    for k, (ls, tr) in sorted(latest.items()):
        if k in cur_ids or not tr.get('valid', True):
            continue
        age = t-ls
        if not (0. < age <= history_s):
            continue
        p = tr['position']
        v = tr.get('estimated_velocity')
        ok = bool(tr.get('velocity_valid')) and v is not None
        lost.append([float(p[0]), float(p[2]), float(v[0]) if ok else 0., float(v[2]) if ok else 0., ls, float(ok),
                     float(tr.get('num_observations', 1))])
    return dict(time=t, pose=pose, hist_t=times[keep].copy(), hist_pose=poses[keep].copy(), res=float(g.res),
                kx=int(ci-r+g.ox), kz=int(cj-r+g.oz), map=st, cur=cur, lost=np.asarray(lost, np.float64).reshape(-1, 7))


def _state(ctx, x, z):
    """Legal-map state of world points (UNKNOWN outside the crop)."""
    x = np.asarray(x, np.float64)
    z = np.asarray(z, np.float64)
    m = ctx['map']
    i = np.floor(x/ctx['res']).astype(np.int64)-ctx['kx']
    j = np.floor(z/ctx['res']).astype(np.int64)-ctx['kz']
    ok = (i >= 0) & (i < m.shape[0]) & (j >= 0) & (j < m.shape[1])
    out = np.zeros(x.shape, np.uint8)
    out[ok] = m[i[ok], j[ok]]
    return out


def _camera(pose, mount):
    x, z, yaw = float(pose[0]), float(pose[2]), float(pose[3])
    if not mount:
        return x, z, yaw
    c, s = math.cos(yaw), math.sin(yaw)
    return x+CAM_FWD*c+CAM_RIGHT*s, z-CAM_FWD*s+CAM_RIGHT*c, yaw+CAM_YAW


def _rays(ctx, cam, rays):
    """Distance to the first legal-map BLOCKED cell along ``rays`` rays across the camera's frustum (RANGE+1: none)."""
    cx, cz, cyaw = cam
    a = -HFOV/2+(np.arange(rays)+0.5)*HFOV/rays
    rs = np.arange(1, int(RANGE/0.1)+1)*0.1
    th = cyaw+a
    hit = _state(ctx, cx+np.cos(th)[:, None]*rs[None], cz-np.sin(th)[:, None]*rs[None]) == BLOCKED
    return np.where(hit.any(1), rs[np.argmax(hit, 1)], RANGE+1.)


def _visible(px, pz, cam, first):
    """Points inside the camera's frustum (HFOV, RANGE) not behind the first BLOCKED cell of their ray."""
    cx, cz, cyaw = cam
    dx, dz = px-cx, pz-cz
    c, s = math.cos(cyaw), math.sin(cyaw)
    fw = dx*c-dz*s
    lf = -dx*s-dz*c
    ang = np.arctan2(lf, fw)
    r = np.hypot(dx, dz)
    rays = len(first)
    b = np.clip(((ang+HFOV/2)/HFOV*rays).astype(np.int64), 0, rays-1)
    return (np.abs(ang) <= HFOV/2) & (r <= RANGE) & (r <= first[b]+0.15)


def _rect_clear(pose, px, pz):
    """Body-rectangle clearance of the robot at ``pose`` to points (``imagine.rectangle_clearance``, vectorised)."""
    from eventwm_graph.querygen.native import ROBOT_BOUNDS
    c, s = math.cos(float(pose[3])), math.sin(float(pose[3]))
    dx, dz = px-float(pose[0]), pz-float(pose[2])
    lx, lz = dx*c-dz*s, dx*s+dz*c
    xmin, xmax, zmin, zmax = ROBOT_BOUNDS
    ex = np.maximum(np.maximum(xmin-lx, 0.), lx-xmax)
    ez = np.maximum(np.maximum(zmin-lz, 0.), lz-zmax)
    return np.hypot(ex, ez)


def _checks(ctx, p, back_s):
    """Past camera checks within ``back_s`` s before t: (times, cameras, first-hit rays), every ``view_stride`` poses."""
    t = ctx['time']
    T, P = ctx['hist_t'], ctx['hist_pose']
    out = []
    i = len(T)-1-int(p['view_stride'])
    while i >= 0 and t-float(T[i]) <= back_s+1e-9:
        cam = _camera(P[i], p['mount'])
        out.append((float(T[i]), cam, _rays(ctx, cam, int(p['rays']))))
        i -= int(p['view_stride'])
    return out


def _view_age(ctx, p, px, pz, back_s):
    """Seconds since each point was last inside the camera's view (line of sight) over the pose history of the last
    ``back_s`` s (inf: not within it).  Position only: how long ago the robot looked there and saw nobody."""
    t = ctx['time']
    age = np.full(len(px), np.inf)
    for tc, cam, first in _checks(ctx, p, back_s):                  # newest first
        live = np.flatnonzero(~np.isfinite(age))
        if not len(live):
            break
        vis = _visible(px[live], pz[live], cam, first)
        age[live[vis]] = t-tc
    return age


def _kmax(ctx, P0, V, steps):
    """Last step each straight path is free of legal-map walls (it stays there afterwards); ``steps`` if never."""
    k = np.arange(1, steps+1)*STEP
    moving = np.flatnonzero((V != 0.).any(1))
    kmax = np.full(len(P0), steps, np.int64)
    if len(moving):
        X = P0[moving, None, 0]+V[moving, None, 0]*k[None]
        Z = P0[moving, None, 1]+V[moving, None, 1]*k[None]
        B = _state(ctx, X, Z) == BLOCKED
        hit = B.any(1)
        kmax[moving[hit]] = np.argmax(B[hit], 1)
    return kmax


# ---------------------------------------------------------------------------------------------- particles
def particles(ctx, p, view_windows=None):
    """Path particles of hypothetical untracked people at t: positions ``P0`` [M, 2], velocities ``V`` [M, 2], wall
    stop ``kmax`` [M] and the attributes their weights are computed from (``weights``).  ``view_windows``: the
    never-seen back-trace windows to tabulate (default: ``p['view_window']`` only)."""
    t = ctx['time']
    pose = ctx['pose']
    x0, z0 = float(pose[0]), float(pose[2])
    vw = tuple(view_windows) if view_windows is not None else (float(p['view_window']),)
    # --- never-seen people: lattice positions x velocities
    g, R = float(p['prior_spacing']), float(p['prior_radius'])
    o = np.arange(-math.floor(R/g), math.floor(R/g)+1)*g
    ox, oz = np.meshgrid(o, o, indexing='ij')
    ox, oz = ox.ravel(), oz.ravel()
    k = np.hypot(ox, oz) <= R+1e-9
    px, pz = x0+ox[k], z0+oz[k]
    st = _state(ctx, px, pz)
    k = (st != BLOCKED) & (_rect_clear(pose, px, pz) >= float(p['exclude_clear']))
    px, pz, unk = px[k], pz[k], st[k] == UNKNOWN
    vel, sp_idx = [], []
    nh = int(p['headings'])
    sw = list(p['speed_weights'])
    for i, s in enumerate(p['speeds']):
        if i < len(sw) and sw[i] <= 0. and view_windows is None:
            continue                                    # zero weight: the particle cannot contribute
        if s <= 0.:
            vel.append((0., 0.)); sp_idx.append(i)
        else:
            for h in range(nh):
                a = 2*math.pi*h/nh
                vel.append((s*math.cos(a), s*math.sin(a))); sp_idx.append(i)
    vel = np.asarray(vel, np.float64).reshape(-1, 2)
    nP, nV = len(px), len(vel)
    P0 = np.repeat(np.stack([px, pz], 1), nV, 0)
    V = np.tile(vel, (nP, 1))
    va = np.full(nP, np.inf)
    if (float(p['view_recover']) > 0. or view_windows is not None) and nP:
        va = _view_age(ctx, p, px, pz, float(p['age_window']))
    a = dict(src=np.zeros(nP*nV, np.int8), sp=np.tile(np.asarray(sp_idx, np.int16), nP),
             unk=np.repeat(unk, nV), ent=np.full(nP*nV, -1, np.int32), age=np.zeros(nP*nV),
             mult=np.full(nP*nV, -1, np.int16), turnw=np.zeros(nP*nV), vw=vw, vage=np.repeat(va, nV))
    # --- remembered people: hypotheses per lost track, from the last-seen point
    L = ctx['lost']
    if len(L):
        L = L[(t-L[:, 4] >= float(p['mem_min_age'])-1e-9) & (t-L[:, 4] <= float(p['mem_max_age'])+1e-9)]
    mP, mV, ment, mage, mmult, mturn, mstart = [], [], [], [], [], [], []
    for e, (lx, lz, vx, vz, ls, ok, _n) in enumerate(L):
        age = t-ls
        s0 = math.hypot(vx, vz)
        known = ok > 0.5 and s0 >= float(p['mem_still'])
        if known:
            h0 = math.atan2(vz, vx)
            turns = np.radians(np.asarray(p['mem_turns'], np.float64))
            tw = np.exp(-0.5*(np.asarray(p['mem_turns'], np.float64)/max(float(p['mem_turn_sigma']), 1e-3))**2)
            tw = tw/tw.sum()
        else:
            s0 = float(p['mem_walk'])
            h0 = 0.
            nm = int(p['mem_headings'])
            turns = 2*math.pi*np.arange(nm)/nm
            tw = np.full(nm, 1./nm)
        for mi, mult in enumerate(p['mem_mult']):
            if mult <= 0.:
                hyp = [((0., 0.), 1.)]
            else:
                hyp = [((mult*s0*math.cos(h0+d), mult*s0*math.sin(h0+d)), float(w)) for d, w in zip(turns, tw)]
            for v, w in hyp:
                mP.append((lx, lz)); mV.append(v); ment.append(e); mage.append(age); mmult.append(mi)
                mturn.append(w); mstart.append(ls)
    tview = np.zeros((nP*nV, len(vw)))
    if mP:
        mP = np.asarray(mP, np.float64); mV = np.asarray(mV, np.float64); mage = np.asarray(mage)
        # walk from the last-seen point to t (stopping at walls); the past path is checked against the camera history
        steps_back = np.ceil(mage/STEP).astype(np.int64)
        kb = _kmax(ctx, mP, mV, int(steps_back.max()))
        kb = np.minimum(kb, steps_back)
        P_t = mP+mV*(np.minimum(mage, kb*STEP))[:, None]
        V_t = np.where((kb < steps_back)[:, None], 0., mV)
        mt = np.zeros(len(mP))
        for tc, cam, first in _checks(ctx, p, float(p['mem_max_age'])):
            el = tc-np.asarray(mstart)
            on = el > 0.
            if not on.any():
                continue
            idx = np.flatnonzero(on)
            dt = np.minimum(el[idx], kb[idx]*STEP)
            vis = _visible(mP[idx, 0]+mV[idx, 0]*dt, mP[idx, 1]+mV[idx, 1]*dt, cam, first)
            mt[idx[vis]] += float(p['view_stride'])*STEP
        P0 = np.concatenate([P0, P_t]); V = np.concatenate([V, V_t])
        n = len(mP)
        a['src'] = np.concatenate([a['src'], np.ones(n, np.int8)])
        a['sp'] = np.concatenate([a['sp'], np.full(n, -1, np.int16)])
        a['unk'] = np.concatenate([a['unk'], np.zeros(n, bool)])
        a['ent'] = np.concatenate([a['ent'], np.asarray(ment, np.int32)])
        a['age'] = np.concatenate([a['age'], mage])
        a['mult'] = np.concatenate([a['mult'], np.asarray(mmult, np.int16)])
        a['turnw'] = np.concatenate([a['turnw'], np.asarray(mturn, np.float64)])
        a['vage'] = np.concatenate([a['vage'], np.full(n, np.inf)])
        a['mem_tview'] = np.concatenate([np.zeros(nP*nV), mt])
        tview = np.concatenate([tview, np.zeros((n, len(vw)))])
        # re-acquired: a hypothesis at a person tracked now
        cur = ctx['cur']
        match = np.zeros(len(P0), bool)
        if len(cur) and float(p['mem_match']) > 0:
            d = np.hypot(P_t[:, None, 0]-cur[None, :, 0], P_t[:, None, 1]-cur[None, :, 1]).min(1)
            match[nP*nV:] = d < float(p['mem_match'])
        a['match'] = match
    else:
        a['mem_tview'] = np.zeros(nP*nV)
        a['match'] = np.zeros(nP*nV, bool)
    # never-seen: back-traced straight paths against the camera history (seconds in view within each window)
    nN = nP*nV
    if nN and len(vw):
        for tc, cam, first in _checks(ctx, p, max(vw)):
            el = t-tc
            vis = _visible(P0[:nN, 0]-V[:nN, 0]*el, P0[:nN, 1]-V[:nN, 1]*el, cam, first)
            for wi, W in enumerate(vw):
                if el <= W+1e-9:
                    tview[:nN, wi] += vis*(float(p['view_stride'])*STEP)
    a['tview'] = tview
    # in view NOW (the camera at t sees nobody there) -> removed
    cam = _camera(pose, p['mount'])
    a['now'] = _visible(P0[:, 0], P0[:, 1], cam, _rays(ctx, cam, int(p['rays']))) if len(P0) else np.zeros(0, bool)
    a['P0'], a['V'] = P0, V
    a['kmax'] = _kmax(ctx, P0, V, int(p['steps'])) if len(P0) else np.zeros(0, np.int64)
    return a


def weights(a, p):
    """Per-particle weight (existence probability of that hypothetical person) under parameters ``p``."""
    n = len(a['src'])
    w = np.zeros(n)
    if not n:
        return w
    nv = a['src'] == 0
    if nv.any():
        sw = np.asarray(p['speed_weights'], np.float64)
        sp = np.asarray(p['speeds'], np.float64)
        per = np.where(sp > 0, sw/max(int(p['headings']), 1), sw)                     # weight per velocity
        g = float(p['prior_spacing'])
        vw = list(a['vw'])
        wi = vw.index(float(p['view_window'])) if float(p['view_window']) in vw else int(np.argmin(np.abs(np.asarray(vw)-float(p['view_window']))))
        w[nv] = (float(p['prior_rate'])*g*g*per[a['sp'][nv]]*np.where(a['unk'][nv], float(p['prior_unknown']), 1.)
                 * np.exp(-float(p['view_kappa'])*a['tview'][nv, wi]))
        if float(p['view_recover']) > 0.:
            # density recovers after a look: 1 - exp(-(time since the point was last in view) / view_recover)
            w[nv] *= -np.expm1(-np.asarray(a['vage'][nv], np.float64)/float(p['view_recover']))
    mm = a['src'] == 1
    if mm.any():
        mw = np.asarray(p['mem_mult_weights'], np.float64)
        mw = mw/max(mw.sum(), 1e-12)
        ex = float(p['mem_exist'])*np.exp(-np.maximum(a['age'][mm]-float(p['mem_min_age']), 0.)/max(float(p['mem_tau']), 1e-3))
        w[mm] = ex*mw[a['mult'][mm]]*a['turnw'][mm]*np.exp(-float(p['mem_kappa'])*a['mem_tview'][mm])
        w[mm & a['match']] = 0.
    w[a['now']] = 0.
    return np.clip(w, 0., 1.)


# ---------------------------------------------------------------------------------------------- contact
def _seq_xz(seq):
    seq = np.asarray(seq, np.float64)
    if seq.ndim == 2 and seq.shape[1] == 4:
        return seq[:, 0], seq[:, 2], seq[:, 3]
    return seq[:, 0], seq[:, 1], np.zeros(len(seq))


def first_hits(a, seq, idx, radii, metric, steps):
    """First hit step (1-based: the robot after i steps vs the particle at row i; 0 = no hit) of particles ``idx`` for
    each radius in ``radii`` -> int16 [len(idx), len(radii)].  Rows of ``seq`` with non-finite poses end it."""
    rx, rz, ryaw = _seq_xz(seq)
    ok = np.isfinite(rx) & np.isfinite(rz)
    n = int(np.argmin(ok)) if not ok.all() else len(rx)
    n = min(n, int(steps))
    out = np.zeros((len(idx), len(radii)), np.int16)
    if not n or not len(idx):
        return out
    rx, rz, ryaw = rx[:n], rz[:n], ryaw[:n]
    P0, V, km = a['P0'][idx], a['V'][idx], a['kmax'][idx]
    kk = np.minimum(np.arange(1, n+1)[None], km[:, None])*STEP                        # [m, n]
    dx = P0[:, None, 0]+V[:, None, 0]*kk-rx[None]
    dz = P0[:, None, 1]+V[:, None, 1]*kk-rz[None]
    if metric == 'rect':
        from eventwm_graph.querygen.native import ROBOT_BOUNDS
        c, s = np.cos(ryaw)[None], np.sin(ryaw)[None]
        lx, lz = dx*c-dz*s, dx*s+dz*c
        xmin, xmax, zmin, zmax = ROBOT_BOUNDS
        d = np.hypot(np.maximum(np.maximum(xmin-lx, 0.), lx-xmax), np.maximum(np.maximum(zmin-lz, 0.), lz-zmax))
    else:
        d = np.hypot(dx, dz)
    for ri, r in enumerate(radii):
        h = d < float(r)
        anyh = h.any(1)
        out[anyh, ri] = np.argmax(h[anyh], 1)+1
    return out


def near(a, seq, reach, steps):
    """Particles whose path can come within ``reach`` of the trajectory (bounding-box test)."""
    rx, rz, _ = _seq_xz(seq)
    ok = np.isfinite(rx) & np.isfinite(rz)
    n = min(int(np.argmin(ok)) if not ok.all() else len(rx), int(steps))
    if not n or not len(a['src']):
        return np.zeros(0, np.int64)
    rx, rz = rx[:n], rz[:n]
    P0, V = a['P0'], a['V']
    e = P0+V*(np.minimum(a['kmax'], n)*STEP)[:, None]
    lo = np.minimum(P0, e)
    hi = np.maximum(P0, e)
    return np.flatnonzero((hi[:, 0] >= rx.min()-reach) & (lo[:, 0] <= rx.max()+reach)
                          & (hi[:, 1] >= rz.min()-reach) & (lo[:, 1] <= rz.max()+reach))


def first_visible(ctx, a, seq, idx, mount, upto, rays=90):
    """First step k (1-based; 0 = never up to ``upto[m]``) at which particle idx[m] is inside the imagined camera's
    view from the robot after k steps (frustum, RANGE, legal-map line of sight: the same ray model as the camera
    history)."""
    rx, rz, ryaw = _seq_xz(seq)
    out = np.zeros(len(idx), np.int16)
    if not len(idx):
        return out
    upto = np.asarray(upto, np.int64)
    P0, V, km = a['P0'][idx], a['V'][idx], a['kmax'][idx]
    for k in range(1, int(upto.max(initial=0))+1):
        live = np.flatnonzero((out == 0) & (upto >= k))
        if not len(live) or k > len(rx) or not np.isfinite(rx[k-1]):
            break
        cam = _camera((rx[k-1], 0., rz[k-1], ryaw[k-1]), mount)
        kk = np.minimum(k, km[live])*STEP
        vis = _visible(P0[live, 0]+V[live, 0]*kk, P0[live, 1]+V[live, 1]*kk, cam, _rays(ctx, cam, rays))
        out[live[vis]] = k
    return out


class UntrackedHazard:
    """The hazard of one decision state (shared by every candidate): built once from the legal state at t."""

    def __init__(self, legal, params, steps=None):
        self.p = dict(DEFAULTS)
        self.p.update(params or {})
        if steps is not None:
            self.p['steps'] = int(steps)
        tick = time.perf_counter()
        self.ctx = legal if isinstance(legal, dict) else legal_context(
            legal, history_s=max(float(self.p['mem_max_age']), float(self.p['view_window']),
                                 float(self.p['age_window']) if float(self.p['view_recover']) > 0. else 0.)+1.)
        self.a = particles(self.ctx, self.p)
        self.w = weights(self.a, self.p)
        self.live = np.flatnonzero(self.w > 0.)
        M = len(self.w)
        self.sub = {k: (v[self.live] if isinstance(v, np.ndarray) and len(v) == M else v) for k, v in self.a.items()}
        self.wl = self.w[self.live]
        self.build_ms = (time.perf_counter()-tick)*1000

    def curve(self, seq):
        """Cumulative P(contact by step k), k = 1..n, for an imagined trajectory (poses after 1..n steps)."""
        p = self.p
        rx, _, _ = _seq_xz(seq)
        ok = np.isfinite(rx)
        n = min(int(np.argmin(ok)) if not ok.all() else len(rx), int(p['steps']))
        if not n:
            return np.zeros(0)
        reach = float(p['radius'])+(0.65 if p['metric'] == 'rect' else 0.)
        sub = self.sub
        cand = near(sub, seq, reach, n)
        fh = first_hits(sub, seq, cand, (float(p['radius']),), p['metric'], n)[:, 0]
        hit = fh > 0
        cand, fh = cand[hit], fh[hit]
        w = self.wl[cand]
        if float(p['perceive']) < 1. and len(cand):
            upto = np.maximum(fh.astype(np.int64)-int(p['react_steps']), 0)
            fv = first_visible(self.ctx, sub, seq, cand, p['mount'], upto, int(p['rays']))
            w = np.where(fv > 0, w*float(p['perceive']), w)
        lg = np.bincount(fh.astype(np.int64)-1, weights=np.log1p(-np.minimum(w, 1.-1e-9)), minlength=n)[:n]
        return 1.-np.exp(np.cumsum(lg))

    def p_contact(self, seq, windows=WINDOWS):
        c = self.curve(seq)
        return [float(c[min(w, len(c))-1]) if len(c) else 0. for w in windows]

    def first_over(self, seq, thr):
        """First step index (0-based, into ``seq``) at which the hazard reaches ``thr``; None if it never does."""
        c = self.curve(seq)
        i = np.flatnonzero(c >= float(thr))
        return int(i[0]) if len(i) else None


# ---------------------------------------------------------------------------------------------- fitting data
def build_cache(data, out, bank_dir=None, first=None):
    """Legal contexts of the dataset's banked states (``legal_context`` of each state's LegalState)."""
    D = np.load(data)
    meta = json.loads(str(D['meta']))
    bank_dir = bank_dir or meta['bank_dir']
    files = [str(f) for f in D['state_file']]
    if first:
        files = files[:int(first)]
    ctxs = []
    t0 = time.perf_counter()
    for i, f in enumerate(files):
        item = pickle.load(open(os.path.join(bank_dir, f), 'rb'))
        ctxs.append(legal_context(item['legal'], history_s=21.))
        del item
        if (i+1) % 25 == 0:
            print('cache %d / %d  %.0f s' % (i+1, len(files), time.perf_counter()-t0), flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(str(out)+'.tmp', 'wb') as fh:
        pickle.dump(dict(files=files, data=str(data), ctx=ctxs), fh, protocol=4)
    os.replace(str(out)+'.tmp', out)
    print('cache: %d states -> %s (%.0f s)' % (len(ctxs), out, time.perf_counter()-t0), flush=True)


def build_tables(ctx_path, data, out, p=None):
    """Per state: the particles of the tabulation geometry (every speed, every back-trace window) and, per candidate
    of the dataset's LEGAL CV imagination (``a_poses``), the first hit step of every particle near it for each
    tabulated contact radius (centre and rectangle) and its first step in the imagined camera's view."""
    p = dict(DEFAULTS, **(p or {}))
    C = pickle.load(open(ctx_path, 'rb'))
    D = np.load(data)
    poses = D['a_poses']
    first, ncand = D['state_first_cand'], D['state_n_cands']
    steps = int(p['steps'])
    rows = []
    t0 = time.perf_counter()
    reach = max(max(R_CENTRE), max(R_RECT)+0.65)
    for s, ctx in enumerate(C['ctx']):
        a = particles(ctx, p, view_windows=VIEW_WINDOWS)
        pc, pi, fhc, fhr, fv = [], [], [], [], []
        for c in range(int(ncand[s])):
            r = int(first[s])+c
            seq = poses[r].astype(np.float64)
            idx = near(a, seq, reach, steps)
            if not len(idx):
                continue
            hc = first_hits(a, seq, idx, R_CENTRE, 'centre', steps)
            hr = first_hits(a, seq, idx, R_RECT, 'rect', steps)
            keep = (hc > 0).any(1) | (hr > 0).any(1)
            idx, hc, hr = idx[keep], hc[keep], hr[keep]
            upto = np.maximum(np.maximum(hc.max(1), hr.max(1)).astype(np.int64)-6, 0)
            v = first_visible(ctx, a, seq, idx, p['mount'], upto, int(p['rays']))
            pc.append(np.full(len(idx), c, np.int16)); pi.append(idx.astype(np.int32)); fhc.append(hc); fhr.append(hr)
            fv.append(v)
        keepa = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 else v)
                 for k, v in a.items() if k not in ('P0', 'V', 'kmax')}
        rows.append(dict(attrs=keepa, n=len(a['src']), cand=np.concatenate(pc) if pc else np.zeros(0, np.int16),
                         part=np.concatenate(pi) if pi else np.zeros(0, np.int32),
                         fh_centre=np.concatenate(fhc) if fhc else np.zeros((0, len(R_CENTRE)), np.int16),
                         fh_rect=np.concatenate(fhr) if fhr else np.zeros((0, len(R_RECT)), np.int16),
                         fvis=np.concatenate(fv) if fv else np.zeros(0, np.int16)))
        if (s+1) % 25 == 0:
            print('tables %d / %d  %.0f s' % (s+1, len(C['ctx']), time.perf_counter()-t0), flush=True)
    with open(str(out)+'.tmp', 'wb') as fh:
        pickle.dump(dict(rows=rows, params=p, data=str(data), radii_centre=R_CENTRE, radii_rect=R_RECT,
                         view_windows=VIEW_WINDOWS), fh, protocol=4)
    os.replace(str(out)+'.tmp', out)
    print('tables: %d states -> %s (%.0f s)' % (len(rows), out, time.perf_counter()-t0), flush=True)


# ---------------------------------------------------------------------------------------------- scoring from tables
class Tables:
    """Tabulated particles + hits of one bank with its labels (labels: privileged, for fitting / evaluation)."""

    def __init__(self, path):
        T = pickle.load(open(path, 'rb'))
        self.p0 = T['params']
        D = np.load(T['data'])
        self.D = D
        R = len(D['cand_state'])
        self.R = R
        first = D['state_first_cand']
        self.rows = T['rows']
        # global particle attributes
        offs = np.cumsum([0]+[r['n'] for r in self.rows])
        self.offs = offs
        keys = ('src', 'sp', 'unk', 'ent', 'age', 'mult', 'turnw', 'mem_tview', 'match', 'now', 'tview', 'vage')
        self.a = {k: np.concatenate([r['attrs'][k] for r in self.rows]) for k in keys}
        self.a['vw'] = tuple(T['view_windows'])
        # global entity ids for remembered people
        ent = self.a['ent'].copy()
        base = 0
        for i, r in enumerate(self.rows):
            sl = slice(offs[i], offs[i+1])
            e = ent[sl]
            m = e >= 0
            e[m] += base
            ent[sl] = e
            base = (int(e[m].max())+1) if m.any() else base
        self.a['ent'] = ent
        self.cand = np.concatenate([np.asarray(r['cand'], np.int64)+int(first[i]) for i, r in enumerate(self.rows)])
        self.part = np.concatenate([np.asarray(r['part'], np.int64)+int(offs[i]) for i, r in enumerate(self.rows)])
        self.fh = dict(centre=np.concatenate([r['fh_centre'] for r in self.rows]),
                       rect=np.concatenate([r['fh_rect'] for r in self.rows]))
        self.fvis = np.concatenate([r['fvis'] for r in self.rows])
        self.radii = dict(centre=tuple(T['radii_centre']), rect=tuple(T['radii_rect']))
        # labels and the legal CV imagination's own contact prediction
        out, rs, cstep = D['cand_real_outcome'], D['cand_real_steps'], D['cand_contact_step']
        an = D['a_n']
        cd = np.where(np.isfinite(D['a_cd_cv']), D['a_cd_cv'], np.inf)
        self.state = D['cand_state']
        self.y, self.cv_hit, self.pop, self.yp = {}, {}, {}, {}
        self.step1 = (out == 'human_collision') & (cstep == 1)
        # dense label (privileged): the legal imagined robot (a) within PROXY_R of a TRUE person absent from the CV
        # imagination -- what TRUE untracked people in the contact scoring (L3_MW_SCORE_TRUTH) would flag
        untr = ~D['person_est_tracked'] & (np.arange(D['truth'].shape[2])[None] < D['state_n_people'][:, None])
        pp = D['a_cd_true_pp']
        mask_p = untr[self.state]
        for w in WINDOWS:
            self.y[w] = (out == 'human_collision') & (rs <= w)
            m = np.arange(64)[None] < np.minimum(an, w)[:, None]
            self.cv_hit[w] = (np.where(m, cd, np.inf).min(1) < 0.75)
            self.pop[w] = ~self.cv_hit[w] & ~self.step1
            dmin = np.where(m[:, :, None] & np.isfinite(pp), pp, np.inf).min(1)
            self.yp[w] = (np.where(mask_p, dmin, np.inf) < PROXY_R).any(1)

    def scores(self, p):
        """P(contact within w) for every candidate, w in WINDOWS."""
        w = weights(self.a, p)
        ri = self.radii[p['metric']].index(float(p['radius']))
        fh = self.fh[p['metric']][:, ri].astype(np.int64)
        ww = w[self.part]
        if float(p['perceive']) < 1.:
            perc = (self.fvis > 0) & (self.fvis.astype(np.int64) <= fh-int(p['react_steps']))
            ww = np.where(perc, ww*float(p['perceive']), ww)
        lg = np.log1p(-np.minimum(ww, 1.-1e-9))
        out = {}
        for W in WINDOWS:
            h = (fh > 0) & (fh <= W)
            s = np.bincount(self.cand[h], weights=lg[h], minlength=self.R)
            out[W] = 1.-np.exp(s)
        return out

    def curves(self, p, steps=64):
        """Cumulative P(contact by step k) [R, steps] for every candidate (k = 1..steps)."""
        w = weights(self.a, p)
        ri = self.radii[p['metric']].index(float(p['radius']))
        fh = self.fh[p['metric']][:, ri].astype(np.int64)
        ww = w[self.part]
        if float(p['perceive']) < 1.:
            perc = (self.fvis > 0) & (self.fvis.astype(np.int64) <= fh-int(p['react_steps']))
            ww = np.where(perc, ww*float(p['perceive']), ww)
        h = (fh > 0) & (fh <= steps)
        lg = np.bincount(self.cand[h]*steps+fh[h]-1, weights=np.log1p(-np.minimum(ww[h], 1.-1e-9)),
                         minlength=self.R*steps).reshape(self.R, steps)
        return 1.-np.exp(np.cumsum(lg, 1))

    def selection(self, curve=None, thr=None, w=32, eps=0.):
        """Offline outcome rule (``multiworld_index`` L3_MW_RULE=outcome on the bank's LEGAL CV imagination over ``w``
        steps; + the hazard when ``curve`` is given) scored against the exact outcomes within ``w`` steps."""
        D = self.D
        an = D['a_n']
        cd = np.where(np.isfinite(D['a_cd_cv']), D['a_cd_cv'], np.inf)
        ctg = D['a_ctg'][:, WINDOWS.index(w)]
        ctg0 = D['state_ctg0']
        out_r, rs, rctg = D['cand_real_outcome'], D['cand_real_steps'], D['cand_real_ctg'][:, WINDOWS.index(w)]
        n = 0
        agree = crit = fixed = broke = dev = 0
        for s in range(len(D['state_first_cand'])):
            r0, nc = int(D['state_first_cand'][s]), int(D['state_n_cands'][s])
            vals, keys = [], []
            for r in range(r0, r0+nc):
                m = int(min(an[r], w))
                hit = np.flatnonzero(cd[r, :m] < 0.75)
                first = int(hit[0]) if len(hit) else None
                if curve is not None and m:
                    hh = np.flatnonzero(curve[r, :m] >= thr)
                    if len(hh):
                        first = int(hh[0]) if first is None else min(first, int(hh[0]))
                if first is not None:
                    v = 0.5*first/max(m, 1)
                elif D['a_stop'][r] and an[r] <= w:
                    v = (3.+0.5*(1.-max(an[r], 1)/w)) if D['a_near_goal'][r] else 0.
                else:
                    g = ctg0[s]-ctg[r] if np.isfinite(ctg[r]) else 0.
                    v = 2.+0.5*float(np.clip(g/max(ctg0[s], 1e-6), 0., 0.999))
                vals.append(v)
                if out_r[r] in ('human_collision', 'stop_fail') and rs[r] <= w:
                    keys.append((0., rs[r]))
                elif out_r[r] == 'success' and rs[r] <= w:
                    keys.append((3., -rs[r]))
                else:
                    keys.append((2., -(rctg[r] if np.isfinite(rctg[r]) else 99.)))
            di = int(D['state_default'][s])
            best = int(np.argmax(vals))
            pick = di if vals[di] >= vals[best]-eps else best
            kb = max(keys)
            n += 1
            agree += int(keys[pick][0] == kb[0])
            dev += int(pick != di)
            if keys[di][0] == 0. and kb[0] > 0.:
                crit += 1
                fixed += int(keys[pick][0] > 0.)
            if keys[di][0] > 0. and keys[pick][0] == 0.:
                broke += 1
        return dict(states=n, category_ok=agree/max(n, 1), critical=crit, fixed=fixed, broke=broke,
                    deviations=dev/max(n, 1))


def auc(score, y):
    y = np.asarray(y, bool)
    s = np.asarray(score, np.float64)
    if not y.any() or y.all():
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    r = np.arange(1, len(s)+1, dtype=np.float64)
    # average ranks for ties
    while i < len(ss):
        j = i
        while j+1 < len(ss) and ss[j+1] == ss[i]:
            j += 1
        r[i:j+1] = 0.5*(i+1+j+1)
        i = j+1
    ranks[order] = r
    npos = y.sum()
    nneg = len(y)-npos
    return float((ranks[y].sum()-npos*(npos+1)/2)/(npos*nneg))


def within_auc(score, y, state):
    """Mean AUC over the states with both outcomes (the selection problem); number of such states."""
    vals = []
    for s in np.unique(state):
        m = state == s
        if y[m].any() and (~y[m]).any():
            vals.append(auc(score[m], y[m]))
    return (float(np.mean(vals)) if vals else float('nan')), len(vals)


def loglik(P, y):
    P = np.clip(P, 1e-6, 1-1e-6)
    return float(np.mean(np.where(y, np.log(P), np.log1p(-P))))


def _boot_auc(s, y, state, n=200, seed=0):
    """State-clustered bootstrap 95% interval of the pooled AUC."""
    rng = np.random.default_rng(seed)
    st = np.unique(state)
    idx = {k: np.flatnonzero(state == k) for k in st}
    vals = []
    for _ in range(int(n)):
        pick = np.concatenate([idx[k] for k in rng.choice(st, len(st))])
        if y[pick].any() and (~y[pick]).any():
            vals.append(auc(s[pick], y[pick]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float('nan')]*2


def metrics(T, p, thr=None, w_fit=32, boot=0):
    """Per window w, on the candidates the legal CV imagination calls safe within w (no contact at step 1): pooled and
    within-state AUC / log-likelihood of the hazard for (real) the exact contacts within w and (proxy) the imagined
    trajectory meeting a TRUE person absent from the CV imagination; contacts caught at fixed false-alarm rates; with
    ``thr``: caught / false alarms at that threshold, the combined legal contact predictor (CV or hazard) on every
    candidate, and the offline outcome rule (``Tables.selection``)."""
    S = T.scores(p)
    out = {}
    for w in WINDOWS:
        m = T.pop[w]
        s = S[w][m]
        st = T.state[m]
        rec = dict(pop=int(m.sum()), mean_p=float(s.mean()))
        for lab, Y in (('real', T.y), ('proxy', T.yp)):
            y = Y[w][m]
            wa, nm = within_auc(s, y, st)
            r = dict(pos=int(y.sum()), pos_states=int(len(np.unique(st[y]))), auc=auc(s, y), within=wa, mixed=nm,
                     ll=loglik(s, y), rate=float(y.mean()))
            if boot:
                r['auc_ci'] = _boot_auc(s, y, st, boot)
            neg = np.sort(s[~y])
            for f in (0.02, 0.05, 0.10):
                tv = float(neg[int(math.ceil((1.-f)*len(neg)))-1]) if len(neg) else 1.
                r['caught@%g' % f] = int((s[y] > tv).sum())
            if thr is not None:
                r['caught'] = int((s[y] >= thr).sum())
                r['false'] = int((s[~y] >= thr).sum())
                r['far'] = float((s[~y] >= thr).mean()) if (~y).any() else float('nan')
            rec[lab] = r
        if thr is not None:
            # combined legal contact predictor on every candidate (step-1 contacts excluded): CV contact, or CV contact
            # or hazard >= thr, against the exact contacts within w
            allm = ~T.step1
            pred = T.cv_hit[w] | (S[w] >= thr)
            ya = T.y[w]
            rec['thr'] = thr
            rec['cv_only'] = dict(tp=int((T.cv_hit[w] & ya & allm).sum()), fp=int((T.cv_hit[w] & ~ya & allm).sum()),
                                  pos=int((ya & allm).sum()))
            rec['cv_plus'] = dict(tp=int((pred & ya & allm).sum()), fp=int((pred & ~ya & allm).sum()),
                                  pos=int((ya & allm).sum()))
        out[w] = rec
    if thr is not None:
        out['selection'] = dict(cv=T.selection(None, None, w=w_fit), cv_hazard=T.selection(T.curves(p), thr, w=w_fit))
    return out, S


FIT_KEYS = ('metric', 'radius', 'prior_rate', 'prior_unknown', 'view_window', 'view_kappa', 'view_recover', 'mem_exist',
            'mem_tau', 'mem_kappa', 'mem_turn_sigma', 'perceive', 'react_steps', 'speed_weights', 'mem_mult_weights')


def fit_params(T, p0, label='proxy', windows=(32, 64), sweeps=3, log=print, fixed=()):
    """Coordinate search of the weight parameters (the particle geometry is fixed by the tables) maximising the TRAIN
    log-likelihood of ``label`` ('proxy': the imagined trajectory meets a TRUE person absent from the CV imagination;
    'real': the exact contacts the CV imagination misses) summed over ``windows``, on the CV-safe candidates."""
    p = dict(p0)
    Y = T.yp if label == 'proxy' else T.y

    def obj(q):
        S = T.scores(q)
        return sum(loglik(S[w][T.pop[w]], Y[w][T.pop[w]]) for w in windows)
    grid = dict(
        prior_rate=[0.0005, 0.001, 0.002, 0.004, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12],
        prior_unknown=[0., 0.25, 0.5, 0.75, 1.],
        view_window=list(VIEW_WINDOWS),
        view_kappa=[0., 0.5, 1., 2., 4., 8., 16., 50.],
        view_recover=[0., 1., 2., 4., 8., 16.],
        mem_exist=[0., 0.1, 0.25, 0.5, 0.75, 1.],
        mem_tau=[1., 2., 4., 8., 16., 64.],
        mem_kappa=[0., 0.5, 1., 2., 4., 8., 50.],
        mem_turn_sigma=[15., 30., 45., 70., 120.],
        perceive=[0., 0.25, 0.5, 0.75, 1.],
        react_steps=[6, 10, 16],
    )
    for k in fixed:
        grid.pop(k, None)
    best = obj(p)
    log('fit start (%s, windows %s): ll %.5f' % (label, list(windows), best))
    for sweep in range(sweeps):
        if 'radius' not in fixed:
            for metric in ('centre', 'rect'):
                for r in T.radii[metric]:
                    q = dict(p, metric=metric, radius=float(r))
                    v = obj(q)
                    if v > best+1e-9:
                        best, p = v, q
        for k, vals in grid.items():
            for v in vals:
                q = dict(p)
                q[k] = v
                o = obj(q)
                if o > best+1e-9:
                    best, p = o, q
        # speed / multiplier weights: move mass between pairs of entries
        for key in ('speed_weights', 'mem_mult_weights'):
            if key in fixed:
                continue
            for i in range(len(p[key])):
                for j in range(len(p[key])):
                    if i == j:
                        continue
                    for f in (0.25, 0.5, 1.0):
                        q = dict(p)
                        ws = list(q[key])
                        d = ws[i]*f
                        if d <= 0:
                            continue
                        ws[i] -= d
                        ws[j] += d
                        q[key] = ws
                        o = obj(q)
                        if o > best+1e-9:
                            best, p = o, q
        log('sweep %d: ll %.5f  %s' % (sweep, best, json.dumps({k: p[k] for k in FIT_KEYS})))
    return p, best


def _fmt(tag, M):
    lines = []
    for w in WINDOWS:
        r = M[w]
        for lab in ('real', 'proxy'):
            x = r[lab]
            s = ('%s w%-2d %-5s CV-safe %d, positives %d (%d states): AUC %.3f%s  within-state %.3f (%d mixed)  ll %.4f  '
                 'caught@FAR 2/5/10%%: %d/%d/%d' % (
                     tag, w, lab, r['pop'], x['pos'], x['pos_states'], x['auc'],
                     (' [%.3f, %.3f]' % tuple(x['auc_ci'])) if 'auc_ci' in x else '', x['within'], x['mixed'], x['ll'],
                     x['caught@0.02'], x['caught@0.05'], x['caught@0.1']))
            if 'caught' in x:
                s += '  | thr: caught %d, false alarms %d (FAR %.3f)' % (x['caught'], x['false'], x['far'])
            lines.append(s)
        if 'cv_only' in r:
            lines.append('%s w%-2d all candidates (exact contacts %d): CV contact rule tp %d fp %d -> CV or hazard>=%.4g tp %d '
                         'fp %d' % (tag, w, r['cv_only']['pos'], r['cv_only']['tp'], r['cv_only']['fp'], r['thr'],
                                    r['cv_plus']['tp'], r['cv_plus']['fp']))
    for k, v in (M.get('selection') or {}).items():
        lines.append('%s offline outcome rule (32 steps, %s): category = best %.3f  critical %d fixed %d broke %d  '
                     'deviations %.2f  (%d states)' % (tag, k, v['category_ok'], v['critical'], v['fixed'], v['broke'],
                                                       v['deviations'], v['states']))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['cache', 'tables', 'fit', 'score', 'check'])
    ap.add_argument('--data')
    ap.add_argument('--bank-dir', default=None)
    ap.add_argument('--ctx')
    ap.add_argument('--out')
    ap.add_argument('--train')
    ap.add_argument('--dev')
    ap.add_argument('--params')
    ap.add_argument('--first', type=int, default=None)
    ap.add_argument('--far', type=float, default=0.05)
    ap.add_argument('--w', type=int, default=32, help='selector window (steps): threshold and offline selection')
    ap.add_argument('--label', default='proxy', choices=['proxy', 'real'])
    ap.add_argument('--windows', default='32,64', help='fit windows (steps)')
    ap.add_argument('--sweeps', type=int, default=3)
    ap.add_argument('--boot', type=int, default=200)
    ap.add_argument('--fix', default='', help="parameters kept at their --params / table values (e.g. 'radius')")
    a = ap.parse_args()
    if a.mode == 'cache':
        build_cache(a.data, a.out, a.bank_dir, a.first)
    elif a.mode == 'tables':
        build_tables(a.ctx, a.data, a.out, load_params(a.params) if a.params else None)
    elif a.mode == 'fit':
        T = Tables(a.train)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        logf = open(out/'fit.log', 'a')

        def log(s):
            print(s, flush=True)
            logf.write(s+'\n'); logf.flush()
        p0 = dict(T.p0)
        if a.params:
            p0.update(json.loads(Path(a.params).read_text()))
        windows = tuple(int(x) for x in a.windows.split(','))
        fixed = tuple(k for k in a.fix.split(',') if k)
        p, ll = fit_params(T, p0, label=a.label, windows=windows, sweeps=a.sweeps, log=log, fixed=fixed)
        # selector threshold: false-alarm rate a.far on the TRAIN CV-safe candidates without an exact contact within
        # a.w steps (hazard >= thr  <=>  above that candidate's score)
        S = T.scores(p)
        neg = np.sort(S[a.w][T.pop[a.w] & ~T.y[a.w]])
        tv = float(neg[int(math.ceil((1.-a.far)*len(neg)))-1])
        p['thr'] = float(np.nextafter(max(tv, 1e-9), 1.))
        p['fit'] = dict(train=str(a.train), label=a.label, windows=list(windows), ll=ll, thr_w=a.w, thr_far=a.far)
        M, _ = metrics(T, p, thr=p['thr'], w_fit=a.w, boot=a.boot)
        log(_fmt('train', M))
        res = dict(train=M)
        if a.dev:
            Md, _ = metrics(Tables(a.dev), p, thr=p['thr'], w_fit=a.w, boot=a.boot)
            log(_fmt('dev  ', Md))
            res['dev'] = Md
        (out/'params.json').write_text(json.dumps(p, indent=1))
        (out/'metrics.json').write_text(json.dumps(res, indent=1, default=float))
        log('-> %s' % (out/'params.json'))
    elif a.mode == 'score':
        p = load_params(a.params)
        for tag, path in (('train', a.train), ('dev  ', a.dev)):
            if path:
                M, _ = metrics(Tables(path), p, thr=p['thr'], w_fit=a.w, boot=a.boot)
                print(_fmt(tag, M), flush=True)
    elif a.mode == 'check':
        # runtime class vs the tabulated scores on the first states of a bank (same particles, same hits)
        p = load_params(a.params)
        T = Tables(a.train)
        C = pickle.load(open(a.ctx, 'rb'))
        S = T.scores(p)
        D = T.D
        worst = 0.
        for s in range(min(a.first or 5, len(C['ctx']))):
            hz = UntrackedHazard(C['ctx'][s], p)
            for c in range(int(D['state_n_cands'][s])):
                r = int(D['state_first_cand'][s])+c
                pc = hz.p_contact(D['a_poses'][r].astype(np.float64))
                for i, w in enumerate(WINDOWS):
                    worst = max(worst, abs(pc[i]-S[w][r]))
            print('state %d: particles %d (live %d), build %.0f ms, max |runtime - tables| so far %.2e' % (
                s, len(hz.w), len(hz.live), hz.build_ms, worst), flush=True)


if __name__ == '__main__':
    main()
