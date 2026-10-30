"""Bank imagination dataset (ANALYSIS; privileged information is used only as labels and for evaluation).

Imagined candidate trajectories on the banked decision states (``statebank.py``), for fitting a calibrated contact
model and an untracked-people hazard.  For every state and every candidate (``sbeval.candidates_of``) the Sandbox
imagines 64 steps (3.07 s): the candidate's interval, then the default policy with a decision boundary every 16 steps
and goal-query inheritance, with imagined core contact physics, on the legal map frozen at t -- exactly the call of
``shortprobe.probe_decision`` (its variants camCVPc / camTPc) -- with the people perceived through the imagined camera
(``perceive.VisibleWorld``: 90 deg frustum, 10 m, legal-map line of sight, 2 s tracker memory) from:

  a  LEGAL: the people tracked at t continuing at the tracker's constant velocity (``oraclebank.CVBank``);
  b  TRUE (privileged): every person's recorded future (the banked ``truth_people``, ``truth_bank``).

Stored per (state, candidate): the imagined poses of a and b; per step and per TRUE person the centre distance and
the body-rectangle clearance (``sandbox.imagine.rectangle_clearance``), their per-step minima over the TRUE and over
the CV-predicted people, and the imagined camera's view of every true person; the candidate's exact outcome and contact
step (``item['probe']``, the branch simulator).  Per state: every true person's path and whether the robot tracked it
at t, the robot and camera poses at t, and the state file name (the legal context -- map, track log -- is re-read from
it).  Fit only on the TRAIN bank; the dev bank is for evaluation.  ``README`` documents every array.

    L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.bankimag run --bank output/statebank/orca_train \\
        --work output/agents/bank/bankimag_work [--shard i --shards n]
    python scripts/l3_run.py eventwm_graph.l3.bankimag merge --bank output/statebank/orca_train \\
        --work output/agents/bank/bankimag_work --out output/agents/bankimag
    python scripts/l3_run.py eventwm_graph.l3.bankimag stats --out output/agents/bankimag
    L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.bankimag check --bank output/statebank/orca_dev --limit 12
"""
import argparse
import glob
import json
import math
import os
import pickle
import time
import zlib
from pathlib import Path

import numpy as np

W = 64                       # imagined steps per candidate (0.048 s each: 3.07 s)
WINDOWS = (16, 32, 64)
VARIANTS = ('a', 'b')        # a: LEGAL constant-velocity tracked people; b: TRUE people (privileged)
MATCH_M = 0.75               # track <-> true person matching radius at t (oraclebank.OracleBank)
VERSION = 1


# ---------------------------------------------------------------------------------------------- people sources
def truth_bank(item, steps=W):
    """TRUE people (privileged; labels / analysis only): the banked ``truth_people`` [65, N, 2] (row j = t + j steps)
    behind the ``oraclebank.OracleBank`` interface in mode 'all' -- the people of shortprobe's camT variants, the same
    track-key matching (a tracked person keeps its legal track id inside ``VisibleWorld``)."""
    from .oraclebank import OracleBank
    T = np.asarray(item['truth_people'], np.float64)
    tape = np.zeros((T.shape[0], T.shape[1], 3))
    tape[..., 0] = T[..., 0]
    tape[..., 2] = T[..., 1]
    return OracleBank(tape, 0, item['legal'], 'all', horizon_steps=steps+1, long_steps=steps)


def cv_bank(legal, steps=W):
    """LEGAL people: the tracked people at the tracker's constant velocity (shortprobe camCV's ``CVBank``)."""
    from .oraclebank import CVBank
    return CVBank(legal, horizon_steps=steps+1)


def people_world(legal, bank, steps=W):
    from .perceive import VisibleWorld
    return VisibleWorld(legal, bank, 0, steps)


_OPTIONS = {}


def process_options(cfg):
    """Set the process-wide legal options of the episode's query generator exactly as the online environment does
    (``sandbox.episode.generator``: ``LegalState.BUMP_MEMORY`` -- 0.25 m for k10, default 0 -- and
    ``gridplan.BLOCKED_TARGET_RELAX_M``).  Without it an offline imagination has no bump memory: 'thru' queries plan
    without the legal bump discs and imagined physics records no bumps, unlike the online probe."""
    from eventwm_graph.querygen import gridplan
    from eventwm_graph.querygen.legal import LegalState
    from eventwm_graph.sandbox.episode import generator
    key = (cfg.gen, cfg.dynamics)
    if _OPTIONS.get('key') != key:
        generator(cfg.gen, cfg.dynamics)
        _OPTIONS.update(key=key, bump_memory_m=float(LegalState.BUMP_MEMORY),
                        blocked_target_relax_m=float(gridplan.BLOCKED_TARGET_RELAX_M))
    return dict(_OPTIONS)


# ---------------------------------------------------------------------------------------------- geometry
def centre_distances(poses, people):
    """[n, P] planar distance of robot poses [n, 4] to people [n, P, 2] (world x, z); NaN for non-finite people."""
    with np.errstate(invalid='ignore'):
        return np.hypot(people[..., 0]-poses[:, None, 0], people[..., 1]-poses[:, None, 2])


def rect_clearances(poses, people):
    """[n, P] body-rectangle clearance (``sandbox.imagine.rectangle_clearance`` per person, vectorised)."""
    from .heuristics import _rect_clearance
    if not people.shape[1]:
        return np.zeros((len(poses), 0))
    return _rect_clearance(poses, people)


def _rowmin(d):
    """Per-row min over the finite entries; +inf for rows without any person."""
    if not d.shape[1]:
        return np.full(len(d), np.inf)
    return np.where(np.isfinite(d), d, np.inf).min(axis=1)


def camera_view(legal, poses, people):
    """[n, P] bool: person p (row k of ``people``) is inside the imagined camera's frustum with a clear legal-map line of
    sight from robot pose k -- ``perceive.VisibleWorld.tracks_at``'s detection test (same camera pose, frustum, range,
    occlusion by the legal map at t)."""
    from .infofeat import _blocked_lookup, _los
    from .perceive import camera_pose, HFOV, RANGE
    n, P = len(poses), people.shape[1]
    out = np.zeros((n, P), bool)
    if not P or not n:
        return out
    blocked = _blocked_lookup(legal.map)
    for k in range(n):
        pts = people[k]
        ok = np.isfinite(pts).all(1)
        cx, cz, cyaw = camera_pose(poses[k])
        dx, dz = pts[:, 0]-cx, pts[:, 1]-cz
        fw = dx*math.cos(cyaw)-dz*math.sin(cyaw)
        lf = -dx*math.sin(cyaw)-dz*math.cos(cyaw)
        with np.errstate(invalid='ignore'):
            inside = ok & (np.abs(np.arctan2(lf, fw)) <= HFOV/2) & (np.hypot(dx, dz) <= RANGE)
        idx = np.flatnonzero(inside)
        if len(idx):
            clear = _los(blocked, np.full(len(idx), cx), np.full(len(idx), cz), pts[idx, 0], pts[idx, 1])
            out[k, idx[clear]] = True
    return out


def _real_camera(item):
    """(origin [3], rotation [3, 3], planar (x, z, yaw)) of the real jaw camera at t (the legal sensor row of the
    decision's DecisionState; the camera looks down its local -Z axis); NaN when unavailable."""
    nan = (np.full(3, np.nan), np.full((3, 3), np.nan), np.full(3, np.nan))
    try:
        fr = item['state'].frames[-1]
        if abs(float(fr['timestamp'])-float(item['legal'].time)) > 1e-3:
            return nan
        o = np.asarray(fr['camera_origin'], np.float64)
        R = np.asarray(fr['camera_rotation'], np.float64)
    except Exception:
        return nan
    look = R @ np.array([0., 0., -1.])
    return o, R, np.array([o[0], o[2], math.atan2(-look[2], look[0])])


# ---------------------------------------------------------------------------------------------- one state
def imagine_state(sb, item, steps=W, variants=VARIANTS, physics='core'):
    """Every candidate of a banked state imagined for ``steps`` steps per variant (the shortprobe.probe_decision call).
    Returns a dict of per-state and per-candidate arrays (see ``README``).  ``physics=''`` reproduces the probe's
    variants without imagined physics (``check``)."""
    from eventwm_graph.querygen.control import human_estimates
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.sandbox.imagine import imagine, frozen_map
    from eventwm_graph.sandbox.episode import default_query
    from .compiler import LegalField
    from .env import ConfigSpec
    from .perceive import camera_pose, CAM_FWD, CAM_RIGHT, CAM_YAW
    from .sbeval import candidates_of
    t0 = time.perf_counter()
    legal = item['legal']
    pr = item['probe']
    cfg = ConfigSpec.parse(item['config'])
    opts = process_options(cfg)
    dyn = FAMILIES[cfg.dynamics]
    goal = default_query()
    gmap = frozen_map(legal.map)
    field = LegalField(legal)
    cands = candidates_of(item)
    K = len(cands)
    if [c['label'] for c in pr['cands']] != list(item['labels']):
        raise ValueError('probe candidates differ from the banked queries')
    T = np.asarray(item['truth_people'], np.float64)                      # [65, N, 2], row j = t + j steps
    N = T.shape[1]
    Tn = np.concatenate([T, np.repeat(T[-1:], max(0, steps+1-len(T)), 0)])[:steps+1]
    tb = truth_bank(item, steps)
    cvb = cv_bank(legal, steps)
    CV = np.asarray(people_world(legal, cvb, steps).P, np.float64)        # [steps+1, M, 2] what (a) perceives
    M = CV.shape[1]
    pose = np.asarray(legal.pose, np.float64)
    # tracked at t (privileged label): OracleBank's rule -- every valid legal track claims the nearest true person
    # within 0.75 m of its (last-seen) position, first come first served
    tracked = ~np.asarray(tb.unseen, bool) if N else np.zeros(0, bool)
    keys = list(tb.track_keys)
    tid = np.asarray([k[len('track:'):] if k.startswith('track:') else '' for k in keys], dtype=object)
    by_id = {str(t['track_id']): t for t in legal.tracks}
    t_dist = np.full(N, np.nan); t_age = np.full(N, np.nan); t_vis = np.zeros(N, bool)
    for n in range(N):
        tr = by_id.get(str(tid[n])) if tid[n] else None
        if tr is not None:
            p = np.asarray(tr['position'], np.float64)
            t_dist[n] = math.hypot(T[0, n, 0]-p[0], T[0, n, 1]-p[2])
            t_age[n] = float(legal.time)-float(tr['last_seen_time'])
            t_vis[n] = bool(tr.get('visible_now'))
    # the CV people (legal estimates, extrapolated to t) -> true person within 0.75 m (sbeval.TrueTrackedBank's rule)
    est = human_estimates(legal.tracks, legal.time)
    cv_ids = np.asarray([str(e.track_id) for e in est], dtype=object)
    cv_true = np.full(M, -1, np.int64)
    for m, e in enumerate(est):
        if not N:
            break
        d0 = np.hypot(T[0, :, 0]-e.position[0], T[0, :, 1]+e.position[1])
        if np.nanmin(d0) < MATCH_M:
            cv_true[m] = int(np.nanargmin(d0))
    est_tracked = np.zeros(N, bool)
    cv_index = np.full(N, -1, np.int64)
    for m in range(M):
        if cv_true[m] >= 0:
            est_tracked[cv_true[m]] = True
            cv_index[cv_true[m]] = m
    cam_o, cam_R, cam_real = _real_camera(item)
    c, s = math.cos(pose[3]), math.sin(pose[3])
    cam_mount = np.array([pose[0]+CAM_FWD*c+CAM_RIGHT*s, pose[2]-CAM_FWD*s+CAM_RIGHT*c, pose[3]+CAM_YAW])
    out = dict(version=VERSION, file=item.get('_file', ''), key=item['key'], config=item['config'],
               decision=int(item['decision']), step=int(item['step']), time=float(legal.time),
               robot_pose=pose, goal=np.asarray(legal.goal, np.float64), ctg0=float(field.start),
               cam_origin=cam_o, cam_rotation=cam_R, cam_real=cam_real,
               cam_center=np.array([pose[0], pose[2], pose[3]]), cam_mount=cam_mount,
               cam_imagined=np.asarray(camera_pose(pose), np.float64),
               default=int(item['default']), n_tracks=len(legal.tracks), n_est=len(est),
               truth=T.astype(np.float32), tracked=tracked, track_id=tid.astype(str), track_dist=t_dist,
               track_age=t_age, track_visible=t_vis, est_tracked=est_tracked, cv_index=cv_index,
               cv_ids=cv_ids.astype(str), cv_paths=CV.astype(np.float32), cv_true=cv_true,
               labels=np.asarray(item['labels'], str), kinds=np.asarray([c['kind'] for c in pr['cands']], str),
               h=np.asarray([int(c['h']) for c in pr['cands']], np.int64), level=int(item['level']),
               bump_memory_m=opts['bump_memory_m'], n_bumps=len(getattr(legal, 'bumps', ()) or ()))
    # exact outcome of every candidate (branch simulator: interval, then the default policy, 64 steps)
    oc = [c['real']['outcome'] for c in pr['cands']]
    st = np.asarray([int(c['real']['steps']) for c in pr['cands']], np.int64)
    out['real_outcome'] = np.asarray(oc, str)
    out['real_steps'] = st
    out['contact_step'] = np.where(np.asarray(oc) == 'human_collision', st, -1)

    def _w(c, key):
        d = c['real'].get(key) or {}
        return [np.inf if d.get(str(w)) is None and key == 'clr' else (np.nan if d.get(str(w)) is None else
                float(d[str(w)])) for w in WINDOWS]
    out['real_clr'] = np.asarray([_w(c, 'clr') for c in pr['cands']], np.float64)
    out['real_ctg'] = np.asarray([_w(c, 'ctg') for c in pr['cands']], np.float64)
    out['real_cd'] = np.asarray([_w(c, 'cd') for c in pr['cands']], np.float64)
    banks = dict(a=cvb, b=tb)                     # read-only inside VisibleWorld (positions copied at construction)
    for v in variants:
        poses = np.full((K, steps, 4), np.nan)
        n_ = np.zeros(K, np.int64); stop = np.zeros(K, bool); near = np.zeros(K, bool)
        ended = np.full(K, '', dtype=object); err = np.full(K, '', dtype=object); ms = np.zeros(K)
        cd_pp = np.full((K, steps, N), np.nan); rc_pp = np.full((K, steps, N), np.nan)
        cd_cv = np.full((K, steps), np.nan); rc_cv = np.full((K, steps), np.nan)
        vis = np.zeros((K, steps+1, N), bool)
        ctg = np.full((K, len(WINDOWS)), np.nan)
        for k, cand in enumerate(cands):
            h = int(out['h'][k])
            t1 = time.perf_counter()
            try:
                ex, _ = sb._execution(cand)
            except Exception as exc:                     # contract violation of this candidate only: recorded
                ex = exc
            if isinstance(ex, Exception):
                ended[k], err[k] = 'error', '%s: %s' % (type(ex).__name__, ex)
                continue
            hum = people_world(legal, banks[v], steps)            # a fresh camera / tracker per candidate (as the probe)
            ro = imagine(legal, ex, dyn, hum, h, level=int(cand.level), gmap=gmap, continuation=max(0, steps-h),
                         cont_query=goal, planner=cfg.planner, cont_boundary=16, cont_inherit=True, physics=physics)
            cc = getattr(ro, 'cont', None)
            im = np.concatenate([ro.poses[1:], cc]) if cc is not None and len(cc) else np.asarray(ro.poses[1:])
            im = np.asarray(im, np.float64).reshape(-1, 4)[:steps]
            n = len(im)
            ms[k] = (time.perf_counter()-t1)*1000
            n_[k], stop[k], ended[k] = n, bool(ro.stop), ro.ended_by
            err[k] = ro.error or ''
            if not n:
                continue
            poses[k, :n] = im
            near[k] = math.hypot(im[-1, 0]-legal.goal[0], im[-1, 2]-legal.goal[2]) <= 0.3
            if N:
                cd_pp[k, :n] = centre_distances(im, Tn[1:n+1])
                rc_pp[k, :n] = rect_clearances(im, Tn[1:n+1])
                vis[k, :n+1] = camera_view(legal, np.concatenate([pose[None], im]), Tn[:n+1])
            cd_cv[k, :n] = _rowmin(centre_distances(im, CV[1:n+1]))
            rc_cv[k, :n] = _rowmin(rect_clearances(im, CV[1:n+1]))
            for i, w in enumerate(WINDOWS):
                p = im[min(w, n)-1]
                ctg[k, i] = float(field.at(p[0], p[2]))
        cd_true = np.where(np.arange(steps)[None] < n_[:, None], _rowmin_3(cd_pp), np.nan)
        rc_true = np.where(np.arange(steps)[None] < n_[:, None], _rowmin_3(rc_pp), np.nan)
        out.update({'%s_poses' % v: poses.astype(np.float32), '%s_n' % v: n_, '%s_stop' % v: stop,
                    '%s_near_goal' % v: near, '%s_ended' % v: ended.astype(str), '%s_error' % v: err.astype(str),
                    '%s_ms' % v: ms, '%s_cd_true_pp' % v: cd_pp.astype(np.float32),
                    '%s_rc_true_pp' % v: rc_pp.astype(np.float32), '%s_cd_true' % v: cd_true.astype(np.float32),
                    '%s_rc_true' % v: rc_true.astype(np.float32), '%s_cd_cv' % v: cd_cv.astype(np.float32),
                    '%s_rc_cv' % v: rc_cv.astype(np.float32), '%s_vis' % v: vis, '%s_ctg' % v: ctg})
    out['seconds'] = round(time.perf_counter()-t0, 2)
    return out


def _rowmin_3(d):
    """[K, T, P] -> [K, T]: min over the finite people entries; +inf without people."""
    if not d.shape[2]:
        return np.full(d.shape[:2], np.inf)
    return np.where(np.isfinite(d), d, np.inf).min(axis=2)


def _guard():
    from eventwm_graph.querygen.executor import QueryExecution
    if not getattr(QueryExecution, 'INHERIT', False):
        raise SystemExit('bankimag: the banked outcomes were labelled with L3_INHERIT=1; set it for the imagination')


def _settings():
    from . import perceive
    return dict(L3_INHERIT=os.environ.get('L3_INHERIT', ''), L3_CAM_MOUNT=int(bool(perceive.CAM_MOUNT)),
                L3_CV_MAX_AGE=os.environ.get('L3_CV_MAX_AGE', '2.0'), physics='core', cont_boundary=16,
                cont_inherit=True, steps=W)


def _sandbox():
    from eventwm_graph.sandbox.core import WorldModelSandbox, SandboxConfig
    from .sbeval import _NoLayer2
    return WorldModelSandbox(_NoLayer2(), SandboxConfig(materialize=False, route=True))


def run(bank, work, shard=0, shards=1, limit=0):
    os.environ.setdefault('QG_FAST_SENSE', '1')
    _guard()
    paths = sorted(glob.glob(os.path.join(bank, '*.pkl')))
    paths = [p for p in paths if zlib.crc32(Path(p).name.encode()) % shards == shard]
    if limit:
        paths = paths[:limit]
    out = Path(work, Path(bank).name)
    out.mkdir(parents=True, exist_ok=True)
    (out/'settings.json').write_text(json.dumps(_settings()))
    sb = _sandbox()
    done = 0
    t0 = time.perf_counter()
    for p in paths:
        target = out/(Path(p).stem+'.pkl')
        if target.exists():
            continue
        item = pickle.load(open(p, 'rb'))
        item['_file'] = Path(p).name
        try:
            r = imagine_state(sb, item)
        except Exception as exc:
            import traceback
            print(json.dumps(dict(state=Path(p).name, error=repr(exc), tb=traceback.format_exc()[-600:])), flush=True)
            continue
        tmp = Path(str(target)+'.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(r, f, protocol=4)
        os.replace(tmp, target)
        done += 1
        print(json.dumps(dict(state=Path(p).name, seconds=r['seconds'], done=done, of=len(paths),
                              elapsed=round(time.perf_counter()-t0, 1))), flush=True)
    sb.close()


# ---------------------------------------------------------------------------------------------- check
def check(bank, limit=12, shard=0, shards=1):
    """Harness check on banks whose probe holds shortprobe's camT variant (camera model, TRUE people, continuation
    fidelity, NO physics, frozen map): re-imagine it offline and compare the imagined length / STOP / end reason and
    the min rectangle clearance to the true people within 16 / 32 / 64 steps with the probe's online record."""
    os.environ.setdefault('QG_FAST_SENSE', '1')
    _guard()
    paths = sorted(glob.glob(os.path.join(bank, '*.pkl')))
    paths = [p for p in paths if zlib.crc32(Path(p).name.encode()) % shards == shard][:limit]
    sb = _sandbox()
    n = same = close = 0
    worst = []
    for p in paths:
        item = pickle.load(open(p, 'rb'))
        if 'camT' not in item['probe']['cands'][0]['imag']:
            print('no camT record in', p)
            continue
        r = imagine_state(sb, item, variants=('b',), physics='')
        for k, c in enumerate(item['probe']['cands']):
            runs = c['imag']['camT']
            if not runs:
                continue
            on = runs[0]
            nk = int(r['b_n'][k])
            mine = [float(np.nanmin(np.where(np.isfinite(r['b_rc_true'][k, :min(w, nk)]), r['b_rc_true'][k, :min(w, nk)],
                                             np.inf))) if nk else math.inf for w in WINDOWS]
            theirs = [math.inf if on['clr'][str(w)] is None else float(on['clr'][str(w)]) for w in WINDOWS]
            dif = max(abs(a-b) if math.isfinite(a) and math.isfinite(b) else (0. if a == b else math.inf)
                      for a, b in zip(mine, theirs))
            ok = nk == int(on['n']) and bool(r['b_stop'][k]) == bool(on['stop']) and r['b_ended'][k] == on['ended']
            n += 1
            same += int(ok and dif < 1e-4)
            close += int(ok and dif < 0.05)
            worst.append((dif, Path(p).stem, c['label'], nk, int(on['n'])))
        print(json.dumps(dict(state=Path(p).name, seconds=r['seconds'])), flush=True)
    worst.sort(reverse=True)
    print('camT reproduction: %d candidates, identical (length, STOP, end, clearance within 1e-4 m) %d, within 0.05 m %d'
          % (n, same, close))
    for x in worst[:8]:
        print('   max |clearance diff| %.4f  %s %s  n %d / %d' % x)
    sb.close()
    return n, same


# ---------------------------------------------------------------------------------------------- merge
STATE_KEYS = ('file', 'key', 'config', 'decision', 'step', 'time', 'robot_pose', 'goal', 'ctg0', 'cam_origin',
              'cam_rotation', 'cam_real', 'cam_center', 'cam_mount', 'cam_imagined', 'default', 'n_tracks', 'n_est',
              'level', 'bump_memory_m', 'n_bumps', 'seconds')
PERSON_KEYS = ('tracked', 'track_id', 'track_dist', 'track_age', 'track_visible', 'est_tracked', 'cv_index')
CAND_KEYS = (('labels', 'label'), ('kinds', 'kind'), ('h', 'h'), ('real_outcome', 'real_outcome'),
             ('real_steps', 'real_steps'), ('contact_step', 'contact_step'), ('real_clr', 'real_clr'),
             ('real_ctg', 'real_ctg'), ('real_cd', 'real_cd'))
VAR_KEYS = ('poses', 'n', 'stop', 'near_goal', 'ended', 'error', 'ms', 'cd_true', 'rc_true', 'cd_cv', 'rc_cv', 'ctg')
VAR_PERSON_KEYS = ('cd_true_pp', 'rc_true_pp', 'vis')


def _pad(arrs, axis, size, fill):
    out = []
    for a in arrs:
        a = np.asarray(a)
        if a.shape[axis] < size:
            shape = list(a.shape)
            shape[axis] = size-a.shape[axis]
            a = np.concatenate([a, np.full(shape, fill, dtype=a.dtype)], axis)
        out.append(a)
    return out


def _fill(dtype):
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if dtype == bool:
        return False
    if np.issubdtype(dtype, np.integer):
        return -1
    return ''


def attribute_contacts(D):
    """Which TRUE person a real contact was with (approximate: the branch's real poses are not banked): the true person
    nearest to the TRUE-people imagined robot (b; else a) at the poses after s-3 .. s+1 steps, s = the contact step.
    Returns (person index or -1, its min centre distance there, source variant)."""
    st = D['cand_real_steps']
    who = np.full(len(st), -1, np.int64)
    dist = np.full(len(st), np.nan)
    src = np.full(len(st), '', dtype='<U1')
    for r in np.flatnonzero(D['cand_real_outcome'] == 'human_collision'):
        s = int(st[r])
        for v in ('b', 'a'):
            n = int(D['%s_n' % v][r])
            lo, hi = max(0, s-4), min(n, s+1)              # rows j = poses after j+1 steps
            if hi <= lo:
                continue
            d = np.asarray(D['%s_cd_true_pp' % v][r, lo:hi], np.float64)
            if not np.isfinite(d).any():
                continue
            per = np.where(np.isfinite(d), d, np.inf).min(axis=0)
            who[r], dist[r], src[r] = int(np.argmin(per)), float(per.min()), v
            break
    return who, dist, src


def merge(bank, work, out):
    """Per-state results -> one compressed ``<out>/<bank>.npz`` (state rows S, candidate rows R; people padded)."""
    name = Path(bank).name
    files = sorted(Path(work, name).glob('*.pkl'))
    rs = [pickle.load(open(f, 'rb')) for f in files]
    if not rs:
        raise SystemExit('no per-state results under %s' % Path(work, name))
    S = len(rs)
    Nmax = max(r['truth'].shape[1] for r in rs)
    Mmax = max(r['cv_paths'].shape[1] for r in rs)
    D = {}
    for k in STATE_KEYS:
        D['state_'+k] = np.stack([np.asarray(r[k]) for r in rs])
    D['state_bank'] = np.full(S, name)
    D['state_n_people'] = np.asarray([r['truth'].shape[1] for r in rs], np.int64)
    D['state_n_cv'] = np.asarray([r['cv_paths'].shape[1] for r in rs], np.int64)
    D['state_n_cands'] = np.asarray([len(r['labels']) for r in rs], np.int64)
    D['state_first_cand'] = np.concatenate([[0], np.cumsum(D['state_n_cands'])[:-1]]).astype(np.int64)
    D['truth'] = np.stack(_pad([r['truth'] for r in rs], 1, Nmax, np.nan))
    for k in PERSON_KEYS:
        dt = '<U32' if k == 'track_id' else np.asarray(rs[0][k]).dtype
        D['person_'+k] = np.stack(_pad([np.asarray(r[k]).astype(dt) for r in rs], 0, Nmax, _fill(np.dtype(dt))))
    D['cv_paths'] = np.stack(_pad([r['cv_paths'] for r in rs], 1, Mmax, np.nan))
    D['cv_ids'] = np.stack(_pad([np.asarray(r['cv_ids']).astype('<U32') for r in rs], 0, Mmax, ''))
    D['cv_true'] = np.stack(_pad([np.asarray(r['cv_true'], np.int64) for r in rs], 0, Mmax, -1))
    D['cand_state'] = np.concatenate([np.full(len(r['labels']), i, np.int64) for i, r in enumerate(rs)])
    D['cand_index'] = np.concatenate([np.arange(len(r['labels'])) for r in rs]).astype(np.int64)
    D['cand_is_default'] = np.concatenate([np.arange(len(r['labels'])) == r['default'] for r in rs])
    for k, k2 in CAND_KEYS:
        D['cand_'+k2] = np.concatenate([np.asarray(r[k]) for r in rs])
    D['cand_censored'] = (D['cand_real_outcome'] == 'running') & (D['cand_real_steps'] < W)
    for v in VARIANTS:
        for k in VAR_KEYS:
            D['%s_%s' % (v, k)] = np.concatenate([np.asarray(r['%s_%s' % (v, k)]) for r in rs])
        for k in VAR_PERSON_KEYS:
            D['%s_%s' % (v, k)] = np.concatenate(_pad([r['%s_%s' % (v, k)] for r in rs], 2, Nmax,
                                                      False if k == 'vis' else np.nan))
    who, dist, src = attribute_contacts(D)
    D['cand_contact_person'], D['cand_contact_person_dist'], D['cand_contact_person_src'] = who, dist, src
    sp = Path(work, name, 'settings.json')
    settings = json.loads(sp.read_text()) if sp.exists() else {}
    D['meta'] = np.asarray(json.dumps(dict(settings, bank=name, bank_dir=str(bank), states=S,
                                           candidates=int(len(D['cand_state'])), people_max=int(Nmax),
                                           cv_people_max=int(Mmax), version=VERSION, windows=list(WINDOWS),
                                           steps=W, step_s=0.048)))
    Path(out).mkdir(parents=True, exist_ok=True)
    target = Path(out, name+'.npz')
    tmp = Path(out, name+'.tmp.npz')
    np.savez_compressed(tmp, **D)
    os.replace(tmp, target)
    print('%s: %d states, %d candidates, people <= %d, CV people <= %d -> %s' % (name, S, len(D['cand_state']), Nmax,
                                                                              Mmax, target))
    return target


def load(path):
    """The merged dataset as a dict of arrays (``meta`` decoded)."""
    with np.load(path, allow_pickle=False) as z:
        D = {k: z[k] for k in z.files}
    D['meta'] = json.loads(str(D['meta']))
    return D


# ---------------------------------------------------------------------------------------------- statistics
def _auc(score, y):
    y = np.asarray(y, bool)
    s = np.asarray(score, np.float64)
    if not y.any() or y.all():
        return float('nan')
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum()-n1*(n1+1)/2.)/(n1*n0))


def _minw(a, w):
    """Min over the first ``w`` steps (NaN = no imagined pose there); +inf when nothing is finite."""
    x = np.asarray(a[:, :w], np.float64)
    return np.where(np.isfinite(x), x, np.inf).min(axis=1)


def _within_state_auc(score, y, sidx, keep):
    vals = []
    for s in np.unique(sidx):
        m = (sidx == s) & keep
        if y[m].any() and (~y[m]).any():
            vals.append(_auc(score[m], y[m]))
    return dict(mean=float(np.mean(vals)) if vals else float('nan'), states=len(vals))


def _q(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, p)) for p in (10, 50, 90)] if len(x) else []


def statistics(D):
    """Contacts per window, which person was hit (tracked at t or not), AUC of the imagined min clearance."""
    oc = D['cand_real_outcome']
    st = D['cand_real_steps']
    coll = oc == 'human_collision'
    sidx = D['cand_state']
    npeople = D['state_n_people']
    tracked = D['person_tracked']
    est_tr = D['person_est_tracked']
    someone = D['state_n_est'][sidx] > 0
    thru = np.char.endswith(D['cand_label'].astype(str), '_th')
    who = D['cand_contact_person']
    whod = D['cand_contact_person_dist']
    out = dict(states=int(len(D['state_file'])), candidates=int(len(oc)), episodes=int(len(set(D['state_key']))),
               outcomes={str(k): int(v) for k, v in zip(*np.unique(oc, return_counts=True))},
               censored_running=int(D['cand_censored'].sum()),
               states_someone_tracked=int((D['state_n_est'] > 0).sum()),
               people_per_state=float(npeople.mean()),
               people_tracked_frac=float(tracked.sum()/max(npeople.sum(), 1)),
               people_est_tracked_frac=float(est_tr.sum()/max(npeople.sum(), 1)),
               imagination_ms_per_candidate={v: float(np.mean(D['%s_ms' % v])) for v in VARIANTS},
               seconds_per_state=float(np.mean(D['state_seconds'])),
               imagined_errors={v: int((D['%s_ended' % v] == 'error').sum()) for v in VARIANTS})
    out['contacts'] = {}
    out['contact_person'] = {}
    for w in WINDOWS:
        y = coll & (st <= w)
        dflt = y & D['cand_is_default']
        safe_alt = sum(1 for s in set(sidx[dflt]) if (~y[sidx == s]).any())
        every = sum(1 for s in set(sidx[y]) if y[sidx == s].all())
        out['contacts'][w] = dict(candidates=int(y.sum()), rate=float(y.mean()), states_with_contact=int(len(set(sidx[y]))),
                                  states_every_candidate=int(every), at_step_1=int((y & (st == 1)).sum()),
                                  default_contacts=int(dflt.sum()), default_contact_with_safe_alternative=int(safe_alt))
        rows = np.flatnonzero(y)
        nobody = np.asarray([not tracked[sidx[r], :npeople[sidx[r]]].any() for r in rows], bool)
        everyone = np.asarray([tracked[sidx[r], :npeople[sidx[r]]].all() for r in rows], bool)
        att = rows[who[rows] >= 0]
        good = att[whod[att] < 1.0]
        unt = np.asarray([not tracked[sidx[r], who[r]] for r in att], bool)
        unt_good = np.asarray([not tracked[sidx[r], who[r]] for r in good], bool)
        unt_est = np.asarray([not est_tr[sidx[r], who[r]] for r in att], bool)
        mean = (lambda a: float(a.mean()) if len(a) else float('nan'))
        pairs = sorted({(int(sidx[r]), int(who[r])) for r in att})            # unique (state, hit person)
        unt_pairs = np.asarray([not tracked[s, p] for s, p in pairs], bool)
        # perceivable vs blind: an untracked hit person enters the imagined camera's view (along the TRUE-people
        # imagination b) at least 10 steps (0.48 s) before the contact step s (rows k <= s - 10; none when s < 10)
        seen = np.asarray([D['b_vis'][r, :max(0, int(st[r])-9), who[r]].any()
                           for r in att if not tracked[sidx[r], who[r]]], bool)
        out['contact_person'][w] = dict(
            contacts=int(len(rows)), in_states_nobody_tracked=mean(nobody), in_states_everyone_tracked=mean(everyone),
            attributed=int(len(att)), untracked_frac=mean(unt), attributed_within_1m=int(len(good)),
            untracked_frac_within_1m=mean(unt_good), untracked_frac_estimate_rule=mean(unt_est),
            state_person_pairs=len(pairs), untracked_frac_pairs=mean(unt_pairs),
            untracked_seen_10_steps_before=mean(seen), untracked_contacts=int(len(seen)))
    auc, rules, geo = {}, {}, {}
    for w in WINDOWS:
        y = coll & (st <= w)
        keep = ~((oc == 'running') & (st < w))          # censored: the episode's step limit cut the window short
        res = {}
        for v in VARIANTS:
            for ppl in ('true', 'cv'):
                for metric in ('rc', 'cd'):
                    name = '%s_%s_%s' % (v, metric, ppl)
                    sc = -np.minimum(_minw(D[name], w), 99.)
                    res[name] = dict(all=_auc(sc[keep], y[keep]),
                                     someone_tracked=_auc(sc[keep & someone], y[keep & someone]),
                                     nobody_tracked=_auc(sc[keep & ~someone], y[keep & ~someone]),
                                     non_thru=_auc(sc[keep & ~thru], y[keep & ~thru]),
                                     thru=_auc(sc[keep & thru], y[keep & thru]),
                                     within_state=_within_state_auc(sc, y, sidx, keep))
                    for thr in ((0.75, 0.6) if metric == 'cd' else (0.2, 0.1)):
                        p = _minw(D[name], w) < thr
                        tp = int((p & y & keep).sum()); fp = int((p & ~y & keep).sum()); fn = int((~p & y & keep).sum())
                        rules.setdefault(w, {})['%s<%.2f' % (name, thr)] = dict(precision=tp/max(tp+fp, 1),
                                                                                recall=tp/max(tp+fn, 1), flagged=tp+fp)
            geo.setdefault(w, {})[v] = dict(contact=_q(_minw(D['%s_cd_true' % v], w)[y & keep]),
                                            no_contact=_q(_minw(D['%s_cd_true' % v], w)[~y & keep]))
        res['n'] = dict(all=int(keep.sum()), contacts=int((y & keep).sum()), someone_tracked=int((keep & someone).sum()),
                        contacts_someone_tracked=int((y & keep & someone).sum()), thru=int((keep & thru).sum()),
                        contacts_thru=int((y & keep & thru).sum()))
        auc[w] = res
    out['auc'] = auc
    out['rules'] = rules
    out['min_centre_distance_to_true_people_p10_p50_p90'] = geo
    div = {}
    for w in WINDOWS:
        j = w-1
        ok = (D['a_n'] > j) & (D['b_n'] > j)
        d = np.hypot(D['a_poses'][ok, j, 0]-D['b_poses'][ok, j, 0], D['a_poses'][ok, j, 2]-D['b_poses'][ok, j, 2])
        div[w] = dict(n=int(ok.sum()), median=float(np.median(d)) if len(d) else float('nan'),
                      p90=float(np.percentile(d, 90)) if len(d) else float('nan'),
                      identical=float((d < 1e-6).mean()) if len(d) else float('nan'))
    out['a_vs_b_divergence'] = div
    return out


def _fmt(name, s):
    L = ['== %s: %d states (%d episodes), %d candidates; outcomes %s; censored by the step limit %d; states with '
         'someone tracked %d; people per state %.2f, tracked at t %.2f (OracleBank rule) / %.2f (estimate rule); '
         'imagination %.0f / %.0f ms per candidate (a / b), %.1f s per state; imagined errors %s' % (
             name, s['states'], s['episodes'], s['candidates'], s['outcomes'], s['censored_running'],
             s['states_someone_tracked'], s['people_per_state'], s['people_tracked_frac'], s['people_est_tracked_frac'],
             s['imagination_ms_per_candidate']['a'], s['imagination_ms_per_candidate']['b'], s['seconds_per_state'],
             s['imagined_errors'])]
    for w in WINDOWS:
        c, a = s['contacts'][w], s['contact_person'][w]
        L.append('  real contacts within %2d steps (%.2f s): %d candidates (%.1f%%; %d at step 1) in %d states (every '
                 'candidate in %d); default candidate %d (%d with a non-contact alternative)' % (
                     w, w*0.048, c['candidates'], 100*c['rate'], c['at_step_1'], c['states_with_contact'],
                     c['states_every_candidate'], c['default_contacts'], c['default_contact_with_safe_alternative']))
        L.append('     hit person untracked at t: %.2f of contacts (attributed %d of %d; %.2f of the %d within 1 m; '
                 'estimate rule %.2f), %.2f of %d unique (state, person) pairs | contacts in states with nobody '
                 'tracked %.2f, everyone tracked %.2f | untracked hit person in the imagined view >= 10 steps before '
                 'contact %.2f (of %d)' % (
                     a['untracked_frac'], a['attributed'], a['contacts'], a['untracked_frac_within_1m'],
                     a['attributed_within_1m'], a['untracked_frac_estimate_rule'], a['untracked_frac_pairs'],
                     a['state_person_pairs'], a['in_states_nobody_tracked'], a['in_states_everyone_tracked'],
                     a['untracked_seen_10_steps_before'], a['untracked_contacts']))
    for w in WINDOWS:
        r = s['auc'][w]
        L.append('  AUC of the imagined min clearance for a real contact within %d steps (%d candidates, %d contacts; '
                 'someone tracked %d / %d; thru %d / %d):' % (
                     w, r['n']['all'], r['n']['contacts'], r['n']['someone_tracked'], r['n']['contacts_someone_tracked'],
                     r['n']['thru'], r['n']['contacts_thru']))
        for k in ('a_rc_cv', 'a_cd_cv', 'a_rc_true', 'a_cd_true', 'b_rc_true', 'b_cd_true'):
            x = r[k]
            L.append('     %-10s all %.3f | someone tracked %.3f | nobody tracked %.3f | non-thru %.3f | thru %.3f | '
                     'within-state %.3f (%d mixed states)' % (k, x['all'], x['someone_tracked'], x['nobody_tracked'],
                                                              x['non_thru'], x['thru'], x['within_state']['mean'],
                                                              x['within_state']['states']))
    for w in WINDOWS:
        rr = s['rules'][w]
        L.append('  rules within %d steps (precision / recall): ' % w + '  '.join(
            '%s %.2f/%.2f' % (k, rr[k]['precision'], rr[k]['recall']) for k in
            ('a_cd_cv<0.75', 'a_cd_true<0.75', 'b_cd_true<0.75', 'b_cd_true<0.60', 'b_rc_true<0.20')))
    L.append('  imagined robot a vs b (m): ' + '  '.join('%s steps median %.2f / p90 %.2f (identical %.2f)' % (
        w, d['median'], d['p90'], d['identical']) for w, d in s['a_vs_b_divergence'].items()))
    return '\n'.join(L)


def stats(out_dir, names=('orca_train', 'orca_dev')):
    res = {}
    for name in names:
        p = Path(out_dir, name+'.npz')
        if p.exists():
            res[name] = statistics(load(p))
    Path(out_dir, 'stats.json').write_text(json.dumps(res, indent=1, default=float))
    text = '\n'.join(_fmt(name, s) for name, s in res.items())
    Path(out_dir, 'stats.txt').write_text(text+'\n')
    Path(out_dir, 'README.md').write_text(README+STATS_HEAD+'```\n'+text+'\n```\n')
    print(text)
    return res


STATS_HEAD = """
## Basic statistics (generated by `bankimag stats`; also `stats.txt`, `stats.json`)

Real contact = `cand_real_outcome == 'human_collision'` with `cand_real_steps <= w`.  The hit person is the APPROXIMATE
`cand_contact_person`; "tracked at t" is `person_tracked` (OracleBank rule) unless stated.  AUC: score = minus the
imagined min clearance within w steps (+inf -> 99 m), label = real contact within w; "all" pools every candidate (it
also ranks states against each other), "within-state" averages the AUC over states with both outcomes (the selection
problem).  "someone tracked" = `state_n_est > 0`; with nobody tracked the CV people are empty (AUC 0.5 by
construction).  `a_rc_cv` / `a_cd_cv` is what a legal Sandbox scores; `a_*_true` the same legal imagined paths scored
against everyone's true path (a perfect people model, no re-imagination); `b_*_true` the TRUE-people imagination.

"""


# ---------------------------------------------------------------------------------------------- README
README = """# Bank imagination dataset (`eventwm_graph/l3/bankimag.py`)

Imagined candidate trajectories on the banked decision states of the closed-loop Sandbox (`l3/statebank.py`), for
fitting a calibrated official-contact model and an untracked-people hazard.  ORCA planner, native dynamics, k10
generator (`orca-native-k10`).

* `orca_train.npz` -- `output/statebank/orca_train` (TRAIN episodes).  Fit / choose parameters on this file only.
* `orca_dev.npz` -- `output/statebank/orca_dev` (dev episodes).  Evaluation only.
* `stats.txt` / `stats.json` -- the basic statistics below (`python scripts/l3_run.py eventwm_graph.l3.bankimag stats`).

Load: `from eventwm_graph.l3.bankimag import load; D = load('output/agents/bankimag/orca_train.npz')` (plain
`np.load(path)` works too; `meta` is a JSON string with the settings).

## What was imagined

For every banked state and every candidate query (`sbeval.candidates_of(item)`), the Sandbox imagines 64 steps
(0.048 s each, 3.07 s): the candidate's interval (`cand_h` steps), then the default policy (keep the running query,
else the goal query) with a decision boundary every 16 steps and goal-query inheritance; imagined core contact
physics (`SANDBOX_PHYSICS=core`); the legal map frozen at t; the people perceived through the imagined camera
(`perceive.VisibleWorld`: 90 deg frustum, 10 m, legal-map line of sight, 2 s tracker memory; camera at the base centre
-- the default, `L3_CAM_MOUNT` off).  This is `shortprobe.probe_decision`'s call for its variants `camCVPc` / `camTPc`.
CPU-only Sandbox (`WorldModelSandbox(_NoLayer2(), SandboxConfig(materialize=False, route=True))`), `L3_INHERIT=1`,
and the generator's process-wide options set as online (`LegalState.BUMP_MEMORY` = 0.25 m; without them the bump
memory is off offline).

Two people sources (the prefix of every per-candidate array):

* `a_` LEGAL: the people tracked at t continuing at the tracker's constant velocity (`oraclebank.CVBank`:
  `control.human_estimates`, tracks at most 2 s old, extrapolated to t).  Nobody else exists.
* `b_` TRUE (privileged): every person's recorded future (`truth`, wrapped as `oraclebank.OracleBank` 'all').

Harness check (`bankimag check`): the same harness re-imagining the dev probe's ONLINE `camT` variant (camera model,
TRUE people, no physics) reproduces all 2830 of 2830 candidates of the 283 dev states exactly (imagined length, STOP,
end reason, min rectangle clearance within 16 / 32 / 64 steps to < 1e-4 m).  Without the generator's process-wide
options 7 of 220 `_th` candidates differed.

## Privileged vs legal

LEGAL (usable as model inputs at run time): `a_*` poses / `*_cv` distances, `cv_*`, `state_robot_pose`, `state_cam_*`,
`state_goal`, `state_ctg0`, `state_n_tracks`, `state_n_est`, `cand_label` / `kind` / `h`, and anything re-read from the
state file except `truth_people` and `probe`.  PRIVILEGED (labels / evaluation only): `truth`, `person_*`,
`*_true*` arrays, all `b_*` arrays, `cand_real_*`, `cand_contact_*`, `cand_censored`.

## Conventions

* World frame: planar (x, z); poses `[x, y, z, yaw]`; the robot's forward direction is (cos yaw, -sin yaw).
* Steps: 0.048 s.  `truth[s, j]` / `cv_paths[s, j]` = positions at t + j steps (j = 0..64).
  Imagined poses `X_poses[r, i]` = the robot after i+1 steps (i = 0..63); the per-step distance arrays use the same
  index i and the people at row i+1.  `X_vis[r, k]` uses the robot after k steps (k = 0..64; k = 0 is the real pose at
  t) and the people at row k (the camera test the imagined tracker applies at step k).
* NaN = no imagined pose at that step (the rollout ended: STOP, executor finished, error) or a padded person slot;
  +inf in the per-step minima = no person of that set exists.
* State rows S, candidate rows R.  Candidate r belongs to state `cand_state[r]`; a state's candidates are the rows
  `state_first_cand[s] .. state_first_cand[s] + state_n_cands[s] - 1`, in the bank's order.
* People are padded to the bank's maximum (`meta['people_max']`, `meta['cv_people_max']`); `state_n_people[s]`,
  `state_n_cv[s]` give the real counts.

## Arrays

State level (S rows):

| array | shape | meaning |
|---|---|---|
| `state_file` | [S] str | state pickle under `meta['bank_dir']` (re-read the legal context: `pickle.load(open(dir/file, 'rb'))['legal']` -- map, tracks, track_log, poses, bumps) |
| `state_bank` | [S] str | bank name (`orca_train` = TRAIN, `orca_dev` = dev) |
| `state_key`, `state_config` | [S] str | episode key, configuration |
| `state_decision`, `state_step`, `state_time` | [S] | decision index, env step of t, time at t (s) |
| `state_robot_pose` | [S, 4] | robot pose at t |
| `state_goal` | [S, 3] | public goal |
| `state_ctg0` | [S] | legal route cost-to-go at t (m; `compiler.LegalField`) |
| `state_cam_origin`, `state_cam_rotation` | [S, 3], [S, 3, 3] | real jaw camera at t (legal sensor row; looks down its local -Z) |
| `state_cam_real` | [S, 3] | its planar pose (x, z, yaw) |
| `state_cam_center` | [S, 3] | base-centre camera (x, z, yaw): the imagination's camera (default) |
| `state_cam_mount` | [S, 3] | `perceive`'s mount model (0.52 m ahead, 0.02 m right, -4.5 deg; `L3_CAM_MOUNT=1`) |
| `state_cam_imagined` | [S, 3] | the camera pose the imagination used at t (= `state_cam_center`) |
| `state_default` | [S] | index of the default candidate within the state |
| `state_n_tracks` | [S] | legal tracks at t (any age) |
| `state_n_est` | [S] | legal people estimates at t (tracks <= 2 s old) = CV people of `a` ("someone tracked" when > 0) |
| `state_level`, `state_bump_memory_m`, `state_n_bumps` | [S] | speed level; bump-memory radius used; legal bumps at t |
| `state_n_people`, `state_n_cv`, `state_n_cands`, `state_first_cand` | [S] | counts and candidate offsets |
| `state_seconds` | [S] | processing time of the state |
| `truth` | [S, 65, P, 2] float32 | PRIVILEGED: every person's true path, rows t + 0..64 steps |
| `person_tracked` | [S, P] bool | PRIVILEGED label: tracked at t by `oraclebank.OracleBank`'s rule (each valid legal track claims the nearest true person within 0.75 m of the track's last-seen position, first come) |
| `person_track_id` | [S, P] str | that track's id ('' if untracked) |
| `person_track_dist` | [S, P] | distance between the track's position and the person at t (m) |
| `person_track_age` | [S, P] | t - the track's last_seen_time (s) |
| `person_track_visible` | [S, P] bool | the track was detected at t (`visible_now`) |
| `person_est_tracked` | [S, P] bool | PRIVILEGED label: some CV person (legal estimate extrapolated to t) lies within 0.75 m (`sbeval.TrueTrackedBank`'s rule) -- i.e. the person exists in `a` |
| `person_cv_index` | [S, P] | index of that CV person (-1) |
| `cv_paths` | [S, 65, Q, 2] float32 | LEGAL: the CV people as `a` perceives them (row 0 = position at t + 1 ms, row j = t + j steps) |
| `cv_ids` | [S, Q] str | their legal track ids |
| `cv_true` | [S, Q] | PRIVILEGED: matched true person (-1) |

Candidate level (R rows):

| array | shape | meaning |
|---|---|---|
| `cand_state`, `cand_index`, `cand_is_default` | [R] | state row, index within the state, default candidate |
| `cand_label`, `cand_kind`, `cand_h` | [R] | query label / kind; interval steps imagined before the default continuation |
| `cand_real_outcome` | [R] str | PRIVILEGED exact branch outcome (interval + default policy, 64 steps): `human_collision`, `success`, `stop_fail`, `running` |
| `cand_real_steps` | [R] | steps the branch ran (terminal step, or 64) |
| `cand_contact_step` | [R] | the step (1..64 after t) at which the official human contact ended the branch, -1 if none.  The robot at contact ~ imagined row `contact_step - 1` |
| `cand_censored` | [R] bool | `running` with fewer than 64 steps: the episode's 500-step limit cut the window (no label after it) |
| `cand_real_clr` | [R, 3] | min body-rectangle clearance of the REAL branch's logged poses to the true people within 16 / 32 / 64 steps (+inf: nobody, or no pose logged -- a contact at step 1; the terminal step is never logged) |
| `cand_real_ctg` | [R, 3] | legal cost-to-go (decision-time map) of the real pose at 16 / 32 / 64 steps |
| `cand_real_cd` | [R, 3] | real min centre distance within 16 / 32 / 64 (NaN: not recorded by the bank's probe) |
| `cand_contact_person` | [R] | APPROXIMATE contacted person (real poses are not banked): nearest true person to the TRUE-people imagined robot (`b`, else `a`) at the poses after s-3 .. s+1 steps, s = `cand_contact_step`; -1 if no contact, or if both imaginations ended (goal / STOP) before s-3 (15 train / 10 dev contacts) |
| `cand_contact_person_dist`, `cand_contact_person_src` | [R] | its centre distance there (m) and the variant used |

Per people source X in {a, b} (R rows):

| array | shape | meaning |
|---|---|---|
| `X_poses` | [R, 64, 4] float32 | imagined robot pose after i+1 steps |
| `X_n` | [R] | imagined steps (< 64: the rollout ended -- STOP, finished, error) |
| `X_stop` | [R] bool | the candidate's interval issued STOP |
| `X_near_goal` | [R] bool | last imagined pose within 0.3 m of the goal |
| `X_ended`, `X_error` | [R] str | interval end reason (horizon / done / stop / failed / error) and error text |
| `X_ms` | [R] | imagination time (ms) |
| `X_cd_true_pp` | [R, 64, P] float32 | centre distance of the imagined robot to each TRUE person |
| `X_rc_true_pp` | [R, 64, P] float32 | body-rectangle clearance (`sandbox.imagine.rectangle_clearance`) to each TRUE person |
| `X_cd_true`, `X_rc_true` | [R, 64] float32 | their per-step minima over the true people |
| `X_cd_cv`, `X_rc_cv` | [R, 64] float32 | per-step min centre distance / rectangle clearance to the CV-predicted people (`cv_paths`) |
| `X_vis` | [R, 65, P] bool | TRUE person p inside the imagined camera's view (frustum, 10 m, legal-map line of sight) from the robot after k steps |
| `X_ctg` | [R, 3] | legal cost-to-go of the imagined pose at 16 / 32 / 64 steps |

Body rectangle (robot frame, forward x): `querygen.native.ROBOT_BOUNDS` = (-0.646, 0.584, -0.243, 0.238) m.  The official
human contact is a physics contact between the articulated robot and the human mesh, not a distance; at the last
logged step before a real contact the person's centre is a median 0.60 m from the robot's centre
(docs/l3/LAYER2_V2_DESIGN.md 1.2; here the attributed person is a median 0.59 / 0.61 m (train / dev) from the
TRUE-people imagined robot around the contact step).

## Notes for fitting

* Contacts at step 1 (47 train / 39 dev candidates) are already in progress at t: every candidate of such a state
  collides.  They inflate pooled AUCs; the within-state AUC (states with both outcomes) is the selection metric.
* `cand_label` ending in `_th` ("thru" queries: observed walls are soft) diverge most between imagination and reality.
* The `a` and `b` imagined robots are identical for most candidates (89% at 16 steps, 77% / 63-67% at 32 / 64 steps):
  the people a legal Sandbox perceives change the robot's own imagined path only near tracked people.
* Untracked people never enter `a`; their hazard can be scored on the `a` paths with `a_*_true_pp` and
  `person_tracked == False` as labels (TRAIN only), and `a_vis` tells whether the imagined camera would see them.
* No state is censored (every branch ran its 64 steps or ended).  `stop_fail` occurs twice (dev).

## Rebuild

    L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.bankimag run --bank output/statebank/orca_train \\
        --work output/agents/bank/bankimag_work --shard 0 --shards 2      # and --shard 1; same for orca_dev
    python scripts/l3_run.py eventwm_graph.l3.bankimag merge --bank output/statebank/orca_train \\
        --work output/agents/bank/bankimag_work --out output/agents/bankimag
    python scripts/l3_run.py eventwm_graph.l3.bankimag stats --out output/agents/bankimag
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'merge', 'stats', 'check', 'readme'])
    ap.add_argument('--bank', default='output/statebank/orca_train')
    ap.add_argument('--work', default='output/agents/bank/bankimag_work')
    ap.add_argument('--out', default='output/agents/bankimag')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    if a.mode == 'run':
        run(a.bank, a.work, a.shard, a.shards, a.limit)
    elif a.mode == 'merge':
        merge(a.bank, a.work, a.out)
        Path(a.out, 'README.md').write_text(README)
    elif a.mode == 'stats':
        stats(a.out)
    elif a.mode == 'check':
        check(a.bank, a.limit or 12, a.shard, a.shards)
    elif a.mode == 'readme':
        Path(a.out).mkdir(parents=True, exist_ok=True)
        Path(a.out, 'README.md').write_text(README)
        print('README ->', Path(a.out, 'README.md'))


if __name__ == '__main__':
    main()
