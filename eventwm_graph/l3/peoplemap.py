"""Map-aware multimodal forecaster of TRACKED people for the Sandbox (Layer 2 v2 people module; legal inputs,
privileged labels).  Default off: the Sandbox uses it only with ``L3_MW_BANK=fc1`` / ``fck`` (``env.py``).

Why: the legal Sandbox imagines tracked people at constant velocity (``oraclebank.CVBank``).  The large 1.5-3 s errors
are people turning or stopping at junctions, doors and waypoints, which the robot's LEGAL map shows.  This model reads
the person's legal track history AND a local patch of the robot's legal map around the person.

* data (``collect``): episodes of a run's spool (legal policies, train scenes) are replayed in querygen's exact branch
  simulator with the legal sensor (the robot's own recorded decisions; as ``peoplefast.py``); every ``--every`` steps,
  for every tracked person matched to a true person: the legal track history (``peopledata.track_histories``), the
  tracker velocity, the age of the last detection, a PATCH x PATCH crop of the robot's legal map (FREE / BLOCKED /
  UNKNOWN, 0.1 m cells, rotated into the frame of the tracker velocity -- ``peoplefc.canonical``) and the true
  displacement every 2 steps over the next 64 steps (the recorded tape: labels only);
* model (``fit``): a small CNN on the map patch (final: ``--cnn-arch small``, 6.4 m at 0.2 m cells) + an MLP on
  history / velocity / age -> K weighted modes, each a trajectory of knots every 4 steps (0 .. 64 steps = 0 .. 3.07 s)
  with per-knot isotropic spreads; trained by the negative log-likelihood of the recorded future (a mixture over whole
  trajectories: a mode is one coherent future) plus a small best-of-K regression term; scene-grouped validation on
  TRAIN;
* offline evaluation (``evaluate``): DEV tracked instances (the DEV state bank, or ``--data`` collected DEV replays)
  against constant velocity, Stage 1's mean world and the history-only MDN (``peoplefc``); ``calib`` fits on TRAIN
  the Gaussian scale mixtures that give the point forecasts (CV, Stage 1 mean) an NLL;
* deployment (``FCBank``): the bank interface of ``oraclebank.CVBank`` -- legal tracks only, CV for a track without a
  usable detection history; ``mode='fc1'``: one world (the weighted mean of the modes or the top mode, chosen on
  TRAIN and stored with the model); ``mode='fck'``: the ``worlds`` most probable joint mode combinations of the
  tracked people (product of the per-person mode weights), weighted -- the multi-world selector imagines each.

    L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.peoplemap collect --run output/runs/R10 \
        --out output/agents/people/data --shard i --shards 10 [--every 4 --until <unix time>]
    python scripts/l3_run.py eventwm_graph.l3.peoplemap fit --data output/agents/people/data,output/agents/people/data2 \
        --estimate-max-age 0.25 --save output/agents/people/fcmap/v1 --device cuda:1 \
        --cnn-arch small --crop 64 --map-drop 0.5 --drop 0.2 --wd 1e-3
    python scripts/l3_run.py eventwm_graph.l3.peoplemap calib
    python scripts/l3_run.py eventwm_graph.l3.peoplemap evaluate --model output/agents/people/fcmap/v1 \
        [--bank output/statebank/orca_dev | --data output/agents/people/dev_floor]
"""
import argparse
import glob
import json
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np

STEP_S = 0.048
PATCH = 96                    # stored crop (cells); the model reads a sub-crop (see Net)
PATCH_RES = 0.1               # metres per patch cell
FUT_STEPS = np.arange(0, 65, 2)              # stored truth: every 2 steps over 0 .. 64 steps
KNOT_STEPS = np.arange(0, 65, 4)             # model output knots: every 4 steps over 0 .. 64 (17 knots)
EVAL_STEPS = (16, 32, 62)                    # 0.77 / 1.5 / 3.0 s
EXCLUDE_SCENES = {'ZNanfzgCdm3', 'ZVPMj4YoZtK', 'ZVScmfktNQ1', 'ZwnLFNzxASM', 'ZxkSUELrWtQ', 'zR6kPe1PsyS',
                  'zUG6FL9TYeR', 'zWydhyFhvcj', 'zmZvNTCxMZE', 'zwzTbNq7xoW'}     # suites.json 'minival_scenes'
FRAME_VEL, FRAME_HIST, FRAME_NONE = 0, 1, 2


# ------------------------------------------------------------------------------------------------ legal features
def frame_of(hist, vel):
    """Heading frame of ``peoplefc.canonical``: (c, s) unit vector (world x, z) and its source (velocity / history
    direction / none -> world axes)."""
    sp = float(np.hypot(*vel))
    if sp > 0.1:
        return (float(vel[0])/sp, float(vel[1])/sp), FRAME_VEL
    valid = hist[:, 2] > 0
    if valid.sum() >= 2:
        d = hist[valid][-1, :2]-hist[valid][0, :2]
        n = float(np.hypot(*d))
        if n > 0.05:
            return (float(d[0])/n, float(d[1])/n), FRAME_HIST
    return (1., 0.), FRAME_NONE


def rot(c, s):
    """World (x, z) -> canonical (along heading, lateral) rotation matrix (as ``peoplefc.canonical``)."""
    return np.array([[c, s], [-s, c]], np.float64)


def map_patch(gmap, ref, c, s, size=PATCH, res=PATCH_RES):
    """[size, size] uint8 legal-map states (0 UNKNOWN / 1 FREE / 2 BLOCKED) around ``ref`` (world x, z) in the
    heading frame (c, s): row a = along the heading, column b = lateral (left of the heading = larger b), cell centres
    at (a - (size-1)/2) * res; nearest legal cell, outside the grid = UNKNOWN."""
    g = np.arange(size, dtype=np.float64)-(size-1)/2.
    cu = g[:, None]*res
    cv = g[None, :]*res
    x = ref[0]+c*cu-s*cv                                                     # world = ref + R^T (cu, cv)
    z = ref[1]+s*cu+c*cv
    i = np.floor(x/gmap.res).astype(np.int64)-gmap.ox
    j = np.floor(z/gmap.res).astype(np.int64)-gmap.oz
    st = gmap.state
    ok = (i >= 0) & (i < st.shape[0]) & (j >= 0) & (j < st.shape[1])
    out = np.zeros((size, size), np.uint8)
    out[ok] = st[i[ok], j[ok]]
    return out


def instances(legal, truth=None, horizons=tuple(FUT_STEPS)):
    """Tracked-person instances at a legal state: per legal track matched to a true person (``truth`` [>=65, N, 2]
    true positions from t; labels only), or (``truth`` None, deployment) every legal track (valid, <= 2 s old), where a
    track without a detection in the last 2 s has ``hist`` None (constant-velocity fallback)."""
    from .peopledata import track_histories
    if truth is not None:
        rows = track_histories(dict(legal=legal, truth_people=truth), horizons=horizons)
    else:
        rows = legal_histories(legal)
    for h in rows:
        if h.get('hist') is None:
            continue
        (c, s), kind = frame_of(h['hist'], h['vel'])
        h['frame'] = np.array([c, s], np.float32)
        h['frame_kind'] = kind
        h['patch'] = map_patch(legal.map, h['ref'], c, s)
    return rows


def instances_by_detection(legal, tape_xz, i0, tol=0.3):
    """Training instances with deployment-identical inputs (``legal_histories``); the LABEL person (privileged) is the
    true person nearest to the track's last detection at the detection's own step of the tape (within ``tol``), and
    the label is its true displacement from that detection every 2 steps over 0 .. 64 steps from t.  Unlike
    ``peopledata.track_histories`` (match of the constant-velocity estimate at t within 0.75 m), lost tracks of people
    who turned or stopped while unseen are kept (that match drops 15-36% of tracks lost 0.5-2 s; TRAIN state bank)."""
    times = np.asarray(legal.times, np.float64)
    tape_xz = np.asarray(tape_xz, np.float64)
    out = []
    for h in legal_histories(legal):
        if h.get('hist') is None:
            continue
        t_last = float(legal.time)-float(h['age'])
        j = int(np.argmin(np.abs(times-t_last)))
        if abs(times[j]-t_last) > 1e-4:
            continue
        i_last = int(i0)-(len(times)-1-j)
        if i_last < 0 or not tape_xz.shape[1]:
            continue
        P = tape_xz[min(i_last, len(tape_xz)-1)]
        d = np.hypot(P[:, 0]-h['ref'][0], P[:, 1]-h['ref'][1])
        if not np.isfinite(d).any() or np.nanmin(d) > tol:
            continue
        n = int(np.nanargmin(d))
        idx = np.minimum(int(i0)+FUT_STEPS, len(tape_xz)-1)
        h['fut'] = (tape_xz[idx, n]-np.asarray(h['ref'], np.float64)[None]).astype(np.float32)
        h['j'] = n
        h['dmatch'] = float(d[n])
        (c, s), kind = frame_of(h['hist'], h['vel'])
        h['frame'] = np.array([c, s], np.float32)
        h['frame_kind'] = kind
        h['patch'] = map_patch(legal.map, h['ref'], c, s)
        out.append(h)
    return out


def legal_histories(legal):
    """``peopledata.track_histories`` without the truth: every legal track (valid, <= 2 s old); ``hist`` None when the
    legal track log holds no detection of it in the last 2 s."""
    from eventwm_graph.querygen.control import human_estimates
    from .peopledata import HIST_S, HIST_N
    now = float(legal.time)
    det = {}
    for tm, trs in (legal.track_log or []):
        if now-float(tm) > HIST_S:
            continue
        for t in trs:
            if t.get('visible_now'):
                det.setdefault(str(t['track_id']), []).append((float(tm), float(t['position'][0]), float(t['position'][2])))
    out = []
    for e in human_estimates(legal.tracks, legal.time):
        hs = sorted(det.get(str(e.track_id), []))
        if not hs:
            out.append(dict(track_id=e.track_id, est=e, hist=None))
            continue
        t_last, x_last, z_last = hs[-1]
        H = np.zeros((HIST_N, 3), np.float32)
        for i, tq in enumerate(np.linspace(now-HIST_S, now, HIST_N)):
            cand = [h for h in hs if abs(h[0]-tq) <= 0.15]
            if cand:
                h = min(cand, key=lambda h: abs(h[0]-tq))
                H[i] = (h[1]-x_last, h[2]-z_last, 1.)
        out.append(dict(hist=H, vel=np.array([e.velocity[0], -e.velocity[1]], np.float32), age=float(now-t_last),
                        track_id=e.track_id, ref=(x_last, z_last), est=e))
    return out


# ------------------------------------------------------------------------------------------------ data collection
def _scene(f):
    return Path(f).name.split('__')[0].split('_')[2]


def collect(a):
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.querygen import native
    native._install()
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.sandbox.episode import generator
    from .actor import read_episode
    from .env import ConfigSpec
    from .cfonline import _query
    from . import sensorcache
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    skip = [x for x in str(a.skip or '').split(',') if x]          # episodes already collected elsewhere
    shared_tapes = Path('output/sandbox_cache/tapes').resolve()
    private_tapes = Path(a.out).resolve().parent/'tapes'
    files = sorted(Path(a.run, 'spool').glob('*.ep'))
    # shard by SCENE: a worker runs all episodes of its scenes back to back (habitat keeps the scene loaded)
    files = [f for f in files if zlib.crc32(_scene(f).encode()) % a.shards == a.shard and _scene(f) not in EXCLUDE_SCENES
             and all(g in f.name for g in str(a.grep or '').split(',') if g)]
    rng = np.random.default_rng(a.shard)
    scenes = sorted({_scene(f) for f in files})
    rng.shuffle(scenes)                                     # a deadline-cut collection still spans many scenes
    order = {sc: i for i, sc in enumerate(scenes)}
    files.sort(key=lambda f: (order[_scene(f)], f.name))
    for f in files:
        if a.until and time.time() > a.until:
            print(json.dumps(dict(stop='deadline')), flush=True)
            break
        target = out/(f.stem+'.npz')
        if target.exists() or any(Path(x, f.stem+'.npz').exists() for x in skip):
            continue
        t0 = time.perf_counter()
        try:
            d = read_episode(f)
            m = d['metrics']
            if 'decision_list' not in m:
                continue
            spec = EpisodeSpec(**d['spec'])
            # the human tape: the shared cache read-only; a miss is recorded into this collection's private cache
            tname = '%s_s%d_500.pkl' % (spec.key.replace('/', '_'), int(spec.seed))
            os.environ['QG_TAPE_CACHE'] = str(shared_tapes if (shared_tapes/tname).exists() else private_tapes)
            Path(os.environ['QG_TAPE_CACHE']).mkdir(parents=True, exist_ok=True)
            cfg = ConfigSpec.parse(d['config'])
            generator(cfg.gen, cfg.dynamics)
            rt = EpisodeRuntime(spec, cfg.planner, FAMILIES[cfg.dynamics], strict=False)
            rt.sim.check_static = False
            sensorcache.attach(rt)
            tape = np.asarray(rt.tape.positions, np.float64)
            rows = []
            st = rt.root
            try:
                for dec in m['decision_list']:
                    if st.terminal:
                        break
                    remaining = int(dec['horizon'])
                    first = True
                    while remaining > 0 and not st.terminal:
                        i0 = int(st.index)
                        if a.match == 'detection':
                            found = instances_by_detection(st.legal, tape[:, :, [0, 2]], i0)
                        else:
                            idx = np.minimum(np.arange(i0, i0+65), len(tape)-1)
                            found = instances(st.legal, tape[idx][:, :, [0, 2]].astype(np.float32))
                        for h in found:
                            rp = np.asarray(st.legal.pose, np.float64)
                            rows.append(dict(hist=h['hist'], vel=h['vel'], age=h['age'], fut=h['fut'], frame=h['frame'],
                                             kind=h['frame_kind'], ref=np.asarray(h['ref'], np.float32),
                                             patch=h['patch'], robot=np.array([rp[0], rp[2], rp[3]], np.float32),
                                             t=float(st.legal.time), step=i0, valid=len(tape)-1-i0,
                                             est=np.array([h['est'].position[0], -h['est'].position[1]], np.float32),
                                             track=str(h['track_id']), j=int(h['j']), dm=float(h.get('dmatch', -1.))))
                        n = min(a.every, remaining)
                        cont = (not first) or bool(dec['cont'])
                        if cont and (st.execution is None or st.execution.status != 'running'):
                            break
                        st = rt.run(st, None if cont else _query(dec), horizon=n, cont=cont, geodesic=False)
                        remaining -= n
                        first = False
                        if st.execution is not None and st.execution.status != 'running':
                            break
            finally:
                try:
                    sensorcache.save(rt)
                except Exception:
                    pass
                rt.close()
            arr = {}
            if rows:
                for k in ('hist', 'vel', 'fut', 'frame', 'ref', 'patch', 'robot', 'est'):
                    arr[k] = np.stack([r[k] for r in rows])
                for k, dt in (('age', np.float32), ('kind', np.int8), ('t', np.float32), ('step', np.int32),
                              ('valid', np.int32), ('j', np.int16), ('dm', np.float32)):
                    arr[k] = np.asarray([r[k] for r in rows], dt)
                arr['track'] = np.asarray([r['track'] for r in rows])
            np.savez_compressed(out/(f.stem+'.tmp.npz'), n=len(rows), config=d['config'], scene=spec.scene,
                                key=spec.key, match=a.match, **arr)
            os.replace(out/(f.stem+'.tmp.npz'), target)
            print(json.dumps(dict(episode=f.name, instances=len(rows), seconds=round(time.perf_counter()-t0, 1))),
                  flush=True)
        except Exception as exc:
            import traceback
            print(json.dumps(dict(episode=f.name, error=repr(exc), tb=traceback.format_exc()[-600:])), flush=True)


# ------------------------------------------------------------------------------------------------ dataset
NK = len(KNOT_STEPS)
KNOT_T = KNOT_STEPS*STEP_S


def canonical_arrays(hist, vel, frame):
    """History (dx, dz) [N, 10, 2], validity [N, 10] and velocity [N, 2] rotated into the heading frame."""
    hist = np.asarray(hist, np.float32).reshape(-1, 10, 3)
    vel = np.asarray(vel, np.float32).reshape(-1, 2)
    c, s = np.asarray(frame, np.float32)[:, 0], np.asarray(frame, np.float32)[:, 1]
    hu = c[:, None]*hist[:, :, 0]+s[:, None]*hist[:, :, 1]
    hv = -s[:, None]*hist[:, :, 0]+c[:, None]*hist[:, :, 1]
    vu = c*vel[:, 0]+s*vel[:, 1]
    vv = -s*vel[:, 0]+c*vel[:, 1]
    return np.stack([hu, hv], -1).astype(np.float32), hist[:, :, 2].astype(np.float32), np.stack([vu, vv], -1).astype(np.float32)


def to_canonical(P, frame):
    """World displacements [N, ..., 2] -> heading frame."""
    c = np.asarray(frame, np.float32)[:, 0].reshape((-1,)+(1,)*(P.ndim-2))
    s = np.asarray(frame, np.float32)[:, 1].reshape((-1,)+(1,)*(P.ndim-2))
    return np.stack([c*P[..., 0]+s*P[..., 1], -s*P[..., 0]+c*P[..., 1]], -1)


def to_world(P, frame):
    """Heading-frame displacements [N, ..., 2] -> world (x, z)."""
    c = np.asarray(frame, np.float64)[:, 0].reshape((-1,)+(1,)*(P.ndim-2))
    s = np.asarray(frame, np.float64)[:, 1].reshape((-1,)+(1,)*(P.ndim-2))
    return np.stack([c*P[..., 0]-s*P[..., 1], s*P[..., 0]+c*P[..., 1]], -1)


def maneuvers(F, near=None):
    """Truth categories from the true positions every 2 steps from t (F [N, 33, 2], any origin; ``near`` an extra
    precomputed mask, e.g. within 2.5 m of the robot).  Falcon people walk short legs with brief pauses, so motion is
    judged on 8-step (0.38 s) windows: 'moving' = faster than 0.5 m/s over the first window; 'turn' = moving and some
    later window faster than 0.5 m/s heads more than 45 deg off the current heading (pauses allowed); 'stop' = moving
    but less than 0.5 m covered between 1.5 and 3 s (a sustained stop); 'start' = slower than 0.2 m/s now and more than
    1 m away at 3 s; 'straight' = moving, no turn, no stop; 'still' = slower than 0.2 m/s now and not a start."""
    F = np.asarray(F, np.float64)
    v = (F[:, 4:]-F[:, :-4])/(8*STEP_S)                                      # [N, 29, 2] 8-step windows
    sp = np.hypot(v[..., 0], v[..., 1])
    v0, s0 = v[:, 0], sp[:, 0]
    moving = s0 >= 0.5
    cosang = (v[..., 0]*v0[:, None, 0]+v[..., 1]*v0[:, None, 1])/np.maximum(sp*s0[:, None], 1e-9)
    turn = moving & ((cosang < math.cos(math.radians(45))) & (sp > 0.5)).any(1)
    stop = moving & (np.hypot(*(F[:, 31]-F[:, 16]).T) < 0.5)
    start = (s0 < 0.2) & (np.hypot(*(F[:, 31]-F[:, 0]).T) > 1.0)
    out = dict(all=np.ones(len(F), bool), straight=moving & ~stop & ~turn, turn=turn, stop=stop, start=start,
               still=(s0 < 0.2) & ~start, maneuver=turn | stop | start, turn_stop=turn | stop)
    if near is not None:
        out['near'] = np.asarray(near, bool)
        out['near_maneuver'] = out['near'] & out['maneuver']
    return out


def near_robot(F, ref, robot, radius=2.5, steps=32):
    """True person within ``radius`` of the robot's position at t at some step <= ``steps`` (F relative to ref)."""
    A = np.asarray(ref, np.float64)[:, None]+np.asarray(F, np.float64)[:, :steps//2+1]
    d = np.linalg.norm(A-np.asarray(robot, np.float64)[:, None, :2], axis=-1)
    return d.min(1) < radius


def load_data(dirs, min_valid=64, estimate_max_age=None):
    """Collected instances (``collect``) with a full 64-step true future: canonical model inputs and targets.
    ``estimate_max_age`` (s): from files labelled by the older estimate match (``--match estimate``, or files written
    before the option existed) keep only tracks detected at most this long ago -- there the estimate match and the
    detection match pick the same person; for older tracks the estimate match drops people who turned or stopped
    while unseen (label bias)."""
    keys = ('hist', 'vel', 'fut', 'frame', 'patch', 'age', 'kind', 'ref', 'robot')
    acc = {k: [] for k in keys}
    scene, epi = [], []
    files = []
    for d in str(dirs).split(','):
        files += sorted(glob.glob(os.path.join(d, '*.npz')))
    for n_f, f in enumerate(files):
        if f.endswith('.tmp.npz'):
            continue
        try:
            z = np.load(f)
            if int(z['n']) == 0:
                continue
            ok = (z['valid'] >= min_valid) & np.isfinite(z['fut']).all((1, 2))
            if estimate_max_age is not None and str(z['match'] if 'match' in z.files else 'estimate') == 'estimate':
                ok &= z['age'] <= float(estimate_max_age)+1e-6
            if not ok.any():
                continue
            for k in keys:
                acc[k].append(z[k][ok])
            scene += [str(z['scene'])]*int(ok.sum())
            epi += [n_f]*int(ok.sum())
        except Exception as exc:                 # a file being written
            print('skip %s: %r' % (f, exc), flush=True)
    D = {k: np.concatenate(v) for k, v in acc.items()}
    D['hc'], D['hvalid'], D['vc'] = canonical_arrays(D['hist'], D['vel'], D['frame'])
    D['futc'] = to_canonical(D['fut'], D['frame']).astype(np.float32)          # [N, 33, 2]
    D['scene'] = np.asarray(scene)
    D['episode'] = np.asarray(epi)
    return D


# ------------------------------------------------------------------------------------------------ model
def _torch():
    import torch
    return torch


RAYS, RAY_STEPS = 16, 45                     # map descriptor: 16 rays from the person, samples every 0.1 m to 4.5 m
_RAY_IDX = {}


def ray_feats(patch):
    """Map descriptor in the heading frame (torch): along 16 rays from the person (every 22.5 deg, ray 0 = heading),
    the distance to the first BLOCKED and to the first UNKNOWN legal cell (/ 4.6 m; 4.6 m = none within 4.5 m) of the
    patch (person at its centre) -> [B, 32].  Computed from the (augmented) patch, so mirror / rotation stay exact."""
    torch = _torch()
    P = patch.shape[-1]
    key = (str(patch.device), P)
    if key not in _RAY_IDX:
        th = np.arange(RAYS)*2*np.pi/RAYS
        r = (np.arange(RAY_STEPS)+1)*PATCH_RES
        a = np.floor(r[None]*np.cos(th)[:, None]/PATCH_RES+P/2.).astype(np.int64).clip(0, P-1)
        b = np.floor(r[None]*np.sin(th)[:, None]/PATCH_RES+P/2.).astype(np.int64).clip(0, P-1)
        _RAY_IDX[key] = (torch.as_tensor(a, device=patch.device), torch.as_tensor(b, device=patch.device))
    a, b = _RAY_IDX[key]
    v = patch[:, a, b]                                                           # [B, RAYS, RAY_STEPS]

    def first(m):
        idx = torch.where(m.any(-1), m.float().argmax(-1), torch.full(m.shape[:-1], RAY_STEPS, device=m.device))
        return (idx.float()+1)*PATCH_RES/4.6
    return torch.cat([first(v == 2), first(v == 0)], -1)


def make_net(n_in=38, K=6, cnn=True, width=128, crop=96, rays=False, drop=0., cnn_arch='v1', map_feat=64,
             map_drop=0.):
    """``cnn_arch``: 'v1' -- 4 strided convolutions at 0.1 m, flatten -> ``width`` features (~0.4 M parameters at
    crop 96; it memorises the training scenes); 'small' -- the one-hot averaged to 0.2 m cells, three max-pooled 3x3
    convolutions (16 / 32 / 32 channels), flatten -> ``map_feat`` features with dropout ``map_drop`` (~50 k parameters
    at crop 64)."""
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.K, self.cnn_on, self.crop, self.rays_on = K, bool(cnn), int(crop), bool(rays)
            n0 = n_in+(2*RAYS if self.rays_on else 0)
            self.mlp = nn.Sequential(nn.Linear(n0, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
            feat = width
            if self.cnn_on and cnn_arch == 'small':
                side = self.crop//2
                for _ in range(3):
                    side //= 2
                self.cnn = nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(), nn.Linear(32*side*side, int(map_feat)), nn.ReLU(), nn.Dropout(float(map_drop)))
                feat += int(map_feat)
            elif self.cnn_on:
                side = self.crop
                for _ in range(4):
                    side = (side+1)//2
                self.cnn = nn.Sequential(
                    nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Flatten(), nn.Linear(64*side*side, width), nn.ReLU())
                feat += width
            self.drop = nn.Dropout(float(drop))
            self.head = nn.Sequential(nn.Linear(feat, 2*width), nn.ReLU(), nn.Linear(2*width, K*(1+NK*3)))

        def forward(self, patch, x, cv):
            """patch uint8 [B, P, P] (crop applied here), x [B, n_in], cv [B, NK, 2] (constant-velocity displacement
            in the heading frame) -> mode logits [B, K], means [B, K, NK, 2], log sigmas [B, K, NK]."""
            if self.rays_on:
                x = torch.cat([x, ray_feats(patch)], 1)
            h = self.mlp(x)
            if self.cnn_on:
                P = patch.shape[-1]
                o = (P-self.crop)//2
                p = patch[:, o:o+self.crop, o:o+self.crop].long()
                oh = torch.nn.functional.one_hot(p, 3).permute(0, 3, 1, 2).float()
                h = torch.cat([h, self.cnn(oh)], 1)
            out = self.head(self.drop(h))
            K = self.K
            logit = out[:, :K]
            mean = out[:, K:K+K*NK*2].reshape(-1, K, NK, 2)+cv[:, None]
            logs = out[:, K+K*NK*2:].reshape(-1, K, NK).clamp(math.log(0.03), math.log(6.))
            return logit, mean, logs
    return Net()


def track_inputs(hc, hvalid, vc, age, kind):
    """Track-branch input vector (torch, heading frame): history (u, v) x 10, validity x 10, velocity (u, v), speed,
    age, frame-kind one-hot (3), valid share -> [B, 38]."""
    torch = _torch()
    sp = torch.linalg.norm(vc, dim=-1, keepdim=True)
    oh = torch.nn.functional.one_hot(kind.long(), 3).float()
    return torch.cat([hc[..., 0], hc[..., 1], hvalid, vc, sp, age[:, None], oh, hvalid.mean(1, keepdim=True)], 1)


def cv_knots(vc, age):
    """Constant-velocity displacement of the last detection at every knot (heading frame): v * (age + t)."""
    torch = _torch()
    t = torch.as_tensor(KNOT_T, dtype=torch.float32, device=vc.device)
    return vc[:, None, :]*(age[:, None, None]+t[None, :, None])


def gmm_terms(logit, mean, logs, Y):
    """Per-sample trajectory-mixture NLL (sum over knots of 2-D isotropic Gaussians) and best-of-K ADE."""
    torch = _torch()
    logw = torch.log_softmax(logit, -1)
    d2 = ((Y[:, None]-mean)**2).sum(-1)                                          # [B, K, NK]
    ll = -d2/(2*torch.exp(2*logs))-2*logs-math.log(2*math.pi)
    nll = -torch.logsumexp(logw+ll.sum(-1), -1)
    ade = torch.sqrt(d2+1e-6).mean(-1)                                           # [B, K]
    return nll, ade


def augment(patch, hc, vc, Y, kind, gen):
    """Exact symmetries in the heading frame: a lateral mirror (p = 0.5); for frames without a heading (no velocity,
    no history direction), a random multiple of 90 deg."""
    torch = _torch()
    B = len(patch)
    flip = torch.rand(B, generator=gen, device=patch.device) < 0.5
    if flip.any():
        patch = torch.where(flip[:, None, None], patch.flip(-1), patch)
        sgn = torch.where(flip, -1., 1.)
        hc = torch.stack([hc[..., 0], hc[..., 1]*sgn[:, None]], -1)
        vc = torch.stack([vc[..., 0], vc[..., 1]*sgn], -1)
        Y = torch.stack([Y[..., 0], Y[..., 1]*sgn[:, None]], -1)
    none = kind == FRAME_NONE
    if none.any():
        k = torch.randint(0, 4, (B,), generator=gen, device=patch.device)
        k = torch.where(none, k, torch.zeros_like(k))
        for r in (1, 2, 3):
            m = k == r
            if not m.any():
                continue
            patch[m] = torch.rot90(patch[m], r, dims=(-2, -1))
            for _ in range(r):          # (u, v) -> (-v, u): the coordinate rotation of np.rot90 on (a, b) cells
                hc[m] = torch.stack([-hc[m][..., 1], hc[m][..., 0]], -1)
                vc[m] = torch.stack([-vc[m][..., 1], vc[m][..., 0]], -1)
                Y[m] = torch.stack([-Y[m][..., 1], Y[m][..., 0]], -1)
    return patch, hc, vc, Y


def fit(D, tr, va, device='cuda:0', epochs=25, lr=2e-3, batch=256, K=6, cnn=True, wta=0.5, seed=0, width=128,
        crop=96, rays=False, drop=0., wd=1e-4, cnn_arch='v1', map_feat=64, map_drop=0., log=print):
    torch = _torch()
    torch.manual_seed(seed)
    dev = torch.device(device)
    T = {k: torch.as_tensor(D[k][np.concatenate([tr, va])]) for k in ('patch', 'hc', 'hvalid', 'vc', 'age', 'kind')}
    T['Y'] = torch.as_tensor(D['futc'][np.concatenate([tr, va])][:, KNOT_STEPS//2])
    T = {k: v.to(dev) for k, v in T.items()}
    ntr = len(tr)
    net = make_net(K=K, cnn=cnn, width=width, crop=crop, rays=rays, drop=drop, cnn_arch=cnn_arch, map_feat=map_feat,
                   map_drop=map_drop).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs*((ntr+batch-1)//batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    best, best_state, hist = float('inf'), None, []

    def run(idx, train):
        p, hc, hv, vc, age, kind, Y = (T[k][idx] for k in ('patch', 'hc', 'hvalid', 'vc', 'age', 'kind', 'Y'))
        if train:
            p, hc, vc, Y = augment(p.clone(), hc.clone(), vc.clone(), Y.clone(), kind, gen)
        x = track_inputs(hc, hv, vc, age, kind)
        logit, mean, logs = net(p, x, cv_knots(vc, age))
        nll, ade = gmm_terms(logit, mean, logs, Y)
        return nll, ade
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(ntr, generator=gen, device=dev)
        tot, nb = 0., 0
        for b in range(0, ntr, batch):
            idx = perm[b:b+batch]
            nll, ade = run(idx, True)
            loss = nll.mean()/NK+wta*ade.min(-1).values.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            opt.step(); sched.step()
            tot += float(loss); nb += 1
        net.eval()
        with torch.no_grad():
            vn, va_ade = [], []
            for b in range(ntr, ntr+len(va), 2048):
                idx = torch.arange(b, min(b+2048, ntr+len(va)), device=dev)
                nll, ade = run(idx, False)
                vn.append(nll); va_ade.append(ade.min(-1).values)
            vn = float(torch.cat(vn).mean())/NK
            vb = float(torch.cat(va_ade).mean())
        hist.append(dict(epoch=ep, train=tot/max(nb, 1), val_nll=vn, val_minade=vb, s=round(time.time()-t0, 1)))
        log(json.dumps(hist[-1]))
        if vn < best:
            best = vn
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net = net.cpu().eval()
    return net, dict(K=K, cnn=bool(cnn), width=int(width), crop=int(crop), n_in=38, rays=bool(rays), drop=float(drop),
                     cnn_arch=str(cnn_arch), map_feat=int(map_feat), map_drop=float(map_drop)), hist


def predict(net, patch, hc, hvalid, vc, age, kind, device='cpu', batch=4096):
    """-> weights [N, K], means [N, K, NK, 2] (heading frame, relative to the last detection), sigmas [N, K, NK]."""
    torch = _torch()
    W, M, S = [], [], []
    net = net.to(device).eval()
    with torch.no_grad():
        for b in range(0, len(patch), batch):
            sl = slice(b, b+batch)
            t = [torch.as_tensor(np.asarray(a[sl])).to(device) for a in (patch, hc, hvalid, vc, age, kind)]
            x = track_inputs(t[1], t[2], t[3], t[4], t[5])
            logit, mean, logs = net(t[0], x, cv_knots(t[3], t[4]))
            W.append(torch.softmax(logit, -1).cpu().numpy()); M.append(mean.cpu().numpy())
            S.append(torch.exp(logs).cpu().numpy())
    if not W:
        return np.zeros((0, net.K)), np.zeros((0, net.K, NK, 2)), np.zeros((0, net.K, NK))
    return np.concatenate(W), np.concatenate(M), np.concatenate(S)


def at_steps(knots, steps):
    """Piecewise-linear value of knot arrays [..., NK, ...] (axis -2 for means [.., NK, 2]) at step offsets."""
    out = []
    for h in steps:
        i = min(int(h)//4, NK-2)
        f = (h-KNOT_STEPS[i])/4.
        out.append(knots[..., i, :]*(1-f)+knots[..., i+1, :]*f)
    return np.stack(out, -2)


def at_steps_scalar(knots, steps):
    out = []
    for h in steps:
        i = min(int(h)//4, NK-2)
        f = (h-KNOT_STEPS[i])/4.
        out.append(knots[..., i]*(1-f)+knots[..., i+1]*f)
    return np.stack(out, -1)


def mixture_nll(w, mean, sig, y):
    """Per-horizon marginal NLL of a 2-D isotropic mixture: w [N, K], mean [N, K, 2], sig [N, K], y [N, 2]."""
    d2 = ((y[:, None]-mean)**2).sum(-1)
    lp = np.log(np.maximum(w, 1e-12))-d2/(2*sig**2)-2*np.log(sig)-math.log(2*math.pi)
    m = lp.max(1, keepdims=True)
    return -(m[:, 0]+np.log(np.exp(lp-m).sum(1)))


def point_errors(w, mean, y):
    """Errors [N] of the weighted mean, the top mode and the best mode: mean [N, K, 2], y [N, 2]."""
    wm = (w[:, :, None]*mean).sum(1)
    top = mean[np.arange(len(w)), np.argmax(w, 1)]
    e_all = np.linalg.norm(mean-y[:, None], axis=-1)
    return np.linalg.norm(wm-y, axis=-1), np.linalg.norm(top-y, axis=-1), e_all.min(1)


def _q(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return dict(n=int(len(v)), median=round(float(np.median(v)), 3) if len(v) else None,
                p90=round(float(np.percentile(v, 90)), 3) if len(v) else None, mean=round(float(v.mean()), 3) if len(v) else None)


def scene_split(scenes, frac=0.12, seed=0):
    u = np.unique(scenes)
    rng = np.random.default_rng(seed)
    va_sc = set(rng.choice(u, max(1, int(round(frac*len(u)))), replace=False).tolist())
    va = np.flatnonzero(np.isin(scenes, list(va_sc)))
    tr = np.flatnonzero(~np.isin(scenes, list(va_sc)))
    return tr, va, sorted(va_sc)


def report_split(net, D, idx, label, log=print):
    """TRAIN-validation metrics: per truth category x horizon (0.77 / 1.5 / 3 s): weighted mean, top mode,
    best-of-K, mixture NLL, constant velocity."""
    w, mean, sig = predict(net, D['patch'][idx], D['hc'][idx], D['hvalid'][idx], D['vc'][idx], D['age'][idx],
                           D['kind'][idx])
    Y = D['futc'][idx]
    cats = maneuvers(D['futc'][idx], near_robot(D['fut'][idx], D['ref'][idx], D['robot'][idx]))
    out = {}
    for hi, h in enumerate(EVAL_STEPS):
        mh = at_steps(mean, [h])[..., 0, :]                                         # [N, K, 2]
        sh = at_steps_scalar(sig, [h])[..., 0]
        y = Y[:, h//2]
        e_w, e_t, e_b = point_errors(w, mh, y)
        nll = mixture_nll(w, mh, sh, y)
        cvp = D['vc'][idx]*(D['age'][idx][:, None]+h*STEP_S)
        e_cv = np.linalg.norm(cvp-y, axis=-1)
        for c, m in cats.items():
            out.setdefault(c, {})['%.2fs' % (h*STEP_S)] = dict(
                mean=_q(e_w[m]), top=_q(e_t[m]), best=_q(e_b[m]), nll=round(float(nll[m].mean()), 3) if m.any() else None,
                cv=_q(e_cv[m]))
    log('%s: %s' % (label, json.dumps({c: {h: {k: (v['median'], v['p90']) if isinstance(v, dict) else v
                                                 for k, v in d.items()} for h, d in r.items()} for c, r in out.items()
                                       if c in ('all', 'maneuver', 'near')})))
    return out


def fit_main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='output/agents/people/data')
    ap.add_argument('--save', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--wta', type=float, default=0.5)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--no-cnn', action='store_true', help='no CNN on the map patch')
    ap.add_argument('--rays', action='store_true', help='ray descriptor of the map patch (ray_feats)')
    ap.add_argument('--drop', type=float, default=0.)
    ap.add_argument('--cnn-arch', default='v1', choices=['v1', 'small'], help='map encoder (make_net)')
    ap.add_argument('--map-feat', type=int, default=64, help="'small' map encoder: output features")
    ap.add_argument('--map-drop', type=float, default=0., help="'small' map encoder: dropout on its features")
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit-train', type=int, default=0, help='learning curve: at most this many training instances')
    ap.add_argument('--estimate-max-age', type=float, default=None,
                    help='estimate-matched files: keep tracks detected at most this long ago (load_data)')
    a = ap.parse_args(argv)
    torch = _torch()
    out = Path(a.save); out.mkdir(parents=True, exist_ok=True)
    logf = open(out/'fit.log', 'a')

    def log(s):
        print(s, flush=True)
        logf.write(s+'\n'); logf.flush()
    t0 = time.time()
    D = load_data(a.data, estimate_max_age=a.estimate_max_age)
    tr, va, va_sc = scene_split(D['scene'])
    if a.limit_train and len(tr) > a.limit_train:
        tr = np.sort(np.random.default_rng(1).choice(tr, a.limit_train, replace=False))
    log('instances %d (train %d / val %d; scenes %d / %d), load %.0f s' % (
        len(D['scene']), len(tr), len(va), len(np.unique(D['scene'][tr])), len(va_sc), time.time()-t0))
    net, arch, hist = fit(D, tr, va, device=a.device, epochs=a.epochs, lr=a.lr, K=a.K, cnn=not a.no_cnn, wta=a.wta,
                          seed=a.seed, width=a.width, crop=a.crop, rays=a.rays, drop=a.drop, wd=a.wd,
                          cnn_arch=a.cnn_arch, map_feat=a.map_feat, map_drop=a.map_drop, log=log)
    rep = report_split(net, D, va, 'TRAIN-val', log=log)
    # the single world for L3_MW_BANK=fc1, chosen on TRAIN-val: weighted mean or top mode, by the mean over the three
    # horizons of (median + p90) error over all instances
    score = {k: float(np.mean([rep['all'][h][k]['median']+rep['all'][h][k]['p90'] for h in rep['all']]))
             for k in ('mean', 'top')}
    one = min(score, key=score.get)
    log('fc1 world (TRAIN-val): %s %s' % (one, json.dumps(score)))
    torch.save(dict(state=net.state_dict(), arch=arch), out/'model.pt')
    meta = dict(arch=arch, K=a.K, knots=KNOT_STEPS.tolist(), patch=PATCH, patch_res=PATCH_RES, data=a.data,
                estimate_max_age=a.estimate_max_age,
                n_train=int(len(tr)), n_val=int(len(va)), val_scenes=va_sc, fc1=one, fc1_score=score, epochs=a.epochs,
                lr=a.lr, wta=a.wta, seed=a.seed, history=hist, train_val=rep)
    (out/'meta.json').write_text(json.dumps(meta, indent=1))
    log('saved %s' % out)


# ------------------------------------------------------------------------------------------------ deployment
DEFAULT_MODEL = 'output/agents/people/fcmap/v1'
_MODELS = {}


class Forecaster:
    """A saved model (``fit --save``) on the CPU."""

    def __init__(self, path):
        torch = _torch()
        ck = torch.load(Path(path, 'model.pt'), map_location='cpu')
        a = ck['arch']
        self.net = make_net(n_in=a.get('n_in', 38), K=a['K'], cnn=a['cnn'], width=a['width'], crop=a.get('crop', 96),
                            rays=a.get('rays', False), drop=a.get('drop', 0.), cnn_arch=a.get('cnn_arch', 'v1'),
                            map_feat=a.get('map_feat', 64), map_drop=a.get('map_drop', 0.))
        self.net.load_state_dict(ck['state'])
        self.net.eval()
        self.meta = json.loads(Path(path, 'meta.json').read_text())
        self.path = str(path)
        self.K = int(a['K'])

    def modes(self, rows):
        """rows: ``instances(legal)`` with a history -> weights [M, K], world knots [M, K, NK, 2] (absolute x, z),
        sigmas [M, K, NK]."""
        frame = np.stack([r['frame'] for r in rows])
        hc, hv, vc = canonical_arrays(np.stack([r['hist'] for r in rows]), np.stack([r['vel'] for r in rows]), frame)
        age = np.asarray([r['age'] for r in rows], np.float32)
        kind = np.asarray([r['frame_kind'] for r in rows], np.int64)
        patch = np.stack([r['patch'] for r in rows])
        w, mean, sig = predict(self.net, patch, hc, hv, vc, age, kind)
        ref = np.asarray([r['ref'] for r in rows], np.float64)
        world = to_world(mean.astype(np.float64), frame)+ref[:, None, None, :]
        return w, world, sig


def load_forecaster(path=None):
    path = str(path or os.environ.get('L3_FC_MODEL', DEFAULT_MODEL))
    if path not in _MODELS:
        _MODELS[path] = Forecaster(path)
    return _MODELS[path]


def top_combos(weights, n):
    """The ``n`` most probable joint choices of independent categorical variables (beam search over the variables;
    exact for products of non-negative factors) -> (choices [W, M] int, product weights [W])."""
    beams = [((), 1.)]
    for w in weights:
        order = np.argsort(-np.asarray(w))
        cand = [(c+(int(k),), p*float(w[k])) for c, p in beams for k in order[:n]]
        cand.sort(key=lambda x: -x[1])
        beams = cand[:n]
    return np.asarray([c for c, _ in beams], np.int64).reshape(len(beams), len(weights)), np.asarray([p for _, p in beams])


class FCBank:
    """LEGAL people bank of the map-aware forecaster (Layer 3 options ``L3_MW_BANK=fc1|fck``, default off; the
    ``oraclebank.CVBank`` interface): every person the robot tracks (valid, at most 2 s old: ``control.human_estimates``)
    -- from the forecaster when the legal track log holds a detection of it in the last 2 s, else at the tracker's
    constant velocity; nobody else exists.

    * ``fc1``: one world -- each person's weighted mean of the modes or its top mode (``one``; default: the choice
      stored with the model, made on TRAIN-validation; ``L3_FC1=mean|top`` overrides);
    * ``fck``: the ``worlds`` most probable joint mode combinations of the forecast people (product of their mode
      weights; constant-velocity people identical in every world), weighted by the renormalised product -- the
      multi-world selector (``heuristics.multiworld_index``, ``L3_MW_WORLDS``) imagines each and weights its outcome.
      Combinations below ``min_w`` (``L3_FC_MINW``, default 0.01) of the total are dropped.
    Positions between the knots (every 4 steps to 3.07 s) are linear; beyond, the last knot segment continues."""

    def __init__(self, legal, forecaster=None, mode='fc1', worlds=6, horizon_steps=63, one=None, min_w=None):
        t0 = time.perf_counter()
        rows = instances(legal)
        self.mode = mode
        N = len(rows)
        self.track_keys = tuple('track:%s' % r['track_id'] for r in rows)
        kt = KNOT_T
        cvp = np.zeros((NK, N, 2))
        for n, r in enumerate(rows):
            e = r['est']
            p0 = np.array([e.position[0], -e.position[1]]); v = np.array([e.velocity[0], -e.velocity[1]])
            cvp[:, n] = p0[None]+kt[:, None]*v[None]
        fidx = [n for n, r in enumerate(rows) if r.get('hist') is not None]
        self.forecast = len(fidx)
        if fidx:
            fc = forecaster or load_forecaster()
            w, knots, _ = fc.modes([rows[n] for n in fidx])
        if mode == 'fc1' or not fidx:
            one = one or os.environ.get('L3_FC1') or (fc.meta.get('fc1', 'mean') if fidx else 'mean')
            paths = cvp[None].copy()
            for m, n in enumerate(fidx):
                paths[0, :, n] = knots[m, int(np.argmax(w[m]))] if one == 'top' else (w[m][:, None, None]*knots[m]).sum(0)
            ww = np.ones(1)
        else:
            choice, pw = top_combos([w[m] for m in range(len(fidx))], max(1, int(worlds)))
            pw = pw/pw.sum()
            keep = pw >= float(os.environ.get('L3_FC_MINW', '0.01') if min_w is None else min_w)
            keep[0] = True
            choice, pw = choice[keep], pw[keep]/pw[keep].sum()
            paths = np.repeat(cvp[None], len(pw), 0)
            for wi in range(len(pw)):
                for m, n in enumerate(fidx):
                    paths[wi, :, n] = knots[m, choice[wi, m]]
            ww = pw
            self.choice = choice
        self.knots = paths                                                       # [W, NK, N, 2]
        self.weights = np.asarray(ww, np.float64)
        self.present = np.ones((len(self.weights), N), bool)
        self.unseen = np.zeros(N, bool)
        self.times = np.arange(1, horizon_steps)*STEP_S
        self.long_steps = 0
        self.build_ms = 1000*(time.perf_counter()-t0)
        if os.environ.get('L3_FC_LOG'):
            # analysis log (L3_FC_LOG=<jsonl>, default off): one line per bank -- tracks, forecast people, worlds
            with open(os.environ['L3_FC_LOG'], 'a') as fh:
                fh.write(json.dumps(dict(time=float(legal.time), mode=mode, tracks=N, forecast=len(fidx),
                                         worlds=len(self.weights), weights=[round(float(x), 4) for x in self.weights],
                                         fc1=(one if mode == 'fc1' or not fidx else None),
                                         ms=round(self.build_ms, 1)))+'\n')

    def sandbox_positions(self, rel):
        rel = np.asarray(rel, np.float64).reshape(-1)
        kt = KNOT_T
        i = np.clip(np.searchsorted(kt, rel, side='right')-1, 0, NK-2)
        f = (rel-kt[i])/(kt[i+1]-kt[i])
        a, b = self.knots[:, i], self.knots[:, i+1]                             # [W, T, N, 2]
        return a+f[None, :, None, None]*(b-a)


# ------------------------------------------------------------------------------------------------ DEV evaluation
def bank_instances(bank_dir, cache=None, max_files=0):
    """Tracked-person instances of a state bank (``statebank.py``): legal model inputs, the true future (labels) and
    Stage 1's errors at 0.77 / 1.5 / 3 s (weighted mean world, best of its worlds, average world; NaN when Stage 1 has
    no slot for the track).  Cached (pickle) -- the bank pickles are large."""
    import pickle
    if cache and Path(cache).exists():
        return pickle.load(open(cache, 'rb'))
    files = sorted(glob.glob(os.path.join(bank_dir, '*.pkl')))
    if max_files:
        files = files[:max_files]
    out = []
    for p in files:
        item = pickle.load(open(p, 'rb'))
        T = np.asarray(item['truth_people'], np.float64)
        if len(T) < 65:
            continue
        legal = item['legal']
        rows = [r for r in instances(legal, T) if np.isfinite(r['fut']).all()]
        if not rows:
            continue
        b = item['bank']
        P1 = np.asarray(b['positions'], np.float64)
        keys = list(b['track_keys'])
        W1 = np.asarray(b['weights'], np.float64)
        W1 = W1/W1.sum()
        for r in rows:
            e = r.pop('est')
            r['est_pos'] = np.asarray(e.position, np.float64)
            r['est_vel'] = np.asarray(e.velocity, np.float64)
            r['state'] = Path(p).stem
            r['robot'] = np.asarray([legal.pose[0], legal.pose[2], legal.pose[3]], np.float64)
            r['config'] = item.get('config')
            ref = np.asarray(r['ref'], np.float64)
            s1 = np.full((len(EVAL_STEPS), 3), np.nan)
            k = 'track:%s' % r['track_id']
            if k in keys and P1.shape[1]:
                for hi, h in enumerate(EVAL_STEPS):
                    y = ref+r['fut'][h//2]
                    pr = P1[:, min(h, P1.shape[1])-1, keys.index(k)]
                    ok = np.isfinite(pr).all(1)
                    if ok.any():
                        mm = (W1[ok, None]*pr[ok]).sum(0)/W1[ok].sum()
                        e1 = np.hypot(pr[ok, 0]-y[0], pr[ok, 1]-y[1])
                        s1[hi] = (float(np.hypot(*(mm-y))), float(e1.min()), float((W1[ok]*e1).sum()/W1[ok].sum()))
            r['s1'] = s1
            out.append(r)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(str(cache)+'.tmp', 'wb') as f:
            pickle.dump(out, f, protocol=4)
        os.replace(str(cache)+'.tmp', cache)
    return out


def fit_scale_mixture(d, C=2, iters=300, sig_min=0.03):
    """2-D isotropic Gaussian scale mixture centred on a point forecast, fitted by EM to its error distances ``d``
    (gives a point forecast an NLL; fitted on TRAIN) -> (weights [C], variances [C]).  Spreads are floored at the
    forecaster's own minimum (``sig_min`` = 0.03 m, ``make_net``): standing people give exactly-zero CV errors, and an
    unfloored component collapses onto them."""
    d2 = np.asarray(d, np.float64)[np.isfinite(d)]**2
    s2 = np.quantile(d2, np.linspace(0.3, 0.9, C))/2.+1e-4
    pi = np.full(C, 1./C)
    for _ in range(iters):
        lp = np.log(pi)[None]-np.log(2*np.pi*s2)[None]-d2[:, None]/(2*s2[None])
        r = np.exp(lp-lp.max(1, keepdims=True))
        r /= r.sum(1, keepdims=True)
        pi = np.maximum(r.mean(0), 1e-6)
        s2 = np.maximum((r*d2[:, None]).sum(0)/(2*np.maximum(r.sum(0), 1e-9)), sig_min**2)
    return pi/pi.sum(), s2


def scale_mixture_nll(d, pi, s2):
    """NLL of 2-D errors of distance ``d`` under ``fit_scale_mixture``'s mixture."""
    d2 = np.asarray(d, np.float64)[:, None]**2
    pi, s2 = np.asarray(pi, np.float64), np.asarray(s2, np.float64)
    lp = np.log(pi)[None]-np.log(2*np.pi*s2)[None]-d2/(2*s2[None])
    m = lp.max(1)
    return -(m+np.log(np.exp(lp-m[:, None]).sum(1)))


def cv_errors(rows):
    """Errors [M, 3] of the tracker's constant-velocity estimate (``oraclebank.CVBank``) at 0.77 / 1.5 / 3 s."""
    fut = np.stack([r['fut'] for r in rows]).astype(np.float64)
    ref = np.stack([np.asarray(r['ref'], np.float64) for r in rows])
    Y = ref[:, None]+fut[:, [h//2 for h in EVAL_STEPS]]
    P0 = np.stack([r['est_pos'] for r in rows]); V = np.stack([r['est_vel'] for r in rows])
    cv = np.zeros((len(rows), 3))
    for hi, h in enumerate(EVAL_STEPS):
        p = P0+V*h*STEP_S                                                                  # planar (x, -z)
        cv[:, hi] = np.hypot(p[:, 0]-Y[:, hi, 0], -p[:, 1]-Y[:, hi, 1])
    return cv


def calib_main(argv):
    """TRAIN calibration of the point forecasts' NLL (``fit_scale_mixture`` per horizon): constant velocity on the
    collected TRAIN instances, Stage 1's mean world on the TRAIN state bank's tracked instances."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='output/agents/people/data,output/agents/people/data2')
    ap.add_argument('--train-bank-cache', default='output/agents/people/train_bank_instances.pkl')
    ap.add_argument('--train-bank', default='output/statebank/orca_train')
    ap.add_argument('--save', default='output/agents/people/point_calib.json')
    a = ap.parse_args(argv)
    rows = data_instances(a.data)
    cv = cv_errors(rows)
    brows = bank_instances(a.train_bank, a.train_bank_cache)
    s1 = np.stack([r['s1'] for r in brows])[:, :, 0]
    s1 = s1[np.isfinite(s1).all(1)]
    out = dict(source=dict(cv=a.data, stage1=a.train_bank, n_cv=int(len(cv)), n_stage1=int(len(s1))))
    for name, E in (('cv', cv), ('stage1', s1)):
        out[name] = []
        for hi in range(3):
            pi, s2 = fit_scale_mixture(E[:, hi], C=3)
            out[name].append(dict(pi=pi.tolist(), s2=s2.tolist(),
                                  train_nll=round(float(scale_mixture_nll(E[:, hi], pi, s2).mean()), 3)))
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    Path(a.save).write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


def score_instances(rows, models, v2_path='output/peoplefc/v2', calib=None):
    """Per-instance errors / NLL of every method at 0.77 / 1.5 / 3 s -> (dict method -> metric -> [M, 3], truth
    categories, Stage 1 coverage mask).  ``calib`` (``calib_main``, TRAIN): NLL of the point forecasts."""
    torch = _torch()
    M = len(rows)
    fut = np.stack([r['fut'] for r in rows]).astype(np.float64)                     # [M, 33, 2] relative to ref
    ref = np.stack([np.asarray(r['ref'], np.float64) for r in rows])
    Y = ref[:, None]+fut[:, [h//2 for h in EVAL_STEPS]]                               # [M, 3, 2] world
    res = {}
    for fc in models:
        w, kn, sg = fc.modes(rows)
        out = {k: np.zeros((M, 3)) for k in ('mean', 'top', 'best', 'nll')}
        for hi, h in enumerate(EVAL_STEPS):
            mh = at_steps(kn, [h])[:, :, 0]                                               # [M, K, 2]
            sh = at_steps_scalar(sg, [h])[:, :, 0]
            out['mean'][:, hi], out['top'][:, hi], out['best'][:, hi] = point_errors(w, mh, Y[:, hi])
            out['nll'][:, hi] = mixture_nll(w, mh, sh, Y[:, hi])
        res[fc.path] = out
    cv = cv_errors(rows)
    res['cv'] = dict(mean=cv)
    if calib:
        res['cv']['nll'] = np.stack([scale_mixture_nll(cv[:, hi], calib['cv'][hi]['pi'], calib['cv'][hi]['s2'])
                                     for hi in range(3)], 1)
    if v2_path and Path(v2_path, 'mdn.pt').exists():
        from .peoplefc import MDN, canonical as v2_canonical
        d = torch.load(Path(v2_path, 'mdn.pt'), map_location='cpu')
        v2 = MDN(int(d['n_in'])); v2.net.load_state_dict(d['state']); v2.mu, v2.sd = d['mu'], d['sd']
        X, Rs = [], []
        for r in rows:
            x, Rm = v2_canonical(r['hist'], r['vel'])
            X.append(np.concatenate([x, [r['age']]])); Rs.append(Rm)
        wv, mv, sv = v2.predict(np.asarray(X, np.float32))                              # [M, K], [M, K, 3, 2], [M, K, 3]
        mw = np.einsum('mkhi,mij->mkhj', mv, np.stack(Rs))+ref[:, None, None]           # canonical -> world (x @ R)
        out = {k: np.zeros((M, 3)) for k in ('mean', 'top', 'best', 'nll')}
        for hi in range(3):
            out['mean'][:, hi], out['top'][:, hi], out['best'][:, hi] = point_errors(wv, mw[:, :, hi], Y[:, hi])
            out['nll'][:, hi] = mixture_nll(wv, mw[:, :, hi], sv[:, :, hi], Y[:, hi])
        res['mdn_v2'] = out
    s1 = np.stack([r['s1'] for r in rows])                                               # [M, 3, 3]
    res['stage1'] = dict(mean=s1[:, :, 0], best=s1[:, :, 1], avg_world=s1[:, :, 2])
    if calib:
        res['stage1']['nll'] = np.stack([scale_mixture_nll(s1[:, hi, 0], calib['stage1'][hi]['pi'],
                                                           calib['stage1'][hi]['s2']) for hi in range(3)], 1)
    cats = maneuvers(fut, near_robot(fut, ref, np.stack([r['robot'] for r in rows])))
    s1ok = np.isfinite(s1[:, :, 0]).all(1)
    return res, cats, s1ok


def summarize(res, cats, s1ok, label=''):
    table = {}
    for name, mets in res.items():
        for c, m in cats.items():
            for sub, msk in (('all', np.ones(len(m), bool)), ('s1cov', s1ok)):
                sel = m & msk
                for hi, h in enumerate(EVAL_STEPS):
                    key = '%s|%s|%s|%.2fs' % (name, c, sub, h*STEP_S)
                    table[key] = {met: ((round(float(np.nanmean(v[sel, hi])), 3) if sel.any() else None) if met == 'nll' else _q(v[sel, hi]))
                                  for met, v in mets.items()}
    return table


def print_table(table, names, cats, s1ok, cats_order=('all', 'maneuver', 'turn_stop', 'turn', 'stop', 'start', 'straight',
                                                     'still', 'near', 'near_maneuver'),
                subs=('all', 's1cov')):
    f = lambda q: '%.2f/%.2f' % (q['median'], q['p90']) if isinstance(q, dict) and q.get('median') is not None else 'n/a'
    for c in cats_order:
        for sub in subs:
            for h in EVAL_STEPS:
                hs = '%.2fs' % (h*STEP_S)
                line = []
                for name in names:
                    r = table.get('%s|%s|%s|%s' % (name, c, sub, hs))
                    if not r:
                        continue
                    s = '%s %s' % (Path(name).name, f(r['mean']))
                    if 'top' in r:
                        s += ' top %s best %s nll %s' % (f(r['top']), f(r['best']), r['nll'])
                    else:
                        if 'best' in r:
                            s += ' best-world %s' % f(r['best'])
                        if 'nll' in r:
                            s += ' nll %s' % r['nll']
                    line.append(s)
                n = int((cats[c] & (s1ok if sub == 's1cov' else True)).sum())
                print('[%s|%s|%s] n=%d  %s' % (c, sub, hs, n, ' | '.join(line)), flush=True)


def data_instances(dirs, min_valid=64):
    """Collected instances (``collect`` format) as evaluation rows (the ``bank_instances`` fields; Stage 1 unavailable
    -> NaN)."""
    rows = []
    for d in str(dirs).split(','):
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            if f.endswith('.tmp.npz'):
                continue
            npz = np.load(f)
            if int(npz['n']) == 0:
                continue
            z = {k: npz[k] for k in npz.files}                  # decompress every array once (not once per row)
            for i in range(int(z['n'])):
                if z['valid'][i] < min_valid or not np.isfinite(z['fut'][i]).all():
                    continue
                est = z['est'][i].astype(np.float64)                                    # world (x, z) at t
                vel = z['vel'][i].astype(np.float64)                                    # world (vx, vz)
                rows.append(dict(hist=z['hist'][i], vel=z['vel'][i], age=float(z['age'][i]), fut=z['fut'][i],
                                 ref=z['ref'][i].astype(np.float64), frame=z['frame'][i], frame_kind=int(z['kind'][i]),
                                 patch=z['patch'][i], robot=z['robot'][i].astype(np.float64),
                                 est_pos=np.array([est[0], -est[1]]), est_vel=np.array([vel[0], -vel[1]]),
                                 state='%s@%d' % (Path(f).stem, int(z['step'][i])), config=str(z['config']),
                                 scene=str(z['scene']), s1=np.full((len(EVAL_STEPS), 3), np.nan)))
    return rows


def evaluate_main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', required=True, help='model dir (repeatable: ablations)')
    ap.add_argument('--bank', default='output/statebank/orca_dev')
    ap.add_argument('--cache', default='output/agents/people/dev_instances.pkl')
    ap.add_argument('--v2', default='output/peoplefc/v2', help='history-only MDN baseline (peoplefc)')
    ap.add_argument('--json', default=None, help='results file')
    ap.add_argument('--max-files', type=int, default=0)
    ap.add_argument('--data', default=None, help='evaluate on collected instances (collect format) instead of a bank')
    ap.add_argument('--calib', default='output/agents/people/point_calib.json',
                    help='TRAIN calibration of the point forecasts (calib_main); NLL of CV / Stage 1 if present')
    a = ap.parse_args(argv)
    t0 = time.time()
    rows = data_instances(a.data) if a.data else bank_instances(a.bank, a.cache, a.max_files)
    fcs = [Forecaster(p) for p in a.model]
    calib = json.loads(Path(a.calib).read_text()) if a.calib and Path(a.calib).exists() else None
    res, cats, s1ok = score_instances(rows, fcs, a.v2, calib=calib)
    table = summarize(res, cats, s1ok)
    print('%s: %d tracked instances from %d states (%d with Stage 1 slots), %.0f s' % (
        a.data or a.bank, len(rows), len({r['state'] for r in rows}), s1ok.sum(), time.time()-t0), flush=True)
    print('categories: %s' % json.dumps({c: int(m.sum()) for c, m in cats.items()}), flush=True)
    print_table(table, [fc.path for fc in fcs]+['mdn_v2', 'stage1', 'cv'], cats, s1ok)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(dict(bank=a.data or a.bank, models=a.model, calib=calib, instances=len(rows),
                                                states=len({r['state'] for r in rows}), stage1_covered=int(s1ok.sum()),
                                                categories={c: int(m.sum()) for c, m in cats.items()}, table=table),
                                           indent=1))


# ------------------------------------------------------------------------------------------------ command line
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['collect', 'fit', 'evaluate', 'calib'])
    ap.add_argument('--run', default='output/runs/R10')
    ap.add_argument('--out', default='output/agents/people/data')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--every', type=int, default=4)
    ap.add_argument('--until', type=float, default=0., help='collect: stop starting episodes after this unix time')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--skip', default='', help='collect: skip episodes already collected in these dirs')
    ap.add_argument('--grep', default='', help='collect: only spool files whose name contains every listed substring')
    ap.add_argument('--match', default='detection', choices=['detection', 'estimate'],
                    help='collect: label person by the last detection (default) or by the CV estimate at t '
                         '(peopledata.track_histories)')
    a, rest = ap.parse_known_args()
    if a.mode == 'collect':
        collect(a)
    elif a.mode == 'fit':
        fit_main(rest)
    elif a.mode == 'calib':
        calib_main(rest)
    else:
        evaluate_main(rest)


if __name__ == '__main__':
    main()
