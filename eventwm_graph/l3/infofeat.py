"""Compiler v3 additions (L3_LAYOUT=v3): query-conditioned candidate descriptors and the robot's INFORMATION state.

Under the legal 90 deg jaw camera the robot must decide when to look: the Oracle never needs to (it knows every person),
so neither its demonstrations nor the v1 layout carry any notion of what the robot does not currently see.  v3 adds:

  ego  EGO_INFO   staleness of the legal map around the robot (per 8 robot-frame sectors x 2 rings: free area weighted
                  by the time since it was last inside the camera frustum), Stage 1's unseen-person mass per sector,
                  tracks that are no longer visible (count, nearest extrapolated position, age), time since any person
                  was visible, distinct people seen so far, recent camera sweep coverage
  cand CAND_SEM   the query itself: kind, label family, the generator's execution-style flags (tight / close / thru /
                  commit / short / ...), duration, speed, heading; the 'continue' candidate describes the running query
  cand CAND_INFO  what the imagined execution would SEE: yaw sweep left / right, stale free area covered by the imagined
                  camera, probability (over Stage 1 worlds) that an unseen person / a no-longer-visible tracked person
                  enters the imagined frustum with a clear legal-map line of sight, unseen-person mass in view at the
                  imagined end pose

Legality: legal map states, the robot's own pose history (camera view history), legal tracks and their history, the robot's own
poses, the candidates' Query objects, imagined poses and Stage 1's hypothetical people.  Nothing privileged.
"""
import math

import numpy as np

from eventwm_graph.querygen.legal import FREE, BLOCKED
from eventwm_graph.querygen.dynamics import STEP_SECONDS

HFOV = math.radians(90.)
VIEW_M = 6.0                       # range used for information features (the detector reaches 10 m; nearby matters)
STALE_CAP = 4.0                    # seconds (a person walks ~4 m; an episode lasts 24 s)
N_BINS = 16                        # direction bins (robot frame) for the sweep / staleness product

KINDS = ('goto', 'goal', 'stgoal', 'wait', 'wait_clear', 'rotate', 'creep', 'kturn', 'stop', 'continue', 'prim')
FAMILIES = ('goal', 'st_goal', 'pass', 'yield', 'sidestep', 'approach', 'advance', 'retreat', 'alt_route', 'frontier',
            'goal_slow', 'turnaround', 'turn', 'look', 'glance', 'scan', 'rot', 'creep', 'kturn', 'wait', 'stop')
FLAGS = ('tight', 'close', 'commit', 'unknown_cost', 'exact_end', 'arrive_exact', 'blind', 'unstick', 'evade', 'ukeep',
         'bumpf', 'shum', 'stop19', 'uprog', 'vclose', 'action', 'thru', 'short')
CAND_SEM = ([f'q_kind_{k}' for k in KINDS] + [f'q_fam_{f}' for f in FAMILIES] + [f'q_{f}' for f in FLAGS]
            + ['q_continued', 'q_duration', 'q_speed', 'q_has_heading', 'q_head_cos', 'q_head_sin', 'q_dir_left',
               'q_dir_right', 'q_n_via'])
CAND_INFO = ['i_sweep_left', 'i_sweep_right', 'i_stale_frac', 'i_stale_abs', 'i_resid_see', 'i_lost_see',
             'i_resid_end']
EGO_INFO = ([f'stale{s}_{r}' for s in range(8) for r in ('n', 'f')] + [f'age{s}' for s in range(8)]
            + [f'rsec{s}' for s in range(8)]
            + ['n_lost', 'lost_fwd', 'lost_left', 'lost_dist', 'lost_age', 't_since_vis', 'n_seen', 'cover_2s',
               'cover_5s'])


def _family(label):
    lab = (label or '').lower()
    for f in sorted(FAMILIES, key=len, reverse=True):
        if lab.startswith(f):
            return f
    return None


def query_semantics(q, frame, continued=False):
    """Descriptor of one Query (``continued``: the running query described by the 'continue' candidate)."""
    v = np.zeros(len(CAND_SEM), np.float32)
    if q is None:
        v[KINDS.index('continue')] = 1.
        v[CAND_SEM.index('q_continued')] = float(continued)
        return v
    kind = q.kind if q.kind in KINDS else 'goto'
    v[KINDS.index(kind)] = 1.
    fam = _family(q.label)
    if fam is not None:
        v[len(KINDS)+FAMILIES.index(fam)] = 1.
    meta = q.meta or {}
    off = len(KINDS)+len(FAMILIES)
    for i, f in enumerate(FLAGS):
        x = meta.get(f)
        if x is None or x is False:
            continue
        if f == 'short':
            v[off+i] = min(float(x), 16.)/16.
        elif f in ('tight', 'unknown_cost'):
            v[off+i] = min(float(x), 4.)/2.
        else:
            v[off+i] = 1. if not isinstance(x, (int, float)) else min(abs(float(x)), 2.)
    v[CAND_SEM.index('q_continued')] = float(continued)
    v[CAND_SEM.index('q_duration')] = min(float(q.duration or 0), 96.)/48.
    v[CAND_SEM.index('q_speed')] = min(float(q.speed if q.speed is not None else 1.), 2.)
    if q.heading is not None:
        d = float(frame.dyaw(q.heading))
        v[CAND_SEM.index('q_has_heading')] = 1.
        v[CAND_SEM.index('q_head_cos')] = math.cos(d)
        v[CAND_SEM.index('q_head_sin')] = math.sin(d)
    direction = meta.get('direction') or meta.get('final')
    v[CAND_SEM.index('q_dir_left')] = float(direction == 'left')
    v[CAND_SEM.index('q_dir_right')] = float(direction == 'right')
    v[CAND_SEM.index('q_n_via')] = min(len(q.via or ()), 4)/2.
    return v


# ----------------------------------------------------------------------------- legal-map helpers
def _window(g, cx, cz, radius):
    """Cells of the legal map within ``radius`` of (cx, cz): world centres and states (UNKNOWN outside the grid)."""
    r = int(math.ceil(radius/g.res))
    ci, cj = g.index(cx, cz)
    ii, jj = np.mgrid[ci-r:ci+r+1, cj-r:cj+r+1]
    st = np.zeros(ii.shape, np.uint8)
    a0, a1 = max(ci-r, 0), min(ci+r+1, g.state.shape[0])
    b0, b1 = max(cj-r, 0), min(cj+r+1, g.state.shape[1])
    if a1 > a0 and b1 > b0:
        st[a0-(ci-r):a1-(ci-r), b0-(cj-r):b1-(cj-r)] = g.state[a0:a1, b0:b1]
    wx = (ii+g.ox+0.5)*g.res
    wz = (jj+g.oz+0.5)*g.res
    return wx, wz, st


def seen_mask(x, z, yaw, wx, wz, blocked, rays=32, step=0.1):
    """Cells (wx, wz) inside the 90 deg frustum of a camera at (x, z, yaw) within VIEW_M and not occluded by the legal
    map's BLOCKED cells (first BLOCKED sample along each of ``rays`` rays)."""
    dx, dz = wx-x, wz-z
    c, s_ = math.cos(yaw), math.sin(yaw)
    fw = dx*c-dz*s_
    lf = -dx*s_-dz*c
    ang = np.arctan2(lf, fw)
    r = np.hypot(dx, dz)
    inside = (np.abs(ang) <= HFOV/2) & (r <= VIEW_M)
    if not inside.any():
        return inside
    a_b = -HFOV/2+(np.arange(rays)+0.5)*HFOV/rays
    rs = np.arange(1, int(VIEW_M/step)+1)*step
    th = yaw+a_b
    hit = blocked(x+np.cos(th)[:, None]*rs[None], z-np.sin(th)[:, None]*rs[None])     # [rays, S]
    first = np.where(hit.any(1), rs[np.argmax(hit, 1)], VIEW_M+1.)
    b = np.clip(((ang+HFOV/2)/HFOV*rays).astype(int), 0, rays-1)
    return inside & (r <= first[b]+0.15)


def view_age(legal, wx, wz, blocked, horizon_s=STALE_CAP, stride=3):
    """Seconds since each cell was last seen by the camera (the robot's own past poses, every ``stride`` steps within
    ``horizon_s``; occlusion by the CURRENT legal map); ``horizon_s`` where not seen within the window.  The legal
    map's own ``last_seen`` is the FIRST certification time (static cells are certified once), so the view history is
    recomputed from the pose history."""
    times = legal.times
    poses = legal.poses
    t_now = float(legal.time)
    age = np.full(wx.shape, float(horizon_s))
    i = len(poses)-1
    while i >= 0 and t_now-float(times[i]) <= horizon_s:
        m = seen_mask(float(poses[i][0]), float(poses[i][2]), float(poses[i][3]), wx, wz, blocked)
        a = t_now-float(times[i])
        age[m] = np.minimum(age[m], a)
        i -= stride
    return age


def _blocked_lookup(g):
    """Vectorised BLOCKED test of world points on the legal map (outside the grid = not blocked)."""
    state = g.state
    def f(x, z):
        i = np.floor(np.asarray(x)/g.res).astype(np.int64)-g.ox
        j = np.floor(np.asarray(z)/g.res).astype(np.int64)-g.oz
        ok = (i >= 0) & (i < state.shape[0]) & (j >= 0) & (j < state.shape[1])
        out = np.zeros(i.shape, bool)
        out[ok] = state[i[ok], j[ok]] == BLOCKED
        return out
    return f


def _los(blocked, ox, oz, px, pz, step=0.15):
    """Clear legal-map line of sight from (ox, oz) to (px, pz) (arrays of equal shape): no BLOCKED cell strictly
    between (the last 0.3 m before the target are not tested: the person's own cell may be marked)."""
    dx, dz = px-ox, pz-oz
    L = np.hypot(dx, dz)
    n = int(np.ceil(max(float(L.max()) if L.size else 0., 0.)/step))
    clear = np.ones(L.shape, bool)
    for s in range(1, n):
        f = s*step/np.maximum(L, 1e-6)
        live = (s*step < L-0.3)
        if not live.any():
            break
        hit = blocked(ox+f*dx, oz+f*dz) & live
        clear &= ~hit
    return clear


@np.errstate(invalid='ignore')
def _in_view(px, pz, x, z, yaw, rng=VIEW_M):
    """Points (px, pz) inside the 90 deg frustum of a robot at (x, z, yaw) within ``rng`` (planar)."""
    dx, dz = px-x, pz-z
    fw = dx*np.cos(yaw)-dz*np.sin(yaw)
    lf = -dx*np.sin(yaw)-dz*np.cos(yaw)
    ang = np.arctan2(lf, fw)
    return np.isfinite(px) & (np.abs(ang) <= HFOV/2) & (np.hypot(dx, dz) <= rng)


# ----------------------------------------------------------------------------- ego information state
class InfoState:
    """Per-decision information state; ``ego()`` -> EGO_INFO row, ``cand(poses_list, HP, W, keys)`` -> CAND_INFO."""

    def __init__(self, legal, frame, keys=(), unseen=None, HP=None, W=None):
        self.legal, self.F = legal, frame
        g = legal.map
        self.blocked = _blocked_lookup(g)
        wx, wz, st = _window(g, frame.x, frame.z, VIEW_M)
        fw, lf = frame.rel(wx, wz)
        rr = np.hypot(fw, lf)
        ang = np.arctan2(lf, fw)
        age = view_age(legal, wx, wz, self.blocked)/STALE_CAP
        free = (st == FREE) & (rr > 0.3) & (rr <= VIEW_M)
        self.age_feats = []
        sec8 = ((ang+math.pi)/(2*math.pi)*8).astype(int) % 8
        for s in range(8):
            for lo, hi in ((0.3, 3.), (3., VIEW_M)):
                m = (sec8 == s) & (rr > lo) & (rr <= hi)
                n = max(int(m.sum()), 1)
                self.age_feats.append(float((age*free)[m].sum())/n)          # stale free mass fraction
        for s in range(8):
            m = (sec8 == s) & free
            self.age_feats.append(float(age[m].mean()) if m.any() else 0.)   # mean age of the sector's free cells
        # stale free mass (the candidates' imagined camera removes what it sees)
        self.wx, self.wz = wx, wz
        self.mass = age*free
        self.mass_tot = float(self.mass.sum())
        # Stage 1 bank: residual (unseen) and tracked-but-not-visible slots
        self.keys = [str(k) for k in keys]
        self.HP, self.W = HP, W
        vis = {str(t['track_id']) for t in legal.tracks if t.get('visible_now')}
        self.res_slots = np.asarray([k.startswith('residual:') for k in self.keys], bool)
        self.lost_slots = np.asarray([k.startswith('track:') and k[len('track:'):] not in vis for k in self.keys], bool)

    @np.errstate(invalid='ignore')
    def ego(self):
        legal, F = self.legal, self.F
        # residual mass per robot-frame sector within 4 m over the Stage 1 horizon (<= 3 s)
        rsec = [0.]*8
        HP, W = self.HP, self.W
        if HP is not None and HP.shape[2] and self.res_slots.any():
            R = HP[:, :, self.res_slots]                                     # [W, T, Nr, 2]
            fw, lf = F.rel(R[..., 0], R[..., 1])
            ok = np.isfinite(fw) & (np.hypot(fw, lf) <= 4.)
            sec = ((np.arctan2(np.where(ok, lf, 0.), np.where(ok, fw, 1.))+math.pi)/(2*math.pi)*8).astype(int) % 8
            for s in range(8):
                hit = (ok & (sec == s)).any(axis=(1, 2))                    # [W]
                rsec[s] = float((W*hit).sum())
        # tracks no longer visible
        lost = []
        for t in legal.tracks:
            if t.get('visible_now') or not t.get('valid', True):
                continue
            age = float(legal.time)-float(t.get('last_seen_time', legal.time))
            pos = np.asarray(t['position'], np.float64)
            vel = np.asarray(t.get('estimated_velocity') or (0., 0., 0.), np.float64)
            px, pz = pos[0]+vel[0]*min(age, 2.), pos[2]+vel[2]*min(age, 2.)
            f, l_ = F.rel(px, pz)
            lost.append((math.hypot(float(f), float(l_)), float(f), float(l_), age))
        lost.sort()
        n_lost = sum(1 for x in lost if x[3] < 5.)
        if lost:
            d, f, l_, a = lost[0]
            lostf = [min(n_lost, 3)/3., max(min(f, 10.), -10.)/5., max(min(l_, 10.), -10.)/5., min(d, 10.)/5., min(a, 10.)/5.]
        else:
            lostf = [0., 0., 0., 2., 2.]
        # time since any person was visible, distinct people seen so far
        last_vis, ids = None, set()
        for tm, trs in legal.track_log:
            for t in trs:
                ids.add(str(t.get('track_id')))
                if t.get('visible_now'):
                    last_vis = tm
        t_since = min(float(legal.time)-last_vis, 10.)/10. if last_vis is not None else 1.
        # recent camera coverage of the 8 sectors (robot's own past headings, relative to the current heading)
        cover = []
        for sec_s in (2., 5.):
            n = int(round(sec_s/STEP_SECONDS))
            yaws = np.asarray([p[3] for p in legal.poses[-n:]], np.float64)
            d = np.angle(np.exp(1j*(yaws-F.yaw)))
            centres = -math.pi+(np.arange(8)+0.5)*2*math.pi/8
            hit = (np.abs(np.angle(np.exp(1j*(centres[:, None]-d[None])))) <= HFOV/2).any(1)
            cover.append(float(hit.mean()))
        return np.asarray(self.age_feats+rsec+lostf+[t_since, min(len(ids), 12)/6.]+cover, np.float32)

    def cand(self, poses, rel_steps=(4, 8, 12, 16)):
        """CAND_INFO for one imagined execution (``poses`` [T+1, 4] world (x, y, z, yaw))."""
        F = self.F
        poses = np.asarray(poses, np.float64)
        d = np.angle(np.exp(1j*(poses[:, 3]-F.yaw)))
        sweep_l = max(float(d.max()), 0.)/math.pi
        sweep_r = max(-float(d.min()), 0.)/math.pi
        T = len(poses)-1
        stale_frac = stale_abs = 0.
        if self.mass_tot > 0 and T > 0:
            seen = np.zeros(self.mass.shape, bool)
            for s in sorted({min(x, T) for x in rel_steps}):
                seen |= seen_mask(poses[s, 0], poses[s, 2], poses[s, 3], self.wx, self.wz, self.blocked)
            got = float(self.mass[seen].sum())
            stale_frac = got/self.mass_tot
            stale_abs = min(got/300., 2.)
        resid_see = lost_see = resid_end = 0.
        HP, W = self.HP, self.W
        if HP is not None and HP.shape[2] and (self.res_slots.any() or self.lost_slots.any()) and T > 0:
            steps = sorted({min(s, T) for s in rel_steps})
            for slots, name in ((self.res_slots, 'resid'), (self.lost_slots, 'lost')):
                if not slots.any():
                    continue
                seen = np.zeros(len(W), bool)
                for s in steps:
                    ti = min(s-1, HP.shape[1]-1)
                    pts = HP[:, ti][:, slots]                                # [W, Ns, 2]
                    x, z, yaw = poses[s, 0], poses[s, 2], poses[s, 3]
                    inv = _in_view(pts[..., 0], pts[..., 1], x, z, yaw)
                    if inv.any():
                        wi, ni = np.nonzero(inv)
                        clear = _los(self.blocked, np.full(len(wi), x), np.full(len(wi), z), pts[wi, ni, 0], pts[wi, ni, 1])
                        vis = np.zeros(inv.shape, bool); vis[wi[clear], ni[clear]] = True
                        seen |= vis.any(1)
                        if s == steps[-1] and name == 'resid':
                            resid_end = float((W*vis.sum(1)).sum())
                p = float((W*seen).sum())
                if name == 'resid':
                    resid_see = p
                else:
                    lost_see = p
        return [sweep_l, sweep_r, stale_frac, stale_abs, resid_see, lost_see, min(resid_end, 3.)]
