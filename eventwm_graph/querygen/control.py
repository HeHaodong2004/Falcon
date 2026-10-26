"""Shared low-level control layer: frames, legal human prediction, body checks.

Planar frame.  All local planners work in the right-handed plane
``P = (X, Y) = (x, -z)`` where the robot yaw ``psi = atan2(-dz, dx)`` is the
ordinary counter-clockwise angle and a left turn increases it.

The robot body is the measured all-gait rectangle (x forward in
[-0.646, 0.584] m, lateral [-0.243, 0.238] m).  The benchmark navmesh only
keeps the *center* 0.25 m from obstacles, so heading-dependent body clearance
is checked here against the legal map (never the navmesh).
"""
import math
import numpy as np

from .legal import UNKNOWN, FREE, BLOCKED
from .dynamics import NATIVE_STEP_M, NATIVE_TURN_RAD, STEP_SECONDS

BODY_X = (-0.646495, 0.584202)
BODY_HALF_WIDTH = 0.243
CIRCUMSCRIBED = math.hypot(0.646495, 0.243)
# Forward steps keep the nose this far from observed wall bands so the robot
# can still rotate in place afterwards (front corners sweep out to ~0.69 m).
FRONT_MARGIN = 0.15


def to_plane(xz):
    xz = np.asarray(xz, dtype=float)
    return np.stack([xz[..., 0], -xz[..., 1]], axis=-1)


def to_world(P):
    P = np.asarray(P, dtype=float)
    return np.stack([P[..., 0], -P[..., 1]], axis=-1)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def heading_vec(psi):
    return np.array([math.cos(psi), math.sin(psi)])


class HumanEstimate:
    """Legal constant-velocity estimate from one track (planar frame)."""
    __slots__ = ('position', 'velocity', 'age', 'visible', 'valid_velocity', 'track_id')

    def __init__(self, track, now):
        p = np.asarray(track['position'], dtype=float)
        v = track.get('estimated_velocity')
        self.valid_velocity = bool(track.get('velocity_valid')) and v is not None
        vel = np.asarray(v, dtype=float) if self.valid_velocity else np.zeros(3)
        self.age = max(0., float(now)-float(track['last_seen_time']))
        self.visible = bool(track.get('visible_now'))
        self.track_id = track['track_id']
        P = to_plane(p[[0, 2]])
        V = to_plane(vel[[0, 2]])
        self.velocity = V
        self.position = P+V*self.age       # extrapolated to now

    def at(self, t):
        return self.position+self.velocity*t


def human_estimates(tracks, now, max_age=2.0):
    out = []
    for t in tracks:
        if not t.get('valid', True):
            continue
        if float(now)-float(t['last_seen_time']) > max_age:
            continue
        out.append(HumanEstimate(t, now))
    return out


def body_axis_points(P, psi, spacing=0.1, x_range=BODY_X):
    n = int(math.ceil((x_range[1]-x_range[0])/spacing))+1
    xs = np.linspace(x_range[0], x_range[1], n)
    h = heading_vec(psi)
    return P[None, :]+xs[:, None]*h[None, :]


def body_cells_state(gmap, P, psi, spacing=0.1, x_range=BODY_X):
    """Legal-map states under the body's longitudinal axis samples.

    BLOCKED certificates are wall hits dilated by 0.25 m (~ the 0.243 m half
    width), so axis samples on non-BLOCKED cells mean the rectangle does not
    reach observed walls.  FREE samples certify full-body clearance.
    """
    pts = to_world(body_axis_points(P, psi, spacing, x_range))
    out = np.empty(len(pts), dtype=np.uint8)
    for k, (x, z) in enumerate(pts):
        out[k] = gmap.state_at(x, z)
    return out


def body_cost(gmap, P, psi, x_range=BODY_X):
    """Number of body-axis samples on observed-BLOCKED cells (0 = clear)."""
    return int((body_cells_state(gmap, P, psi, x_range=x_range) == BLOCKED).sum())


FORWARD_X = (BODY_X[0], BODY_X[1]+FRONT_MARGIN)


def body_ok(gmap, P, psi, allow_unknown=True, budget=0, x_range=BODY_X):
    """Body admissible at (P, psi): no more BLOCKED samples than ``budget``.

    ``budget`` is the cost of the robot's current pose, so a robot already
    touching an inflated wall band may still move or turn as long as it does
    not get worse (otherwise every action would be refused and it would sit
    in place).  ``allow_unknown=False`` additionally demands FREE samples.
    """
    s = body_cells_state(gmap, P, psi, x_range=x_range)
    if int((s == BLOCKED).sum()) > budget:
        return False
    return allow_unknown or bool(((s == FREE) | (s == BLOCKED)).all())


def turn_ok(gmap, P, psi0, dpsi, allow_unknown=True, budget=0):
    n = max(1, int(math.ceil(abs(dpsi)/0.02)))
    for k in range(1, n+1):
        if not body_ok(gmap, P, psi0+dpsi*k/n, allow_unknown, budget):
            return False
    return True


def blocked_points_near(gmap, P, radius=1.6, sectors=24):
    """Nearest observed-BLOCKED cell centre per angular sector (planar)."""
    x, z = float(P[0]), -float(P[1])
    i0, j0 = gmap.index(x, z)
    r = int(math.ceil(radius/gmap.res))
    a0, a1 = max(0, i0-r), min(gmap.state.shape[0], i0+r+1)
    b0, b1 = max(0, j0-r), min(gmap.state.shape[1], j0+r+1)
    if a1 <= a0 or b1 <= b0:
        return np.zeros((0, 2))
    sub = gmap.state[a0:a1, b0:b1]
    ii, jj = np.nonzero(sub == BLOCKED)
    if not len(ii):
        return np.zeros((0, 2))
    cx = (ii+a0+gmap.ox+.5)*gmap.res
    cz = (jj+b0+gmap.oz+.5)*gmap.res
    Q = np.stack([cx, -cz], axis=1)
    d = Q-P[None, :]
    dist = np.hypot(d[:, 0], d[:, 1])
    keep = dist <= radius
    Q, d, dist = Q[keep], d[keep], dist[keep]
    if not len(Q):
        return np.zeros((0, 2))
    sec = ((np.arctan2(d[:, 1], d[:, 0])+math.pi)/(2*math.pi)*sectors).astype(int) % sectors
    out = []
    for s in np.unique(sec):
        m = np.flatnonzero(sec == s)
        out.append(Q[m[np.argmin(dist[m])]])
    return np.asarray(out)


# --------------------------------------------------------------------------
# Kinematic adapters: planner velocity -> robot command
# --------------------------------------------------------------------------
NATIVE_VMAX = NATIVE_STEP_M/STEP_SECONDS    # ~4.34 m/s in benchmark time


class NativeAdapter:
    """Discrete robot (forward | left | right, no stop/pause) tracking a velocity.

    The planner supplies a desired planar velocity ``v_des`` (m/s).  Its
    magnitude below ``v_max`` is realised by *duty cycling*: desired
    displacement accumulates and a full forward step is taken once half a
    step is owed along the (aligned) heading -- the average velocity tracks
    ``v_des`` within one step, which planners add to their safety radius.
    Otherwise the robot turns toward ``v_des``; a zero velocity is held by
    alternating in-place turns (net zero drift).  Every forward step must keep
    the body axis off observed walls and pass the planner's one-step
    ``hard_ok`` safety predicate; every turn must keep the swept body axis off
    observed walls.
    """
    kind = 'native'

    def __init__(self, align_tol=math.radians(20.), hold_frac=0.12):
        self.align_tol = align_tol
        self.hold_frac = hold_frac
        self.vmax = NATIVE_VMAX
        self.step_m = NATIVE_STEP_M
        self.turn = NATIVE_TURN_RAD
        self._last_wait = 'left'
        self.debt = np.zeros(2)
        self.plan = []
        self._budget = 0

    def reset(self):
        self.debt = np.zeros(2)
        self.plan = []

    def _maneuver(self, P, psi, err, gmap, allow_unknown):
        """Committed maneuver when the short in-place rotation is body-blocked.

        Advance up to four steps if that frees the rotation sweep, else rotate
        the long way round if its whole sweep is clear; None if neither.
        """
        if gmap is None:
            return None
        h = heading_vec(psi)
        n_short = int(math.ceil(abs(err)/self.turn))
        short = 'left' if err > 0 else 'right'
        fwd_budget = body_cost(gmap, P, psi, FORWARD_X)
        for k in range(1, 5):
            Pk = P+k*self.step_m*h
            if not body_ok(gmap, Pk, psi, allow_unknown, fwd_budget, FORWARD_X):
                break
            if turn_ok(gmap, Pk, psi, math.copysign(min(abs(err), math.pi/2), err), allow_unknown,
                       body_cost(gmap, Pk, psi)):
                return ['forward']*k+[short]*n_short
        long_total = 2*math.pi-abs(err)
        long_dir = 'right' if err > 0 else 'left'
        if turn_ok(gmap, P, psi, math.copysign(long_total, -err), allow_unknown, self._budget):
            return [long_dir]*int(math.ceil(long_total/self.turn))
        return None

    def predicted_pose(self, P, psi, action):
        if action == 'forward':
            return P+self.step_m*heading_vec(psi), psi
        if action == 'left':
            return P, psi+self.turn
        if action == 'right':
            return P, psi-self.turn
        return P, psi

    def _turn_dir_ok(self, gmap, P, psi, a, allow_unknown):
        return gmap is None or turn_ok(gmap, P, psi, self.turn if a == 'left' else -self.turn, allow_unknown,
                                       self._budget)

    def wait_action(self, P, psi, gmap, face=None, allow_unknown=True):
        """Native 'hold position': turn toward ``face`` or alternate turns."""
        if gmap is not None:
            self._budget = body_cost(gmap, P, psi)
        if face is not None and abs(wrap(face-psi)) > self.turn*0.6:
            a = 'left' if wrap(face-psi) > 0 else 'right'
            if self._turn_dir_ok(gmap, P, psi, a, allow_unknown):
                return a
        a = 'right' if self._last_wait == 'left' else 'left'
        if not self._turn_dir_ok(gmap, P, psi, a, allow_unknown):
            a = 'left' if a == 'right' else 'right'
        self._last_wait = a
        return a

    def command(self, P, psi, v_des, hard_ok=None, gmap=None, face=None, allow_unknown_body=True):
        """Return (primitive, mode)."""
        P = np.asarray(P, dtype=float)
        v_des = np.asarray(v_des, dtype=float)
        speed = float(np.hypot(*v_des))
        h = heading_vec(psi)
        P1 = P+self.step_m*h
        self._budget = body_cost(gmap, P, psi) if gmap is not None else 0
        fwd_budget = body_cost(gmap, P, psi, FORWARD_X) if gmap is not None else 0
        fwd_ok = ((gmap is None or body_ok(gmap, P1, psi, allow_unknown_body, fwd_budget, FORWARD_X)) and
                  (hard_ok is None or hard_ok(P1, psi)))
        if self.plan:
            prim = self.plan[0]
            safe_here = hard_ok is None or hard_ok(P, psi)
            valid = ((prim == 'forward' and fwd_ok) or
                     (prim != 'forward' and safe_here and self._turn_dir_ok(gmap, P, psi, prim, allow_unknown_body)))
            still_wanted = speed*STEP_SECONDS >= self.hold_frac*self.step_m
            if valid and still_wanted:
                self.plan.pop(0)
                return prim, 'maneuver'
            self.plan = []
        if speed*STEP_SECONDS < self.hold_frac*self.step_m:
            self.debt *= 0.
            if hard_ok is not None and not hard_ok(P, psi) and fwd_ok:
                return 'forward', 'escape'
            return self.wait_action(P, psi, gmap, face, allow_unknown_body), 'hold'
        theta = math.atan2(v_des[1], v_des[0])
        err = wrap(theta-psi)
        self.debt = self.debt+v_des*STEP_SECONDS
        n = float(np.hypot(*self.debt))
        if n > 1.5*self.step_m:
            self.debt *= 1.5*self.step_m/n
        if abs(err) <= self.align_tol:
            if float(np.dot(self.debt, h)) >= 0.5*self.step_m and fwd_ok:
                self.debt = self.debt-self.step_m*h
                return 'forward', 'track'
            if hard_ok is not None and not hard_ok(P, psi) and fwd_ok:
                return 'forward', 'escape'
            if abs(err) > 0.55*self.turn:
                a = 'left' if err > 0 else 'right'
                if self._turn_dir_ok(gmap, P, psi, a, allow_unknown_body):
                    return a, 'align'
            return self.wait_action(P, psi, gmap, theta, allow_unknown_body), 'duty'
        self.debt *= 0.5
        if hard_ok is not None and not hard_ok(P, psi) and fwd_ok:
            return 'forward', 'escape'
        a = 'left' if err > 0 else 'right'
        sweep = math.copysign(min(abs(err), math.pi/2), err)
        if gmap is not None and not turn_ok(gmap, P, psi, sweep, allow_unknown_body, self._budget):
            # The intended rotation is body-blocked somewhere: decide the
            # maneuver *before* turning into the blocked heading.
            plan = self._maneuver(P, psi, err, gmap, allow_unknown_body)
            if plan:
                self.plan = plan[1:]
                return plan[0], 'maneuver'
            if self._turn_dir_ok(gmap, P, psi, a, allow_unknown_body):
                return a, 'turn_partial'
            if fwd_ok:
                return 'forward', 'escape'
            return self.wait_action(P, psi, gmap, None, allow_unknown_body), 'jammed'
        return a, 'turn'


class VelocityAdapter:
    """Continuous unicycle dynamics with limits (see ``dynamics.DynamicsModel``).

    Tracks a desired planar velocity: angular command proportional to the
    heading error, linear command ``|v| cos(err)`` (no reversing), both then
    clipped by the dynamics model's speed/acceleration/braking limits.
    """
    kind = 'velocity'

    def __init__(self, model, k_heading=3.0):
        self.model = model
        self.k = k_heading

    def command(self, P, psi, v_des, admissible=None, gmap=None, face=None, allow_unknown_body=True,
                state=None):
        lin0, ang0 = state if state is not None else (0., 0.)
        speed = float(np.hypot(*v_des))
        if speed < 1e-3:
            target_ang = 0. if face is None else self.k*wrap(face-psi)
            return self.model.limit((0., target_ang), (lin0, ang0)), 'hold'
        theta = math.atan2(v_des[1], v_des[0])
        err = wrap(theta-psi)
        lin = speed*max(0., math.cos(err))
        ang = self.k*err
        return self.model.limit((lin, ang), (lin0, ang0)), 'track'
