"""Realise kinematic primitives under a dynamics family (legal inputs only).

Planners and the static lattice speak in kinematic primitives: ``forward``
(advance along the heading), ``left``/``right`` (rotate in place).  The
realiser turns them into executable commands for the robot's dynamics:

* native / instant-speed families: unchanged (``forward`` = full step).
* inertia families: speed is a state on discrete levels with bounded
  acceleration/braking; rotation requires standing still.  ``forward`` takes
  the highest admissible level such that (a) the robot can still stop before
  the next planned rotation or the plan end, (b) the full braking trajectory
  keeps the declared clearance from legally predicted humans, and (c) the
  body stays off observed walls; otherwise it brakes.  A rotation request
  while moving brakes first.  ``idle`` (zero velocity) only arises while
  braking to rest.
"""
import math
import numpy as np

from .control import heading_vec, body_ok, body_cost, FORWARD_X, BODY_X, BODY_HALF_WIDTH

# Layer 3 option (process-wide, default OFF -> released behaviour unchanged; L3_MOM_SIMPLE=1): robots with speed state
# realise a commanded 'forward' like the native robot (no extra body / braking-clearance / straight-budget gates in the
# realiser), only within their acceleration limit -- the planner already decided the advance; braking before turns
# and at STOP is kept.  The extra gates make momentum robots refuse advances the native robot takes in tight spots
# (dev floors: in-place dithering and stalls in the timeouts of every momentum family).
MOM_SIMPLE = __import__('os').environ.get('L3_MOM_SIMPLE') == '1'
from .dynamics import STEP_SECONDS


def braking_safe(dyn, P, psi, level, humans, clearance, t0=0):
    """Braking from ``level`` (after one step at that level) stays clear of humans."""
    if not humans:
        return True
    h = heading_vec(psi)
    d = 0.; l = level; k = 0
    c, s = math.cos(psi), math.sin(psi)
    while True:
        d += dyn.levels[l]; k += 1
        Q = P+d*h
        t = (t0+k)*STEP_SECONDS
        for hu in humans:
            q = hu.at(t)-Q
            lx = q[0]*c+q[1]*s; ly = -q[0]*s+q[1]*c
            dx = max(BODY_X[0]-lx, 0., lx-BODY_X[1]); dy = max(abs(ly)-BODY_HALF_WIDTH, 0.)
            if math.hypot(dx, dy) < clearance:
                return False
        if l == 0:
            return True
        l = max(0, l-dyn.brake)


class MotionRealizer:
    def __init__(self, dyn, clearance=0.30):
        self.dyn = dyn
        self.clearance = clearance

    def realize(self, prim, level, P, psi, gmap, humans, plan_prims=None, cap=None):
        """Return (command primitive, new level).

        Command primitives: 'forward' | 'left' | 'right' (native/instant) or
        ('forward', level) | 'left' | 'right' | 'idle' (stateful families).
        """
        dyn = self.dyn
        if prim == 'stop':
            return 'stop', 0
        if not dyn.stateful:
            return prim, (len(dyn.levels)-1 if prim == 'forward' else 0)
        if prim in ('left', 'right'):
            if level > 0 and dyn.turn_needs_stop:
                l2 = max(0, level-dyn.brake)
                return (('forward', l2) if l2 > 0 else 'idle'), l2
            return prim, 0
        if MOM_SIMPLE:
            top = len(dyn.levels)-1 if cap is None else max(1, min(int(cap), len(dyn.levels)-1))
            l2 = max(1, min(top, int(level)+dyn.accel)) if int(level) < top else max(top, int(level)-dyn.brake)
            return ('forward', l2), l2
        # forward with speed planning
        budget = math.inf
        if plan_prims:
            n = 0
            for p in plan_prims:
                if p != 'forward':
                    break
                n += 1
            budget = n*dyn.step_m
            if n == len(plan_prims):
                budget += 0.5*dyn.step_m    # plan end: arrive within half a step
        h = heading_vec(psi)
        for l2 in sorted(dyn.next_levels(level), reverse=True):
            if l2 == 0 or (cap is not None and l2 > cap):
                continue
            if dyn.levels[l2]+dyn.stop_distance(l2) > budget+1e-9:
                continue
            P1 = P+dyn.levels[l2]*h
            if gmap is not None and not body_ok(gmap, P1, psi, True, body_cost(gmap, P, psi, FORWARD_X), FORWARD_X):
                continue
            if not braking_safe(dyn, P, psi, l2, humans, self.clearance):
                continue
            return ('forward', l2), l2
        l2 = max(0, level-dyn.brake)
        return (('forward', l2) if l2 > 0 else 'idle'), l2
