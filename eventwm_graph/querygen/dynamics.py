"""Robot dynamics families and their native control interface.

The benchmark-native robot is the discrete Falcon action set (forward 25/120 m
nominal per action, turns +-10/120 rad, no velocity state).  Non-native
families command a continuous base velocity per control interval through
``QGBaseVelocityAction``, which uses the *same* ``BaseVelAction.update_base``
integration, navmesh step filter and Falcon leg animation as the discrete
actions; only the commanded (linear, angular) velocity differs.  The dynamics
model (limits, inertia) lives in ``DynamicsModel`` and is shared verbatim by
the online executor, the sandbox predictor and the Oracle.
"""
from dataclasses import dataclass, field, asdict
import math
import numpy as np

VELOCITY_ARG = 'agent_0_qg_base_vel'
CTRL_FREQ = 120.          # habitat simulator ctrl_freq; one action integrates 1/120 s
NATIVE_LIN = 25.          # discrete forward lin_speed
NATIVE_ANG = 10.          # discrete turn ang_speed (rad/s in integrator units)
NATIVE_STEP_M = NATIVE_LIN/CTRL_FREQ     # nominal 0.20833 m per action
NATIVE_TURN_RAD = NATIVE_ANG/CTRL_FREQ   # 0.08333 rad per action
STEP_SECONDS = 0.048      # measured world-time per Env.step (physics 1/60 + 4 x 0.008)


def velocity_moves_legs(lin, ang):
    """Gait animation advances whenever the base translates (as discrete forward)."""
    return lin != 0.




@dataclass(frozen=True)
class DynamicsModel:
    """Robot motion capability in per-control-step units (one Env.step).

    All families keep Falcon's structure -- per step the robot either
    translates straight ahead or turns in place -- and vary capabilities
    independently:

    ``turn_rad``    in-place yaw per step (native 1/12 rad).
    ``levels``      forward speed levels (m/step); native = (0, 25/120).
    ``accel``/``brake``  max level change per step (native: instant).
    ``turn_needs_stop``  rotation only when the speed level is 0.

    ``native_actions`` marks the unmodified Falcon action set (no velocity
    state).  All other families command the continuous base-velocity action
    with lin = 120 * metres-per-step, ang = 120 * rad-per-step.
    """
    name: str
    turn_rad: float = NATIVE_TURN_RAD
    levels: tuple = (0., NATIVE_STEP_M)
    accel: int = 1
    brake: int = 1
    turn_needs_stop: bool = True
    native_actions: bool = False

    @property
    def discrete(self):
        return True

    @property
    def step_m(self):
        return self.levels[-1]

    @property
    def v_max(self):
        return self.levels[-1]

    @property
    def stateful(self):
        return len(self.levels) > 2

    @property
    def n_levels(self):
        return len(self.levels)

    def next_levels(self, level):
        """Admissible speed levels after one step from ``level``."""
        lo = max(0, level-self.brake); hi = min(len(self.levels)-1, level+self.accel)
        return range(lo, hi+1)

    def stop_distance(self, level):
        """Distance travelled while braking to 0 from ``level`` (excl. current step)."""
        d = 0.; l = level
        while l > 0:
            l = max(0, l-self.brake); d += self.levels[l]
        return d

    def stop_steps(self, level):
        return int(math.ceil(level/self.brake)) if level > 0 else 0

    def to_command(self, v, w):
        """Velocity-action integrator arguments for (m/step, rad/step)."""
        return (v*CTRL_FREQ, w*CTRL_FREQ)

    def describe(self):
        d = asdict(self)
        d.update(v_max_mps=self.v_max/STEP_SECONDS, turn_deg_per_step=math.degrees(self.turn_rad),
                 steps_to_full_speed=int(math.ceil((len(self.levels)-1)/self.accel)),
                 steps_to_stop=self.stop_steps(len(self.levels)-1),
                 stop_distance_m=self.stop_distance(len(self.levels)-1))
        return d


def dynamics_violations(dyn, commands):
    """Number of executed commands that break the family's motion model: a speed-level change beyond
    ``accel``/``brake``, turning while moving (``turn_needs_stop``) or with translation, STOP while moving.
    Native (instant-speed) robots have no velocity state: always 0."""
    if dyn.native_actions or not dyn.stateful:
        return 0
    unit = dyn.levels[1]*CTRL_FREQ
    prev, bad = 0, 0
    for c in commands:
        if isinstance(c, str):
            bad += int(c == 'stop' and prev > 0)
            continue
        lin, ang = float(c[0]), float(c[1])
        lvl = int(round(lin/unit))
        bad += int(lvl-prev > dyn.accel or prev-lvl > dyn.brake or abs(lin-dyn.levels[min(lvl, len(dyn.levels)-1)]*CTRL_FREQ) > 1e-6
                   or (ang != 0. and (lin != 0. or (dyn.turn_needs_stop and prev > 0))))
        prev = lvl
    return bad


def _levels(n):
    return tuple(NATIVE_STEP_M*k/n for k in range(n+1))


NATIVE = DynamicsModel('native', native_actions=True)
FAMILIES = {
    'native': NATIVE,
    'slowturn': DynamicsModel('slowturn', turn_rad=NATIVE_TURN_RAD/2),
    'inertia': DynamicsModel('inertia', levels=_levels(12), accel=2, brake=2),
    'slowturn_inertia': DynamicsModel('slowturn_inertia', turn_rad=NATIVE_TURN_RAD/2, levels=_levels(12), accel=2, brake=2),
    # held-out families (never used for generator development)
    'heavy_brake': DynamicsModel('heavy_brake', levels=_levels(12), accel=2, brake=1),
    'slowturn_heavy': DynamicsModel('slowturn_heavy', turn_rad=NATIVE_TURN_RAD/3, levels=_levels(12), accel=2, brake=1),
    # mild momentum (added 2026-10-30; the existing families are unchanged): full speed or a stop within 3 steps
    # (0.14 s), stop distance 0.21 m from full speed (inertia: 6 steps, 0.52 m)
    'light_inertia': DynamicsModel('light_inertia', levels=_levels(12), accel=4, brake=4),
    # milder still: full speed or a stop within 2 steps (0.10 s), stop distance 0.10 m from full speed
    'quick_inertia': DynamicsModel('quick_inertia', levels=_levels(12), accel=6, brake=6),
    # inertia in acceleration only: full speed after 6 steps (0.29 s, as 'inertia'), strong brakes (a stop within one
    # step, no stopping distance)
    'accel_inertia': DynamicsModel('accel_inertia', levels=_levels(12), accel=2, brake=12),
}
