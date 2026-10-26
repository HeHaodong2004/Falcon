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

    ``discrete``: the robot either translates ``step_m`` straight ahead or
    turns in place by ``turn_rad`` (the Falcon structure); no velocity state.
    Otherwise a unicycle: linear speed ``v`` in [0, v_max] m/step changes by
    at most ``accel``/``brake`` m/step per step, yaw rate |w| <= w_max
    rad/step changing by at most ``w_accel`` per step; simultaneous v and w
    are allowed (arcs).  ``min_radius`` > 0 forbids in-place turning
    (|w| <= v / min_radius).  Native actions map to the velocity action's
    integrator arguments as lin = 120 * metres-per-step, ang = 120 * rad-per-step.
    """
    name: str
    discrete: bool = False
    step_m: float = NATIVE_STEP_M
    turn_rad: float = NATIVE_TURN_RAD
    v_max: float = NATIVE_STEP_M
    accel: float = math.inf
    brake: float = math.inf
    w_max: float = NATIVE_TURN_RAD
    w_accel: float = math.inf
    min_radius: float = 0.
    native_actions: bool = False   # True only for the unmodified Falcon action set

    @property
    def stateful(self):
        return not self.discrete

    def limit(self, desired, current):
        """Clip a desired (v, w) [m/step, rad/step] given the current state."""
        v_d, w_d = float(desired[0]), float(desired[1])
        v0, w0 = float(current[0]), float(current[1])
        v = min(max(v_d, 0.), self.v_max)
        v = min(v, v0+self.accel) if v >= v0 else max(v, v0-self.brake)
        w = min(max(w_d, -self.w_max), self.w_max)
        w = min(max(w, w0-self.w_accel), w0+self.w_accel)
        if self.min_radius > 0.:
            w_cap = v/self.min_radius
            w = min(max(w, -w_cap), w_cap)
        return v, w

    def to_command(self, v, w):
        """Velocity-action integrator arguments for (m/step, rad/step)."""
        return (v*CTRL_FREQ, w*CTRL_FREQ)

    def describe(self):
        d = asdict(self)
        d['v_max_mps'] = self.v_max/STEP_SECONDS
        d['stop_steps'] = (math.inf if self.brake == 0 else
                           (0 if math.isinf(self.brake) else math.ceil(self.v_max/self.brake)))
        return d


NATIVE = DynamicsModel('native', discrete=True, native_actions=True)
FAMILIES = {
    'native': NATIVE,
}
