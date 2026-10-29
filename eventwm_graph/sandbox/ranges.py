"""Environment-side range sensors (``ideal_current_jaw_range_dynamic_occlusion_1``) for the sandbox's legal stream.

The range frame is the legal sensor extension Stage 2 was trained on: 256 x 256 Euclidean first-hit distances on
the jaw camera's 90 x 90 degree frustum, NaN beyond 10 m, active humans occlude, the robot and inactive agents are
ignored.  It is produced by the ENVIRONMENT (like any camera); the sandbox only receives the frame.

``native``: ``stage2_repair.current_range_sensor.CurrentRangeSensor`` at 256 px (the reference: one native Bullet
ray per pixel, ~1.5 s per frame).
"""
from types import SimpleNamespace

from .context import RESOLUTION, RANGE_SIZE


def runtime_view(task, observation_config):
    """The minimal runtime namespace the project's geometry observers expect (same as querygen LegalSensor)."""
    return SimpleNamespace(sim=task.sim, observation_config=observation_config, count=task.human_count,
                           episode=SimpleNamespace(scene_id=task.episode.scene_id,
                                                   info={'graph_navmesh': 'native_benchmark_navmesh'}))


class NativeRangeSensor:
    kind = 'native'

    def __init__(self, task, observation_config, size=RANGE_SIZE):
        from eventwm_graph.stage2_repair.current_range_sensor import CurrentRangeSensor
        self.sensor = CurrentRangeSensor(runtime_view(task, observation_config), RESOLUTION, size=size)

    def measure(self, timestamp=None):
        out = self.sensor.measure()
        if timestamp is not None:
            out['timestamp'] = float(timestamp)
        return out


def make_range_sensor(kind, task, observation_config, size=RANGE_SIZE):
    if kind == 'native':
        return NativeRangeSensor(task, observation_config, size)
    if kind in ('fast', 'fast-driver'):      # 'fast-driver': the same kernel without a PyTorch CUDA context
        from .fastrange import FastRangeSensor
        return FastRangeSensor(task, observation_config, size, backend='driver' if kind == 'fast-driver' else 'torch')
    if hasattr(kind, 'measure'):
        return kind
    raise ValueError('unknown range sensor %r' % (kind,))
