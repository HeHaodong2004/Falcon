"""Necessary clearance gate at the EXACT pose used for query generation.

Never call this on an episode start in place of a later selected root. The map
adapter must be read-only and use the same NavMesh as the canonical checker.
Every compiled query checks this same first pose, including turns and WAIT.
"""
from dataclasses import asdict
import math
import time
import numpy as np
from collector.core import read, write
from collector.static_feasibility import is_static_feasible


class RootStaticPreflightRejected(RuntimeError):
    pass


def check_root_clearance(root, static_map, geometry, config):
    clearance = float(static_map.clearance(np.asarray(root.position, dtype=float)))
    # Run the unmodified canonical checker on a stationary two-pose probe.
    # The adapter intentionally tests only its clearance component: no Bullet
    # calls or state mutations, and no claim that a pass proves native validity.
    class ClearanceOnly:
        def clearance(self, position): return clearance
        def pose_free(self, position, yaw): return True
        def segment_free(self, a, b): return True
    pose = [0., *root.position, root.yaw]
    end = [root.control_dt, *root.position, root.yaw]
    probe = dict(nominal_timed_trajectory=[pose, end], executable_controls=[[0., 0.]],
                 control_dt=root.control_dt, integration_dt=root.integration_dt)
    certificate = is_static_feasible(root, probe, ClearanceOnly(), geometry, config)
    reason = certificate['rejection_reason']
    if reason not in (None, 'insufficient_swept_footprint_clearance'):
        raise RuntimeError('Invalid root preflight probe: ' + reason)
    reject = not certificate['static_feasible']
    # Diagnostic only; the canonical checker above makes the decision.
    required = max(0., geometry.radius-geometry.navmesh_radius) + config.safety_margin + config.max_sample_distance/2
    return dict(passed=not reject, rejection_reason='root_static_clearance_preflight' if reject else None,
                clearance=clearance if math.isfinite(clearance) else None,
                required_clearance=required, position=list(root.position), yaw=root.yaw,
                geometry=asdict(geometry), config=asdict(config))


def run_root_preflight(root, static_map, geometry, config, output, phase):
    began = time.perf_counter()
    check = check_root_clearance(root, static_map, geometry, config)
    elapsed = time.perf_counter() - began
    path = output/'audits/root_preflight.json'
    audit = read(path) if path.exists() else dict(preflight_checked=0, preflight_pass=0,
        preflight_reject=0, preflight_seconds=0., checks=[])
    audit['preflight_checked'] += 1
    audit['preflight_pass' if check['passed'] else 'preflight_reject'] += 1
    audit['preflight_seconds'] += elapsed
    audit['checks'].append(dict(check, phase=phase, seconds=elapsed))
    write(path, audit)
    if not check['passed']:
        raise RootStaticPreflightRejected('root_static_clearance_preflight: ' + phase)
    return check
