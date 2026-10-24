"""Canonical planner/collector static feasibility, independent of human state."""
from dataclasses import asdict, dataclass
import math
import numpy as np

CHECKER_VERSION = 'eventwm-static-swept-v2.2'


@dataclass(frozen=True)
class StaticConfig:
    safety_margin: float = .03
    max_sample_distance: float = .02
    max_sample_angle: float = .04

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError('Static sampling/margin must be positive and finite')


@dataclass(frozen=True)
class RobotGeometry:
    radius: float
    navmesh_radius: float = .25
    height: float = 1.41
    source: str = 'configured_robot'

    def __post_init__(self):
        if not math.isfinite(self.radius) or self.radius < self.navmesh_radius or self.navmesh_radius <= 0:
            raise ValueError('Invalid conservative robot footprint')


def is_static_feasible(root, query, static_map, geometry, config=StaticConfig()):
    """Check full fixed trajectory, including root/turns/WAIT, against a static map.

    static_map.clearance(p): distance to edge of navmesh already eroded by navmesh_radius.
    static_map.pose_free(p,yaw): native static geometry overlap; must ignore all humans.
    static_map.segment_free(a,b): same static navmesh step-filter consistency as execution.
    Map adapters cannot inspect query labels or human futures. Native adapters restore poses.
    Dense sampling includes a half-spacing clearance allowance for intervening swept centers.
    """
    q = query.data if hasattr(query, 'data') else query
    result = dict(static_feasible=False, rejection_reason=None, min_static_clearance=None,
                  failed_segment=None, failed_pose=None, checker_version=CHECKER_VERSION,
                  geometry=asdict(geometry), config=asdict(config), checked_poses=0)
    trajectory = np.asarray(q['nominal_timed_trajectory'], dtype=float)
    if trajectory.ndim != 2 or len(trajectory) < 2 or trajectory.shape[1] != 5 or not np.isfinite(trajectory).all():
        result['rejection_reason'] = 'invalid_trajectory'; return result
    if not np.allclose(trajectory[0,1:4],root.position,atol=1e-7,rtol=0) or abs(trajectory[0,4]-root.yaw)>1e-7:
        result['rejection_reason'] = 'root_query_mismatch'; return result
    controls=np.asarray(q['executable_controls'],dtype=float)
    if controls.shape != (len(trajectory)-1,2) or not np.isfinite(controls).all():
        result['rejection_reason']='invalid_fixed_controls';return result
    rebuilt=[trajectory[0].copy()]
    for i,(v,w) in enumerate(controls):
        pose=rebuilt[-1].copy();pose[0]=(i+1)*q['control_dt']
        pose[4]+=w*q['integration_dt']
        pose[1:4]+=np.array([math.cos(pose[4]),0.,-math.sin(pose[4])])*v*q['integration_dt']
        rebuilt.append(pose)
    if not np.allclose(rebuilt,trajectory,atol=1e-7,rtol=0):
        result['rejection_reason']='fixed_controls_trajectory_mismatch';return result
    required = max(0., geometry.radius - geometry.navmesh_radius) + config.safety_margin + config.max_sample_distance/2
    minimum = float('inf')
    for i, (a,b) in enumerate(zip(trajectory[:-1],trajectory[1:])):
        distance = float(np.linalg.norm(b[1:4]-a[1:4]))
        n = max(1, math.ceil(distance/config.max_sample_distance), math.ceil(abs(b[4]-a[4])/config.max_sample_angle))
        for j in range(n+1):
            pose = a+(b-a)*j/n
            clearance = float(static_map.clearance(pose[1:4]))
            minimum = min(minimum, clearance)
            reason = None
            if not math.isfinite(clearance) or clearance < required:
                reason = 'insufficient_swept_footprint_clearance'
            elif not static_map.pose_free(pose[1:4], float(pose[4])):
                reason = 'native_static_geometry_overlap'
            result['checked_poses'] += 1
            if reason:
                result.update(rejection_reason=reason, failed_segment=i, failed_pose=pose.tolist(),
                              min_static_clearance=minimum if math.isfinite(minimum) else None)
                return result
        if distance > 1e-8 and not static_map.segment_free(a[1:4],b[1:4]):
            result.update(rejection_reason='controller_navmesh_clipping',failed_segment=i,
                          failed_pose=b.tolist(),min_static_clearance=minimum)
            return result
    result.update(static_feasible=True,min_static_clearance=minimum if math.isfinite(minimum) else None)
    return result
