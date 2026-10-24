"""Static ego-link geometry calibrated from legal observed articulation.

Native reset randomizes the seven arm joints. Physics then clamps positions to
the URDF limits, even for the kinematic robot. The cached gait archive describes
the independent leg animation; it cannot certify a noisy arm posture. This
adapter uses a renderer-free robot copy, without a live environment, people,
episode state or physics stepping. Only changed arm observations invoke FK.
"""
from dataclasses import dataclass
import itertools
import time

import numpy as np


def _readonly(array):
    value = np.ascontiguousarray(array, dtype=np.float64)
    value.setflags(write=False)
    return value


def footprint_requirements(gait_boxes):
    """Conservative bounds for yaw sweep and the three dynamic-cover disks.

    For collinear equal-radius disks, nearest-center distance on a rectangle
    is maximal at an X endpoint or an interior center bisector, and at a Z
    endpoint. Thus this checks the entire boxes, not just their corners.
    """
    boxes = np.asarray(gait_boxes, dtype=float)
    minimum = boxes[:, :, :3].min(0)
    maximum = boxes[:, :, 3:6].max(0)
    z = np.maximum(np.abs(minimum[:, 2]), np.abs(maximum[:, 2]))
    x = np.maximum(np.abs(minimum[:, 0]), np.abs(maximum[:, 0]))
    rotation = float(np.max(np.hypot(x, z)))
    candidates = np.stack((minimum[:, 0], maximum[:, 0],
        np.clip(-.19, minimum[:, 0], maximum[:, 0]),
        np.clip(.19, minimum[:, 0], maximum[:, 0])), axis=1)
    nearest = np.min(np.abs(candidates[:, :, None] - np.array([-.38, 0., .38])), axis=2)
    cover = float(np.max(np.hypot(nearest, z[:, None])))
    return rotation, cover


@dataclass(frozen=True)
class EgoGeometryReceipt:
    """Immutable numeric data; no simulator handles cross this boundary."""

    gait_boxes: np.ndarray
    gait_error: float
    elapsed_ms: float
    changed: bool
    phase_index: int
    arm_limit_correction: float
    rotation_radius_m: float
    dynamic_cover_radius_m: float
    arm_obbs: np.ndarray


class EgoGeometryCalibration:
    """Adapt cached per-gait leg boxes to measured native arm articulation.

    Call ``update(actual_robot_joint_positions, acknowledged_forward_count)``
    after every actual acquisition, including reset and legal prefix actions.
    Pass its ``gait_boxes`` to the navigator. The navigator still selects the
    current and next acknowledged leg phase and performs its static sweep.

    The helper owns a robot-only static FK copy and must be closed. Construction
    and changed-arm calibration occur outside action search. The arm bounds
    enclose both the currently observed and the next native joint-limit-clamped
    posture, since clamping can occur on the first action. No arm command or
    continuously actuated arm motion is supported; unexpected changes fail
    explicitly instead of silently invalidating that prediction.
    """

    def __init__(self, robot_config, gait_boxes, gait_joints):
        from eventwm_graph.native import StaticGeometry

        boxes = np.asarray(gait_boxes, dtype=float)
        joints = np.asarray(gait_joints, dtype=float)
        if boxes.shape != (757, 21, 7) or joints.shape != (757, 20):
            raise ValueError("Expected installed Spot reset plus 756-frame gait calibration")
        if not np.isfinite(boxes).all() or not np.isfinite(joints).all():
            raise ValueError("Nonfinite ego calibration")
        if np.any(boxes[:, :, :3] > boxes[:, :, 3:6]):
            raise ValueError("Invalid ego link bounds")
        self._reference_boxes = _readonly(boxes.copy())
        self._reference_joints = _readonly(joints.copy())
        self._geometry = StaticGeometry("NONE", robot_config)
        robot = self._geometry.robot
        self._links = [-1, *robot.sim_obj.get_link_ids()]
        if self._links != [-1, *range(20)] or list(robot.params.leg_joints) != list(range(8, 20)):
            self.close()
            raise ValueError("Unsupported robot link/joint order")
        limits = np.asarray(robot.sim_obj.joint_position_limits, dtype=float)
        if limits.shape != (2, 20):
            self.close()
            raise ValueError("Unsupported robot joint limits")
        self._limits = limits.copy()
        self._last_arm = None
        self._last_clamped_arm = None
        self._boxes = None
        self._latest_arm_obbs=np.empty((0,15))
        self._arm_obbs=_readonly(np.empty((0,15)))
        self._rotation_radius = self._dynamic_cover_radius = None
        self._last_forward_count = -1
        self._closed = False

    def close(self):
        geometry = getattr(self, "_geometry", None)
        if geometry is not None:
            geometry.close()
            self._geometry = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _native_boxes(self, joints):
        """Native asset FK only: no physics, contact queries or live agent."""
        import magnum as mn

        robot = self._geometry.robot
        robot.sim_obj.joint_positions = np.asarray(joints, dtype=np.float32)
        robot.base_pos = np.zeros(3)
        robot.base_rot = 0.0
        boxes = []
        obbs=[]
        for link in self._links:
            node = robot.sim_obj.get_link_scene_node(link)
            bounds = node.cumulative_bb
            transform = node.absolute_transformation()
            points = np.asarray([
                transform.transform_point(mn.Vector3(point))
                for point in itertools.product(*zip(bounds.min, bounds.max))
            ], dtype=float)
            boxes.append([*points.min(0), *points.max(0), float(link in (10, 13, 16, 19))])
            if link<8:
                origin=np.asarray(transform.transform_point(mn.Vector3(0,0,0)),float)
                axes=np.column_stack([np.asarray(transform.transform_point(mn.Vector3(v)),float)-origin
                                      for v in np.eye(3)])
                if not np.allclose(axes.T@axes,np.eye(3),rtol=0,atol=2e-6):
                    raise ValueError("Native link bounds require rigid, unscaled FK axes")
                center=np.asarray(transform.transform_point((bounds.min+bounds.max)*.5),float)
                half=np.asarray((bounds.max-bounds.min)*.5,float)
                obbs.append(np.r_[center,half,axes.ravel()])
        self._latest_arm_obbs=np.asarray(obbs)
        return np.asarray(boxes)

    def update(self, robot_joint_positions, acknowledged_forward_count):
        began = time.perf_counter()
        if self._closed:
            raise RuntimeError("Ego geometry calibration is closed")
        joints = np.asarray(robot_joint_positions, dtype=float)
        if joints.shape != (20,) or not np.isfinite(joints).all():
            raise ValueError("Expected 20 finite legal ego joint positions")
        n = int(acknowledged_forward_count)
        if n != acknowledged_forward_count or n < 0 or n < self._last_forward_count:
            raise ValueError("Acknowledged forward count must be a nondecreasing nonnegative integer")
        phase = 0 if n == 0 else 1 + ((n - 1) * 5) % 756
        gait_error = float(np.max(np.abs(joints[8:] - self._reference_joints[phase, 8:])))
        if gait_error > 2e-5:
            raise ValueError("Actual ego legs differ from acknowledged native gait contract")
        arm = joints[:8].copy()
        clamped = np.clip(joints, self._limits[0], self._limits[1])
        correction = float(np.max(np.abs(clamped[:8] - arm)))
        changed = self._last_arm is None or not np.array_equal(arm, self._last_arm)
        if changed:
            if self._last_arm is not None and not np.allclose(
                    arm, self._last_clamped_arm, rtol=0, atol=2e-5):
                raise ValueError("Unexpected arm motion outside observed/native-clamp contract")
            observed_boxes = self._native_boxes(joints)
            # Arm joints must not change the independent leg geometry. Check
            # this on every geometry rebuild against the observed gait phase.
            if not np.allclose(observed_boxes[9:], self._reference_boxes[phase, 9:],
                               rtol=0, atol=2e-6):
                raise ValueError("Native leg FK no longer matches static gait calibration")
            clamped_boxes = self._native_boxes(clamped) if correction else observed_boxes
            arm_bounds = observed_boxes[:9].copy()
            arm_bounds[:, :3] = np.minimum(arm_bounds[:, :3], clamped_boxes[:9, :3])
            arm_bounds[:, 3:6] = np.maximum(arm_bounds[:, 3:6], clamped_boxes[:9, 3:6])
            adapted = self._reference_boxes.copy()
            adapted[:, :9] = arm_bounds
            rotation_radius, dynamic_cover_radius = footprint_requirements(adapted)
            if rotation_radius > .72 + 1e-7 or dynamic_cover_radius > .36 + 1e-7:
                raise ValueError("Unsupported physical ego envelope: observed arm posture "
                    f"requires yaw radius {rotation_radius:.6f} m (limit .72) and "
                    f"three-disk radius {dynamic_cover_radius:.6f} m (limit .36)")
            self._boxes = _readonly(adapted)
            # At reset, clamping may change arm posture on the first native
            # step. Keep the enclosing two-posture AABB until the arrived arm
            # is stable; only then refine its empty corners with exact FK axes.
            self._arm_obbs=_readonly(self._latest_arm_obbs.copy() if not correction else np.empty((0,15)))
            self._rotation_radius, self._dynamic_cover_radius = rotation_radius, dynamic_cover_radius
            self._last_arm = arm
            self._last_clamped_arm = clamped[:8].copy()
        self._last_forward_count = n
        return EgoGeometryReceipt(self._boxes, gait_error,
            1000 * (time.perf_counter() - began), changed, phase, correction,
            self._rotation_radius, self._dynamic_cover_radius,self._arm_obbs)
