"""Original Falcon/Habitat benchmark episodes, exogenous human tapes, robot branching.

Only this module touches the simulator. It exposes two execution modes:

* ``NativeTask.step``: the unchanged ``habitat.Env.step`` of the original
  ``social_nav_v2/falcon_{hm3d,mp3d}.yaml`` benchmark (original ``RearrangeSim-v2``
  with its own recomputed navmesh), used for authoritative continuous replay.
* ``BranchSimulator``: privileged offline Oracle branching.  Falcon humans
  never consult robot agent 0 (``OracleNavObstacleAction`` only avoids agents
  ``1..N``; the robot-distance stop/walk coupling is disabled), so one recorded
  ``HumanTape`` is a valid exogenous future for every robot branch.

Exact restoration.  Bullet stores articulated bases as double quaternions, so
writing back a float32 transform read from the simulator is lossy (~1 ulp).
Branch states therefore keep the *setter inputs*: the robot's last base target
computed by an exact replica of ``BaseVelAction.update_base`` and each human's
last ``set_joint_transform`` arguments.  Re-applying them reproduces the
internal state bit for bit.  Equivalence to ``Env.step`` (poses, contacts,
legal observations) is a tested property, never assumed.

No module outside the Oracle may read a tape: it is privileged future truth.
"""
from dataclasses import dataclass
import math
import random
import time
import numpy as np

CONFIGS = {'hm3d': 'social_nav_v2/falcon_hm3d.yaml', 'mp3d': 'social_nav_v2/falcon_mp3d.yaml'}
# Original socialnav_task_detail measures minus the pixel top-down map and the
# training reward; their relative (dependency) order is preserved.
OFFICIAL_MEASURES = ('distance_to_goal', 'success', 'did_multi_agents_collide', 'num_steps',
                     'spl', 'psc', 'human_collision')
NATIVE_ACTIONS = ('forward', 'left', 'right', 'stop')
ACTION_NAMES = {'forward': 'agent_0_discrete_move_forward', 'left': 'agent_0_discrete_turn_left',
                'right': 'agent_0_discrete_turn_right', 'stop': 'agent_0_discrete_stop'}
VELOCITY_ACTION = 'agent_0_qg_base_velocity'
MAX_EPISODE_STEPS = 500
SUCCESS_DISTANCE = 0.2
# All-gait robot body bounds in the base frame (x forward, z right), measured
# by the earlier audit over 756 leg-animation frames.
ROBOT_BOUNDS = (-.646495, .584202, -.242846, .238225)
# Robot-human contact cannot occur beyond this rectangle-to-human-center
# distance (audited human enclosing radius .616 m plus margin).  Tests verify
# that the exact detection never finds a human contact beyond it.
CONTACT_GATE_M = 1.0
# The benchmark humanoid collision model is a stick skeleton of 2 cm boxes
# (female_0.urdf etc.), so the official measure can miss a robot driving
# through a person's body.  Declared additional physical criterion: the
# all-gait robot rectangle must stay >= BODY_PROXY_M from every active
# same-floor human centre (torso/limb volume proxy).
BODY_PROXY_M = 0.25
SAME_FLOOR_M = 0.5
# Declared static-collision rule.  The benchmark sometimes spawns the robot
# already touching static geometry (e.g. the arm wrist 0.6 m ahead).  Then,
# and only then, static contacts are exempt during an escape phase that ends
# at the first contact-free step and must end within ESCAPE_STEPS steps; every
# later static contact (and any contact in episodes spawned clear) counts.
ESCAPE_STEPS = 24


@dataclass(frozen=True)
class EpisodeSpec:
    dataset: str
    split: str
    scene: str
    episode_id: str
    seed: int

    @property
    def key(self):
        return f'{self.dataset}/{self.split}/{self.scene}/{self.episode_id}'

    def asdict(self):
        return dict(dataset=self.dataset, split=self.split, scene=self.scene,
                    episode_id=self.episode_id, seed=self.seed)


def matrix_yaw(m):
    """Yaw of local +x (robot forward) from ``np.asarray(Matrix4)``.

    The numpy view is row-major (``m[3] == (0, 0, 0, 1)``); column 0 is the
    image of local +x.  Matches ``robotmap.execution.signed_yaw`` (tested).
    """
    return math.atan2(-float(m[2][0]), float(m[0][0]))


def benchmark_config(spec, velocity_action=False):
    from habitat.config import read_write
    from habitat_baselines.config.default import get_config
    cfg = get_config(CONFIGS[spec.dataset])
    with read_write(cfg):
        cfg.habitat.dataset.split = spec.split
        cfg.habitat.dataset.content_scenes = [spec.scene]
        cfg.habitat.simulator.concur_render = False
        cfg.habitat.task.measurements = {k: v for k, v in cfg.habitat.task.measurements.items()
                                         if k in OFFICIAL_MEASURES}
        # Goal vectors are computed exactly from the public goal and pose; the
        # lab sensors only produce observations and never alter dynamics.
        cfg.habitat.task.lab_sensors = {}
        cfg.habitat.seed = int(spec.seed)
        for agent in cfg.habitat.simulator.agents.values():
            agent.sim_sensors = {}
        if velocity_action:
            from .native_velocity import velocity_action_config
            cfg.habitat.task.actions[VELOCITY_ACTION] = velocity_action_config()
    return cfg


def _install():
    import habitat_sim  # noqa: F401  (native bindings before Magnum users)
    import habitat  # noqa: F401
    import falcon.additional_action  # noqa: F401
    import falcon.additional_metric  # noqa: F401
    from eventwm_graph.stage1.runtime import install_headless
    install_headless()
    from .native_velocity import _register
    _register()


@dataclass
class RobotState:
    """Exact robot branch state.

    ``matrix`` is the float32 base transform the next ``update_base`` reads;
    ``target`` the last setter input that reproduces it (None only at reset);
    ``joints`` the float32 joint vector (stored as float, exact round trip);
    ``gait`` Falcon's module-global leg-animation counter ``play_i``.
    """
    matrix: np.ndarray
    target: object
    joints: np.ndarray
    gait: int



class NativeTask:
    """One original-benchmark episode in its own ``habitat.Env``."""

    def __init__(self, spec, velocity_action=False):
        _install()
        import habitat
        import torch
        from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
        self.spec = spec
        self.velocity_action = bool(velocity_action)
        cfg = benchmark_config(spec, velocity_action)
        full = PointNavDatasetV1(cfg.habitat.dataset)
        episodes = [e for e in full.episodes if str(e.episode_id) == str(spec.episode_id)]
        if len(episodes) != 1:
            raise ValueError(f'Episode {spec.key} not found exactly once')
        dataset = PointNavDatasetV1()
        dataset.episodes = episodes
        # Falcon draws each human's speed scale from the global NumPy RNG when
        # the task actions are constructed; seeding first makes it reproducible.
        random.seed(spec.seed); np.random.seed(spec.seed); torch.manual_seed(spec.seed)
        torch.set_num_threads(1)
        self.config = cfg
        self.env = habitat.Env(config=cfg.habitat, dataset=dataset)
        self.env.seed(spec.seed)
        self.episode = episodes[0]
        self.sim = self.env.sim
        self.task = self.env.task
        self.human_count = int(self.episode.info.get('human_num', 0))
        self.human_actions = tuple(f'agent_{i}_oracle_nav_randcoord_action_obstacle'
                                   for i in range(1, self.human_count+1))
        self.goal = np.asarray(self.episode.goals[0].position, dtype=float)
        self.reset()

    def reset(self):
        import falcon.additional_action as fa
        self.env.reset()
        fa.play_i = 0  # Falcon's module-global gait counter is replay state.
        self.steps = 0
        self.robot = self.sim.get_agent_data(0).articulated_agent
        self.humans = [self.sim.get_agent_data(i).articulated_agent for i in range(1, self.human_count+1)]
        self.human_speed_scales = [float(self.task.actions[a].rand_human_speed_scale)
                                   for a in self.human_actions]
        return self

    # ---------------------------------------------------------------- stepping
    def _action_dict(self, action):
        args = {name: np.array([1.]) for name in self.human_actions}
        if isinstance(action, str):
            if action not in ACTION_NAMES:
                raise ValueError(f'Unknown native action {action!r}')
            robot = ACTION_NAMES[action]
        else:
            if not self.velocity_action:
                raise ValueError('Velocity commands require velocity_action=True')
            from .dynamics import VELOCITY_ARG
            robot = VELOCITY_ACTION
            args[VELOCITY_ARG] = np.asarray(action, dtype=np.float64)
        return {'action': (robot, *self.human_actions), 'action_args': args}

    def step(self, action):
        """Authoritative unchanged ``habitat.Env.step``."""
        if self.env.episode_over:
            raise RuntimeError('Refusing to step a finished episode')
        self.env.step(self._action_dict(action))
        self.steps += 1
        return self

    @property
    def episode_over(self):
        return bool(self.env.episode_over)

    def metrics(self):
        return {k: (float(v) if isinstance(v, (bool, int, float, np.number)) else v)
                for k, v in self.env.get_metrics().items()}

    # ------------------------------------------------------------------- state
    def world_time(self):
        return float(self.sim.get_world_time())

    def robot_matrix(self):
        return np.asarray(self.robot.sim_obj.transformation, dtype=np.float32).copy()

    def robot_pose(self):
        p = np.asarray(self.robot.base_pos, dtype=float)
        return np.array([p[0], p[1], p[2], matrix_yaw(self.robot_matrix())])

    def robot_state(self):
        """Current state read from the simulator (exact as a motion state)."""
        import falcon.additional_action as fa
        return RobotState(self.robot_matrix(), None,
                          np.asarray(self.robot.sim_obj.joint_positions, dtype=np.float32).copy(),
                          int(fa.play_i))

    def human_positions(self):
        return np.asarray([h.base_pos for h in self.humans], dtype=float).reshape(-1, 3)

    def geodesic_to_goal(self, position=None):
        p = np.asarray(self.robot.base_pos if position is None else position, dtype=np.float32)
        return float(self.sim.geodesic_distance(p, [self.goal.astype(np.float32)], self.episode))

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def rectangle_distance(pose, points):
    """Distance from points (N,3) to the all-gait robot rectangle at pose."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros(0)
    c, s = math.cos(pose[3]), math.sin(pose[3])
    d = points[:, [0, 2]] - np.asarray(pose)[[0, 2]]
    lx = d[:, 0]*c - d[:, 1]*s          # forward (c, -s)
    lz = d[:, 0]*s + d[:, 1]*c          # right   (s,  c)
    xmin, xmax, zmin, zmax = ROBOT_BOUNDS
    dx = np.maximum(np.maximum(xmin-lx, 0.), lx-xmax)
    dz = np.maximum(np.maximum(zmin-lz, 0.), lz-zmax)
    return np.hypot(dx, dz)


def same_floor_rect_distance(pose, points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    d = rectangle_distance(pose, points)
    if len(points):
        d = np.where(np.abs(points[:, 1]-pose[1]) < SAME_FLOOR_M, d, np.inf)
    return d


def classify_contacts(task, gate_humans=True, need_static=True, pose=None):
    """One discrete detection, classified like the benchmark and project.

    ``human`` reproduces ``DidMultiAgentsCollide`` (robot object id in a contact
    whose other id is an active human).  ``static`` reproduces
    ``partial.metrics.contact_labels`` (non-support environment contacts).
    The detection is skipped only if static contacts are not requested and no
    human is within ``CONTACT_GATE_M`` of the robot rectangle.
    """
    sim = task.sim
    pose = task.robot_pose() if pose is None else pose
    near = same_floor_rect_distance(pose, task.human_positions()) if task.human_count else np.zeros(0)
    min_rect = float(near.min()) if len(near) else math.inf
    if gate_humans and not need_static and min_rect > CONTACT_GATE_M:
        return {'human': False, 'static': None, 'static_witnesses': [], 'checked': False,
                'min_rect_distance': min_rect}
    sim.perform_discrete_collision_detection()
    points = sim.get_physics_contact_points()
    robot = task.robot
    rid = robot.sim_obj.object_id
    human_ids = {h.sim_obj.object_id for h in task.humans}
    human = False
    static = []
    base = np.asarray(robot.base_pos)
    human_ray_ids = None
    for cp in points:
        ids = {cp.object_id_a, cp.object_id_b}
        if rid not in ids:
            continue
        if ids & human_ids:
            human = True
            continue
        if not need_static or len(ids) == 1:
            continue
        link = cp.link_id_a if cp.object_id_a == rid else cp.link_id_b
        contact_y = max(cp.position_on_a_in_ws[1], cp.position_on_b_in_ws[1])
        low_foot = (link in robot.params.leg_joints[2::3] and contact_y <= base[1]+.2)
        support = low_foot and abs(float(cp.contact_normal_on_b_in_ws[1])) >= math.cos(math.radians(45.))
        if low_foot and not support:
            import habitat_sim
            if human_ray_ids is None:
                human_ray_ids = set(human_ids)
                for h in task.humans:
                    human_ray_ids.update(h.sim_obj.link_object_ids.keys())
            contact = np.asarray(cp.position_on_b_in_ws if cp.object_id_a == rid else cp.position_on_a_in_ws,
                                 dtype=float).copy()
            normal = np.asarray(cp.contact_normal_on_b_in_ws, dtype=float).copy(); normal[1] = 0.
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal /= norm; contact[1] = base[1]+.22; blocked = False
                for sign in (-1, 1):
                    ray = habitat_sim.geo.Ray(contact+sign*.05*normal, -sign*normal)
                    if any(h.object_id != rid and h.object_id not in human_ray_ids
                           for h in sim.cast_ray(ray, .1).hits):
                        blocked = True
                support = not blocked
        if not support:
            static.append(dict(robot_link=int(link), distance=float(cp.contact_distance),
                               point=np.asarray(cp.position_on_a_in_ws).tolist()))
    return {'human': human, 'static': (bool(static) if need_static else None),
            'static_witnesses': static, 'checked': True, 'min_rect_distance': min_rect}


class HumanTape:
    """Privileged exogenous human future recorded from the reset state.

    ``inputs[k][h]`` holds human ``h``'s last ``set_joint_transform`` setter
    input at step ``k`` (None until first set: reset placement untouched).
    """

    def __init__(self, times, inputs, positions, speed_scales):
        self.times = np.asarray(times, dtype=float)
        self.inputs = inputs
        self.positions = np.asarray(positions, dtype=float)
        self.speed_scales = list(speed_scales)

    def __len__(self):
        return len(self.times)

    @classmethod
    def record(cls, task, steps=MAX_EPISODE_STEPS):
        """Step only the human actions with the task's own step (no measures)."""
        from habitat.articulated_agents.humanoids.kinematic_humanoid import KinematicHumanoid
        if task.steps != 0:
            raise ValueError('Tapes must start at the episode reset state')
        index = {h.sim_obj.object_id: k for k, h in enumerate(task.humans)}
        current = [None]*task.human_count
        original = KinematicHumanoid.set_joint_transform

        def recording(self, joint_list, offset_transform, base_transform):
            original(self, joint_list, offset_transform, base_transform)
            k = index.get(self.sim_obj.object_id)
            if k is not None:
                import magnum as mn
                add_rot = self.offset_transform_base.inverted()
                final = (base_transform @ add_rot) @ offset_transform
                current[k] = (np.asarray(joint_list, dtype=np.float32).copy(),
                              np.asarray(mn.Matrix4(offset_transform), dtype=np.float32).copy(),
                              np.asarray(final, dtype=np.float32).copy())
        times = [task.world_time()]
        inputs = [list(current)]
        positions = [task.human_positions()]
        action = {'action': task.human_actions,
                  'action_args': {n: np.array([1.]) for n in task.human_actions}}
        KinematicHumanoid.set_joint_transform = recording
        try:
            for _ in range(steps):
                task.task.step(action=dict(action, action_args=dict(action['action_args'])),
                               episode=task.env.current_episode)
                times.append(task.world_time())
                inputs.append(list(current))
                positions.append(task.human_positions())
        finally:
            KinematicHumanoid.set_joint_transform = original
        return cls(times, inputs, positions, task.human_speed_scales)

    def apply(self, task, k):
        import magnum as mn
        for human, row in zip(task.humans, self.inputs[k]):
            if row is None:
                continue
            joints, offset, final = row
            human.sim_obj.joint_positions = joints
            human.offset_transform = mn.Matrix4(offset)
            human.sim_obj.transformation = mn.Matrix4(final)


@dataclass
class StepResult:
    index: int
    time: float
    pose: np.ndarray
    action: object
    human_collision: bool
    static_contact: object
    body_overlap: bool
    stop: bool
    success: bool
    episode_over: bool
    min_rect_distance: float
    geodesic: float = math.nan


class BranchSimulator:
    """Robot-only native stepping against a recorded human tape (Oracle only)."""

    def __init__(self, task, tape, check_static=True):
        self.task = task
        self.tape = tape
        self.check_static = check_static
        self.state = None
        self.index = None
        forward = task.task.actions[ACTION_NAMES['forward']]
        lower, upper = task.robot.sim_obj.joint_position_limits
        # Physics stepping clamps kinematic joint positions to their limits
        # (Spot's arm init value -3.1686 rad is clamped to -pi on step 1).
        self._joint_lower = np.asarray(lower, dtype=np.float32)
        self._joint_upper = np.asarray(upper, dtype=np.float32)
        self._leg_data = forward._leg_data
        self._leg_length = forward._play_length_data
        self._leg_increment = forward._play_i_perframe
        self.performance = dict(steps=0, action_s=0., restore_s=0., contact_s=0., contact_checks=0)

    def root(self):
        """Robot state at reset (call before any branch stepping)."""
        if self.task.steps != 0:
            raise ValueError('Root must be captured from the reset state')
        return self.task.robot_state()

    def load(self, state, index):
        """Install ``state`` at tape index ``index`` (exact for index >= 1)."""
        import magnum as mn
        import falcon.additional_action as fa
        t = time.perf_counter()
        obj = self.task.robot.sim_obj
        obj.joint_positions = state.joints
        obj.transformation = mn.Matrix4(state.target if state.target is not None else state.matrix)
        fa.play_i = int(state.gait)
        self.tape.apply(self.task, index)
        self.task.robot.update()
        self.state, self.index = state, index
        self.performance['restore_s'] += time.perf_counter()-t

    def _base_velocity(self, action_obj, lin, ang):
        """Exact replica of ``BaseVelAction.update_base`` (dyn-slide enabled)."""
        import habitat_sim
        import magnum as mn
        ctrl = action_obj.base_vel_ctrl
        ctrl.linear_velocity = mn.Vector3(lin, 0, 0)
        ctrl.angular_velocity = mn.Vector3(0, ang, 0)
        trans = mn.Matrix4(self.state.matrix)
        rigid_state = habitat_sim.RigidState(mn.Quaternion.from_matrix(trans.rotation()), trans.translation)
        target_rigid_state = ctrl.integrate_transform(1 / self.task.sim.ctrl_freq, rigid_state)
        end_pos = self.task.sim.step_filter(rigid_state.translation, target_rigid_state.translation)
        did_try_step_fail = end_pos == rigid_state.translation
        if not did_try_step_fail:
            end_pos -= self.task.robot.params.base_offset
        target = mn.Matrix4.from_(target_rigid_state.rotation.to_matrix(), end_pos)
        self.task.robot.sim_obj.transformation = target
        return np.asarray(target, dtype=np.float32).copy()

    def _advance_gait(self, gait):
        import falcon.additional_action as fa
        cur = int(gait % self._leg_length)
        self.task.robot.leg_joint_pos = self._leg_data[cur]
        fa.play_i = gait + self._leg_increment
        return fa.play_i

    def step(self, action, compute_geodesic=False):
        """Advance the loaded state by one action; humans move to index+1.

        Mirrors ``Env.step``: robot action, humans to their next state, agent
        update (camera nodes), then the benchmark contact measure.
        """
        import falcon.additional_action as fa
        task = self.task
        index = self.index+1
        if index >= len(self.tape):
            raise ValueError('Tape does not cover this step')
        t = time.perf_counter()
        state = self.state
        stop = False
        target, gait = state.target, state.gait
        if isinstance(action, str):
            if action == 'stop':
                stop = True
            else:
                obj = task.task.actions[ACTION_NAMES[action]]
                target = self._base_velocity(obj, obj.lin_vel, obj.ang_vel)
                if action == 'forward':
                    gait = self._advance_gait(gait)
        else:
            from .dynamics import velocity_moves_legs
            lin, ang = float(action[0]), float(action[1])
            obj = task.task.actions[ACTION_NAMES['forward']]
            if lin != 0. or ang != 0.:
                target = self._base_velocity(obj, lin, ang)
                if velocity_moves_legs(lin, ang):
                    gait = self._advance_gait(gait)
        fa.play_i = gait
        matrix = np.asarray(task.robot.sim_obj.transformation, dtype=np.float32).copy()
        joints = np.asarray(task.robot.sim_obj.joint_positions, dtype=np.float32).copy()
        clamped = np.minimum(np.maximum(joints, self._joint_lower), self._joint_upper)
        if not np.array_equal(clamped, joints):
            task.robot.sim_obj.joint_positions = clamped
            joints = np.asarray(task.robot.sim_obj.joint_positions, dtype=np.float32).copy()
        self.tape.apply(task, index)
        task.robot.update()
        self.performance['action_s'] += time.perf_counter()-t
        self.state = RobotState(matrix, target, joints, gait)
        self.index = index
        t = time.perf_counter()
        pose = task.robot_pose()
        c = classify_contacts(task, gate_humans=True, need_static=self.check_static, pose=pose)
        self.performance['contact_s'] += time.perf_counter()-t
        self.performance['contact_checks'] += int(c['checked'])
        self.performance['steps'] += 1
        geodesic = task.geodesic_to_goal() if (compute_geodesic or stop) else math.nan
        success = bool(stop and geodesic < SUCCESS_DISTANCE)
        over = bool(stop or c['human'] or index >= MAX_EPISODE_STEPS)
        overlap = bool(c['min_rect_distance'] < BODY_PROXY_M)
        return StepResult(index, float(self.tape.times[index]), pose, action, bool(c['human']),
                          c['static'], overlap, stop, success, over, c['min_rect_distance'], geodesic)
