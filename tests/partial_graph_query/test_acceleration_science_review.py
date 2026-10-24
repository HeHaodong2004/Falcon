"""Reviewer A's exact observer differential tests.

Synthetic geometry isolates observation decisions without advancing a simulator.
The reference is loaded from the accepted commit and verified by SHA-256. These
tests complement real-Habitat ray/endpoint comparisons; they do not replace them.
Run with the repository's native eventwm-falcon Python environment.
"""
import copy
import hashlib
import math
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import habitat_sim  # Installed native import order, before Magnum/Falcon.
import numpy as np

from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.partial.query import generate_queries
from scripts.collector.observation import ObservationConfig


ACCEPTED_COMMIT = '84cb0847930f40024fdf6be8ffb94887efd97f19'
ACCEPTED_OBSERVER_SHA256 = '779dde9eadb3753782fe10b3148687230b07b2f927147078b32a6ad404a07989'


def accepted_observer_class():
    root = Path(__file__).resolve().parents[2]
    source = subprocess.run(
        ['git', 'show', ACCEPTED_COMMIT+':eventwm_graph/partial/observer.py'],
        cwd=root, check=True, capture_output=True).stdout
    if hashlib.sha256(source).hexdigest() != ACCEPTED_OBSERVER_SHA256:
        raise AssertionError('Frozen observer identity changed')
    reference = types.ModuleType('eventwm_graph.partial._reviewer_a_frozen_observer')
    reference.__package__ = 'eventwm_graph.partial'
    exec(compile(source, 'accepted:'+ACCEPTED_COMMIT+'/observer.py', 'exec'), reference.__dict__)
    return reference.GeometryObserver


def y_rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


class Transform:
    def __init__(self, rotation=None, translation=None):
        self.matrix = np.eye(3) if rotation is None else np.asarray(rotation)
        self.translation = np.zeros(3) if translation is None else np.asarray(translation)

    def rotation(self):
        return self.matrix

    def transform_vector(self, vector):
        return self.matrix @ np.asarray(vector)


class Support:
    def __init__(self, blocked=False):
        self.blocked = blocked
        self.calls = 0
        self.requests = []

    def snap_point(self, position):
        self.calls += 1
        p = np.asarray(position, dtype=float).copy()
        self.requests.append(('snap', p.tolist()))
        # Unknown space and a discontinuous multi-height patch must stay unknown.
        if self.blocked or p[0] > 2.25 or p[2] < -2.35:
            return np.full(3, np.nan)
        if p[0] < -1.85:
            p[1] += .24
        return p

    def distance_to_closest_obstacle(self, position, distance):
        self.requests.append(('clearance', np.asarray(position).tolist(), float(distance)))
        return float(distance)

    def try_step_no_sliding(self, start, target):
        self.requests.append(('step', np.asarray(start).tolist(), np.asarray(target).tolist()))
        return np.asarray(target, dtype=float).copy()


class RayFixture:
    """Read-only analytic hits, with real Habitat Ray float conversion.

    Robot and inactive objects intercept every ray and must be ignored. Other
    human spheres remain occluders even when their centers are outside camera
    FOV. The static wall can be revealed after a moving human uncovers it.
    """
    def __init__(self, blocked=False):
        self.pathfinder = Support(blocked)
        self.time = 0.
        self.rays = []
        self.camera = Transform(translation=[.08, .82, .06])
        self.actors = []
        for i, pos in enumerate(([.1, 0., .1], [-.45, 0., -1.05],
                                 [.45, 0., -1.65], [1.65, 0., -.8], [8., 0., 8.])):
            obj = types.SimpleNamespace(object_id=100+i, link_object_ids={200+i:0},
                                        transformation=Transform())
            self.actors.append(types.SimpleNamespace(base_pos=np.asarray(pos), sim_obj=obj))
        self.agents_mgr = types.SimpleNamespace(articulated_agents_iter=self.actors)
        node = types.SimpleNamespace(absolute_transformation=lambda:self.camera)
        self._sensors = {'agent_0_articulated_agent_jaw_depth':
                         types.SimpleNamespace(_sensor_object=types.SimpleNamespace(node=node))}

    def get_agent_data(self, index):
        return types.SimpleNamespace(articulated_agent=self.actors[index])

    def get_world_time(self):
        return self.time

    def cast_ray(self, ray, max_distance):
        origin = np.asarray(ray.origin, dtype=float)
        direction = np.asarray(ray.direction, dtype=float)
        self.rays.append((origin.copy(), direction.copy(), float(max_distance)))
        hits = []

        def add(object_id, distance):
            if 0. <= distance <= max_distance:
                hits.append(types.SimpleNamespace(object_id=object_id, ray_distance=float(distance),
                    point=np.asarray(origin+direction*distance, dtype=np.float32)))

        add(100, .01)  # Robot self-occlusion is excluded by contract.
        add(104, .02)  # Inactive actor must not occlude or become a static wall.
        if direction[2] < -1e-9:
            distance = (-1.95-origin[2])/direction[2]
            p = origin+direction*distance
            if -.85 <= p[0] <= .85 and .2 < p[1] < 1.45:
                add(700, distance)
        for i in (1, 2, 3):
            center = self.actors[i].base_pos+[0., .9, 0.]
            delta = origin-center
            a = float(direction@direction)
            b = float(delta@direction)
            c = float(delta@delta)-.31**2
            discriminant = b*b-a*c
            if discriminant >= 0.:
                roots = [(-b-math.sqrt(discriminant))/a, (-b+math.sqrt(discriminant))/a]
                for distance in roots:
                    add(100+i, distance)
        return types.SimpleNamespace(hits=sorted(hits, key=lambda h:h.ray_distance))


def runtime(scene='review_scene_a', blocked=False):
    sim = RayFixture(blocked)
    return types.SimpleNamespace(sim=sim, count=3,
        observation_config=ObservationConfig(120., 120., 2.8),
        episode=types.SimpleNamespace(scene_id=scene, info={'graph_navmesh':scene+'.navmesh'}))


class AccelerationScientificDifferential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from eventwm_graph.partial.fast_observer import FastGeometryObserver
        cls.fast_class = FastGeometryObserver
        cls.reference_class = accepted_observer_class()

    def pair(self, rt, resolution=.4, shared_cache=None):
        # Candidate/probe vectorization still runs with the scalar ray API;
        # actual native batch ABI parity is tested on real simulators.
        expected = self.reference_class(rt, resolution, {})
        actual = self.fast_class(rt, resolution, {} if shared_cache is None else shared_cache,
                                 native_batch=False)
        return expected, actual

    def exact_rows(self, expected, actual, timestamp=None):
        start = len(expected.sim.rays)
        support_start = len(expected.sim.pathfinder.requests)
        expected_row = expected.observe(timestamp)
        middle = len(expected.sim.rays)
        support_middle = len(expected.sim.pathfinder.requests)
        actual_row = actual.observe(timestamp)
        # Wavefront batching may alter call order. Every ray used by the frozen
        # short-circuit implementation must retain EXACT native-converted start,
        # direction and double clipping/decision distance in the optimized path.
        def signature(ray):
            origin, direction, distance = ray
            return origin.tobytes(), direction.tobytes(), distance.hex()
        before = {signature(ray) for ray in expected.sim.rays[start:middle]}
        after = {signature(ray) for ray in expected.sim.rays[middle:]}
        self.assertLessEqual(before, after)
        self.assertEqual(expected.sim.pathfinder.requests[support_start:support_middle],
                         expected.sim.pathfinder.requests[support_middle:])
        self.assertEqual(expected_row, actual_row)
        self.assertEqual(expected.seen, actual.seen)
        return expected_row, actual_row

    def test_full_causal_rows_graphs_and_queries_match_across_moving_occlusion(self):
        rt = runtime()
        expected, actual = self.pair(rt)
        graphs = [PartialGraph(.4, 0.), PartialGraph(.4, 0.)]
        saw_free = saw_wall = saw_visible = saw_hidden = False
        for index, angle in enumerate((0., .18, -.22, .65, 1.4, -.1)):
            rt.sim.time = index*.048
            tilt = .11
            pitch = np.array([[1., 0., 0.], [0., math.cos(tilt), -math.sin(tilt)],
                              [0., math.sin(tilt), math.cos(tilt)]])
            rt.sim.camera.matrix = y_rotation(angle) @ pitch
            rt.sim.camera.translation = np.array([.08+index*.025, .82, .06])
            rt.sim.actors[1].base_pos = np.array([-.45+index*.22, 0., -1.05])
            rt.sim.actors[0].sim_obj.transformation = Transform(y_rotation(angle+.25))
            rows = self.exact_rows(expected, actual)
            for graph, row in zip(graphs, rows):
                graph.update(row)
            self.assertEqual(graphs[0].snapshot(), graphs[1].snapshot())
            left = generate_queries(graphs[0], rows[0]['tracks'], rows[0], [.1, 0., -2.], limit=8)
            right = generate_queries(graphs[1], rows[1]['tracks'], rows[1], [.1, 0., -2.], limit=8)
            self.assertEqual(left, right)
            saw_free |= any(r['source']=='visible_native_cspace_square_and_clearance_halo'
                            for r in rows[0]['static_updates'])
            saw_wall |= any(r['source']=='observed_static_wall_inflated_025m'
                            for r in rows[0]['static_updates'])
            saw_visible |= any(t['visible_now'] for t in rows[0]['tracks'])
            saw_hidden |= any(not t['visible_now'] for t in rows[0]['tracks'])
        self.assertTrue(saw_free)
        self.assertTrue(saw_wall)
        self.assertTrue(saw_visible)
        self.assertTrue(saw_hidden)

    def test_timestamp_memoization_returns_detached_legal_row(self):
        rt = runtime()
        expected, actual = self.pair(rt)
        initial, _ = self.exact_rows(expected, actual)
        leaked_copy = actual.observe()
        leaked_copy['camera_origin'][0] = 999.
        leaked_copy['static_updates'][0]['height'] = 999.
        rt.sim.actors[1].base_pos += np.array([10., 0., 0.])
        self.assertEqual(initial, actual.observe())
        self.assertEqual(initial, expected.observe())

    def test_branch_restore_prevents_same_clock_visibility_and_tracker_reuse(self):
        rt = runtime()
        expected, actual = self.pair(rt)
        self.exact_rows(expected, actual, 0.)
        saved = [expected.state(), actual.state()]
        branch_a, _ = self.exact_rows(expected, actual, .048)
        expected.restore(saved[0])
        actual.restore(saved[1])
        rt.sim.camera.matrix = y_rotation(math.pi)
        rt.sim.actors[1].base_pos += np.array([1., 0., 0.])
        branch_b, _ = self.exact_rows(expected, actual, .048)
        self.assertNotEqual(branch_a, branch_b)
        self.assertTrue(any(not t['visible_now'] for t in branch_b['tracks']))
        self.assertEqual(saved[0]['seen'], saved[1]['seen'])

    def test_private_certificates_do_not_seed_legal_unknown_space(self):
        rt = runtime()
        expected, actual = self.pair(rt)
        private_key = (3, 3)  # Behind the articulated camera, on valid support.
        self.assertIsNotNone(expected._support(private_key, 0.))
        self.assertIsNotNone(actual._support(private_key, 0.))
        left, right = self.exact_rows(expected, actual)
        self.assertNotIn(list(private_key), [record['cell'] for record in left['static_updates']])
        self.assertNotIn(private_key, actual.seen)
        graph = PartialGraph(.4, 0.)
        graph.update(right)
        self.assertNotIn(private_key, graph.cells)
        self.assertEqual(set(left), {'timestamp', 'robot_pose', 'tracks', 'camera_origin',
            'camera_rotation', 'static_updates', 'sensor_protocol', 'observation_valid'})

    def test_private_cache_scene_and_resolution_do_not_cross_contaminate(self):
        shared = {}
        first = self.fast_class(runtime(), .4, shared, native_batch=False)
        self.assertIsNotNone(first._support((0, 0), 0.))
        second = self.fast_class(runtime('review_scene_b', blocked=True), .4, shared, native_batch=False)
        self.assertIsNone(second._support((0, 0), 0.))
        finer = self.fast_class(runtime(), .2, shared, native_batch=False)
        center, _ = finer._support((0, 0), 0.)
        np.testing.assert_array_equal(center, [.1, 0., .1])

    def test_clearance_threshold_and_target_human_exclusion_match_exactly(self):
        rt = runtime()
        expected, actual = self.pair(rt)
        origin = np.array([0., .8, 0.])
        target = np.array([0., .8, -2.])
        for distance in (1.984999999, 1.985, 1.985000001):
            rt.sim.cast_ray = lambda *args, distance=distance, **kwargs:types.SimpleNamespace(
                hits=[types.SimpleNamespace(object_id=101, ray_distance=distance)])
            self.assertEqual(expected.clear_ray(origin, target), actual.clear_ray(origin, target))
            self.assertEqual(expected.clear_ray(origin, target, {101}),
                             actual.clear_ray(origin, target, {101}))
            self.assertEqual(actual.clear_ray(origin, target), distance >= 2.-.015)
            self.assertTrue(actual.clear_ray(origin, target, {101}))

    def test_immutable_probe_coordinates_equal_original_expression_bitwise(self):
        rt = runtime()
        for resolution in (.1, .2, .4):
            expected, actual = self.pair(rt, resolution)
            for height in (0., -.049, .051, -3.2405707836151123):
                for key in ((0, 0), (-1, -1), (2, -3)):
                    center, points = expected._support(key, height)
                    radius = .25+resolution/math.sqrt(2)
                    halo = np.array([center+[math.cos(a)*radius, 0., math.sin(a)*radius]
                        for a in np.linspace(0., 2*math.pi, 16, endpoint=False)])
                    probes = np.concatenate([points[[0, 1, 2, 3, 4]], halo])
                    frozen = np.concatenate([probes+[0., .025, 0.], probes+[0., .7, 0.],
                                             probes+[0., 1.4, 0.]])
                    _, optimized = actual._targets(key, height)
                    np.testing.assert_array_equal(frozen, optimized)
                    self.assertEqual(frozen.tobytes(), optimized.tobytes())
                    self.assertFalse(optimized.flags.writeable)
                    with self.assertRaises(ValueError):
                        optimized[0, 0] = 999.

    def test_flattened_frustum_batch_has_exact_reference_boundary_decisions(self):
        frustum = self.reference_class.observe.__globals__['in_frustum']
        config = ObservationConfig(120., 90., 3.)
        rng = np.random.default_rng(994730)
        origin = np.array([1.1834431, -3.2405707836151123, -.124])
        for iteration in range(12):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(rotation) < 0.:
                rotation[:, 0] *= -1.
            local = rng.uniform(-1., 1., size=(64, 63, 3))
            local[..., 2] = -rng.uniform(.1, 1.2, size=(64, 63))
            # Adjacent representable points straddle the exact horizontal,
            # vertical, forward and range boundaries. No epsilon is accepted.
            direction = (-np.inf, np.inf, 0.)[iteration % 3]
            if iteration % 4 == 0:
                edge = -local[..., 2]*math.tan(math.radians(config.horizontal_fov_deg/2))
                local[..., 0] = np.nextafter(edge, direction)
            elif iteration % 4 == 1:
                edge = -local[..., 2]*math.tan(math.radians(config.vertical_fov_deg/2))
                local[..., 1] = np.nextafter(edge, direction)
            elif iteration % 4 == 2:
                local[..., :2] = 0.
                local[..., 2] = -np.nextafter(1e-6, direction)
            else:
                local[..., 2] = -2.
                local *= np.nextafter(config.max_range_m, direction)/np.linalg.norm(local, axis=-1)[..., None]
            world = local @ rotation.T+origin
            scalar = np.array([frustum(points, origin, rotation, config) for points in world])
            flattened = frustum(world.reshape(-1, 3), origin, rotation, config).reshape(64, 63)
            np.testing.assert_array_equal(scalar, flattened)

    def test_batch_lengths_preserve_original_scalar_norm_bits(self):
        from eventwm_graph.partial.fast_observer import ray_lengths
        rng = np.random.default_rng(91037)
        vectors = rng.normal(size=(40000, 3))*rng.uniform(0., 15., size=(40000, 1))
        axes = np.vstack([np.eye(3), -np.eye(3)])
        edges = np.concatenate([axes*scale for scale in (0., 1e-7, .015, .25, 1., 10.)])
        vectors = np.concatenate([vectors, edges, np.nextafter(edges, np.inf),
                                  np.nextafter(edges, -np.inf)])
        scalar = np.array([np.linalg.norm(vector) for vector in vectors])
        batched = ray_lengths(vectors)
        np.testing.assert_array_equal(scalar, batched)
        self.assertEqual(scalar.tobytes(), batched.tobytes())

    def test_static_index_preserves_first_evaluation_across_height_and_root_changes(self):
        caches = [{}, {}]
        for root_index, base_height in enumerate((.041, .049, .151)):
            rt = runtime()
            rt.sim.actors[0].base_pos[1] = base_height
            rt.sim.camera.translation[1] += base_height
            rt.sim.camera.matrix = y_rotation(root_index*.2)
            expected = self.reference_class(rt, .4, caches[0])
            actual = self.fast_class(rt, .4, caches[1], native_batch=False)
            self.exact_rows(expected, actual, root_index+.01)

    def test_surface_ties_and_float32_inflation_boundary_keep_exact_raw_updates(self):
        boundary = np.float32(.35)
        for x in (np.nextafter(boundary, np.float32(-np.inf)), boundary,
                  np.nextafter(boundary, np.float32(np.inf))):
            for tied_order in ((700, 101), (101, 700)):
                with self.subTest(x=float(x), tied_order=tied_order):
                    rt = runtime(blocked=True)
                    expected, actual = self.pair(rt, .2)

                    def cast(ray, max_distance):
                        rt.sim.rays.append((np.asarray(ray.origin, dtype=float),
                                            np.asarray(ray.direction, dtype=float), float(max_distance)))
                        return types.SimpleNamespace(hits=[types.SimpleNamespace(
                            object_id=object_id, ray_distance=.8,
                            point=np.array([x, 1., .1], dtype=np.float32)) for object_id in tied_order])

                    rt.sim.cast_ray = cast
                    row, _ = self.exact_rows(expected, actual)
                    walls = [u for u in row['static_updates']
                             if u['source']=='observed_static_wall_inflated_025m']
                    if tied_order[0] == 101:
                        self.assertEqual(walls, [])
                    else:
                        self.assertGreater(len(walls), 0)
                        # Every sparse ray sees the same wall witness; raw
                        # multiplicity is retained instead of deduplicated.
                        self.assertEqual(len(walls) % 105, 0)


class AccelerationLiveBranchDifferential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from eventwm_graph.partial.fast_execution import LiveGraphPrefixFuture
        root = Path(__file__).resolve().parents[2]
        source = subprocess.run(
            ['git', 'show', ACCEPTED_COMMIT+':eventwm_graph/partial/execution.py'],
            cwd=root, check=True, capture_output=True).stdout
        if hashlib.sha256(source).hexdigest() != '889d771263177cef59db907b462a727bf22b708e1e81d03f84303c930c10e63b':
            raise AssertionError('Frozen execution identity changed')
        module = types.ModuleType('eventwm_graph.partial._reviewer_a_frozen_execution')
        module.__package__ = 'eventwm_graph.partial'
        exec(compile(source, 'accepted:'+ACCEPTED_COMMIT+'/execution.py', 'exec'), module.__dict__)
        cls.reference_class = module.GraphPrefixFuture
        cls.fast_class = LiveGraphPrefixFuture

    def evaluate(self, executor_class, horizon, obstruction, collision_enabled):
        import magnum as mn
        import falcon.additional_action as native_actions
        from eventwm_graph.partial.query import TURN_STEP, FORWARD_STEP
        from eventwm_graph.robotmap.execution import signed_yaw

        calls = []

        class Robot:
            def __init__(self):
                self.sim_obj = types.SimpleNamespace(
                    transformation=mn.Matrix4.translation(mn.Vector3(.1, 0., .1)),
                    joint_positions=np.zeros(2), joint_velocities=np.zeros(2))

            @property
            def base_pos(self):
                return np.asarray(self.sim_obj.transformation.translation)

            def update(self):
                calls.append(('update', self.base_pos.tolist(), signed_yaw(self)))

        robot = Robot()
        sim = types.SimpleNamespace(index=0, get_agent_data=lambda i:
                                    types.SimpleNamespace(articulated_agent=robot))

        class Collision:
            _ever_collide = False

            def update_metric(self, **kwargs):
                self._ever_collide = collision_enabled and signed_yaw(robot) < -1.5*TURN_STEP
                if self._ever_collide:
                    task.should_end = True

            def get_metric(self):
                return self._ever_collide

        collision = Collision()

        class Action:
            def __init__(self, name):
                self.name = name

            def step(self, **kwargs):
                yaw = signed_yaw(robot)
                if self.name == 'forward':
                    robot.sim_obj.transformation.translation = mn.Vector3(robot.base_pos+
                        [FORWARD_STEP*math.cos(yaw), 0., -FORWARD_STEP*math.sin(yaw)])
                else:
                    yaw += TURN_STEP if self.name == 'left' else -TURN_STEP
                    robot.sim_obj.transformation = mn.Matrix4.from_(
                        mn.Matrix4.rotation_y(mn.Rad(yaw)).rotation(), mn.Vector3(robot.base_pos))
                native_actions.play_i += 1
                robot.sim_obj.joint_positions = np.array([native_actions.play_i, yaw])
                calls.append(('action', self.name, robot.base_pos.tolist(), yaw, native_actions.play_i))

        task = types.SimpleNamespace(should_end=False, is_stop_called=False,
            actions={name:Action(name) for name in ('forward', 'left', 'right')},
            measurements=types.SimpleNamespace(measures={'human_collision':collision,
                'did_multi_agents_collide':types.SimpleNamespace(update_metric=lambda **kwargs:None)}))

        class Sensor:
            def __init__(self):
                self.last = 1.
                self.count = 1
                self.path = []

            def state(self):
                return copy.deepcopy(self.__dict__)

            def restore(self, state):
                self.__dict__ = copy.deepcopy(state)

            def observe(self, timestamp):
                if timestamp <= self.last:
                    raise AssertionError('Lost causal branch restoration')
                self.last = timestamp
                self.count += 1
                self.path.append([*robot.base_pos.tolist(), signed_yaw(robot)])
                updates = []
                if obstruction and robot.base_pos[0] > .45:
                    updates.append(dict(cell=[1, 0], state=2, height=0., timestamp=timestamp,
                                        source='observed_static_wall_inflated_025m'))
                return dict(timestamp=timestamp, robot_pose=self.path[-1],
                    tracks=[dict(track_id='track_000000', position=[sim.index*.03, 0., -1.],
                        first_seen_time=1., last_seen_time=timestamp, observation_age=0.,
                        num_observations=self.count, visible_now=True)],
                    camera_origin=(robot.base_pos+[.5, .8, 0.]).tolist(),
                    camera_rotation=np.asarray(robot.sim_obj.transformation.rotation()).tolist(),
                    static_updates=updates, sensor_protocol='independent_branch_fixture',
                    observation_valid=True)

        sensor = Sensor()
        initial = dict(timestamp=1., robot_pose=[.1, 0., .1, 0.], tracks=[],
            static_updates=[dict(cell=[x, z], state=1, height=0., timestamp=1., source='fixture')
                            for x in range(-8, 16) for z in range(-8, 9)])
        graph = PartialGraph(.2, 0.)
        graph.update(initial)
        tape = executor_class.__new__(executor_class)
        tape.sim = sim; tape.task = task; tape.sensor = sensor; tape.sensor_root = sensor.state()
        tape.graph_root = graph.snapshot(); tape.root_observation = initial
        tape.robot_transform = mn.Matrix4(robot.sim_obj.transformation)
        tape.robot_joints = np.zeros(2); tape.robot_velocities = np.zeros(2); tape.gait_frame = 0
        tape.times = [1.+i*.048 for i in range(9)]; tape.pointgoal = [2., 0., .1]
        tape.prefix_compliant = 1; tape.prefix_steps = 2; tape.max_steps = horizon
        tape.nodes = []; tape.performance = {}
        tape.rt = types.SimpleNamespace(action_names={a:a for a in task.actions}, episode=types.SimpleNamespace(),
            config=types.SimpleNamespace(habitat=types.SimpleNamespace(
                environment=types.SimpleNamespace(max_episode_steps=20))))

        def restore_humans(index):
            sim.index = index
            calls.append(('restore_humans', index))

        tape._restore_humans = restore_humans
        schedules = [('left',), ('forward',), ('left', 'left', 'forward', 'forward'),
            ('left', 'left', 'right'), ('left', 'right', 'right', 'forward'),
            ('right', 'right', 'forward', 'left'), ('forward', 'forward', 'forward', 'right'),
            ('left', 'left', 'forward', 'forward', 'forward', 'forward')]
        queries = [dict(query_id='independent_query_'+str(i), waypoints=[[.1, 0., .1]], final_yaw=0.,
            executor_version='known_space_discrete_route_follower_1',
            compiled=dict(actions=list(actions), position_tolerance=.155, yaw_tolerance=.042))
            for i, actions in enumerate(schedules)]

        def labels(runtime, timestamp):
            distance = 1.2 if sim.index % 2 else .9
            return dict(timestamp=timestamp, official_center_distances_m=[distance],
                euclidean_center_distances_m=[distance], official_dmin_m=distance,
                euclidean_dmin_m=distance, psc_violation=distance < 1.,
                euclidean_ps_violation=distance < 1., human_mesh_contact=collision.get_metric(),
                static_obstacle_contact=False)

        with patch.dict(executor_class.evaluate_many.__globals__, step_labels=labels), \
                patch('falcon.additional_action.play_i', 0):
            outcomes = tape.evaluate_many(queries)
        return outcomes, tape.nodes, calls

    def test_live_unary_edges_match_frozen_restoration_at_forks_and_termination(self):
        for horizon, obstruction, collision in ((1, False, False), (4, False, False),
                                                (6, True, False), (6, False, True)):
            with self.subTest(horizon=horizon, obstruction=obstruction, collision=collision):
                expected = self.evaluate(self.reference_class, horizon, obstruction, collision)
                actual = self.evaluate(self.fast_class, horizon, obstruction, collision)
                self.assertEqual(expected, actual)
                reasons = {outcome['summary']['termination'] for outcome in actual[0]}
                self.assertIn('query_complete', reasons)
                if horizon < 6:
                    self.assertIn('query_timeout', reasons)
                if obstruction:
                    self.assertIn('route_invalidated', reasons)
                if collision:
                    self.assertIn('human_collision', reasons)


class AccelerationGraphDifferential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2]/'eventwm_graph/partial/map.py'
        if hashlib.sha256(path.read_bytes()).hexdigest() != '41eec5df355c257ab6f40b1093f027d50b737accb5c899d0cafec51e946b0291':
            raise AssertionError('Accepted graph reference identity changed')

    @staticmethod
    def initial_graph():
        graph = PartialGraph(.2, 0.)
        graph.update(dict(timestamp=1., robot_pose=[.1, 0., .1, 0.], tracks=[],
            static_updates=[dict(cell=[x, 0], state=1, height=0., timestamp=1., source='fixture')
                            for x in (0, 1, 2)]))
        graph.global_nodes = {name:dict(id=name, cell=[cell, 0], position=[position, 0., .1],
            kind='fixture', created_at=1.) for name, cell, position in (('a', 0, .1), ('b', 2, .5))}
        graph.global_edges = [dict(source='a', target='b', path=[[.1, 0., .1], [.5, 0., .1]],
                                  length=.4, revision=1, certificate='known_free_cell_supercover')]
        return graph

    @staticmethod
    def row(timestamp, height=0., key=(2, 0), pose=(.1, 0., .1, 0.)):
        return dict(timestamp=timestamp, robot_pose=list(pose), tracks=[], static_updates=[
            dict(cell=list(key), state=1, height=height, timestamp=timestamp, source='fixture')])

    def test_tiny_height_debt_is_recertified_on_later_distant_revision(self):
        from eventwm_graph.partial.fast_graph import IncrementalPartialGraph
        for restore_between in (False, True):
            with self.subTest(restore_between=restore_between):
                expected = self.initial_graph()
                actual = IncrementalPartialGraph.restore(expected.snapshot())
                for row in (self.row(2., .21999999999), self.row(3., .22000000001),
                            self.row(4., 0., (10, 10))):
                    self.assertEqual(expected.update(copy.deepcopy(row)), actual.update(copy.deepcopy(row)))
                    self.assertEqual(expected.snapshot(), actual.snapshot())
                    if restore_between:
                        expected = PartialGraph.restore(expected.snapshot())
                        actual = IncrementalPartialGraph.restore(actual.snapshot())
                self.assertEqual(expected.global_edges, [])

    def test_moved_singleton_debt_survives_without_revision_and_after_restore(self):
        from eventwm_graph.partial.fast_graph import IncrementalPartialGraph
        for restore_between in (False, True):
            with self.subTest(restore_between=restore_between):
                expected = self.initial_graph()
                actual = IncrementalPartialGraph.restore(expected.snapshot())
                moved = self.row(3., pose=(.15, 0., .1, 0.))
                moved['static_updates'] = []
                observations = (self.row(2., .23, (0, 0)), moved,
                                self.row(4., 0., (10, 10), pose=(.15, 0., .1, 0.)))
                for row in observations:
                    self.assertEqual(expected.update(copy.deepcopy(row)), actual.update(copy.deepcopy(row)))
                    self.assertEqual(expected.snapshot(), actual.snapshot())
                    if restore_between:
                        expected = PartialGraph.restore(expected.snapshot())
                        actual = IncrementalPartialGraph.restore(actual.snapshot())
                self.assertEqual(expected.global_edges, [])


class AccelerationValidationDifferential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2]/'eventwm_graph/partial/storage.py'
        if hashlib.sha256(path.read_bytes()).hexdigest() != '76ef55901fbdb3a0b672ecaf1ba5f2a926853790d86de06bbce6b3fbc54b8137':
            raise AssertionError('Accepted validator reference identity changed')

    def make_root(self, directory):
        # Reuse the accepted independent review's legal fixture, then build
        # three distinct compiled heading queries sharing one prefix endpoint.
        from test_stage1_review import fixture, row, step_label
        from eventwm_graph.config import digest
        from eventwm_graph.partial.metrics import summarize
        from eventwm_graph.partial.query import compile_query, TURN_STEP
        from eventwm_graph.partial.storage import seal_root
        record, tape, queries, outcomes = fixture()
        template = queries[0]
        initial = record['history'][-1]
        graph = PartialGraph.restore(record['partial_graph'])
        queries = []
        for yaw in (TURN_STEP, 2*TURN_STEP, -TURN_STEP):
            query = {key:copy.deepcopy(value) for key, value in template.items()
                     if key not in ('query_id', 'compiled', 'route_distance')}
            query['waypoints'] = [initial['robot_pose'][:3]]
            query['final_yaw'] = yaw
            query['provenance'] = [dict(source='current_position')]
            query['compiled'] = compile_query(graph, query, initial)
            self.assertTrue(query['compiled']['ok'])
            query['route_distance'] = query['compiled']['route_distance']
            query['query_id'] = digest(query)
            queries.append(query)
        nodes = []
        paths = [[0], [0, 1], [2]]
        for query_index, action_index, parent in ((0, 0, None), (1, 1, 0), (2, 0, None)):
            query = queries[query_index]
            graph = PartialGraph.restore(record['partial_graph'])
            if parent is not None:
                graph.update(nodes[parent]['observation'])
            timestamp = 1.+(action_index+1)*.048
            observation = row(timestamp, query['compiled']['planned_poses'][action_index+1])
            nodes.append(dict(node_id=len(nodes), parent=parent,
                action=query['compiled']['actions'][action_index], observation=observation,
                graph_delta=graph.update(observation), labels=step_label(timestamp),
                future_index=action_index+1))
        tape.nodes = nodes
        tape.times = [1., 1.048, 1.096]
        tape.truth = [[], [], []]
        outcomes = []
        for query, path in zip(queries, paths):
            selected = [nodes[i] for i in path]
            observations = [n['observation'] for n in selected]
            poses = [initial['robot_pose'], *[r['robot_pose'] for r in observations]]
            outcomes.append(dict(query_id=query['query_id'], node_ids=path,
                summary=summarize([n['labels'] for n in selected], poses, 1.,
                    query['compiled']['actions'], 'query_complete', query, record['pointgoal']),
                successor=dict(timestamp=observations[-1]['timestamp'], robot_pose=poses[-1], tracks=[],
                               graph_revision=record['partial_graph']['revision']),
                executor_version=query['executor_version'], execution_engine='independent_fixture'))
        record['collection_config']['queries_per_root'] = 3
        return Path(seal_root(directory, record, tape, queries, outcomes, min_free_bytes=0)['folder'])

    def test_every_shared_node_and_successor_passes_same_complete_validator(self):
        from eventwm_graph.partial.storage import validate_root
        from eventwm_graph.partial.fast_validation import validate_root as fast_validate
        with tempfile.TemporaryDirectory(prefix='independent_acceleration_validation_') as directory:
            root = self.make_root(directory)
            self.assertEqual(validate_root(root), fast_validate(root))

    def test_required_corruption_checks_remain_active(self):
        from eventwm_graph.config import file_digest, read_json, write_json
        from eventwm_graph.stage1.storage import read_gzip, write_gzip
        from eventwm_graph.partial.storage import validate_root
        from eventwm_graph.partial.fast_validation import validate_root as fast_validate
        mutations = [
            ('future_observations.json.gz', lambda x:x[0]['graph_delta'].__setitem__('revision', 99)),
            ('future_observations.json.gz', lambda x:x[1]['graph_delta'].__setitem__('changed_cells', [[99, 99]])),
            ('outcomes.json.gz', lambda x:x[0]['successor'].__setitem__('graph_revision', 99)),
            ('outcomes.json.gz', lambda x:x[1]['summary'].__setitem__('executed_steps', 99)),
            ('future_observations.json.gz', lambda x:x[2].__setitem__('parent', 0)),
            ('future_observations.json.gz', lambda x:x[1]['observation'].__setitem__('timestamp', 1.048)),
            ('privileged_step_labels.json.gz', lambda x:x[0].__setitem__('psc_violation', True)),
            ('queries.json.gz', lambda x:x[0].__setitem__('final_yaw', 9.)),
            ('legal_root.json.gz', lambda x:x['partial_graph']['cells'][0].__setitem__('height', 7.)),
        ]
        with tempfile.TemporaryDirectory(prefix='independent_acceleration_corruption_') as directory:
            root = self.make_root(directory)
            original_manifest = read_json(root/'manifest.json')
            for name, mutate in mutations:
                with self.subTest(member=name, mutation=mutate):
                    original = read_gzip(root/name)
                    changed = copy.deepcopy(original)
                    mutate(changed)
                    write_gzip(root/name, changed)
                    manifest = copy.deepcopy(original_manifest)
                    manifest['files'][name] = file_digest(root/name)
                    write_json(root/'manifest.json', manifest)
                    with self.assertRaises(ValueError):
                        validate_root(root)
                    with self.assertRaises(ValueError):
                        fast_validate(root)
                    write_gzip(root/name, original)
                    original_manifest['files'][name] = file_digest(root/name)
                    write_json(root/'manifest.json', original_manifest)


if __name__ == '__main__':
    unittest.main()
