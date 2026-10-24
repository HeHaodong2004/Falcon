"""Independent equivalence and invalidation tests for exact body memoization."""
import math
import unittest
from unittest.mock import patch

import numpy as np

from eventwm_graph.partial.map import PartialGraph, KNOWN_FREE, KNOWN_OCCUPIED
from eventwm_graph.partial.query import static_body_clearance, generate_queries, compile_query


# Original audited all-gait bounds, intentionally independent of helper caches.
BOUNDS = (-.646495, .584202, -.242846, .238225)


def original_clearance(graph, position, yaw):
    """Pre-optimization vector/public-API certificate used as reference."""
    spacing = min(.10, graph.resolution/2)
    n = int(math.ceil((BOUNDS[1]-BOUNDS[0])/spacing))+1
    c, s = math.cos(yaw), math.sin(yaw)
    for x in np.linspace(BOUNDS[0], BOUNDS[1], n):
        p = np.asarray(position)+[x*c, 0., -x*s]
        if graph.state(graph.key(p)) != KNOWN_FREE or not graph.point_free(p):
            return False
    return True


def update(graph, timestamp, cells=(), ego=(5., 0., 5.)):
    graph.update(dict(timestamp=timestamp, robot_pose=[*ego, 0.], tracks=[],
        static_updates=[dict(cell=list(k), state=state, height=height, timestamp=timestamp,
                             source='independent_legal_observation') for k, state, height in cells]))


def room(resolution=.2, half_extent=2.):
    graph = PartialGraph(resolution, 0.)
    limit = math.ceil(half_extent/resolution)
    update(graph, 0., [((x,z), KNOWN_FREE, 0.) for x in range(-limit,limit+1)
                      for z in range(-limit,limit+1)])
    return graph


class IndependentClearanceCacheReview(unittest.TestCase):
    def check_equivalent(self, graph, position, yaw):
        expected = original_clearance(graph, position, yaw)
        self.assertEqual(static_body_clearance(graph, position, yaw), expected)
        self.assertEqual(static_body_clearance(graph, position, yaw), expected)
        return expected

    def test_randomized_legal_graphs_match_original_certificate(self):
        rng = np.random.default_rng(310210099)
        results = []
        for resolution in (.6101305376787912/9, .1, .2, .3):
            for variant in range(6):
                graph = PartialGraph(resolution, 0.)
                limit = math.ceil(2./resolution)
                cells = []
                for x in range(-limit,limit+1):
                    for z in range(-limit,limit+1):
                        draw = rng.random()
                        if variant and draw < .015:continue  # UNKNOWN stays absent.
                        state = KNOWN_OCCUPIED if variant and draw < .03 else KNOWN_FREE
                        height = float(.02*math.sin(x/7.)+.015*math.cos(z/5.)) if variant else 0.
                        cells.append(((x,z), state, height))
                update(graph, 0., cells)
                for _ in range(128):
                    position = [float(rng.uniform(-1.2,1.2)),
                                float(rng.choice([0., .219999999, .220000001, -.22])),
                                float(rng.uniform(-1.2,1.2))]
                    yaw = float(rng.uniform(-math.pi,math.pi))
                    results.append(self.check_equivalent(graph, position, yaw))
        self.assertTrue(any(results));self.assertTrue(not all(results))

    def test_new_wall_invalidates_cached_positive(self):
        graph = room();pose = [.1, 0., .1]
        self.assertTrue(self.check_equivalent(graph, pose, 0.))
        update(graph, 1., [((0,0), KNOWN_OCCUPIED, 0.)])
        self.assertFalse(self.check_equivalent(graph, pose, 0.))

    def test_new_known_space_invalidates_cached_unknown_failure(self):
        graph = PartialGraph(.2, 0.);update(graph, 0.)
        pose = [.1, 0., .1]
        self.assertFalse(self.check_equivalent(graph, pose, 0.))
        update(graph, 1., [((x,z), KNOWN_FREE, 0.) for x in range(-8,9) for z in range(-8,9)])
        self.assertTrue(self.check_equivalent(graph, pose, 0.))

    def test_height_change_invalidates_cached_positive(self):
        graph = room();pose = [.1, 0., .1]
        self.assertTrue(self.check_equivalent(graph, pose, 0.))
        update(graph, 1., [((0,0), KNOWN_FREE, .220001)])
        self.assertFalse(self.check_equivalent(graph, pose, 0.))

    def test_sub_revision_height_change_cannot_keep_threshold_result(self):
        graph = room();pose = [.1, .22, .1]
        self.assertTrue(self.check_equivalent(graph, pose, 0.))
        update(graph, 1., [((0,0), KNOWN_FREE, -5e-9)])
        self.assertFalse(self.check_equivalent(graph, pose, 0.))

    def test_ego_change_invalidates_with_unchanged_static_revision(self):
        graph = room();pose = [.199-BOUNDS[0], 0., .1]
        ego = [.199, 0., .1]
        update(graph, 1., [((0,0), KNOWN_FREE, .23)], ego=ego)
        self.assertTrue(self.check_equivalent(graph, pose, 0.))
        revision = graph.revision
        update(graph, 2., ego=[ego[0]+.01, 0., .1])
        self.assertEqual(graph.revision, revision)
        self.assertFalse(self.check_equivalent(graph, pose, 0.))
        update(graph, 3., ego=ego)
        self.assertTrue(self.check_equivalent(graph, pose, 0.))

    def test_singleton_ego_never_fills_unknown_body_support(self):
        graph = PartialGraph(.2, 0.);update(graph, 0., ego=(.1,0.,.1))
        self.assertFalse(self.check_equivalent(graph, [.1,0.,.1], 0.))
        update(graph, 1., ego=(.2,0.,.1))
        self.assertFalse(self.check_equivalent(graph, [.1,0.,.1], 0.))

    def test_nearby_positions_across_unknown_boundary_have_distinct_keys(self):
        graph = PartialGraph(.05, 0.)
        update(graph, 0., [((x,z), KNOWN_FREE, 0.) for x in range(40) for z in range(-5,6)])
        before = [-BOUNDS[0]-1e-12, 0., .1]
        after = [-BOUNDS[0]+1e-12, 0., .1]
        self.assertFalse(self.check_equivalent(graph, before, 0.))
        self.assertTrue(self.check_equivalent(graph, after, 0.))
        self.assertEqual(len(graph._body_cache), 2)

    def test_snapshot_restore_does_not_share_memoized_results(self):
        graph = room();pose = [.1,0.,.1]
        self.assertTrue(self.check_equivalent(graph, pose, 0.))
        restored = PartialGraph.restore(graph.snapshot())
        update(restored, 1., [((0,0), KNOWN_OCCUPIED, 0.)])
        self.assertFalse(self.check_equivalent(restored, pose, 0.))
        self.assertTrue(self.check_equivalent(graph, pose, 0.))

    def test_nan_height_or_pose_remains_fail_closed_like_original(self):
        graph = room()
        self.assertFalse(self.check_equivalent(graph, [.1,float('nan'),.1], 0.))
        update(graph, 1., [((0,0), KNOWN_FREE, float('nan'))])
        self.assertFalse(self.check_equivalent(graph, [.1,0.,.1], 0.))

    def test_default_band_parameters_preserve_full_queries(self):
        graph = room(.2, 4.)
        ego = dict(timestamp=0., robot_pose=[.1,0.,.1,0.])
        human = dict(track_id='track_000000', position=[1.8,0.,.1], estimated_velocity=[.2,0.,0.],
                     velocity_valid=True, visible_now=True, valid=True, first_seen_time=-.2,
                     last_seen_time=0., observation_age=0., num_observations=4)
        default, _ = generate_queries(graph, [human], ego, [3.,0.,3.], limit=32)
        explicit, _ = generate_queries(PartialGraph.restore(graph.snapshot()), [human], ego, [3.,0.,3.],
                                       limit=32, safe_margin=.2, intrusion_radius=.96)
        bands = {p['intended_band'] for q in default for p in q['provenance'] if p['source']=='human'}
        self.assertEqual(bands, {'safe','intrusion'})
        self.assertEqual(default, explicit)

    def test_float32_inputs_preserve_original_threshold_arithmetic(self):
        graph = room()
        for py in (0., .22, .220000001, -.22):
            for yaw in (0., math.pi/2, -math.pi+.00001):
                self.check_equivalent(graph, np.asarray([.1,py,.1],dtype=np.float32), yaw)

    def test_complete_schedules_match_original_for_ordered_points_and_headings(self):
        graph = room(.2, 3.)
        ego = dict(timestamp=0., robot_pose=[.1,0.,.1,0.])
        for waypoints, yaw in (([[1.1,0.,.1]], math.pi/2),
                               ([[1.1,0.,.1],[-.9,0.,.1]], math.pi),
                               ([[1.1,0.,.1]], -.5),
                               ([[.1,0.,.1]], .25)):
            query = dict(waypoints=waypoints, final_yaw=yaw)
            actual = compile_query(PartialGraph.restore(graph.snapshot()), query, ego)
            with patch('eventwm_graph.partial.query.static_body_clearance', original_clearance):
                expected = compile_query(PartialGraph.restore(graph.snapshot()), query, ego)
            self.assertTrue(actual['ok'], actual)
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
