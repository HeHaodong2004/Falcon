"""Finite-world oracles and real frozen-model integration; no protected data."""
from dataclasses import replace
import inspect
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from eventwm_graph.l2_intermittent.api_example import inputs
from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
from eventwm_graph.l2_intermittent.observations import predict_observations
from eventwm_graph.l2_partial.state import ObservedGraph, EXECUTOR_VERSION
from eventwm_graph.l25 import QueryOutcomeCompiler, CompiledReference, ComputeBudget
from eventwm_graph.l25.numeric import aggregate_joint, joint_events, finite_event_mi
from eventwm_graph.l25.spatial import build_spatial
from eventwm_graph.l25.graph import batch_graphs, message_passing_logits, QUERY_FEATURES
from eventwm_graph.partial.query import physical_clearance

CHECKPOINT = Path('/research-user/workspace/projects/eventwm-falcon-l2-intermittent-500k-20261022/output/l2_intermittent_500k_20261022/training/checkpoint.pt')


def native(b, key='forward', steps=24, query_id=None):
    """Mathematical reference fixture, not a new nominal compiler."""
    p = np.tile(b.robot_pose, (steps+1, 1))
    if key == 'forward':
        p[:, 0] += np.arange(steps+1)*.08
    elif key == 'left':
        p[:, 3] += np.arange(steps+1)*.083333333333
    return dict(query_id=query_id or key, executor_version=EXECUTOR_VERSION,
                waypoints=[p[-1, :3].tolist()], final_yaw=float(p[-1, 3]),
                compiled=dict(planned_poses=p.tolist(), actions=[key]*steps, graph_revision=b.graph.version))


class NumericOracles(unittest.TestCase):
    def test_joint_union_across_time_and_humans(self):
        p = np.full((4, 3, 2, 2), 10.)
        p[0, 0, 0] = [0, 0]; p[0, 2, 1] = [0, 0]
        p[1, :, 0] = [0, 0]
        p[2, 2, 1] = [0, 0]
        r = aggregate_joint(p, np.zeros((1, 3, 4)), [3], np.array([.1, .2, .3, .4]))
        self.assertAlmostEqual(r['p_collision'][0], .6)
        np.testing.assert_allclose(r['first_collision_mass'][0], [.3, 0, .3])
        # Independent occupancy would incorrectly double count repeated worlds.
        self.assertNotAlmostEqual(r['p_collision'][0], 1-(1-.3)*(1-.3)*(1-.6))

    def test_against_stage01_body_geometry(self):
        rng = np.random.default_rng(13)
        robot = np.array([.1, 0., -.3, .9])
        positions = rng.normal(size=(40, 5, 1, 2))*2
        robots = np.tile(robot, (2, 5, 1))
        events, _ = joint_events(positions, robots)
        expected = np.array([[not physical_clearance(robot[:3], robot[3],
                         [dict(position=[p[0], 0., p[1]])], margin=0.) for p in world[:, 0]] for world in positions])
        np.testing.assert_array_equal(events[0], expected)
        np.testing.assert_array_equal(events[1], expected)

    def test_finite_bank_brute_force_reference(self):
        rng = np.random.default_rng(92)
        positions = rng.normal(size=(7, 9, 3, 2))*2
        robots = rng.normal(size=(4, 9, 4))
        weights = np.arange(1, 8)/28.
        lengths = [9, 4, 8, 2]
        result = aggregate_joint(positions, robots, lengths, weights)
        for q in range(4):
            hits, social_duration = [], []
            for world in positions:
                hit, duration = False, 0.
                for t in range(lengths[q]):
                    humans = [dict(position=[p[0], 0., p[1]]) for p in world[t]]
                    hit |= not physical_clearance(robots[q, t, :3], robots[q, t, 3], humans, margin=0.)
                    duration += .048*any(np.linalg.norm(p-robots[q, t, [0, 2]]) < 1 for p in world[t])
                hits.append(hit); social_duration.append(duration)
            self.assertAlmostEqual(result['p_collision'][q], weights@hits)
            self.assertAlmostEqual(result['expected_social_duration'][q], weights@social_duration)

    def test_no_humans_is_finite_with_explicit_infinite_separation(self):
        r = aggregate_joint(np.empty((3, 4, 0, 2)), np.zeros((2, 4, 4)), [4, 2], np.ones(3)/3)
        np.testing.assert_equal(r['p_collision'], [0, 0])
        np.testing.assert_equal(r['p_social'], [0, 0])
        self.assertEqual(r['minimum_quantiles'], [[None]*3]*2)
        np.testing.assert_allclose(r['p_no_humans'], [1, 1])

    def test_weighted_minimum_quantiles_preserve_no_human_mass(self):
        from eventwm_graph.l25.numeric import weighted_quantiles
        self.assertEqual(weighted_quantiles(np.array([.2, np.inf]), np.array([.25, .75])), [.2, None, None])

    def test_nan_padding_is_absent_not_origin(self):
        p = np.full((2, 3, 2, 2), np.nan)
        p[1, 1, 0] = [0, 0]
        r = aggregate_joint(p, np.zeros((1, 3, 4)), [3], np.array([.7, .3]))
        self.assertAlmostEqual(r['p_collision'][0], .3)

    def test_padding_beyond_query_cannot_add_risk(self):
        p = np.full((1, 6, 1, 2), 10.)
        p[0, 3:] = 0.
        r = aggregate_joint(p, np.zeros((1, 6, 4)), [3], np.ones(1))
        self.assertEqual(r['p_collision'][0], 0)

    def test_observationally_indistinguishable_worlds_have_zero_mi(self):
        self.assertAlmostEqual(finite_event_mi(np.full(16, .2), np.ones(16)/16), 0)
        self.assertAlmostEqual(finite_event_mi(np.array([0., 1.]), np.array([.5, .5])), np.log(2))


class SpatialSemantics(unittest.TestCase):
    def test_tiles_do_not_connect_disconnected_free_patches(self):
        g = ObservedGraph('split', 0, .2, 0., {(0, 0): (1, 0.), (2, 0): (1, 0.), (1, 0): (2, 0.)})
        s = build_spatial(g)
        self.assertNotEqual(s.locate([.1, .1]), s.locate([.5, .1]))
        self.assertEqual(s.edge_index.shape, (2, 0))

    def test_unknown_is_not_traversable(self):
        g = ObservedGraph('unknown', 0, .2, 0., {(0, 0): (1, 0.), (1, 0): (0, 0.)})
        s = build_spatial(g)
        self.assertNotIn(s.unknown_index, s.edge_index)
        self.assertEqual(s.locate([100, 100]), s.unknown_index)
        self.assertTrue(np.isinf(s.goal_distances([100, 0, 100])).all())

    def test_height_discontinuity_blocks_edges(self):
        g = ObservedGraph('height', 0, .2, 0., {(0, 0): (1, 0.), (1, 0): (1, 1.)})
        s = build_spatial(g)
        self.assertEqual(s.edge_index.shape[1], 0)

    def test_submillimeter_noise_merges_but_does_not_remove_cell_ownership(self):
        g = ObservedGraph('noise', 0, .2, 0., {(0, 0): (1, 0.), (1, 0): (1, .0003)})
        s = build_spatial(g)
        self.assertEqual(s.locate([.1, .1]), s.locate([.3, .1]))
        with self.assertRaises(TypeError):
            s.cell_to_node[(0, 0)] = 999


class FrozenModelIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.model = IntermittentWorldModel.load(CHECKPOINT)
        cls.history, cls.graph, row = inputs()
        cls.row = staticmethod(row)
        cls.belief = cls.model.initialize(cls.history, cls.graph)

    def setUp(self):
        self.c = QueryOutcomeCompiler(self.model)
        self.b = self.belief
        self.ctx = self.c.build_context(self.b, self.b.graph, self.b.robot_pose, [3., 0., 0.])
        self.refs = tuple(CompiledReference.from_native(native(self.b, k), self.b.timestamp) for k in ('forward', 'left'))
        self.budget = ComputeBudget(soft_budget_ms=10000.)

    def evaluate(self, refs=None, budget=None):
        return self.c.evaluate_candidates(self.ctx, self.refs if refs is None else refs, budget or self.budget)

    def test_two_stage_context_needs_no_forecast(self):
        self.assertEqual(self.c.forecast_calls, 0)
        self.assertLess(len(self.ctx.spatial.ids), self.ctx.metadata['static_cells'])
        self.assertEqual(self.ctx.timestamp, self.b.timestamp)
        self.assertEqual(self.ctx.metadata['role'], 'actual')

    def test_no_human_scene_through_actual_model_interface(self):
        # A declared zero-population mathematical scene, using frozen learned
        # weights, has a different model version from the deployed population.
        m = IntermittentWorldModel(learned=self.model.learned, law=dict(self.model.law), population=[1.])
        history = [{**r, 'tracks': []} for r in self.history]
        b = m.initialize(history, self.graph)
        compiler = QueryOutcomeCompiler(m)
        ctx = compiler.build_context(b, b.graph, b.robot_pose, [3, 0, 0])
        ref = CompiledReference.from_native(native(b), b.timestamp)
        p = compiler.evaluate_candidates(ctx, [ref], self.budget)
        self.assertEqual(p.outcomes[0]['p_nominal_collision'], 0)
        self.assertEqual(p.outcomes[0]['p_social_euclidean_proxy'], 0)
        self.assertEqual(p.outcomes[0]['minimum_separation_quantiles'], [None]*3)
        self.assertTrue(np.isfinite(p.graph['node_features']).all())

    def test_one_forecast_shared_by_every_candidate(self):
        packet = self.evaluate()
        self.assertEqual(self.c.forecast_calls, 1)
        self.assertEqual(len({o['source']['future_identity'] for o in packet.outcomes}), 1)
        self.assertTrue(packet.metadata['joint_future_shared_across_all_candidates'])

    def test_optical_vectorization_matches_actual_public_api(self):
        packet = self.evaluate()
        future = self.c._future(self.ctx, self.budget)
        for ref, out in zip(self.refs, packet.outcomes):
            direct = predict_observations(self.model, self.b, ref.native(), future)
            self.assertAlmostEqual(direct['any_probability'], out['p_any_observation'], places=14)
            np.testing.assert_allclose(direct['first_detection_mass'], out['observation_first_mass'], atol=1e-14)
            np.testing.assert_allclose(direct['endpoint_any_probability'], out['observation_endpoint_probability'], atol=1e-14)

    def test_nominal_risk_matches_direct_joint_bank(self):
        packet = self.evaluate()
        future = self.c._future(self.ctx, self.budget)
        allx = np.concatenate([future.positions, future.hidden_positions], 2)
        for ref, out in zip(self.refs, packet.outcomes):
            n = len(ref.actions)
            r = aggregate_joint(allx[:, 1:n+1], ref.poses[None, 1:n+1], [n], future.weights)
            self.assertAlmostEqual(out['p_nominal_collision'], r['p_collision'][0])
            self.assertAlmostEqual(sum(out['collision_first_mass']), out['p_nominal_collision'])
            self.assertAlmostEqual(sum(out['collision_histogram']), out['p_nominal_collision'])

    def test_query_and_gnn_permutation_equivariance(self):
        a, b = self.evaluate(), self.evaluate(tuple(reversed(self.refs)))
        ao, bo = {o['query_id']: o for o in a.outcomes}, {o['query_id']: o for o in b.outcomes}
        self.assertEqual(ao, bo)
        al, _ = message_passing_logits(a.graph)
        bl, _ = message_passing_logits(b.graph)
        np.testing.assert_allclose(al, bl[::-1], atol=1e-7)
        self.assertEqual({(x.query_id, x.event) for x in a.branches}, {(x.query_id, x.event) for x in b.branches})

    def test_absolute_timing_and_native_root_exclusion(self):
        p = self.evaluate()
        o = p.outcomes[0]
        self.assertAlmostEqual(o['collision_first_times'][0], self.b.timestamp+.048)
        self.assertAlmostEqual(o['prediction_end_time'], self.b.timestamp+24*.048)
        self.assertEqual(p.metadata['prediction_start_time'], self.b.timestamp)

    def test_horizon_censoring_never_claims_completion(self):
        p = self.evaluate(budget=replace(self.budget, horizon=.5))
        for out in p.outcomes:
            self.assertTrue(out['censored_running'])
            self.assertAlmostEqual(out['covered_duration'], .48)
            self.assertGreater(out['unsupported_duration'], 0)
            self.assertFalse(out['support']['horizon_complete'])
            self.assertIsNone(out['p_l4_query_completion'])

    def test_nominal_and_closed_loop_success_are_distinct(self):
        p = self.evaluate()
        for out in p.outcomes:
            for key in ('p_l4_query_completion', 'p_episode_success', 'p_official_psc', 'p_nominal_mesh_contact'):
                self.assertIsNone(out[key])
                self.assertFalse(out['gnn_support'][QUERY_FEATURES.index(key)])

    def test_branch_probability_conservation_and_marginalization(self):
        p = self.evaluate(budget=replace(self.budget, min_branch_probability=0.))
        future = self.c._future(self.ctx, self.budget)
        branches = list(p.branches)
        self.assertEqual(len(branches), 2)
        self.assertAlmostEqual(sum(b.probability for b in branches), 1)
        reconstructed = sum(b.probability*np.array([c.weight for c in b.successor.components]) for b in branches)
        np.testing.assert_allclose(reconstructed, future.weights, atol=1e-14)
        for o in p.outcomes:
            self.assertAlmostEqual(o['branching']['represented_mass']+o['branching']['residual_model_supported_mass'], 1.)
            self.assertIsNone(o['branching']['model_external_unknown_mass'])

    def test_pruning_keeps_residual_mass(self):
        p = self.evaluate(budget=replace(self.budget, max_branches=1, min_branch_probability=0.))
        self.assertEqual(len(p.branches), 1)
        o = next(o for o in p.outcomes if o['branching']['represented_mass'])
        self.assertGreater(o['branching']['residual_model_supported_mass'], 0)
        self.assertAlmostEqual(o['branching']['represented_mass']+o['branching']['residual_model_supported_mass'], 1)

    def test_budget_keeps_core_outputs_and_marks_unexpanded_events(self):
        p = self.evaluate(budget=replace(self.budget, soft_budget_ms=0.))
        self.assertFalse(p.branches)
        self.assertTrue(p.metadata['budget_exceeded'])
        self.assertTrue(any(o['branching']['status'] == 'BUDGET_UNEXPANDED' for o in p.outcomes))
        self.assertTrue(all(o['p_nominal_collision'] is not None for o in p.outcomes))

    def test_conditional_is_hypothetical_and_time_ordered(self):
        before = self.b.fingerprint()
        p = self.evaluate()
        for branch in p.branches:
            self.assertEqual(branch.successor.role, 'imagined')
            self.assertGreater(branch.timestamp, self.b.timestamp)
            self.assertFalse(branch.summary['bellman_transition_supported'])
            self.assertFalse(branch.root_executable)
            with self.assertRaises(ValueError):
                self.model.observe(branch.successor, self.row(branch.timestamp, 2.4))
            tail = self.model.predict_shared(branch.successor, .096, 16, 731, times=[0., .048, .096])
            self.assertAlmostEqual(tail.times[0], branch.timestamp)
        self.assertEqual(before, self.b.fingerprint())

    def test_no_centroid_is_fabricated_as_an_observation(self):
        p = self.evaluate()
        for b in p.branches:
            self.assertNotIn('observed_position', b.summary)
            self.assertFalse(b.summary['exact_reobservation_identity_supported'])
            self.assertEqual(b.summary['event_semantics'].startswith('at least') or b.summary['event_semantics'].startswith('no detection'), True)

    def test_world_indistinguishable_events_are_not_expanded(self):
        def indistinguishable(model, future, robots, lengths):
            q, t, _ = robots.shape
            first = np.zeros((q, t)); first[:, 0] = .5
            return dict(per_world_none=np.full((q, len(future.weights)), .5), p_any=np.full(q, .5),
                        first_detection_mass=first, endpoint_any=first)
        with patch('eventwm_graph.l25.compiler.optical_batch', indistinguishable):
            packet = self.evaluate()
        self.assertFalse(packet.branches)
        self.assertTrue(all(o['branching']['status'] == 'NO_INFORMATIVE_EVENT' for o in packet.outcomes))
        self.assertTrue(all(o['branching']['residual_model_supported_mass'] == 1 for o in packet.outcomes))

    def test_cache_matches_uncached_output(self):
        a = self.evaluate()
        b = self.evaluate()
        self.assertEqual(self.c.forecast_calls, 1)
        self.assertEqual(a.outcomes, b.outcomes)
        self.c.clear_dynamic_caches()
        c = self.evaluate()
        self.assertEqual(a.outcomes, c.outcomes)
        np.testing.assert_array_equal(a.graph['node_features'], c.graph['node_features'])

    def test_actual_update_invalidates_predictions_and_stale_context(self):
        self.evaluate()
        updated = self.model.observe(self.b, self.row(self.b.timestamp+.048, 2.06))
        context = self.c.build_context(updated, updated.graph, updated.robot_pose, [3, 0, 0])
        self.assertNotEqual(context.belief_version, self.ctx.belief_version)
        with self.assertRaises(ValueError):
            self.evaluate()
        ref = CompiledReference.from_native(native(updated), updated.timestamp)
        packet = self.c.evaluate_candidates(context, [ref], self.budget)
        self.assertEqual(packet.metadata['belief_version'], updated.fingerprint())
        self.assertEqual(self.c.forecast_calls, 2)

    def test_query_specific_optics_never_reuses_another_reference(self):
        p = self.evaluate()
        self.assertNotEqual(p.outcomes[0]['p_any_observation'], p.outcomes[1]['p_any_observation'])
        q = native(self.b, 'left', query_id='forward')
        ref = CompiledReference.from_native(q, self.b.timestamp)
        p2 = self.evaluate([ref])
        self.assertAlmostEqual(p2.outcomes[0]['p_any_observation'], p.outcomes[1]['p_any_observation'])
        self.assertNotEqual(ref.content_hash, self.refs[0].content_hash)

    def test_other_model_cannot_use_cached_belief(self):
        self.evaluate()
        c = QueryOutcomeCompiler(IntermittentWorldModel(seed=222))
        with self.assertRaises(ValueError):
            c.build_context(self.b, self.b.graph, self.b.robot_pose, [3, 0, 0])

    def test_graph_indices_masks_and_candidate_mapping(self):
        p = self.evaluate()
        g = p.graph
        self.assertTrue(np.isfinite(g['node_features']).all())
        self.assertGreaterEqual(g['edge_index'].min(), 0)
        self.assertLess(g['edge_index'].max(), len(g['node_features']))
        self.assertEqual(tuple(g['query_ids']), tuple(q.query_id for q in self.refs))
        self.assertEqual(int(g['candidate_mask'].sum()), len(self.refs))
        self.assertTrue(np.all(g['node_type'][np.flatnonzero(g['candidate_mask'])] == 1))
        self.assertFalse(g['candidate_mask'][g['node_type'] == 2].any())

    def test_graph_batch_matches_separate_consumption(self):
        a = self.evaluate([self.refs[0]])
        b = self.evaluate()
        la, _ = message_passing_logits(a.graph)
        lb, _ = message_passing_logits(b.graph)
        batched = batch_graphs([a.graph, b.graph])
        logits, mask = message_passing_logits(batched)
        np.testing.assert_allclose(logits, np.r_[la, lb], atol=1e-7)
        np.testing.assert_array_equal(batched['query_batch_ptr'], [0, 1, 3])
        self.assertTrue(mask.all())

    def test_variable_and_zero_candidate_counts(self):
        for refs in ([], [self.refs[0]], self.refs):
            p = self.evaluate(refs)
            logits, _ = message_passing_logits(p.graph)
            self.assertEqual(len(logits), len(refs))

    def test_unsupported_route_has_false_candidate_and_prediction_masks(self):
        q = native(self.b)
        q['compiled']['planned_poses'][-1][0] = 100.
        p = self.evaluate([CompiledReference.from_native(q, self.b.timestamp)])
        self.assertFalse(p.outcomes[0]['candidate_legal'])
        self.assertIsNone(p.outcomes[0]['p_nominal_collision'])
        self.assertFalse(p.graph['candidate_validity'][0])
        self.assertFalse(p.branches)

    def test_repeated_route_region_has_distinct_order_and_time(self):
        q = native(self.b, steps=24)
        poses = np.asarray(q['compiled']['planned_poses'])
        poses[13:, 0] = poses[11::-1, 0]
        q['compiled']['planned_poses'] = poses.tolist()
        p = self.evaluate([CompiledReference.from_native(q, self.b.timestamp)])
        occurrences = p.outcomes[0]['route_occurrences']
        self.assertGreater(len(occurrences), len({o['node'] for o in occurrences}))
        self.assertEqual([o['order'] for o in occurrences], list(range(len(occurrences))))

    def test_json_packet_is_finite_and_successors_are_opaque(self):
        p = self.evaluate()
        encoded = json.dumps(p.to_dict(), allow_nan=False)
        decoded = json.loads(encoded)
        self.assertNotIn('successor', decoded['branches'][0])
        self.assertEqual(decoded['schema'], 'l25-query-outcomes-1')

    def test_context_rejects_unarrived_graph_and_ego(self):
        with self.assertRaises(ValueError):
            self.c.build_context(self.b, replace(self.b.graph, namespace='future'), self.b.robot_pose, [3, 0, 0])
        with self.assertRaises(ValueError):
            self.c.build_context(self.b, self.b.graph, self.b.robot_pose+1, [3, 0, 0])

    def test_imagined_state_cannot_be_a_root_context(self):
        p = self.evaluate()
        child = p.branches[0].successor
        with self.assertRaises(ValueError):
            self.c.build_context(child, child.graph, child.robot_pose, [3, 0, 0])

    def test_adapter_content_is_immutable(self):
        q = native(self.b)
        r = CompiledReference.from_native(q, self.b.timestamp)
        q['compiled']['planned_poses'][0][0] = 99.
        self.assertEqual(r.poses[0, 0], self.b.robot_pose[0])
        with self.assertRaises(ValueError):
            r.poses.flags.writeable = True

    def test_changed_same_hash_reference_cannot_reuse_risk_cache(self):
        self.evaluate()
        with self.assertRaises(ValueError):
            self.evaluate([replace(self.refs[0], poses=self.refs[0].poses+1.)])

    def test_cached_support_metadata_cannot_be_modified_through_outputs(self):
        packet = self.evaluate()
        with self.assertRaises(TypeError):
            packet.context.metadata['expected_hidden_count'] = 99
        with self.assertRaises(TypeError):
            packet.branches[0].summary['bellman_transition_supported'] = True
        with self.assertRaises(ValueError):
            packet.context.tracks[0]['mean'][0] = 99

    def test_brier_metrics_keep_scenes_as_resampling_units(self):
        from eventwm_graph.l25.validation import risk_metrics
        rows = [dict(scene_id='a', root_id='r', prediction=0., label=False),
                dict(scene_id='b', root_id='s', prediction=1., label=True)]
        m = risk_metrics(rows, bootstrap=100)
        self.assertEqual(m['brier_scene_root_balanced'], 0.)
        self.assertEqual(m['brier_95_scene_bootstrap'], [0., 0.])
        self.assertEqual(m['auroc'], 1.)

    def test_invalid_references_are_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate([self.refs[0], self.refs[0]])
        with self.assertRaises(ValueError):
            self.evaluate([replace(self.refs[0], start_timestamp=100.)])
        with self.assertRaises(ValueError):
            self.evaluate([replace(self.refs[0], graph_revision=10000)])
        with self.assertRaises(ValueError):
            self.evaluate(budget=replace(self.budget, max_candidates=1))

    def test_unknown_goal_progress_is_masked(self):
        context = self.c.build_context(self.b, self.b.graph, self.b.robot_pose, [100, 0, 100])
        p = self.c.evaluate_candidates(context, self.refs, self.budget)
        self.assertIsNone(p.outcomes[0]['observed_graph_progress'])
        self.assertFalse(p.outcomes[0]['gnn_support'][14])

    def test_zero_hit_is_not_certain_safety(self):
        p = self.evaluate([self.refs[1]])
        self.assertTrue(p.metadata['no_guaranteed_safety'])
        out = p.outcomes[0]
        if out['p_nominal_collision'] == 0:
            self.assertGreater(out['zero_hit_upper_95_iid'], .17)

    def test_online_modules_have_no_truth_reader_or_training(self):
        import eventwm_graph.l25.compiler as compiler
        source = inspect.getsource(compiler)
        for forbidden in ('privileged_future', 'future_observations.json', 'np.load(', 'fit.train(', 'torch.optim'):
            self.assertNotIn(forbidden, source)


if __name__ == '__main__':
    unittest.main()
