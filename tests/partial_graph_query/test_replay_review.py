"""Independent replay-tool and native-height acceptance regressions."""
import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from eventwm_graph.config import write_json
from eventwm_graph.partial.replay_review import Comparison, select_branches, review_root, reconstruct_root_lineage
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.partial.query import compile_query, TURN_STEP
from eventwm_graph.stage1.storage import write_gzip, scene_split


class IndependentReplayToolReview(unittest.TestCase):
    def query(self, target, source, yaw=0., band=None):
        provenance = dict(source=source)
        if band:provenance['intended_band'] = band
        return dict(waypoints=[target], final_yaw=yaw, provenance=[provenance])

    def test_selection_covers_social_bands_frontier_inplace_backtrack_and_yaw_pair(self):
        qs = [self.query([1., 0., 0.], 'human', band='safe'),
              self.query([.96, 0., 0.], 'human', band='intrusion'),
              self.query([0., 0., 0.], 'current_position', yaw=.25),
              self.query([0., 0., 0.], 'current_position', yaw=-.25),
              self.query([2., 0., 0.], 'frontier'),
              self.query([-1., 0., 0.], 'topology'),
              self.query([3., 0., 0.], 'goal')]
        selected, coverage = select_branches(qs, [0., 0., 0., 0.], 8)
        self.assertEqual(set(selected), set(range(7)))
        self.assertTrue(coverage['same_target_heading_pair_selected'])
        self.assertIn('backtracking', coverage['selected'])

    def test_small_budget_cannot_silently_drop_available_required_categories(self):
        qs = [self.query([1., 0., 0.], 'human', band='safe'),
              self.query([.96, 0., 0.], 'human', band='intrusion'),
              self.query([2., 0., 0.], 'frontier')]
        with self.assertRaisesRegex(ValueError, 'max_queries_too_small'):
            select_branches(qs, [0., 0., 0., 0.], 2)

    def test_categorical_visibility_mismatch_is_never_hidden_by_position_tolerance(self):
        comparison = Comparison()
        comparison.tree({'visible_now':True}, {'visible_now':False}, 'track')
        self.assertFalse(comparison.report()['passed'])

    def test_wrapped_native_yaw_parity_and_submillimeter_motion(self):
        comparison = Comparison()
        comparison.numeric(math.pi-1e-7, -math.pi+1e-7, 'yaw', 1e-5, angular=True)
        comparison.numeric([0., 0., 0.], [6e-6, 6e-6, 0.], 'native_position', 1e-4)
        self.assertTrue(comparison.report()['passed'])
        comparison.numeric([0., 0., 0.], [0., .002, 0.], 'native_position', 1e-4)
        self.assertFalse(comparison.report()['passed'])

    def test_nonfinite_comparison_fails(self):
        comparison = Comparison()
        comparison.numeric([0.], [float('nan')], 'native_position', 1e-4)
        self.assertFalse(comparison.report()['passed'])

    def test_identical_static_witness_duplicates_are_set_evidence(self):
        from test_stage1_review import row
        update = dict(cell=[0,0], state=2, height=0., timestamp=1., source='observed_wall')
        expected = row(1., [.1,0.,.1,0.], [update,copy.deepcopy(update)])
        actual = row(1., [.1,0.,.1,0.], [copy.deepcopy(update)])
        comparison = Comparison(); comparison.observation(expected,actual,'frame')
        self.assertTrue(comparison.report()['passed'])
        self.assertEqual(len(comparison.report()['raw_static_multiplicity_differences']),1)

    def test_static_set_comparison_keeps_new_cells_states_and_clocks_strict(self):
        from test_stage1_review import row
        update = dict(cell=[0,0], state=2, height=0., timestamp=1., source='observed_wall')
        expected = row(1., [.1,0.,.1,0.], [update])
        for changes in (dict(cell=[1,0]),dict(state=1),dict(height=.001),
                        dict(timestamp=1.000001),dict(source='different_observation')):
            altered = {**update,**changes}
            actual = row(1., [.1,0.,.1,0.], [copy.deepcopy(update),altered])
            comparison = Comparison(); comparison.observation(expected,actual,'frame')
            self.assertFalse(comparison.report()['passed'],changes)

    def test_static_set_order_does_not_depend_on_subnumeric_height_order(self):
        from test_stage1_review import row
        occupied = dict(cell=[0,0], state=2, height=0., timestamp=1., source='observed_wall')
        free = dict(cell=[0,0], state=1, height=1e-9, timestamp=1., source='observed_free')
        expected = row(1., [.1,0.,.1,0.], [occupied,free])
        actual = row(1., [.1,0.,.1,0.], [{**free,'height':-1e-9},occupied])
        comparison = Comparison(); comparison.observation(expected,actual,'frame')
        self.assertTrue(comparison.report()['passed'])

    def test_held_out_identity_is_refused_before_private_files_are_opened(self):
        scene = next('independent_held_out_'+str(i) for i in range(1000)
                     if scene_split('independent_held_out_'+str(i)) != 'train')
        with tempfile.TemporaryDirectory() as folder:
            write_json(Path(folder)/'manifest.json', dict(scene_id=scene, split='confirmation'))
            with patch('eventwm_graph.partial.replay_review.read_gzip', side_effect=AssertionError('private read')):
                with self.assertRaisesRegex(ValueError, 'held_out_scene_refused'):
                    review_root(folder)


class IndependentKnownWallRefreshReview(unittest.TestCase):
    source='observed_static_wall_inflated_025m'

    def observation(self, timestamp, updates):
        from test_stage1_review import row
        return row(timestamp,[.1,0.,.1,0.],updates)

    def wall(self, timestamp, **changes):
        return {**dict(cell=[0,0],state=2,height=0.,timestamp=timestamp,source=self.source),**changes}

    def initial(self):
        c=Comparison(resolution=.2);first=self.observation(1.,[self.wall(1.)])
        c.observation(first,copy.deepcopy(first),'initial')
        self.assertTrue(c.report()['passed'])
        return c

    def test_only_exact_prior_known_wall_refresh_can_differ(self):
        c=self.initial()
        c.observation(self.observation(2.,[self.wall(2.)]),self.observation(2.,[]),'future')
        self.assertTrue(c.report()['passed'])
        proof=c.report()['known_occupied_refresh_differences'][0]
        self.assertEqual(proof['stored_prior']['last_seen'],1.)
        self.assertEqual(proof['native_prior']['last_seen'],1.)
        self.assertEqual(proof['stored_last_seen_after'],2.)
        self.assertEqual(proof['native_last_seen_after'],1.)

    def test_new_free_occupied_height_and_source_changes_never_get_exception(self):
        for changes in (dict(cell=[1,0]),dict(state=1),dict(height=1e-12),dict(source='other')):
            c=self.initial()
            c.observation(self.observation(2.,[self.wall(2.,**changes)]),self.observation(2.,[]),'future')
            self.assertFalse(c.report()['passed'],changes)
            self.assertFalse(c.report()['known_occupied_refresh_differences'])

    def test_prior_occupied_certificate_is_required_in_both_histories(self):
        c=Comparison(resolution=.2)
        c.observation(self.observation(1.,[self.wall(1.)]),self.observation(1.,[]),'initial')
        c.observation(self.observation(2.,[self.wall(2.)]),self.observation(2.,[]),'future')
        self.assertFalse(c.report()['passed'])
        self.assertFalse(c.report()['known_occupied_refresh_differences'])

    def test_future_heartbeat_timestamp_fails_causality(self):
        c=self.initial()
        with self.assertRaisesRegex(ValueError,'Future/stale'):
            c.observation(self.observation(2.,[self.wall(2.000001)]),self.observation(2.,[]),'future')


class IndependentRootLineageReview(unittest.TestCase):
    def fixture(self):
        from test_stage1_review import fixture,row
        from eventwm_graph.partial.query import generate_queries
        record,_,_,_=fixture();history=[record['history'][0]]
        graph=PartialGraph.restore(record['partial_graph']);queries=[]
        for i in range(1,14):
            r=row(1.+.048*i,[.1,0.,.1,TURN_STEP*i]);history.append(r);graph.update(r)
            if i in (11,12,13):
                queries,_=generate_queries(graph,[],r,record['pointgoal'],8)
        legal=dict(history=history,partial_graph=graph.snapshot(),pointgoal=record['pointgoal'])
        return legal,queries,dict(queries_per_root=8,safe_margin_m=.2,intrusion_radius_m=.96)

    def test_root_topology_and_full_queries_rebuild_at_causal_checkpoints(self):
        legal,queries,configuration=self.fixture()
        result=reconstruct_root_lineage(legal,queries,configuration)
        self.assertTrue(result['passed'])
        self.assertEqual(result['checkpoints_replayed'],[11,12,13])
        self.assertTrue(result['full_queries_reproduced'])

    def test_certified_but_fabricated_global_lineage_is_rejected(self):
        legal,queries,configuration=self.fixture()
        node=next(iter(legal['partial_graph']['global_nodes'].values()))
        node['kind']='fabricated_historical_provenance'
        with self.assertRaisesRegex(ValueError,'topology_lineage'):
            reconstruct_root_lineage(legal,queries,configuration)


class IndependentNativeHeightReview(unittest.TestCase):
    def run_probe(self, dx=0., dy=.006, static_contact=False, target_extra_x=0.):
        import habitat_sim
        import magnum as mn
        from eventwm_graph.partial.acceptance import native_static_probe
        # Import the existing synthetic fixture, never a real research episode.
        from test_stage1_review import fixture
        record, _, queries, _ = fixture()
        query = copy.deepcopy(queries[0])
        if target_extra_x:
            query['waypoints'][0][0] += target_extra_x
            query['compiled'] = compile_query(PartialGraph.restore(record['partial_graph']),
                                               query, record['history'][0])
            self.assertTrue(query['compiled']['ok'])
        trace = dict(scene_id=record['scene_id'], seed=record['seed'], prefix_actions=[],
                     private_initialization=record['private_initialization'], snapshots=[{'queries':[query]}])
        class Robot:
            def __init__(self):
                self.sim_obj = SimpleNamespace(transformation=mn.Matrix4.translation(
                    mn.Vector3(query['compiled']['planned_poses'][0][:3])))
            @property
            def base_pos(self):return np.asarray(self.sim_obj.transformation.translation)
        class Runtime:
            def __init__(self, *args, **kwargs):
                self.robot = Robot(); self.i = 0
                self.sim = SimpleNamespace(get_agent_data=lambda _:SimpleNamespace(articulated_agent=self.robot))
                self.env = SimpleNamespace(episode_over=False)
            def __enter__(self):return self
            def __exit__(self, *args):pass
            def step(self, action):
                self.i += 1
                p = np.asarray(query['compiled']['planned_poses'][self.i]).copy()
                p[:3] += [dx, dy, 0.]
                self.robot.sim_obj.transformation = mn.Matrix4.from_(
                    mn.Matrix4.rotation_y(mn.Rad(p[3])).rotation(), mn.Vector3(p[:3]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'independent_height.json.gz'; write_gzip(path, trace)
            with patch('eventwm_graph.partial.acceptance.scene_manifest', return_value='unit'), \
                    patch('eventwm_graph.partial.acceptance.load_certified_map', return_value=SimpleNamespace()), \
                    patch('eventwm_graph.partial.acceptance.NativeEpisode', Runtime), \
                    patch('eventwm_graph.partial.acceptance.contact_labels',
                          return_value={'static_obstacle_contact':static_contact}):
                return native_static_probe(path, Path(folder)/'probes', limit=1)

    def test_small_vertical_floor_error_passes_while_terminal_semantics_stay_strict(self):
        result = self.run_probe(dx=6e-6, dy=.006)
        self.assertTrue(result['passed'])
        self.assertLess(result['queries'][0]['maximum_planar_error_m'], .001)
        self.assertGreater(result['queries'][0]['maximum_height_error_m'], .001)

    def test_planar_error_is_not_excused_as_floor_quantization(self):
        self.assertFalse(self.run_probe(dx=.0011, dy=0.)['passed'])

    def test_excessive_height_error_is_rejected(self):
        self.assertFalse(self.run_probe(dy=.081)['passed'])

    def test_terminal_full_3d_distance_is_not_relaxed_by_height_cap(self):
        # Planned endpoint has .145m remaining X distance. A .079m native floor
        # offset is below the height cap, but puts the actual endpoint too far.
        self.assertFalse(self.run_probe(dy=.079, target_extra_x=.145)['passed'])

    def test_native_static_contact_always_rejects_probe(self):
        self.assertFalse(self.run_probe(static_contact=True)['passed'])


if __name__ == '__main__':
    unittest.main()
