"""Real native observations plus adversarial counterexamples for OnlineContext."""
import copy
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import unittest
import torch

from eventwm_graph.l2_two_stage.context import OnlineContext
from eventwm_graph.l2_two_stage.data import advance_graph, jobs_for, read, selected_branch
from eventwm_graph.l2_two_stage.features import extract
from eventwm_graph.l2_two_stage.model import Stage1WorldModel
from eventwm_graph.l2_two_stage.types import LegalContext


def load_native():
    torch.set_num_threads(1)
    jobs = jobs_for('dev')
    assert jobs, 'actual fixed development roots are required'
    folder, row = jobs[0]
    folder = Path(folder)
    legal = read(folder / 'legal_root.json.gz')
    _, _, branch = selected_branch(folder)
    assert len(branch) >= 3
    return legal, branch, row


def factory(legal):
    return OnlineContext.from_raw(legal['history'], legal['partial_graph'], legal['history_actions'])


def assert_same_features(left, right):
    for key in ('x', 'graph', 'global_graph', 'global_features', 'origin',
                'velocity', 'age', 'last_positions', 'last_times', 'robotxy'):
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)
    assert left['keys'] == right['keys']
    assert left['timestamp'] == right['timestamp']


def height_fixture(middle_height=0., end_height=0.):
    heights = [0., middle_height, end_height]
    updates = [dict(cell=[i, 0], state=1, height=heights[i],
                    source='visible_native_cspace_square_and_clearance_halo', timestamp=0.)
               for i in range(3)]
    obs = dict(timestamp=0., robot_pose=[.1, 0., .1, 0.], camera_origin=[.1, 1.2, .1],
               camera_rotation=np.eye(3).tolist(), tracks=[], static_updates=updates,
               sensor_protocol='ideal_frustum_observed_cspace_dynamic_occlusion_1', observation_valid=True,
               human_sensor_available=True)
    nodes = {str(i):dict(id=str(i), cell=[i, 0], position=[(i+.5)*.2, heights[i], .1],
                        kind='visibility_support', created_at=0.) for i in (0, 2)}
    path = [nodes['0']['position'], nodes['2']['position']]
    edge = dict(source='0', target='2', path=path, length=float(np.linalg.norm(np.diff(path, axis=0))),
                revision=1, certificate='known_free_cell_supercover')
    graph = dict(schema='causal_partial_cspace_graph_1', resolution=.2, height=0., clearance=.25,
                 timestamp=0., revision=1, ego_support=[.1, 0., .1],
                 visited=[dict(position=[.1, 0., .1], timestamp=0.)],
                 cells=[dict(cell=u['cell'], state=1, height=u['height'], source=u['source'],
                             first_seen=0., last_seen=0.) for u in updates],
                 global_nodes=nodes, global_edges=[edge])
    return obs, graph


class TestOnlineContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = load_native()

    def test_real_native_factory_and_three_online_updates_match_raw(self):
        native = self.native
        legal, branch, _ = native
        context = factory(legal)
        history, actions = list(legal['history']), list(legal['history_actions'])
        graph = copy.deepcopy(legal['partial_graph'])
        model = Stage1WorldModel()
        for node in branch[:3]:
            old = context
            context = context.ingest(node['observation'], node['action'])
            assert len(context.history) == len(old.history) + 1
            assert context.history[-2] is old.history[-1]
            assert context.actions == old.actions + (node['action'],)
            history.append(node['observation'])
            actions.append(node['action'])
            graph = advance_graph(graph, node['observation'], node['graph_delta'])
            raw = extract(LegalContext.from_inputs(history, history[-1], graph, actions))
            fast = extract(context)
            assert_same_features(raw, fast)
            expected = model.predict_features(raw, seed=41)
            actual = model.predict_features(fast, seed=41)
            np.testing.assert_array_equal(expected.positions, actual.positions)
            np.testing.assert_array_equal(expected.present, actual.present)
            assert context.partial_graph['revision'] == graph['revision']
            assert context.partial_graph['timestamp'] == node['observation']['timestamp']


    def test_interior_wrong_height_is_unknown_support(self):
        obs, graph = height_fixture(middle_height=10.)
        with self.assertRaisesRegex(ValueError, 'UNKNOWN|obstructed'):
            OnlineContext.from_raw([obs], graph, ())

    def test_arrived_interior_height_change_removes_old_link(self):
        obs, graph = height_fixture()
        context = OnlineContext.from_raw([obs], graph, ())
        assert len(context.partial_graph['global_edges']) == 1
        arrived = copy.deepcopy(obs)
        arrived['timestamp'] = 1.
        arrived['static_updates'] = [dict(arrived['static_updates'][1], height=10., timestamp=1.)]
        updated = context.ingest(arrived, 'wait')
        assert updated.partial_graph['revision'] == 2
        assert len(updated.partial_graph['global_edges']) == 0
        assert len(context.partial_graph['global_edges']) == 1

    def test_supported_sloped_path_is_retained(self):
        obs, graph = height_fixture(middle_height=.1, end_height=.2)
        context = OnlineContext.from_raw([obs], graph, ())
        assert len(context.partial_graph['global_edges']) == 1

    def test_mutable_tensor_input_is_rejected_or_copied(self):
        legal, _, _ = self.native
        history = copy.deepcopy(legal['history'])
        pose = torch.tensor(history[-1]['robot_pose'], dtype=torch.float64)
        history[-1]['robot_pose'] = pose
        try:
            context = OnlineContext.from_raw(history, legal['partial_graph'], legal['history_actions'])
        except (ValueError, TypeError):
            return
        before = float(context.history[-1]['robot_pose'][0])
        pose[0] += 100.
        assert float(context.history[-1]['robot_pose'][0]) == before

    def test_raw_history_iterator_is_consumed_once(self):
        legal, _, _ = self.native
        context = OnlineContext.from_raw(iter(legal['history']), legal['partial_graph'], legal['history_actions'])
        assert len(context.history) == len(legal['history'])

    def test_direct_constructor_cannot_certify(self):
        native = self.native
        with self.assertRaises(TypeError):
            OnlineContext((), {}, 0)
        with self.assertRaisesRegex(ValueError, 'uncertified'):
            OnlineContext().validate_certificate()
        with self.assertRaisesRegex(ValueError, 'uncertified'):
            OnlineContext().ingest(native[1][0]['observation'], 'wait')


    def test_old_and_new_facts_are_deeply_immutable(self):
        native = self.native
        legal, branch, _ = native
        mutable = copy.deepcopy(legal)
        context = factory(mutable)
        key = next(iter(context.certified_cells))
        with self.assertRaises(TypeError):
            context.certified_cells[key]['state'] = 99
        with self.assertRaises(TypeError):
            context.partial_graph['resolution'] = 100.
        with self.assertRaises(FrozenInstanceError):
            context.actions = ()
        old_pose = tuple(context.history[-1]['robot_pose'])
        mutable['history'][-1]['robot_pose'][0] += 100.
        mutable['partial_graph']['cells'][0]['state'] = 99
        assert tuple(context.history[-1]['robot_pose']) == old_pose
        obs = copy.deepcopy(branch[0]['observation'])
        new = context.ingest(obs, branch[0]['action'])
        actual_pose = tuple(new.history[-1]['robot_pose'])
        obs['robot_pose'][0] += 100.
        assert tuple(new.history[-1]['robot_pose']) == actual_pose
        assert len(context.history) + 1 == len(new.history)


    def test_history_actions_and_future_clocks(self):
        native = self.native
        legal, branch, _ = native
        with self.assertRaisesRegex(ValueError, 'executed action'):
            OnlineContext.from_raw(legal['history'], legal['partial_graph'], ())
        context = factory(legal)
        with self.assertRaisesRegex(ValueError, 'chronological|noncausal'):
            context.ingest(legal['history'][-1], 'wait')
        with self.assertRaisesRegex(ValueError, 'unknown executed action'):
            context.ingest(branch[0]['observation'], 'oracle_go_to')
        graph = copy.deepcopy(legal['partial_graph'])
        graph['timestamp'] += 1.
        with self.assertRaisesRegex(ValueError, 'clock'):
            OnlineContext.from_raw(legal['history'], graph, legal['history_actions'])


    def test_hidden_map_fields_and_duplicate_cells_rejected(self):
        native = self.native
        legal, _, _ = native
        graph = copy.deepcopy(legal['partial_graph'])
        graph['navmesh'] = 'privileged completion'
        with self.assertRaisesRegex(ValueError, 'privileged graph'):
            OnlineContext.from_raw(legal['history'], graph, legal['history_actions'])
        graph = copy.deepcopy(legal['partial_graph'])
        graph['cells'][-1] = copy.deepcopy(graph['cells'][0])
        with self.assertRaisesRegex(ValueError, 'duplicated|omitted'):
            OnlineContext.from_raw(legal['history'], graph, legal['history_actions'])


    def test_known_free_path_certification_rejects_unknown_shortcut(self):
        native = self.native
        legal, _, _ = native
        graph = copy.deepcopy(legal['partial_graph'])
        assert graph['global_edges'], 'real known-path example required'
        edge = graph['global_edges'][0]
        a, b = edge['path'][0], edge['path'][-1]
        cells = [c['cell'] for c in graph['cells']]
        outside = [(max(k[0] for k in cells) + 8) * graph['resolution'], a[1],
                   (max(k[1] for k in cells) + 8) * graph['resolution']]
        edge['path'] = [a, outside, b]
        edge['length'] = float(np.linalg.norm(np.diff(edge['path'], axis=0), axis=-1).sum())
        with self.assertRaisesRegex(ValueError, 'UNKNOWN|obstructed'):
            OnlineContext.from_raw(legal['history'], graph, legal['history_actions'])


    def test_optional_replacement_graph_is_fully_certified(self):
        native = self.native
        legal, branch, _ = native
        context = factory(legal)
        node = branch[0]
        graph = advance_graph(copy.deepcopy(legal['partial_graph']), node['observation'], node['graph_delta'])
        fast = context.ingest(node['observation'], node['action'])
        replaced = context.ingest(node['observation'], node['action'], graph)
        assert_same_features(extract(fast), extract(replaced))
        graph['resolution'] *= 2
        with self.assertRaisesRegex(ValueError, 'grid contract'):
            context.ingest(node['observation'], node['action'], graph)


    def test_valid_empty_counterexample_preserves_legally_seen_episodes(self):
        native = self.native
        legal, branch, _ = native
        context = factory(legal)
        initial = set(extract(context)['keys'])
        assert initial, 'real observed prefix required'
        empty = copy.deepcopy(branch[0]['observation'])
        empty['tracks'] = []
        arrived = context.ingest(empty, branch[0]['action'])
        assert initial <= set(extract(arrived)['keys'])

if __name__ == '__main__':
    unittest.main()
