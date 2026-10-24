"""Independent full-replay comparisons for immutable actual motion summaries."""
import copy
from pathlib import Path
import unittest

import numpy as np
import torch

from eventwm_graph.l2_two_stage.context import OnlineContext
from eventwm_graph.l2_two_stage.data import jobs_for, read, selected_branch
from eventwm_graph.l2_two_stage.features import extract
from eventwm_graph.l2_two_stage.motion import MotionSummary
from eventwm_graph.l2_two_stage.types import LegalContext


class TestMotionSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        folder, _ = jobs_for('dev')[0]
        cls.legal = read(Path(folder)/'legal_root.json.gz')
        cls.branch = selected_branch(Path(folder))[2]

    def context(self):
        z = self.legal
        return OnlineContext.from_raw(z['history'], z['partial_graph'], z['history_actions'])

    def assert_replay_equal(self, context):
        fast = extract(context)
        replay = extract(LegalContext(context.history, context.actions, context.partial_graph))
        assert fast['keys'] == replay['keys']
        assert fast['minimum_count'] == replay['minimum_count']
        for field in ('x', 'graph', 'global_graph', 'global_features', 'origin', 'velocity',
                      'age', 'last_positions', 'last_times', 'robotxy'):
            np.testing.assert_array_equal(fast[field], replay[field], err_msg=field)

    def test_actual_history_and_native_updates_match_independent_replay(self):
        context = self.context()
        self.assert_replay_equal(context)
        for node in self.branch[:8]:
            previous = context.motion_summary
            previous_counts = tuple((key, row['count']) for key, row in previous.memory.items())
            context = context.ingest(node['observation'], node['action'])
            self.assert_replay_equal(context)
            assert previous_counts == tuple((key, row['count']) for key, row in previous.memory.items())
            assert previous.frame_count + 1 == context.motion_summary.frame_count

    def test_summary_direct_constructor_cannot_bypass_certification(self):
        with self.assertRaises(TypeError):
            MotionSummary({}, 1, 2)
        with self.assertRaisesRegex(ValueError, 'uncertified'):
            MotionSummary().validate_certificate()

    def test_all_motion_vector_storage_is_immutable(self):
        context = self.context()
        summary = context.motion_summary
        assert summary.memory
        key = next(iter(summary.memory))
        with self.assertRaises(TypeError):
            summary.memory[key]['count'] = 999
        for field in ('pos', 'velocity', 'acceleration', 'mean'):
            array = summary.memory[key][field]
            with self.assertRaises(ValueError):
                array[0] = 999
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_invalid_and_empty_counterexamples_keep_old_humans_and_counts(self):
        context = self.context()
        keys = set(context.motion_summary.memory)
        obs = copy.deepcopy(self.branch[0]['observation'])
        obs['tracks'] = []
        empty = context.ingest(obs, self.branch[0]['action'])
        assert keys <= set(empty.motion_summary.memory)
        assert empty.motion_summary.empty == context.motion_summary.empty + 1
        self.assert_replay_equal(empty)
        obs['observation_valid'] = False
        obs['static_updates'] = []
        invalid = context.ingest(obs, self.branch[0]['action'])
        assert keys <= set(invalid.motion_summary.memory)
        assert invalid.motion_summary.invalid == context.motion_summary.invalid + 1
        assert invalid.motion_summary.empty == context.motion_summary.empty
        self.assert_replay_equal(invalid)


if __name__ == '__main__':
    unittest.main()
