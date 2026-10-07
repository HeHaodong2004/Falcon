"""Durable execution bookkeeping tests; no Habitat world is created."""
import copy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import eventwm_combined_runner as runner
from collector.core import read


class CombinedBookkeepingTests(unittest.TestCase):
    def test_durable_json_round_trip(self):
        with TemporaryDirectory() as temp:
            p = Path(temp)/'sub/manifest.json'
            runner.durable_json(p, {'query_ids':['a'], 'seconds':0.25})
            self.assertEqual(read(p), {'query_ids':['a'], 'seconds':0.25})
            self.assertFalse(p.with_suffix('.json.tmp').exists())

    def test_recover_completed_and_interrupted_without_resampling(self):
        with TemporaryDirectory() as temp:
            root = dict(root_id='r', state='running', recipe={'seed':17},
                        attempts=[{'state':'running','path':'roots/r/attempt_01'}])
            m = {'roots':[root]}
            with patch.object(runner.mass, 'recover_completed', return_value=False):
                runner.recover(Path(temp), m)
            self.assertEqual(root['attempts'][-1]['reason'], 'interrupted_before_commit')
            self.assertEqual(root['recipe'], {'seed':17})
            self.assertEqual(len(root['attempts']), 1)

    def test_validated_before_manifest_is_recovered(self):
        with TemporaryDirectory() as temp:
            p = Path(temp)
            root = dict(root_id='r', state='running', attempts=[{'state':'running','path':'roots/r/attempt_01'}])
            runner.durable_json(p/'roots/r/attempt_01/VALIDATED.json', {'root_id':'r'})
            def commit(output, entry):
                entry.update(state='completed',query_ids=['q'])
            with patch.object(runner.mass, 'recover_completed', return_value=False), patch.object(runner, 'commit_root', side_effect=commit) as fn:
                runner.recover(p, {'roots':[root]})
            fn.assert_called_once()
            self.assertTrue(root['attempts'][0]['recovered_validation'])

    def test_logical_order_blocks_later_validation(self):
        roots = [dict(root_id='a',state='running'), dict(root_id='b',state='running')]
        m = {'roots':roots, 'execution_cursor':0}
        ready = {'b'}
        with patch.object(runner, 'save_manifest'), patch.object(runner, 'commit_root') as commit:
            runner.ordered_commit(Path('/tmp/test'), m, ready)
        commit.assert_not_called()
        self.assertEqual(m['execution_cursor'], 0)
        self.assertEqual(ready, {'b'})

    def test_gate_is_specific_and_unexpected_errors_are_fatal(self):
        result = {'error':{'type':'RootStaticPreflightRejected','message':'root_static_clearance_preflight'}}
        self.assertEqual(runner.failure_reason(result), 'root_static_clearance_preflight')
        with self.assertRaises(RuntimeError):
            runner.failure_reason({'error':{'type':'ValueError','message':'unexpected'}})

    def test_completed_corruption_is_not_ignored(self):
        root = dict(root_id='r',state='completed',seal_file='r.json')
        with patch.object(runner, 'read', return_value={}), patch.object(runner, 'verify_root', side_effect=RuntimeError('corruption')):
            with self.assertRaisesRegex(RuntimeError,'corruption'):
                runner.recover(Path('/tmp/test'), {'roots':[root]})

    def test_unknown_only_root_remains_nonfatal_and_unretained(self):
        root = dict(root_id='r',state='running',attempts=[{'state':'running','path':'roots/r/attempt_01'}])
        m = {'roots':[root], 'execution_cursor':0}
        ready = {'r'}
        with patch.object(runner, 'save_manifest'), patch.object(runner.mass, 'inspect_root', side_effect=runner.mass.NoRetainedValidQueries('no_retained_valid_queries')):
            runner.ordered_commit(Path('/tmp/test'), m, ready)
        self.assertEqual(root['attempts'][0]['reason'], 'no_retained_valid_queries')
        self.assertEqual(m['execution_cursor'], 1)
        self.assertFalse(ready)

    def test_stop_refuses_unrelated_pid(self):
        with patch.object(runner, 'read', return_value={'runner':{'pid':1,'host':runner.socket.gethostname()}}):
            with self.assertRaisesRegex(RuntimeError,'PID identity'):
                runner.request_stop(Path('/tmp/test'))

    def test_missing_completion_receipt_is_not_published(self):
        with TemporaryDirectory() as temp:
            root = dict(root_id='r',attempts=[{'path':'roots/r/attempt_01'}])
            with self.assertRaisesRegex(RuntimeError,'unvalidated'):
                runner.commit_root(Path(temp), root)

    def test_root_gate_uses_measured_state_not_fixed_threshold(self):
        from collector.core import LegalRoot
        from collector.root_preflight import check_root_clearance
        from collector.static_feasibility import RobotGeometry, StaticConfig
        from types import SimpleNamespace
        root = LegalRoot('r',(0.,0.,0.),0.,.1,.01)
        world = SimpleNamespace(clearance=lambda p:.405)
        a = check_root_clearance(root, world, RobotGeometry(.610), StaticConfig())
        b = check_root_clearance(root, world, RobotGeometry(.625), StaticConfig())
        self.assertTrue(a['passed'])
        self.assertFalse(b['passed'])
        self.assertAlmostEqual(a['required_clearance'], .400)
        self.assertAlmostEqual(b['required_clearance'], .415)

    def test_timeout_keeps_original_proposal_and_execution_evidence(self):
        with TemporaryDirectory() as temp:
            p = Path(temp)
            root = dict(root_id='r',state='running',recipe={'seed':42},
                        attempts=[{'state':'running','path':'roots/r/attempt_01'}])
            runner.durable_json(p/'roots/replay.json', root['recipe'])
            runner.mark_timeout(root, p, 'b007', 300.25)
            self.assertEqual(root['attempts'][0]['reason'],'worker_timeout')
            self.assertEqual(read(p/'roots/replay.json'), {'seed':42})
            self.assertEqual(root['recipe'], {'seed':42})
            self.assertEqual(read(p/'audits/workers/b007/collection_error.json')['timeout_seconds'],300)


if __name__ == '__main__':
    unittest.main()
