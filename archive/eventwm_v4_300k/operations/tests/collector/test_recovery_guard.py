"""Operational fault injection: no native workers or production writes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/eventwm_production'))
import control
import recovery_guard as guard


class EndLoop(BaseException):
    pass


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.patch = mock.patch.object(control, 'LOGS', self.base)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        guard.save(dict(enabled=True, restarts=[]))

    def test_integrity_crash_never_restarts(self):
        for reason in ('Branch/static integrity failure', 'Duplicate retained query IDs',
                       'seal mismatch', 'source mismatch', 'hash mismatch'):
            self.assertIsNotNone(guard.refusal({'state': 'interrupted_error'}, reason, [], 5000))

    def test_budget_survives_new_guard_process(self):
        guard.save(dict(enabled=True, restarts=[4001, 4002, 4003]))
        self.assertIn('budget', guard.refusal({'state': 'running'}, '', guard.state()['restarts'], 5000))
        self.assertIsNone(guard.refusal({'state': 'running'}, '', [1, 2, 3], 5000))

    def test_intentional_and_unknown_stops_refused(self):
        for state in ('stopped_by_user', 'stopped_resource_guard', 'target_reached', 'bounded_finished'):
            self.assertIsNotNone(guard.refusal({'state': state}, '', [], 5000))
        self.assertIsNotNone(guard.refusal({'state': 'interrupted_error'}, '', [], 5000))

    def watch_once(self, reason='', session=False, check_error=None, live=None):
        m = dict(state='running', runner={'pid': 123})
        with mock.patch.object(control, 'PROJECT', self.base), \
             mock.patch.object(control, 'metadata', return_value=(m, {}, [])), \
             mock.patch.object(control, 'identity', return_value=live), \
             mock.patch.object(control, 'session_exists', return_value=session), \
             mock.patch.object(guard, 'stop_reason', return_value=reason), \
             mock.patch.object(control, 'check', side_effect=check_error) as check, \
             mock.patch.object(control, 'verify_index') as verify, \
             mock.patch.object(control, 'reject_active') as ownership, \
             mock.patch.object(control, 'launch') as launch, \
             mock.patch.object(guard.time, 'sleep', side_effect=EndLoop):
            with self.assertRaises(EndLoop): guard.watch()
            return check, verify, ownership, launch

    def test_collector_death_resumes_same_checkpoint_through_owner_guard(self):
        check, verify, ownership, launch = self.watch_once()
        check.assert_called_once(); verify.assert_called_once(); ownership.assert_called_once()
        launch.assert_called_once_with('resume', enable_recovery=False)
        self.assertEqual(len(guard.state()['restarts']), 1)
        events = [json.loads(line)['event'] for line in (self.base / 'recovery.jsonl').read_text().splitlines()]
        self.assertEqual(events, ['collector_dead', 'restart_attempt', 'restart_launched'])

    def test_integrity_failure_blocks_without_launch(self):
        _, _, _, launch = self.watch_once(reason='Branch/static integrity failure')
        launch.assert_not_called()
        self.assertFalse(guard.state()['enabled'])
        self.assertEqual(guard.state()['status'], 'blocked')

    def test_failed_preflight_blocks_without_loop(self):
        _, verify, _, launch = self.watch_once(check_error=RuntimeError('checkpoint mismatch'))
        verify.assert_not_called(); launch.assert_not_called()
        self.assertFalse(guard.state()['enabled'])

    def test_startup_session_and_live_collector_are_never_duplicated(self):
        for session, live in ((True, None), (False, {'pid': 123})):
            check, _, _, launch = self.watch_once(session=session, live=live)
            check.assert_not_called(); launch.assert_not_called()

    def test_operator_stop_disables_before_signalling(self):
        with mock.patch.object(control, 'metadata', return_value=({}, {}, [])), \
             mock.patch.object(control, 'identity', return_value={'pid': 123}), \
             mock.patch.object(control.os, 'kill') as kill:
            kill.side_effect = lambda *args: self.assertFalse(guard.state()['enabled'])
            control.stop()
        self.assertEqual(guard.state()['reason'], 'operator_stop')


if __name__ == '__main__':
    unittest.main()
