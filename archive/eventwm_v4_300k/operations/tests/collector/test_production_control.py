"""Fault-injection tests for launcher safety, using isolated temporary data."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/eventwm_production'))
import control


class ProductionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / 'dataset'
        self.output.mkdir()
        self.data_patch = mock.patch.object(control, 'DATA', self.output)
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)
        self.addCleanup(self.temp.cleanup)
        self.legacy = dict(accepted_query_ids=['legacy'], accepted_root_ids=['legacy_root'], retained=1)
        self.release = dict(implementation_hashes={'x': 'hash'}, protected_scientific_hashes={'f': 'frozen'},
                            configuration={'master_seed': 731000000})
        self.m = dict(frozen_commit=control.C['scientific_commit'], source_hashes={'x': 'hash'},
                      pipeline_release=control.canonical_digest(self.release),
                      pipeline_settings=copy.deepcopy(control.C['pipeline_settings']),
                      master_seed=731000000, protected_scientific_hashes={'f': 'frozen'},
                      next_sequence=1, production_source_epochs=[])
        root_dir = self.output / 'roots/root/attempt_01'; root_dir.mkdir(parents=True)
        (root_dir / 'COMPLETE.json').write_text('{}')
        (self.output / 'seal.json').write_text('{}')
        self.root = dict(root_id='root', state='completed', source_commit=control.C['scientific_commit'],
                         seal_file='seal.json', path='roots/root/attempt_01',
                         query_ids=['q' + str(i) for i in range(33)], stats={'primary_queries': 33})
        self.progress = {'primary_queries': 33}

    def checkpoint(self, m=None, roots=None, progress=None):
        with mock.patch.object(control, 'legacy', return_value=self.legacy):
            return control.checkpoint_check(self.release, m or self.m, progress or self.progress,
                                            roots if roots is not None else [self.root])

    def test_compatible_checkpoint_preserves_ids_and_metadata(self):
        before = copy.deepcopy((self.m, self.root))
        result = self.checkpoint()
        self.assertEqual(result['total_retained'], 34)
        self.assertEqual((self.m, self.root), before)

    def test_duplicate_across_legacy_and_new_refused(self):
        self.legacy['accepted_query_ids'] = ['q0']
        with self.assertRaisesRegex(RuntimeError, 'Duplicate query'): self.checkpoint()

    def test_duplicate_within_root_refused(self):
        self.root['query_ids'][1] = 'q0'
        with self.assertRaisesRegex(RuntimeError, 'Invalid completed query'): self.checkpoint()

    def test_duplicate_completed_root_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'Duplicate root'): self.checkpoint(roots=[self.root, self.root])

    def test_missing_seal_refused(self):
        (self.output / 'seal.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'seal/COMPLETE'): self.checkpoint()

    def test_missing_complete_refused(self):
        (self.output / self.root['path'] / 'COMPLETE.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'seal/COMPLETE'): self.checkpoint()

    def test_changed_checkpoint_settings_refused(self):
        self.m['pipeline_settings']['backlog'] = 72
        with self.assertRaisesRegex(RuntimeError, 'Resume settings'): self.checkpoint()

    def test_different_source_refused_without_migration(self):
        self.m['frozen_commit'] = 'unknown'
        with self.assertRaisesRegex(RuntimeError, 'no implicit migration'): self.checkpoint()

    def test_changed_release_digest_refused(self):
        self.m['pipeline_release'] = 'unknown'
        with self.assertRaisesRegex(RuntimeError, 'scientific release differs'): self.checkpoint()

    def test_changed_seed_refused(self):
        self.m['master_seed'] += 1
        with self.assertRaisesRegex(RuntimeError, 'seed differs'): self.checkpoint()

    def test_lost_epoch_refused(self):
        self.root['source_commit'] = 'previous'
        with self.assertRaisesRegex(RuntimeError, 'source epoch'): self.checkpoint()

    def test_historical_epoch_accepted_unchanged(self):
        self.root['source_commit'] = 'previous'
        self.m['production_source_epochs'] = [{'commit': 'previous'}]
        self.checkpoint()
        self.assertEqual(self.root['source_commit'], 'previous')

    def test_progress_cannot_count_unfinished_work(self):
        with self.assertRaisesRegex(RuntimeError, 'count differs'): self.checkpoint(progress={'primary_queries': 34})

    def test_resume_missing_checkpoint_never_creates_dataset(self):
        with mock.patch.object(control, 'reject_active'), mock.patch.object(control, 'git', return_value=''), mock.patch.object(control, 'read', return_value={'hashes': {}}):
            # Test the final initialization guard under the launcher lock.
            with self.assertRaisesRegex(RuntimeError, 'initialization changed'): control.prepare_launch_after_lock('resume')
        self.assertFalse((self.output / 'pipeline.sqlite3').exists())

    def test_start_refuses_existing_checkpoint(self):
        (self.output / 'pipeline.sqlite3').write_bytes(b'checkpoint')
        with mock.patch.object(control, 'reject_active'):
            with self.assertRaisesRegex(RuntimeError, 'initialization changed'): control.prepare_launch_after_lock('start')
        self.assertEqual((self.output / 'pipeline.sqlite3').read_bytes(), b'checkpoint')

    def test_start_refuses_unrecognized_existing_files(self):
        (self.output / 'precious').write_text('keep')
        with mock.patch.object(control, 'reject_active'):
            with self.assertRaisesRegex(RuntimeError, 'nonempty'): control.prepare_launch_after_lock('start')
        self.assertEqual((self.output / 'precious').read_text(), 'keep')

    def test_live_duplicate_refused_before_other_checks(self):
        with mock.patch.object(control, 'metadata', return_value=(self.m, None, [])), mock.patch.object(control, 'identity', return_value={'pid': 123}), mock.patch.object(control, 'session_exists') as tmux:
            with self.assertRaisesRegex(RuntimeError, 'already running'): control.reject_active()
            tmux.assert_not_called()

    def test_execution_lock_held_is_detected_without_writes(self):
        path = self.output / '.lock'; path.write_bytes(b'keep')
        with path.open('rb') as held:
            control.fcntl.flock(held, control.fcntl.LOCK_EX | control.fcntl.LOCK_NB)
            self.assertTrue(control.lock_busy(path))
        self.assertEqual(path.read_bytes(), b'keep')
        self.assertFalse(control.lock_busy(path))

    def test_no_reset_action(self):
        with mock.patch.object(sys, 'argv', ['control.py', 'reset']):
            with self.assertRaises(SystemExit) as error: control.main()
            self.assertEqual(error.exception.code, 2)

    def test_reused_pid_is_never_treated_as_collector(self):
        proc_root = self.output / 'proc'; proc = proc_root / '123'; proc.mkdir(parents=True)
        (proc / 'cwd').symlink_to(self.output)
        (proc / 'cmdline').write_bytes(b'/usr/bin/python3\0unrelated.py\0')
        m = dict(runner={'pid': 123, 'host': control.socket.gethostname()})
        self.assertIsNone(control.identity(m, output=self.output, proc_root=proc_root))

    def test_absent_pid_after_reboot_is_stopped(self):
        m = dict(runner={'pid': 123, 'host': control.socket.gethostname()})
        self.assertIsNone(control.identity(m, output=self.output, proc_root=self.output / 'absent_proc'))

    def test_missing_resume_checkpoint_refused_before_preflight(self):
        with mock.patch.object(control, 'reject_active'), mock.patch.object(control, 'operations_check'), mock.patch.object(control, 'check') as preflight:
            with self.assertRaisesRegex(RuntimeError, 'No saved dataset'): control.prepare_launch('resume')
            preflight.assert_not_called()
        self.assertFalse((self.output / 'pipeline.sqlite3').exists())

    def test_existing_start_refused_before_preflight(self):
        (self.output / 'pipeline.sqlite3').write_bytes(b'keep')
        with mock.patch.object(control, 'reject_active'), mock.patch.object(control, 'operations_check'), mock.patch.object(control, 'check') as preflight:
            with self.assertRaisesRegex(RuntimeError, 'Use resume'): control.prepare_launch('start')
            preflight.assert_not_called()
        self.assertEqual((self.output / 'pipeline.sqlite3').read_bytes(), b'keep')

    def test_tampered_operational_release_refused(self):
        manifest = self.output / 'configs/eventwm_300k_operations.json'; manifest.parent.mkdir()
        manifest.write_text(json.dumps(dict(release_version=control.C['release_version'], scientific_commit=control.C['scientific_commit'], hashes={'payload': 'wrong'})))
        (self.output / 'payload').write_text('tampered')
        with mock.patch.object(control, 'RELEASE_REPO', self.output), mock.patch.object(control, 'git', return_value=control.C['release_branch']):
            with self.assertRaisesRegex(RuntimeError, 'Operational release file differs'): control.operations_check()

    def test_low_disk_fails_without_publication(self):
        before = set(self.output.iterdir())
        with mock.patch.object(control.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 1})()), mock.patch.object(control.os, 'stat', return_value=type('Stat', (), {'st_dev': 1})()):
            with self.assertRaisesRegex(RuntimeError, 'space'): control.disk_check(34)
        self.assertEqual(set(self.output.iterdir()), before)

    def test_reboot_preserve_restore_keeps_complete_tree(self):
        backup = Path(self.temp.name) / 'persistent_backup'
        (self.output / 'pipeline.sqlite3').write_bytes(b'checkpoint')
        self.m['state'] = 'stopped_by_user'
        with mock.patch.dict(control.C, reboot_backup=str(backup)), mock.patch.object(control, 'LOGS', Path(self.temp.name) / 'logs'), mock.patch.object(control, 'operations_check'), mock.patch.object(control, 'reject_active'), mock.patch.object(control, 'stop_support_sessions'), mock.patch.object(control, 'source_check', return_value=self.release), mock.patch.object(control, 'metadata', return_value=(self.m, self.progress, [self.root])), mock.patch.object(control, 'legacy', return_value=self.legacy):
            control.reboot_storage('preserve-for-reboot')
            self.assertFalse(self.output.exists())
            self.assertEqual((backup / 'pipeline.sqlite3').read_bytes(), b'checkpoint')
            control.reboot_storage('restore-after-reboot')
        self.assertEqual((self.output / 'pipeline.sqlite3').read_bytes(), b'checkpoint')
        self.assertTrue((self.output / self.root['path'] / 'COMPLETE.json').is_file())
        self.assertFalse(backup.exists())

    def test_reboot_backup_collision_refuses_overwrite(self):
        backup = Path(self.temp.name) / 'persistent_backup'; backup.mkdir()
        with mock.patch.dict(control.C, reboot_backup=str(backup)):
            with self.assertRaisesRegex(RuntimeError, 'refusing overwrite'): control.relocate_stopped_dataset('preserve-for-reboot')
        self.assertTrue((self.output / 'seal.json').is_file())

    def test_reboot_restore_existing_output_refuses_overwrite(self):
        backup = Path(self.temp.name) / 'persistent_backup'
        with mock.patch.dict(control.C, reboot_backup=str(backup)):
            with self.assertRaisesRegex(RuntimeError, 'refusing overwrite'): control.relocate_stopped_dataset('restore-after-reboot')
        self.assertTrue((self.output / 'seal.json').is_file())

    def test_reboot_preservation_refuses_live_collector(self):
        backup = Path(self.temp.name) / 'persistent_backup'
        with mock.patch.dict(control.C, reboot_backup=str(backup)), mock.patch.object(control, 'metadata', return_value=(self.m, self.progress, [self.root])), mock.patch.object(control, 'reject_active', side_effect=RuntimeError('Collector already running')):
            with self.assertRaisesRegex(RuntimeError, 'already running'): control.relocate_stopped_dataset('preserve-for-reboot')
        self.assertTrue(self.output.exists()); self.assertFalse(backup.exists())

    def test_reboot_support_stop_excludes_collector(self):
        with mock.patch.object(control, 'session_exists', return_value=True), mock.patch.object(control.subprocess, 'run') as run:
            control.stop_support_sessions()
        sessions = [call.args[0][-1] for call in run.call_args_list]
        self.assertNotIn('=' + control.C['session'], sessions)
        self.assertEqual(set(sessions), {'=' + values[1] for values in control.HELPERS.values()})


if __name__ == '__main__': unittest.main()
