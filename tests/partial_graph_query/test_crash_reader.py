"""Publication failure recovery and explicit supervised-reader boundaries."""
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from eventwm_graph.partial.storage import seal_root,validate_root,online_input,check_legal
from eventwm_graph.partial.dataset import supervised_example,iter_roots
from test_stage1_review import fixture


class PublicationRecovery(unittest.TestCase):
    def test_failure_before_publication_retains_staging_then_resumes(self):
        record,tape,queries,outcomes=fixture()
        original=os.replace
        def fail_publish(source,target):
            if Path(source).is_dir() and Path(source).name.startswith('.partial_'):
                raise OSError('injected publication failure')
            return original(source,target)
        with tempfile.TemporaryDirectory() as directory:
            with patch('eventwm_graph.partial.storage.os.replace',side_effect=fail_publish):
                with self.assertRaisesRegex(OSError,'injected'):
                    seal_root(directory,record,tape,queries,outcomes,min_free_bytes=0)
            failed=list(Path(directory).glob('roots/*/.partial_*/FAILED.json'))
            self.assertEqual(len(failed),1)
            self.assertFalse([p for p in Path(directory).glob('roots/*/*') if not p.name.startswith('.')])
            result=seal_root(directory,record,tape,queries,outcomes,min_free_bytes=0)
            self.assertTrue(failed[0].exists())
            self.assertEqual(validate_root(result['folder'])['root_id'],result['root_id'])
            self.assertEqual(list(iter_roots(directory,result['split'])),[Path(result['folder'])])
            resumed=seal_root(directory,record,tape,queries,outcomes,min_free_bytes=0)
            self.assertEqual(resumed['root_id'],result['root_id'])
            self.assertEqual(resumed['files'],result['files'])

    def test_offline_scan_skips_torn_unpublished_manifest_before_open(self):
        record,tape,queries,outcomes=fixture()
        with tempfile.TemporaryDirectory() as directory:
            result=seal_root(directory,record,tape,queries,outcomes,min_free_bytes=0)
            torn=Path(result['folder']).parent/'.partial_simulated_crash'
            torn.mkdir();(torn/'manifest.json').write_text('{ torn unpublished manifest')
            self.assertEqual(list(iter_roots(directory,result['split'])),[Path(result['folder'])])
            from eventwm_graph.partial.report import summarize
            if result['split']=='train':
                report=summarize([directory],Path(directory)/'report.json')
                self.assertEqual(len(report['roots']),1)

    def test_failure_after_atomic_rename_preserves_valid_receipt(self):
        record,tape,queries,outcomes=fixture();original=os.fsync
        def fail_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):raise OSError('injected post-publication failure')
            return original(fd)
        with tempfile.TemporaryDirectory() as directory:
            with patch('eventwm_graph.partial.storage.os.fsync',side_effect=fail_directory):
                with self.assertRaisesRegex(OSError,'post-publication'):
                    seal_root(directory,record,tape,queries,outcomes,min_free_bytes=0)
            roots=list(Path(directory).glob('roots/*/*/manifest.json'))
            roots=[p for p in roots if not p.parent.name.startswith('.')]
            self.assertEqual(len(roots),1)
            before=validate_root(roots[0].parent)
            resumed=seal_root(directory,record,tape,queries,outcomes,min_free_bytes=0)
            self.assertEqual(resumed['files'],before['files'])

    def test_supervised_reader_keeps_privileged_targets_out_of_inputs(self):
        record,tape,queries,outcomes=fixture()
        with tempfile.TemporaryDirectory() as directory:
            result=seal_root(directory,record,tape,queries,outcomes,min_free_bytes=0)
            example=supervised_example(result['folder'])
            self.assertEqual(example['inputs'],online_input(result['folder']))
            check_legal(example['inputs'])
            self.assertEqual(example['targets']['privileged_step_labels'][0]['node_id'],0)
            self.assertEqual(len(example['targets']['future_observations']),1)
            self.assertEqual(list(iter_roots(directory,result['split'])),[Path(result['folder'])])
            with self.assertRaises(ValueError):list(iter_roots(directory,'all'))


if __name__=='__main__':unittest.main()
