"""Recovery regressions: wall-clock limits and unknown outcomes are not labels."""
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import eventwm_collect as collect
import eventwm_mass_collection as mass


class RecoveryTests(unittest.TestCase):
    def test_worker_timeout_preserves_execution_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            args=SimpleNamespace(output=out,protocol='v3',storage_profile='production')
            with patch.object(collect.subprocess,'run',side_effect=subprocess.TimeoutExpired(['worker'],300)):
                with self.assertRaises(subprocess.TimeoutExpired):collect.launch(args,'root')
            execution=mass.read(out/'audits/workers/root/execution.json')
            self.assertTrue(execution['timed_out']);self.assertIsNone(execution['returncode'])
            self.assertEqual(mass.read(out/'audits/workers/root/collection_error.json')['termination_reason'],'worker_timeout')

    def test_unknown_only_root_does_not_stop_other_roots(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'output').mkdir();out=base/'dataset'
            root=mass.schedule_spec(0,[dict(name='scene',physical_scene_id='scene',family='test',scene_role='training')])
            mass.write(out/'recipes'/f"{root['root_id']}.json",root['recipe'])
            manifest=dict(roots=[root],scenes=[{'name':'scene'}],max_completed_roots=1,
                target_primary_queries=33,collection_seconds=0.,sessions=[],state='prepared',root_timeout_seconds=12000)
            args=SimpleNamespace(output=out,benchmark=True,stop_after_attempts=1,concurrency=1,
                gpus=[0],storage_profile='structured_production',min_free_gb=150,target=33,max_roots=1)
            with patch.object(mass,'REPO',base),patch.object(mass,'init',return_value=manifest), \
                 patch.object(mass,'catalog_link'),patch.object(mass,'hardware',return_value={}), \
                 patch.object(mass.shutil,'disk_usage',return_value=SimpleNamespace(free=1000*10**9)), \
                 patch.object(mass.subprocess,'Popen') as launch,patch.object(mass.time,'sleep'), \
                 patch.object(mass,'inspect_root',side_effect=mass.NoRetainedValidQueries('no_retained_valid_queries')):
                launch.return_value.pid=123;launch.return_value.poll.return_value=0
                self.assertEqual(mass.run(args),0)
            self.assertEqual(manifest['state'],'bounded_benchmark_finished')
            self.assertEqual(root['attempts'][-1]['reason'],'no_retained_valid_queries')
            self.assertEqual(mass.aggregate(manifest)['primary_queries'],0)


if __name__=='__main__':unittest.main()
