import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import eventwm_mass_collection as mass
from eventwm_mass_collection import schedule_spec,region,CATEGORIES

class MassSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.scenes=[dict(name=f's{i}',physical_scene_id=f'MP3D:s{i}',family='MP3D',scene_role='unseen_evaluation' if i==2 else 'training') for i in range(5)]
    def test_round_robin_resume_prefix_and_unique_roots(self):
        first=[schedule_spec(i,self.scenes) for i in range(30)]
        extended=[schedule_spec(i,self.scenes) for i in range(100)]
        self.assertEqual(first,extended[:30]);self.assertEqual(len({x['root_id'] for x in extended}),100)
        for offset in range(0,30,5):self.assertEqual({r['scene'] for r in first[offset:offset+5]},{s['name'] for s in self.scenes})
    def test_holdout_and_natural_roles_do_not_leak(self):
        jobs=[schedule_spec(i,self.scenes) for i in range(800)]
        for r in jobs:
            self.assertEqual(r['split_group'],r['root_id'])
            if r['scene_role']=='unseen_evaluation':self.assertEqual(r['role'],'unseen_evaluation')
            elif r['sampling_profile']=='natural':self.assertEqual(r['role'],'natural_reference')
            else:self.assertEqual(r['role'],'train')
            self.assertNotIn('label',r['recipe']);self.assertFalse(r['recipe']['debug_static_reference'])
        self.assertEqual(sum(r['sampling_profile']=='natural' for r in jobs),100)
        self.assertEqual({r['human_count'] for r in jobs},{0,1,2,4,6})
    def test_no_unsupported_human_stratum(self):
        for i in range(200):
            r=schedule_spec(i,self.scenes)
            if r['requested_category']=='visible_lost':self.assertGreaterEqual(r['human_count'],2)
            if r['requested_category']=='zero':self.assertEqual(r['human_count'],0)
    def test_canonical_stratum_mass_independent_of_natural_slots(self):
        from collections import Counter
        # Any prefix of targeted opportunities has the same deterministic sequence,
        # regardless of scene count and interleaved natural-reference opportunities.
        expected=None
        for n in (1,5,952):
            scenes=[dict(name=f's{i}',physical_scene_id=f's{i}',family='other',scene_role='training') for i in range(n)]
            jobs=[]
            for i in range(5000):
                r=schedule_spec(i,scenes)
                if r['sampling_profile']=='wm_train':jobs.append(r)
                if len(jobs)==4000:break
            cats=Counter(r['requested_category'] for r in jobs)
            self.assertEqual(cats,{'zero':400,'unseen':600,'visible':1200,'lost':1200,'visible_lost':600})
            vs=Counter(r['recipe']['sampling']['target_visible_bucket'] for r in jobs if r['requested_category'] in ('visible','visible_lost'))
            ls=Counter(r['recipe']['sampling']['target_lost_bucket'] for r in jobs if r['requested_category'] in ('lost','visible_lost'))
            self.assertEqual(vs,{'near':540,'medium':900,'far':360})
            self.assertEqual(ls,{'recent':900,'medium':630,'stale':270})
            sequence=[(r['requested_category'],r['recipe']['sampling']['target_visible_bucket'],r['recipe']['sampling']['target_lost_bucket']) for r in jobs]
            if expected is not None:self.assertEqual(sequence,expected)
            expected=sequence
    def test_regions_are_diagnostic_and_deterministic(self):
        self.assertEqual(region([3.1,2.1,-.1]),'1_1_-1')
        r=schedule_spec(0,self.scenes)
        self.assertNotIn('spatial_region_id',r['recipe'])

class MassSafetyTests(unittest.TestCase):
    def test_disk_guard_stops_before_any_worker(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'output').mkdir()
            args=SimpleNamespace(output=base/'dataset',benchmark=True,stop_after_attempts=1,concurrency=1,
                 gpus=[1],storage_profile='structured_production',min_free_gb=150,target=33,max_roots=1)
            manifest=dict(roots=[],scenes=[],max_completed_roots=1,target_primary_queries=33,
                 collection_seconds=0.,sessions=[],state='prepared',source_hashes={})
            with patch.object(mass,'REPO',base),patch.object(mass,'init',return_value=manifest), \
                 patch.object(mass,'catalog_link'),patch.object(mass,'hardware',return_value={}), \
                 patch.object(mass.shutil,'disk_usage',return_value=SimpleNamespace(free=149*10**9)), \
                 patch.object(mass.subprocess,'Popen') as launch:
                self.assertEqual(mass.run(args),1)
                launch.assert_not_called()
                self.assertEqual(manifest['state'],'stopped_disk_guard')
    def test_benchmark_is_hard_bounded(self):
        args=SimpleNamespace(concurrency=1,benchmark=True,stop_after_attempts=None)
        with self.assertRaisesRegex(ValueError,'bounded'):mass.run(args)
    def test_high_benchmark_ceiling_and_stagger_validation(self):
        with self.assertRaisesRegex(ValueError,'1..24'):
            mass.run(SimpleNamespace(concurrency=25))
        with self.assertRaisesRegex(ValueError,'bounded'):
            mass.run(SimpleNamespace(concurrency=24,benchmark=True,stop_after_attempts=25))
        with self.assertRaisesRegex(ValueError,'stagger'):
            mass.run(SimpleNamespace(concurrency=24,benchmark=True,stop_after_attempts=24,startup_stagger=-1))
    def test_static_replay_and_oom_failures_are_global(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);log=root/'run.log'
            for marker,expected in [('STATIC_FEASIBILITY_VIOLATION','static_violation'),
                                    ('Root replay mismatch','integrity_failure'),('CUDA_ERROR_OUT_OF_MEMORY','oom')]:
                log.write_text(marker)
                reason,global_failure,_=mass.failure_reason(root,log)
                self.assertTrue(global_failure);self.assertEqual(reason,expected)
    def test_interrupted_manifest_recovers_completed_root_without_new_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);path=out/'roots/r/attempt_01';path.mkdir(parents=True)
            (path/'data.json').write_text('{}')
            root=dict(root_id='r',role='train',scene_role='training',physical_scene_id='scene',state='running',
                      attempts=[dict(path='roots/r/attempt_01',state='running')])
            root_seals=mass.seals(path);mass.write(out/'root_seals/r.json',root_seals)
            mass.write(path/'COMPLETE.json',dict(root_id='r',seal_digest=mass.digest(root_seals)))
            with patch.object(mass,'inspect_root',return_value=({'primary_queries':33},[{'query_id':'q'}])):
                self.assertTrue(mass.recover_completed(out,root))
            self.assertEqual(root['state'],'completed');self.assertEqual(len(root['attempts']),1)
            self.assertTrue(root['attempts'][0]['recovered_completion'])
            self.assertTrue((out/'primary_indices/r.json').exists())
            (path/'data.json').write_text('corrupt')
            with self.assertRaisesRegex(RuntimeError,'corruption'):mass.recover_completed(out,root)
    def test_monitor_does_not_modify_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            m=dict(state='prepared_not_started',roots=[],scenes=[],max_completed_roots=1,
                   target_primary_queries=33,collection_seconds=0.)
            mass.save(out,m)
            before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
            with patch.object(mass,'hardware',return_value={'gpus':[]}),patch.object(mass.time,'sleep'),patch('builtins.print'):
                mass.status(SimpleNamespace(output=out))
            self.assertEqual(before,{str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()})
    def test_resume_rejects_protocol_drift(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);mass.write(out/'manifest.json',dict(protocol='legacy'))
            with self.assertRaisesRegex(RuntimeError,'protocol mismatch'):
                mass.init(SimpleNamespace(output=out))

if __name__=='__main__':unittest.main()
