"""Real candidate orchestration with fake native physics; NOT Habitat speed tests."""
from collections import deque
import concurrent.futures as cf
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import eventwm_pipeline_runner as runner
import eventwm_pipeline_io as io
from collector.core import read, write


def manifest():
    scenes=[dict(name='scene'+str(i),scene={'name':'scene'+str(i)}) for i in range(3)]
    return dict(roots=[],scenes=scenes,master_seed=410000,next_sequence=0,max_attempts=2,
                max_completed_roots=100,target_primary_queries=3300,collection_seconds=0.,
                sessions=[],state='running',pipeline_settings={})


def root_spec(index,scenes,seed):
    scene=scenes[index%len(scenes)]['name']
    return dict(root_id='r'+str(index),sequence=index,scene=scene,seed=seed+index,human_count=0,
                family='fake',physical_scene_id=scene,role='train',scene_role='train',state='pending',attempts=[],
                recipe=dict(root_id='r'+str(index),case='case'+str(index),seed=seed+index))


def fake_prepare(output,scene,root,covered):
    time.sleep(.025 if root['sequence']%3==0 else .005)
    ep=dict(position=len(covered),seed=root['seed'])
    p=Path(output)/'catalogue/episodes'/f"{root['root_id']}.json"
    io.immutable_json(p,ep)
    io.immutable_json(Path(output)/'recipes'/f"{root['root_id']}.json",root['recipe'])
    return dict(error=None,start_region=str(len(covered)),episode_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),seconds=.005)


class FakeWorker:
    def __init__(self,index,repo,output,session,pids,checkpoints,started):
        self.index=index
        self.last_scene=None
        self.proc=SimpleNamespace(poll=lambda:0,pid=99999999)
    def execute(self,j):
        rid,wid=j['root_id'],j['worker_id']
        time.sleep(.12 if rid=='r0' and wid=='root' else .001)
        result=dict(error=None,dispatch_wall=.001,label={'static_feasibility_violation':False})
        if wid=='root':
            write(Path(j['output'])/'queries/queries.json',[{'query':{'query_id':rid+'/q'+str(i)}} for i in range(33)])
            result['checkpoint']=dict(checkpoint_pid=99999999,socket='/fake')
        self.last_scene=j['scene']
        return result
    def close(self): pass


class FakeValidator:
    def __init__(self,*args): self.proc=SimpleNamespace(poll=lambda:0)
    def execute(self,path):
        time.sleep(.02)
        write(path/'VALIDATED.json',{})
        return dict(seconds=.02,result={'retained':33})
    def close(self): pass


def fake_commit(output,root):
    if not (output/root['attempts'][-1]['path']/'VALIDATED.json').exists():
        raise AssertionError('Publishing before validation')
    st=dict(primary_queries=33,spatial_region_id=root['scene'],bytes=0,legal_state=dict(category='zero',visible_distance_bucket=None,lost_age_bucket=None),
            robot_state='stationary',static_violations=0,invalid=0)
    root.update(state='completed',stats=st,query_ids=[root['root_id']+'/q'+str(i) for i in range(33)])
    root['attempts'][-1]['state']='completed'


class FakeMonitor:
    def __init__(self,*args): self.samples=deque()
    def start(self): pass
    def stop(self): pass


class PipelineTests(unittest.TestCase):
    def test_native_comparator_rejects_changed_scientific_hash(self):
        import eventwm_pipeline_benchmark as benchmark
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for name in ('a','b'):
                out=p/name;m=manifest();r=root_spec(0,m['scenes'],410000)
                r['attempts']=[dict(state='failed',path='roots/r0/attempt_01',reason='root_static_clearance_preflight')]
                r['state']='failed';m['roots']=[r];m['sessions']=[{'seconds':1.}]
                write(out/'manifest.json',m)
                result=dict(scientific_hashes={'terminal':'same'},label=None,error={'type':'RootStaticPreflightRejected'})
                write(out/'roots/r0/attempt_01/diagnostic_jobs/root/result.json',result)
            self.assertTrue(benchmark.compare(p/'a',p/'b')['scientific_matches'])
            file=p/'b/roots/r0/attempt_01/diagnostic_jobs/root/result.json'
            data=read(file);data['scientific_hashes']['terminal']='different';write(file,data)
            self.assertFalse(benchmark.compare(p/'a',p/'b')['scientific_matches'])

    def run_fake(self, path, m, args, prepare=fake_prepare, worker=FakeWorker):
        store=io.StateStore(path);store.flush(m,all_roots=True)
        prep_factory=lambda **kw: cf.ThreadPoolExecutor(max_workers=kw['max_workers'])
        with patch.object(runner,'PipelineWorker',worker),patch.object(runner,'Validator',FakeValidator),\
             patch.object(runner,'Monitor',FakeMonitor),patch.object(runner,'prepare_episode',prepare),\
             patch.object(runner.mass,'schedule_spec',side_effect=root_spec),patch.object(runner.old,'commit_root',side_effect=fake_commit),\
             patch.object(runner,'stop_checkpoint'),patch.object(runner,'compact_validated'),patch.object(runner.cf,'ProcessPoolExecutor',side_effect=prep_factory),\
             patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=10**12)):
            code=runner.coordinate(args,path,m,store)
        store.close()
        return code

    def args(self,**kwargs):
        d=dict(target=3300,concurrency=3,preparers=2,validators=2,prefetch=4,backlog=6,max_proposals=9,fixture=None)
        d.update(kwargs)
        return SimpleNamespace(**d)

    def test_stop_drains_preparations_and_native_work(self):
        import os,signal
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m=manifest()
            def prepare(*args):
                value=fake_prepare(*args)
                if args[2]['root_id']=='r0': os.kill(os.getpid(),signal.SIGTERM)
                return value
            self.assertEqual(self.run_fake(p,m,self.args(max_proposals=100),prepare=prepare),0)
            self.assertEqual(m['state'],'stopped_by_user')
            self.assertLess(m['next_sequence'],100)
            self.assertTrue(all(r['state']=='completed' for r in m['roots']))

    def test_timeout_retries_same_proposal_without_resampling(self):
        calls=[]
        class OnceTimeout(FakeWorker):
            def execute(self,j):
                if j['root_id']=='r0' and j['worker_id']=='root' and not calls:
                    calls.append(j)
                    raise TimeoutError('injected native timeout')
                return super().execute(j)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m=manifest()
            with patch.object(runner.os,'killpg'):
                self.assertEqual(self.run_fake(p,m,self.args(max_proposals=3),worker=OnceTimeout),0)
            self.assertEqual(m['next_sequence'],3)
            self.assertEqual(len(m['roots'][0]['attempts']),2)
            self.assertEqual(m['roots'][0]['attempts'][0]['reason'],'worker_timeout')
            self.assertEqual(m['roots'][0]['state'],'completed')
            self.assertEqual(read(p/'roots/r0/attempt_01/roots/replay.json'),read(p/'roots/r0/attempt_02/roots/replay.json'))

    def test_resume_preparing_after_target_still_drains_reserved_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m=manifest()
            r=root_spec(0,m['scenes'],410000)
            r.update(state='preparing',_pipeline_ordinal=0,_preparation_covered={})
            m['roots']=[r];m['next_sequence']=1
            # A zero target is used only in this internal test to exercise the
            # reached-target restart branch with a still-reserved preparation.
            self.assertEqual(self.run_fake(p,m,self.args(target=0)),0)
            self.assertEqual(m['next_sequence'],1)
            self.assertEqual(m['roots'][0]['state'],'completed')

    def test_staging_budget_reserves_space_for_active_roots(self):
        self.assertTrue(runner.executing_capacity({'first'},0,35,36,72))
        self.assertFalse(runner.executing_capacity(set(range(36)),0,0,36,72))
        self.assertFalse(runner.executing_capacity({'first'},0,71,36,72))

    def test_incremental_store_and_crash_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m=manifest();r=root_spec(0,m['scenes'],410000)
            r['_pipeline_ordinal']=0;m['roots']=[r]
            result=fake_prepare(p,{},r,{})
            r['episode_sha256']=result['episode_sha256']
            store=io.StateStore(p);store.flush(m,all_roots=True)
            def persist(root): store.touch(root);store.flush(m)
            with patch.object(io,'immutable_json',side_effect=KeyboardInterrupt('crash after reservation')):
                with self.assertRaises(KeyboardInterrupt): io.reserve_attempt(p,r,p/'logs',persist)
            restored=store.load();r=restored['roots'][0]
            self.assertEqual(r['attempts'][0]['state'],'reserved')
            path=io.reserve_attempt(p,r,p/'logs',lambda r:None)
            self.assertEqual(len(r['attempts']),1)
            self.assertEqual(read(path/'roots/replay.json'),r['recipe'])
            self.assertEqual((path/'roots/replay.json').read_bytes(),(p/'recipes/r0.json').read_bytes())
            store.close()

    def test_different_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'value.json';io.immutable_json(p,{'x':1})
            with self.assertRaises(RuntimeError): io.immutable_json(p,{'x':2})
            self.assertEqual(read(p),{'x':1})

    def test_indexed_lookup_tracks_each_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'catalogue/scenes.json',[{'name':'s1'},{'name':'s2'}])
            rt=SimpleNamespace(read=read);rc=SimpleNamespace(read=read)
            io.install_indexed_reads(rt,rc,{},p)
            for rid,scene in [('r1','s1'),('r2','s2')]:
                case=dict(tag=rid,scene=scene)
                write(p/'catalogue/cases'/f'{rid}.json',case)
                write(p/rid/'roots/replay.json',dict(root_id=rid))
                rt.read(p/rid/'roots/replay.json')
                self.assertEqual(rt.read(p/'catalogue/cases.json'),[case])
                self.assertEqual(rc.read(p/'catalogue/scenes.json'),[{'name':scene}])

    def test_preparation_integrity_exception_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'mesh').write_text('mesh')
            scene=dict(scene={'navmesh':str(p/'mesh')},navmesh_sha256='wrong')
            with self.assertRaisesRegex(RuntimeError,'prepared_navmesh_hash_mismatch'):
                io.prepare_episode(p,scene,{'root_id':'r'}, {})

    def test_known_preparation_rejection_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'mesh').write_text('mesh')
            scene=dict(scene={'navmesh':str(p/'mesh')},navmesh_sha256=hashlib.sha256(b'mesh').hexdigest())
            r=dict(root_id='r',seed=1,human_count=0)
            with patch.object(runner.mass,'episode_for',side_effect=RuntimeError('static_episode_unavailable: no connected 4-10m goal')):
                self.assertIsNotNone(io.prepare_episode(p,scene,r,{})['error'])
            with patch.object(runner.mass,'episode_for',side_effect=RuntimeError('programming failure')):
                with self.assertRaisesRegex(RuntimeError,'programming failure'): io.prepare_episode(p,scene,r,{})

    def test_whole_coordinator_order_coverage_and_replenishment(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m=manifest();store=io.StateStore(p);store.flush(m,all_roots=True)
            args=SimpleNamespace(target=3300,concurrency=3,preparers=2,validators=2,prefetch=4,backlog=6,max_proposals=9,fixture=None)
            prep_factory=lambda **kw: cf.ThreadPoolExecutor(max_workers=kw['max_workers'])
            with patch.object(runner,'PipelineWorker',FakeWorker),patch.object(runner,'Validator',FakeValidator),\
                 patch.object(runner,'Monitor',FakeMonitor),patch.object(runner,'prepare_episode',fake_prepare),\
                 patch.object(runner.mass,'schedule_spec',side_effect=root_spec),patch.object(runner.old,'commit_root',side_effect=fake_commit),\
                 patch.object(runner,'stop_checkpoint'),patch.object(runner,'compact_validated'),patch.object(runner.cf,'ProcessPoolExecutor',side_effect=prep_factory),\
                 patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=10**12)):
                code=runner.coordinate(args,p,m,store)
            self.assertEqual(code,0)
            self.assertEqual(m['next_sequence'],9)
            self.assertTrue(all(r['state']=='completed' for r in m['roots']))
            for scene in ('scene0','scene1','scene2'):
                self.assertEqual([r['start_region'] for r in m['roots'] if r['scene']==scene],['0','1','2'])
            self.assertEqual(store.load()['roots'],m['roots'])
            log=next((p/'logs').glob('*/scheduler_events.jsonl'))
            events=[json.loads(l) for l in log.read_text().splitlines()]
            first_dispatch=next(i for i,e in enumerate(events) if e['kind']=='dispatch')
            self.assertLess(sum(e['kind']=='preparation_end' for e in events[:first_dispatch]),4)
            self.assertEqual(len([e for e in events if e['kind']=='result']),9*37)
            # Later roots can start before the earliest long root has validated.
            r3=next(e['time'] for e in events if e['kind']=='dispatch' and e['root_id']=='r3')
            r0=next(e['time'] for e in events if e['kind']=='validation_end' and e['root_id']=='r0')
            self.assertLess(r3,r0)
            store.close()


if __name__=='__main__': unittest.main()
