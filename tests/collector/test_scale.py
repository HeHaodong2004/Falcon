"""Bounds, atomic completion, failure isolation and resume without a simulator."""
import argparse
import copy
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import eventwm_scale as scale
from collector.core import write,read
from collector.root_sampling import prefix_commands


class ScaleTests(unittest.TestCase):
    def args(self,path):
        return argparse.Namespace(output=Path(path),scenes=['castle'],human_counts=[0,2],roots_per_config=1,
            max_total_roots=2,max_total_branches=144,max_attempts=2,seed=2301,
            profiles=['wait','seek_visible'],resume=False,stop_after_roots=None,root_timeout=30)

    def test_plan_bounds_and_determinism(self):
        a=self.args('/tmp/unused')
        x=scale.plan(a); self.assertEqual(x,scale.plan(a))
        self.assertEqual(len({r['root_id'] for r in x['roots']}),2)
        a.max_total_branches=71
        with self.assertRaises(ValueError): scale.plan(a)
        a.max_total_branches=144; a.max_total_roots=1
        with self.assertRaises(ValueError): scale.plan(a)
        a.max_total_roots=2; a.human_counts=[0,0]
        with self.assertRaises(ValueError): scale.plan(a)

    def test_prefix_uses_only_legal_history_and_is_bounded(self):
        h=[{'human_tracks':[]}]; r={'sampling':{'profile':'seek_visible'}}
        g=prefix_commands(r,h); self.assertEqual(next(g),[0,6])
        h.append({'human_tracks':[{'visible_now':True}]})
        self.assertEqual(list(g),[[0,0]]*12)
        for profile,total in [('seek_visible',138),('seek_move',138),('seek_lost',150)]:
            self.assertEqual(len(list(prefix_commands({'sampling':{'profile':profile}},[{'human_tracks':[]}]))),total)
        self.assertEqual(list(prefix_commands({'prefix_controls':[[3,4]]},h)),[[3,4]])

    def test_failure_isolation_resume_no_overwrite_and_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            a=self.args(td); launches=[]
            class Process:
                def __init__(self,cmd,**kwargs):
                    path=Path(cmd[cmd.index('--output')+1]); launches.append(str(path)); self.path=path
                    path.mkdir(parents=True); write(path/'diagnostic.json',{'attempt':len(launches)})
                def wait(self,timeout):
                    if len(launches)==1: return 1
                    recipe=read(Path(td)/'recipes'/f'{self.path.parent.name}.json')
                    write(self.path/'roots/replay.json',recipe)
                    write(self.path/'legal_observations/observation_config.json',{'gate':.8})
                    write(self.path/'queries/queries.json',[{'query':{'generation_config':{}}}])
                    return 0
            stats=dict(queries=33,branches=36,primary_branches=33,legal_detection_observations=0,
                visible_history=False,visible_at_root=False,lost_at_root=False,lost_history=False,
                moving_at_root=False,collisions=0,completion={'completed':33},termination={'query_completed':33},
                wait=[0],speed=[1],duration=[3])
            original_handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
            try:
                with patch.object(scale.subprocess,'Popen',Process),patch.object(scale,'provenance',return_value={'test':1}),patch.object(scale,'root_stats',return_value=stats):
                    self.assertEqual(scale.run(a),1)
                    m=read(Path(td)/'manifest.json'); self.assertEqual([r['state'] for r in m['roots']],['failed','completed'])
                    completed=copy.deepcopy(m['roots'][1]); a.resume=True
                    self.assertEqual(scale.run(a),0)
                    m=read(Path(td)/'manifest.json'); self.assertEqual(m['roots'][1],completed)
                    self.assertEqual(len(launches),3); self.assertEqual(len(set(launches)),3)
                    self.assertEqual(scale.run(a),0); self.assertEqual(len(launches),3)
                    corrupt=Path(td)/completed['path']/'diagnostic.json'; corrupt.write_text('corrupt')
                    with self.assertRaisesRegex(RuntimeError,'Global corruption'): scale.run(a)
            finally:
                for s,handler in original_handlers.items(): signal.signal(s,handler)

    def test_free_space_guard_starts_no_root(self):
        with tempfile.TemporaryDirectory() as td:
            a=self.args(td); a.min_free_gb=150.
            from types import SimpleNamespace
            original={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
            try:
                with patch.object(scale,'provenance',return_value={'test':1}),patch.object(scale.shutil,'disk_usage',return_value=SimpleNamespace(free=149_000_000_000)),patch.object(scale.subprocess,'Popen') as process:
                    self.assertEqual(scale.run(a),0)
                    process.assert_not_called()
                m=read(Path(td)/'manifest.json')
                self.assertEqual(m['runner']['state'],'stopped_low_disk')
                self.assertEqual(m['reserved_branches'],0)
                self.assertTrue(all(r['state']=='pending' for r in m['roots']))
            finally:
                for s,handler in original.items(): signal.signal(s,handler)

    def test_global_integrity_failure_stops_further_roots(self):
        with tempfile.TemporaryDirectory() as td:
            a=self.args(td)
            class Process:
                def __init__(self,cmd,stdout,**kwargs):
                    stdout.write('Collector integrity audit failed\n'); stdout.flush()
                def wait(self,timeout): return 1
            original={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
            try:
                with patch.object(scale,'provenance',return_value={'test':1}),patch.object(scale.subprocess,'Popen',Process):
                    with self.assertRaises(scale.GlobalIntegrityError): scale.run(a)
                m=read(Path(td)/'manifest.json')
                self.assertEqual([r['state'] for r in m['roots']],['failed','pending'])
                self.assertEqual(m['runner']['state'],'global_integrity_failure')
            finally:
                for s,handler in original.items(): signal.signal(s,handler)

    def test_completion_requires_manifest_and_marker(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); (p/'partial').mkdir()
            with self.assertRaises(FileNotFoundError): scale.verify_root(p,{'path':'partial','root_id':'x'})


if __name__=='__main__': unittest.main()
