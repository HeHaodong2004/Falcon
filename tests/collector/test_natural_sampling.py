"""Natural selection invariance, coverage compatibility, and frozen contracts."""
import copy
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from collector.natural_sampling import natural_selection,natural_commands
from collector.root_sampling import prefix_commands
from collector.core import read,write,load_wm_input
from collector.sampling_audit import root_diagnostics
from test_v3 import fixture

REPO=Path(__file__).resolve().parents[2]


def recipe():
    return dict(seed=24001,sampling=dict(profile='natural',sampler_seed=771,window_start_step=20,
        window_end_step=180,prefix_movement_queries=3,prelude_controls=[]))


class NaturalSamplingTests(unittest.TestCase):
    def test_time_deterministic_from_two_seeds(self):
        r=recipe();a=natural_selection(r['seed'],r['sampling'])
        self.assertEqual(a,natural_selection(r['seed'],r['sampling']))
        self.assertGreaterEqual(a['selected_root_step'],20);self.assertLessEqual(a['selected_root_step'],180)
        choices={natural_selection(i,r['sampling'])['selected_root_step'] for i in range(100)}
        self.assertGreater(len(choices),40)
    def test_no_global_rng_consumption(self):
        random.seed(1);before=random.getstate();r=recipe();natural_selection(r['seed'],r['sampling'])
        self.assertEqual(before,random.getstate())
    def check_poison(self,field):
        r=recipe();a=natural_selection(r['seed'],r['sampling']);r['sampling'][field]={'poison':999}
        self.assertEqual(a,natural_selection(r['seed'],r['sampling']))
    def test_no_gt_distance_selection(self):self.check_poison('nearest_human_distance')
    def test_no_future_labels_selection(self):self.check_poison('future_labels')
    def test_no_future_trajectory_selection(self):self.check_poison('future_human_trajectory')
    def test_no_visibility_selection(self):self.check_poison('visible_now')
    def test_no_lost_selection(self):self.check_poison('lost_tracks')
    def test_history_not_even_read(self):
        class Poison:
            def __getitem__(self,key):raise AssertionError('History read during natural selection')
        commands=list(prefix_commands(recipe(),Poison(),natural_motion=lambda _: [[1.,0.]]*3))
        self.assertGreater(len(commands),3)
    def test_visible_or_zero_visible_both_selected(self):
        outputs=[]
        for tracks in ([],[{'visible_now':True}],[{'visible_now':False}]):
            outputs.append(list(prefix_commands(recipe(),[{'human_tracks':tracks}],natural_motion=lambda _: [[1.,0.]]*3)))
        self.assertEqual(outputs[0],outputs[1]);self.assertEqual(outputs[1],outputs[2])
    def test_prelude_counts_in_absolute_window(self):
        r=recipe();r['sampling'].update(prelude_controls=[[0.,2.]]*10,window_start_step=21,window_end_step=21)
        commands=list(natural_commands(r,lambda _: [[1.,0.]]*4))
        self.assertEqual(len(commands),21);self.assertEqual(commands[:10],[[0.,2.]]*10)
        self.assertEqual(commands[14:],[[0.,0.]]*7)
    def test_window_rejects_t0_and_malformed(self):
        for a,b in [(0,20),(30,20)]:
            r=recipe();r['sampling'].update(window_start_step=a,window_end_step=b)
            with self.assertRaises(ValueError):natural_selection(r['seed'],r['sampling'])
    def test_frozen_explicit_replay_wins(self):
        r=recipe();r['prefix_controls']=[[0,0],[0,6]]
        self.assertEqual(list(prefix_commands(r,None)),r['prefix_controls'])
    def check_seek(self,profile):
        old={};exec(subprocess.check_output(['git','show','82097ac:scripts/collector/root_sampling.py'],cwd=REPO),old)
        for visible_after in (0,3,500):
            r=dict(sampling={'profile':profile});histories=[[{'human_tracks':[]}],[{'human_tracks':[]}]];outputs=[]
            for fn,h in zip((old['prefix_commands'],prefix_commands),histories):
                commands=[]
                for i,c in enumerate(fn(r,h)):
                    commands.append(c)
                    if i==visible_after:h[-1]['human_tracks']=[{'visible_now':True}]
                outputs.append(commands)
            self.assertEqual(*outputs)
    def test_seek_visible_unchanged(self):self.check_seek('seek_visible')
    def test_seek_lost_unchanged(self):self.check_seek('seek_lost')
    def test_wait_and_seek_move_unchanged(self):
        self.check_seek('seek_move')
        self.assertEqual(list(prefix_commands(dict(sampling=dict(profile='wait',wait_steps=5)),None)),[[0.,0.]]*5)
    def test_profile_persisted(self):
        from eventwm_v3_sampling_pilot import job_table
        cfg=read(REPO/'configs/eventwm_v3_natural_pilot.json');a=job_table(cfg)
        self.assertEqual(a,job_table(cfg));self.assertEqual(len(a),110)
        self.assertEqual(len({r['root_id'] for r in a}),110)
        self.assertTrue(all(r['sampling_profile_family']=='natural' for r in a))
        self.assertTrue(all(r['recipe']['sampling']['profile']=='natural' for r in a))
        comparison=job_table(cfg,True);self.assertEqual(len(comparison),16)
        self.assertTrue(all(r['sampling_profile_family']=='coverage' for r in comparison))
    def test_diagnostics_are_post_selection_and_not_model_input(self):
        b,q=fixture(distance=.8)
        history=[dict(timestamp=0,robot_position=[0,0,0],robot_yaw=0,previous_command=[0,0],
            human_tracks=[],tracking_available=True,sensor_keys=[])]
        diag=root_diagnostics(recipe(),b['truth'][0],history,33)
        self.assertTrue(diag['already_in_social_violation_at_root']);self.assertFalse(diag['used_for_selection'])
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);write(p/'legal_observations/history.json',history);write(p/'queries/queries.json',[dict(query=q)])
            before=load_wm_input(p,q['query_id'])
            write(p/'audits/root_sampling.json',diag);history[0].update(diag);q.update(diag)
            write(p/'legal_observations/history.json',history);write(p/'queries/queries.json',[dict(query=q)])
            self.assertEqual(before,load_wm_input(p,q['query_id']))
    def test_labels_static_humans_core_byte_unchanged(self):
        paths=['scripts/collector/v2.py','scripts/collector/static_feasibility.py',
            'scripts/collector/static_habitat.py','scripts/collector/social_habitat.py','scripts/collector/core.py',
            'scripts/collector/observation.py','scripts/controlled/runtime.py','scripts/controlled/run_case.py',
            'falcon/additional_action.py','falcon/additional_metric.py','falcon/rearrange_sim_v2.py',
            'habitat-lab/habitat/tasks/rearrange/social_nav_v2/socialnav_task.py']
        for path in paths:
            # Compare repaired static files at the preserved previous epoch.
            source=Path('/home/haodong/workspace/projects/eventwm-falcon/production/eventwm-v4-science') if '/static_' in path else REPO
            self.assertEqual((source/path).read_bytes(),subprocess.check_output(['git','show','82097ac:'+path],cwd=REPO),path)
    def test_visuals_off(self):
        from eventwm_v3_sampling_pilot import job_table
        c=read(REPO/'configs/eventwm_v3_natural_pilot.json')
        self.assertFalse(c['visualization']);self.assertTrue(all(not r['recipe']['debug_static_reference'] for r in job_table(c)))
    def test_zero_humans_diagnostic(self):
        b,q=fixture(count=0);d=root_diagnostics(recipe(),b['truth'][0],[{'human_tracks':[]}],22)
        self.assertIsNone(d['nearest_social_distance_at_root']);self.assertFalse(d['already_in_social_violation_at_root'])

if __name__=='__main__':unittest.main()
