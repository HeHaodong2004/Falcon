import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from collector.wm_sampling import legal_root_state,matches,wm_commands
from collector.root_sampling import prefix_commands
from collector.core import read,digest
from eventwm_wm_train_pilot import job_table


def track(visible=True,d=2.,age=0.):
    return dict(valid=True,visible_now=visible,position=[d,0.,0.],observation_age=age)
def row(tracks=()):
    return dict(timestamp=1.,robot_position=[0.,0.,0.],human_tracks=list(tracks))
def sampling(category='visible'):
    return dict(profile='wm_train',human_count=2,target_category=category,
        target_visible_bucket=None,target_lost_bucket=None,min_prefix_steps=1,
        max_prefix_steps=4,scan_angular_velocity=6.,allow_visible_inside_1m=False)


class WMTrainSamplingTests(unittest.TestCase):
    def test_categories_mutually_exclusive(self):
        for tracks,expected in [([], 'unseen'),([track()], 'visible'),([track(False,age=.2)],'lost'),
                                ([track(),track(False,age=.2)],'visible_lost')]:
            self.assertEqual(legal_root_state([row(tracks)],2,{})['category'],expected)
        self.assertEqual(legal_root_state([row()],0,{})['category'],'zero')
    def test_expired_history_not_mislabeled_unseen(self):
        self.assertEqual(legal_root_state([row([track()]),row()],1,{})['category'],'observed_expired')
    def test_visible_distance_boundaries(self):
        for d,b in [(.99,'inside_1m'),(1.,'near'),(1.499,'near'),(1.5,'medium'),(3.,'medium'),(3.001,'far')]:
            self.assertEqual(legal_root_state([row([track(d=d)])],1,{})['visible_distance_bucket'],b)
    def test_lost_age_boundaries(self):
        for age,b in [(.1,'recent'),(.5,'recent'),(.501,'medium'),(1.,'medium'),(1.001,'stale'),(2.,'stale')]:
            self.assertEqual(legal_root_state([row([track(False,age=age)])],1,{})['lost_age_bucket'],b)
    def test_oldest_lost_and_nearest_visible(self):
        s=legal_root_state([row([track(d=2),track(d=4),track(False,age=.1),track(False,age=1.2)])],4,{})
        self.assertEqual(s['nearest_legal_visible_distance'],2);self.assertEqual(s['oldest_lost_age'],1.2)
    def test_forbidden_fields_do_not_affect_state(self):
        h=[row([track()])];before=legal_root_state(h,2,{})
        for field in ('hidden_humans','gt_velocity','future_labels','future_human_trajectory','social_metric','collision'):
            h[-1][field]={'poison':99};h[-1]['human_tracks'][0][field]={'poison':99}
        self.assertEqual(before,legal_root_state(h,2,{}))
    def test_submeter_visible_rejected_without_gt(self):
        s=legal_root_state([row([track(d=.8)])],1,{})
        self.assertFalse(matches(s,sampling()))
    def test_target_bucket_required(self):
        cfg=sampling();cfg['target_visible_bucket']='far'
        self.assertFalse(matches(legal_root_state([row([track(d=2)])],1,{}),cfg))
    def test_generator_observes_arrived_transition(self):
        h=[row()];g=wm_commands(dict(sampling=sampling()),h)
        self.assertEqual(next(g),[0.,6.]);h.append(row([track()]))
        with self.assertRaises(StopIteration):next(g)
    def test_unfilled_target_fails_without_unseen_fallback(self):
        with self.assertRaisesRegex(RuntimeError,'wm_train_stratum_unavailable'):
            list(wm_commands(dict(sampling=sampling()),[row()]))
    def test_unseen_does_not_accept_expired(self):
        with self.assertRaisesRegex(RuntimeError,'wm_train_stratum_unavailable'):
            list(wm_commands(dict(sampling=sampling('unseen')),[row([track()]),row()]))
    def test_explicit_replay_wins_for_wm_train(self):
        self.assertEqual(list(prefix_commands(dict(sampling=sampling(),prefix_controls=[[0,0]]),None)),[[0,0]])
    def test_job_table_persistable_deterministic_n8(self):
        cfg=read(Path(__file__).resolve().parents[2]/'configs/eventwm_wm_train_pilot.json')
        jobs=job_table(cfg);self.assertEqual(jobs,job_table(cfg));self.assertEqual(len(jobs),60)
        self.assertEqual(len({j['root_id'] for j in jobs}),60)
        self.assertTrue(all(j['recipe']['query_config']['movement_queries']==8 for j in jobs))
        self.assertTrue(all(j['human_count']>=2 for j in jobs if j['target_category']=='visible_lost'))
        self.assertTrue(all(j['sampling_profile_family']=='targeted' for j in jobs))
        self.assertFalse(cfg['visualization'])
        self.assertEqual(digest(jobs),digest(json.loads(json.dumps(jobs))))
    def test_requested_root_allocations(self):
        from collections import Counter
        cfg=read(Path(__file__).resolve().parents[2]/'configs/eventwm_wm_train_pilot.json')
        self.assertEqual(Counter(j['target_category'] for j in job_table(cfg)),dict(zero=6,unseen=9,visible=18,lost=18,visible_lost=9))

if __name__=='__main__':unittest.main()
