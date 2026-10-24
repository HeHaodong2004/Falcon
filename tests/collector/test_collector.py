"""Semantic regression tests; no GPU or simulator needed."""
import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from collector.core import LegalRoot, Query, build_labels, generate_queries, legal_observation


class StraightStaticPlanner:
    def path(self,start,end,cfg):
        return [list(start),list(end)]


def branch(distances, times, completed):
    return dict(branch_id='test',query_id='test',completed=completed,planned_duration=4.,
        termination_reason='query_completed' if completed else 'timeout',truth=[
        dict(timestamp=t,robot={'position':[0,0,0]},humans=[{'position':[d,0,0]}],
             native_human_contact=False,native_static_contact=False) for d,t in zip(distances,times)])


class Contracts(unittest.TestCase):
    def test_censored_is_unknown_even_without_collision(self):
        label=build_labels(branch([3,3],[0,1],False))
        self.assertIsNone(label['full_query_safe']); self.assertTrue(label['censored'])

    def test_disc_and_native_are_distinct_and_timestamps_authoritative(self):
        b=branch([2.,1.,.5],[7.,7.13,7.8],False)
        l=build_labels(b)
        self.assertTrue(l['research_disc_contact']); self.assertFalse(l['native_robot_human_contact'])
        self.assertAlmostEqual(l['first_event_time'],.8)
        self.assertAlmostEqual(l['near_exposure_seconds'],.8)
        self.assertAlmostEqual(l['minimum_human_clearance'],-.05)

    def test_native_contact_without_disc_overlap(self):
        b=branch([3.,3.],[0.,.024],False); b['truth'][-1]['native_human_contact']=True
        l=build_labels(b)
        self.assertTrue(l['native_robot_human_contact']); self.assertFalse(l['research_disc_contact'])
        self.assertEqual(l['first_event_time'],.024)

    def test_empty_humans(self):
        b=branch([3.,3.],[0.,4.],True)
        for r in b['truth']: r['humans']=[]
        l=build_labels(b)
        self.assertIsNone(l['minimum_human_clearance']); self.assertTrue(l['full_query_safe'])

    def test_fixed_queries_and_timing_variants(self):
        root=LegalRoot('root',(0.,0.,0.),0.,.0486666666667,1/120)
        qs,rejected=generate_queries(root,StraightStaticPlanner())
        self.assertEqual(len(qs),33); self.assertFalse(rejected)
        self.assertEqual([q.data['wait_duration'] for q in qs[:4]],[0,.5,1,0])
        self.assertEqual([q.data['speed_scale'] for q in qs[:4]],[1,1,1,.5])
        before=qs[0].sha256; d=qs[0].data; d['executable_controls'][0][0]=999
        self.assertEqual(before,qs[0].sha256)
        with self.assertRaises(FrozenInstanceError): qs[0].serialized='changed'
        for q in qs:
            x=q.data
            self.assertAlmostEqual(len(x['executable_controls'])*x['control_dt'],x['planned_duration'])
            self.assertTrue(all(abs(c[0])<=25 and abs(c[1])<=10 for c in x['executable_controls']))
            if x['direction']!='WAIT':
                self.assertTrue(np.allclose(np.array(x['nominal_timed_trajectory'][-1][1:4])[[0,2]],np.array(x['spatial_path'][-1])[[0,2]]))

    def test_legal_observation_poison_and_future_mutation(self):
        obs={'agent_0_articulated_agent_jaw_rgb':np.zeros((2,2,3)),
             'agent_0_articulated_agent_jaw_depth':np.ones((2,2,1)),
             'human_future_trajectory':[999], 'hidden_human_position':[999,999,999]}
        meta,im=legal_observation(obs,[0,0,0],0,1,[0,0]); old=copy.deepcopy(im)
        obs['agent_0_articulated_agent_jaw_depth'][:]=0
        self.assertTrue(np.array_equal(im['agent_0_articulated_agent_jaw_depth'],old['agent_0_articulated_agent_jaw_depth']))
        self.assertNotIn('human_future_trajectory',meta); self.assertEqual(meta['human_tracks'],[])


if __name__=='__main__': unittest.main()
