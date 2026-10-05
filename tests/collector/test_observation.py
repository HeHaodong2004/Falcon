"""Visibility, causal memory, metadata poison and nested loader boundary."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
import ast
import subprocess
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from collector.observation import (ObservationConfig, IdealHumanDetector, LegalHumanTracker,
                                   HumanDetection, TRACK_FIELDS)
from collector.core import LegalRoot, generate_queries, legal_observation, load_wm_input, write


CFG = ObservationConfig(90.,90.,10.)
ORIGIN = np.array([0.,1.,0.])
ROTATION = np.eye(3)


def world(position=(0.,0.,-2.), **extra):
    return [dict(position=list(position),active=True,diagnostic_id=999,**extra)]


def visible(a,b): return True,None

def blocked(a,b): return False,{'object_id':-1,'distance_m':1.}


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.detector=IdealHumanDetector(CFG); self.tracker=LegalHumanTracker(CFG)

    def observe(self,t,w,los=visible):
        ds,diag=self.detector.detect(t,ORIGIN,ROTATION,w,los)
        return self.tracker.update(t,ds),ds,diag

    def test_a_visible_position_and_history_velocity(self):
        tracks,ds,_=self.observe(1.,world())
        self.assertEqual(ds[0].position,(0.,0.,-2.)); self.assertTrue(tracks[0]['visible_now'])
        self.assertIsNone(tracks[0]['estimated_velocity']); self.assertFalse(tracks[0]['velocity_valid'])
        tracks,_,_=self.observe(1.056,world((.112,0.,-2.),gt_velocity=[999,999,999]))
        self.assertTrue(tracks[0]['velocity_valid'])
        np.testing.assert_allclose(tracks[0]['estimated_velocity'],[2.,0.,0.],atol=1e-12)

    def test_b_out_of_range(self):
        tracks,ds,diag=self.observe(0.,world((0.,0.,-11.)))
        self.assertFalse(tracks); self.assertFalse(ds); self.assertEqual(diag[0]['reason'],'out_of_range')

    def test_c_outside_fov(self):
        for p in [(0.,0.,2.),(3.,0.,-1.)]:
            ds,diag=self.detector.detect(0.,ORIGIN,ROTATION,world(p),visible)
            self.assertFalse(ds); self.assertEqual(diag[0]['reason'],'outside_fov')

    def test_d_static_occlusion_and_f_never_seen_hidden(self):
        tracks,ds,diag=self.observe(0.,world(),blocked)
        self.assertFalse(ds); self.assertFalse(tracks); self.assertEqual(diag[0]['reason'],'static_occlusion')

    def test_e_hidden_memory_and_g_reappearance(self):
        first,_,_=self.observe(0.,world())
        seen,_,_=self.observe(.048,world((.048,0.,-2.)))
        hidden,_,_=self.observe(.104,world((.5,0.,-2.),gt_velocity=[100,0,0]),blocked)
        t=hidden[0]
        self.assertEqual(t['position'],seen[0]['position']); self.assertEqual(t['estimated_velocity'],seen[0]['estimated_velocity'])
        self.assertFalse(t['visible_now']); self.assertEqual(t['last_seen_time'],.048)
        self.assertAlmostEqual(t['observation_age'],.056)
        self.assertEqual(t['num_observations'],2)
        again,_,_=self.observe(.2,world((.2,0.,-2.)))
        self.assertEqual(again[0]['track_id'],first[0]['track_id']); self.assertEqual(again[0]['observation_age'],0.)
        np.testing.assert_allclose(again[0]['estimated_velocity'],[1.,0.,0.])

    def test_expiration_and_long_gap_velocity_reset(self):
        self.observe(0.,world()); self.observe(.05,world((.05,0.,-2.)))
        tracks,_,_=self.observe(.8,world((.1,0.,-2.)))
        self.assertEqual(tracks[0]['num_observations'],3); self.assertFalse(tracks[0]['velocity_valid'])
        tracks,_,_=self.observe(2.800001,[],blocked)
        self.assertFalse(tracks)
        tracks,_,_=self.observe(3.,world((.1,0.,-2.)))
        self.assertEqual(tracks[0]['track_id'],'track_000001')

    def test_identity_is_not_oracle_and_input_order_not_identity(self):
        tracks,_,_=self.observe(0.,world())
        far=world((2.,0.,-3.)); far[0]['diagnostic_id']=999
        tracks,_,_=self.observe(.05,far)
        self.assertEqual(len(tracks),2); self.assertFalse(tracks[0]['visible_now'])
        self.assertTrue(tracks[1]['visible_now'])
        a=world((.1,0.,-2.))+world((-.1,0.,-2.))
        d1,_=self.detector.detect(0.,ORIGIN,ROTATION,a,visible)
        d2,_=self.detector.detect(0.,ORIGIN,ROTATION,a[::-1],visible)
        self.assertEqual(d1,d2)

    def test_gt_poison_all_visibility_states(self):
        a,b=LegalHumanTracker(CFG),LegalHumanTracker(CFG)
        for time,p,los in [(0.,(0,0,-2),visible),(.056,(.1,0,-2),visible),
                           (.11,(.2,0,-2),blocked),(.21,(.4,0,-2),blocked),(.3,(.2,0,-2),visible)]:
            clean=world(p); dirty=copy.deepcopy(clean)
            dirty[0].update(gt_velocity=[1e9]*3,goal=[1e9]*3,waypoint_progress=999,
                future_position=[1e9]*3,controller_state={'secret':999},future_visibility=True)
            if los is blocked:
                dirty[0]['position']=[-.4,0.,-2.]  # Different hidden GT, same observable empty detection.
            dirty[0]['diagnostic_id']=-100
            d1,_=self.detector.detect(time,ORIGIN,ROTATION,clean,los)
            d2,_=self.detector.detect(time,ORIGIN,ROTATION,dirty,los)
            self.assertEqual(a.update(time,d1),b.update(time,d2))

    def test_duplicate_tiny_dt_nonfinite_and_inactive(self):
        before,_,_=self.observe(0.,world())
        self.assertEqual(before,self.tracker.update(0.,[HumanDetection(0.,(999,0,-2))]))
        tracks=self.tracker.update(1e-8,[HumanDetection(1e-8,(0.,0.,-2.))])
        self.assertFalse(tracks[0]['velocity_valid'])
        self.assertFalse(self.detector.detect(.1,ORIGIN,ROTATION,world((math.nan,0,-2)),visible)[0])
        w=world(); w[0]['active']=False
        self.assertFalse(self.detector.detect(.1,ORIGIN,ROTATION,w,visible)[0])
        with self.assertRaises(ValueError): self.tracker.update(-1.,[])
        with self.assertRaises(ValueError): self.tracker.update(math.nan,[])

    def test_loader_recursively_rejects_privileged_fields(self):
        class Planner:
            def path(self,a,b,c): return [list(a),list(b)]
        qs,_=generate_queries(LegalRoot('root',(0,0,0),0,.0486666666667,1/120),Planner())
        tracks,_,_=self.observe(0.,world())
        pixels={'agent_0_articulated_agent_jaw_rgb':np.zeros((2,2,3)),
                'agent_0_articulated_agent_jaw_depth':np.ones((2,2,1))}
        meta,_=legal_observation(pixels,[0,0,0],0,0,[0,0],tracks)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); q=qs[0].data
            write(root/'legal_observations/history.json',[meta]); write(root/'queries/queries.json',[{'query':q}])
            before=load_wm_input(root,q['query_id'])
            poison=dict(future_position=[999]*3,gt_velocity=[999]*3,goal=[999]*3,
                        hidden_human_position=[999]*3,controller_state={'secret':999})
            meta.update(poison); meta['human_tracks'][0].update(poison)
            q.update(poison); q['generation_config'].update(poison); q['robot_spec'].update(poison)
            write(root/'legal_observations/history.json',[meta]); write(root/'queries/queries.json',[{'query':q}])
            # Poison privileged files too: loader must not open/parse them.
            (root/'roots').mkdir(); (root/'roots/privileged_root.json').write_text('INVALID JSON SECRET')
            self.assertEqual(before,load_wm_input(root,q['query_id']))
            self.assertEqual(set(before['history'][0]['human_tracks'][0]),set(TRACK_FIELDS))

    def test_protected_phase1_components_unchanged(self):
        repo=Path(__file__).resolve().parents[2]
        for path,names in {
            'scripts/collector/core.py':('compile_motion','generate_queries','build_labels','LegalRoot','Query'),
            'scripts/collector/runtime.py':('StaticPlanner','Recorder','state','controller_state'),
        }.items():
            old=subprocess.check_output(['git','show',f'eventwm-collector-phase1:{path}'],cwd=repo,text=True)
            new=(repo/path).read_text()
            def definitions(source):
                return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
                        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
            self.assertEqual(definitions(old),definitions(new))


if __name__=='__main__': unittest.main()
