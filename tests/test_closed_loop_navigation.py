"""Meaningful policy, temporal execution and legal-input boundary tests."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collector.core import LegalRoot,compile_motion
from eventquerywm.navigation import NavigationConfig,RiskAwareSelector,TimedWaypointController,prefix
from eventquerywm.inputs import tensorize


class NavigationTests(unittest.TestCase):
    def query(self,scale=1.,wait=0.):
        return compile_motion(LegalRoot('root',(0.,0.,0.),0.,.048,1/120),[[0.,0.,0.],[1.5,0.,0.]],
                              [1.5,0.],0,wait,scale,dict(speed=.7,yaw_speed=1.5)).data

    def candidates(self):
        return [dict(query=self.query(s),family=name,progress_m=1.5,waiting=False)
                for s,name in ((1.,'normal'),(.5,'slow'))]

    def test_risk_predictions_change_online_choice(self):
        selector=RiskAwareSelector(NavigationConfig())
        self.assertEqual(selector.select(self.candidates(),[.95,.70])[0],0)
        selector=RiskAwareSelector(NavigationConfig())
        self.assertEqual(selector.select(self.candidates(),[.70,.95])[0],1)

    def test_static_ablation_never_uses_predictions(self):
        selector=RiskAwareSelector(NavigationConfig(),False)
        self.assertEqual(selector.select(self.candidates(),[None,None])[0],0)
        self.assertEqual(selector.select(self.candidates(),[0.,1.])[0],0)

    def test_static_and_wm_share_motion_hysteresis(self):
        candidates=self.candidates()
        candidates[1]['family']='right/normal'
        candidates[1]['query']['planned_duration']=.95*candidates[0]['query']['planned_duration']
        for use_wm in (False,True):
            selector=RiskAwareSelector(NavigationConfig(),use_wm)
            selector.last_family='normal'
            self.assertEqual(selector.select(candidates,[.99,.99])[0],0)

    def test_slow_control_is_actually_slower(self):
        controller=TimedWaypointController()
        normal,slow=self.query(),self.query(.5)
        index=1  # initial compile segment is a zero turn.
        a,_=controller.command(normal,index,[0.,0.,0.],0.)
        b,_=controller.command(slow,index,[0.,0.,0.],0.)
        self.assertLess(b[0],.6*a[0])
        self.assertGreater(slow['planned_duration'],1.8*normal['planned_duration'])

    def test_wait_is_exact_zero_even_with_pose_error(self):
        command,_=TimedWaypointController().command(self.query(wait=.5),0,[.1,0.,.1],.2)
        self.assertEqual(command,[0.,0.])

    def test_repeated_wait_and_delay_are_bounded(self):
        candidates=self.candidates()
        candidates.append(dict(query=self.query(wait=.5),family='delay',progress_m=1.5,waiting=False))
        q=self.query(wait=3.);q['direction']='WAIT'
        candidates.append(dict(query=q,family='wait',progress_m=0.,waiting=True))
        selector=RiskAwareSelector(NavigationConfig());selector.wait_s=2.
        chosen,_=selector.select(candidates,[.5,.6,.99,.99])
        self.assertIn(chosen,(0,1))

    def test_fixed_query_tensors_keep_clock_and_legal_history(self):
        history=[dict(timestamp=t,robot_position=[0.,0.,0.],robot_yaw=0.,previous_command=[0.,0.],
                      human_tracks=[],tracking_available=True) for t in (0.,.048)]
        value=tensorize(dict(history=history,history_images=None,query=self.query()))
        self.assertEqual(tuple(value.query.shape),(16,10))
        self.assertGreater(float(value.query[-1,2]),0.)
        self.assertEqual(len(value.robot),2)

    def test_prefix_obeys_path_not_straight_chord(self):
        p=prefix([[0,0,0],[1,0,0],[1,0,1]],1.5)
        np.testing.assert_allclose(p,[[0,0,0],[1,0,0],[1,0,.5]])


if __name__=='__main__':
    unittest.main()
