"""Progress-first regressions; native collision results are separately audited."""
import unittest
from unittest.mock import Mock
import numpy as np
from eventwm_graph.l4 import Executor, Status, RuntimeContext
from eventwm_graph.l4.progress import RouteProgress
from test_contract import reference


def packet(h, human=None):
    return dict(timestamp=h.timestamps[-1], robot_pose=h.poses[-1], tracks=[] if human is None else [
        dict(position=human,last_seen_time=h.timestamps[-1],visible_now=True,velocity_valid=False)])


def step(e,h,d,**extra):
    return e.on_transition(h,dict(action=d.action,timestamp=h.timestamps[-1]+.048,
        robot_pose=h.context.action_spec.step(h.poses[-1],d.action),observation_ref='actual',**extra))


class ProgressFirstTests(unittest.TestCase):
    def test_empty_route_matches_nominal_without_turns(self):
        q,r,c=reference();e=Executor();h=e.start(q,r,c)
        while h.status==Status.RUNNING:
            ob=packet(h);d=e.act(h,ob,ob);self.assertEqual(d.action,'forward');step(e,h,d)
        self.assertEqual(tuple(h.actions),r.native_action_schedule)

    def test_visible_human_causes_committed_side_and_translation(self):
        q,r,c=reference(((0,0,0),(6,0,0)));e=Executor();h=e.start(q,r,c)
        chosen=[]
        for _ in range(10):
            ob=packet(h,[3.,0,0]);d=e.act(h,ob,ob);chosen.append(d.action);step(e,h,d)
        turns=chosen[:chosen.index('forward')]
        self.assertTrue(turns);self.assertEqual(len(set(turns)),1)
        self.assertFalse(any(a!=b and a in ('left','right') and b in ('left','right') for a,b in zip(chosen,chosen[1:])))
        self.assertIn('forward',chosen);self.assertGreater(h.tracker.path_length,.5)

    def test_visible_pedestrian_avoidance_completes_and_rejoins(self):
        from eventwm_graph.l4.scenarios import run
        result=run('head_on')
        self.assertEqual(result['status'],'QUERY_COMPLETED')
        self.assertFalse(result['human_collision'])
        self.assertGreater(result['feedback']['detours'],0)
        self.assertEqual(result['feedback']['detours'],result['feedback']['rejoins'])

    def test_static_and_motion_mismatch_continue_from_actual_pose(self):
        q,r,c=reference();e=Executor();h=e.start(q,r,c);ob=packet(h);d=e.act(h,ob,ob)
        e.on_transition(h,dict(action=d.action,timestamp=.048,robot_pose=q.start_pose,
            observation_ref='contact',static_collision=True))
        self.assertEqual(h.status,Status.RUNNING);self.assertEqual(h.tracker.path_length,0.)
        self.assertTrue(e.feedback(h)['transition_events'][-1]['static_collision'])
        self.assertEqual(e.act(h,packet(h),packet(h)).action,'forward')

    def test_stale_uncertainty_and_watchdog_do_not_stop_motion(self):
        q,r,c=reference();e=Executor();h=e.start(q,r,c)
        for _ in range(110):
            ob=packet(h,[0.,0,0]);d=e.act(h,ob,ob)
            e.on_transition(h,dict(action=d.action,timestamp=h.timestamps[-1]+.048,
                robot_pose=q.start_pose,observation_ref='actual'))
        self.assertEqual(h.status,Status.RUNNING);self.assertNotIn('stop',h.actions)

    def test_waypoint_plane_rejoin_and_adjacent_corner(self):
        q,r,c=reference(((0,0,0),(2,0,0),(4,0,0)))
        t=RouteProgress(r);t.update(q.start_pose,0.);t.update([2.4,0,.1,0],.048)
        self.assertEqual(t.next_waypoint,2);self.assertGreater(t.progress,2.)
        q,r,c=reference(((0,0,0),(2,0,0),(2,0,-2)))
        t=RouteProgress(r);t.update([2.05,0,-.4,0],.048)
        self.assertEqual(t.next_waypoint,2)

    def test_no_online_mesh_calls(self):
        q,r,c=reference();c.geometry.precise=Mock()
        e=Executor();h=e.start(q,r,c);ob=packet(h);d=e.act(h,ob,ob)
        self.assertEqual(d.action,'forward');c.geometry.precise.swept.assert_not_called()

    def test_native_truncation_is_authoritative(self):
        q,r,c=reference();e=Executor();h=e.start(q,r,c);ob=packet(h);d=e.act(h,ob,ob)
        step(e,h,d,episode_truncation=True)
        self.assertEqual(h.status,Status.INTERRUPTED)
        self.assertIsNone(e.act(h,packet(h),packet(h)).action)


if __name__=='__main__':unittest.main()
