import copy
import math
import unittest
import numpy as np
from eventwm_graph.l4 import ActionSpec, QuerySpec, compile_reference, Executor, RuntimeContext, Status
from eventwm_graph.l4.geometry import StaticGrid
from eventwm_graph.l4.progress import RouteProgress
from eventwm_graph.l4.search import human_cost


def room():
    return StaticGrid(np.ones((200,200),bool),np.zeros((200,200)),[-10,-10],.1,'test-room')


def reference(route=((0,0,0),(2,0,0)),yaw=0.,limit=200):
    g=room();q=QuerySpec('test',route,(*route[0],0.),yaw,execution_limit=limit)
    return q,compile_reference(q,g),RuntimeContext(g)


class ContractTests(unittest.TestCase):
    def test_immutable_and_deterministic(self):
        points=[[0,0,0],[2,0,0]];g=room();q=QuerySpec('q',points,[0,0,0,0],0)
        a=compile_reference(q,g);points[1][0]=9
        self.assertEqual(a,compile_reference(q,g));self.assertEqual(q.target,(2.,0.,0.))
        self.assertEqual(len(a.timestamped_nominal_poses),len(a.native_action_schedule)+1)
        self.assertEqual(a.l2_segment()['parent_reference_hash'],a.reference_hash)

    def test_completion_and_actual_feedback(self):
        q,r,c=reference();e=Executor();h=e.start(q,r,c)
        while h.status==Status.RUNNING:
            ob=dict(timestamp=h.timestamps[-1],robot_pose=h.poses[-1],tracks=[])
            d=e.act(h,ob,ob);self.assertIn(d.action,c.action_spec.actions)
            # Repeated act before a transition cannot invent time or a new action.
            self.assertIs(e.act(h,ob,ob),d)
            pose=c.action_spec.step(h.poses[-1],d.action)
            e.on_transition(h,dict(action=d.action,timestamp=h.timestamps[-1]+.048,
                                  robot_pose=pose,observation_ref=f'actual:{len(h.actions)}'))
        self.assertEqual(h.status,Status.QUERY_COMPLETED)
        f=e.feedback(h);self.assertAlmostEqual(f['actual_duration'],len(h.actions)*.048)
        self.assertNotIn('stop',f['actual_actions']);self.assertFalse(f['episode_termination'])
        self.assertEqual(len(f['actual_observation_refs']),len(h.actions))

    def test_orientation_progress(self):
        q,r,c=reference(((0,0,0),),yaw=1.5);e=Executor();h=e.start(q,r,c)
        for action in r.native_action_schedule:
            ob=dict(timestamp=h.timestamps[-1],robot_pose=h.poses[-1],tracks=[])
            d=e.act(h,ob,ob)
            e.on_transition(h,dict(action=d.action,timestamp=h.timestamps[-1]+.048,
                                  robot_pose=c.action_spec.step(h.poses[-1],d.action),observation_ref='actual'))
        self.assertEqual(h.status,Status.QUERY_COMPLETED)
        self.assertEqual(h.tracker.path_length,0.)
        self.assertGreater(h.tracker.last_meaningful,q.timestamp)

    def test_progress_can_retreat_and_never_accumulates(self):
        q,r,c=reference();t=RouteProgress(r)
        for i,x in enumerate([0,.5,.2,.5,.2]):t.update([x,0,0,0],i*.048)
        self.assertAlmostEqual(t.progress,.2)
        self.assertGreater(t.path_length,t.progress)

    def test_hairpin_does_not_skip_ordered_route(self):
        q,r,c=reference(((0,0,0),(2,0,0),(2,0,.2),(0,0,.2)))
        t=RouteProgress(r);t.update([0,0,.2,0],.1)
        self.assertLess(t.progress,.01);self.assertFalse(t.complete([0,0,.2,0]))

    def test_zero_time_and_unsolicited_transition_rejected(self):
        q,r,c=reference();e=Executor();h=e.start(q,r,c)
        with self.assertRaises(ValueError):e.on_transition(h,dict(action='forward'))
        ob=dict(timestamp=0.,robot_pose=q.start_pose,tracks=[]);d=e.act(h,ob,ob)
        with self.assertRaises(ValueError):e.on_transition(h,dict(action=d.action,timestamp=0.,robot_pose=q.start_pose,observation_ref='bad'))

    def test_collision_cannot_be_success(self):
        q,r,c=reference();e=Executor();h=e.start(q,r,c)
        ob=dict(timestamp=0.,robot_pose=q.start_pose,tracks=[]);d=e.act(h,ob,ob)
        e.on_transition(h,dict(action=d.action,timestamp=.048,robot_pose=[*q.target,0.],observation_ref='actual',human_collision=True,episode_termination=True))
        self.assertEqual(h.status,Status.INTERRUPTED)

    def test_impossible_avoidance_preserves_nominal_no_stop(self):
        q,r,c=reference();e=Executor();h=e.start(q,r,c)
        ob=dict(timestamp=0.,robot_pose=q.start_pose,tracks=[dict(position=[0,0,0],last_seen_time=0.,visible_now=True)])
        d=e.act(h,ob,ob)
        # Progress-first revision: failed collision-free search is diagnostic,
        # never a BLOCKED handoff or experiment-driver STOP.
        self.assertEqual(d.action,'forward');self.assertEqual(h.status,Status.RUNNING)
        self.assertEqual(h.avoidance_failures,1)
        self.assertEqual(e.feedback(h)['actual_duration'],0.)

    def test_temporal_crossing_rejected(self):
        a=np.array([0.,0,0,0]);b=np.array([.208333,0,0,0])
        people=np.array([[.1,-1.1,0,40.,0,1.]])
        safe,_,_=human_cost(a,b,0,people,.056)
        self.assertFalse(safe)

    def test_actual_clock_is_recorded_without_artificial_episode_end(self):
        q,r,c=reference();e=Executor();h=e.start(q,r,c)
        ob=dict(timestamp=0.,robot_pose=q.start_pose,tracks=[]);d=e.act(h,ob,ob)
        e.on_transition(h,dict(action=d.action,timestamp=.10,robot_pose=d.planned_ego_trajectory[1],observation_ref='actual'))
        self.assertEqual(h.status,Status.RUNNING)
        self.assertTrue(h.transition_events[-1]['clock_outside_verified_bounds'])
        self.assertEqual(h.timestamps[-1],.10)

    def test_query_schedule_is_owned_by_l3(self):
        q,r,c=reference();e=Executor();h=e.start(q,r,c)
        h.actions=['left']*q.execution_limit
        ob=dict(timestamp=0.,robot_pose=q.start_pose,tracks=[])
        self.assertEqual(e.act(h,ob,ob).action,'forward');self.assertEqual(h.status,Status.RUNNING)

    def test_stale_current_measurement_rejected(self):
        with self.assertRaises(ValueError):
            Executor._people(dict(tracks=[dict(position=[1,0,0],visible_now=True,last_seen_time=0)]),1.)

    def test_static_swept_body(self):
        g=room();free=g.free.copy();free[105,:]=False
        blocked=StaticGrid(free,g.heights,g.origin,g.resolution,'wall')
        self.assertFalse(blocked.motion_free([0,0,0,0],[.208333,0,0,0]))
        with self.assertRaises(ValueError):compile_reference(QuerySpec('q',((0,0,0),(2,0,0)),(0,0,0,0),0),blocked)

    def test_enclosing_region_shortcut_cannot_clear_sampled_obstacle(self):
        from eventwm_graph.l4.geometry import body_free
        rng=np.random.default_rng(22117)
        for _ in range(250):
            free=rng.uniform(size=(40,40))>.005
            g=StaticGrid(free,np.zeros((40,40)),[-2,-2],.1,'random-geometry-regression')
            a=np.array([*rng.uniform(-1,1,size=1),0.,*rng.uniform(-1,1,size=1),rng.uniform(-math.pi,math.pi)])
            b=ActionSpec().step(a,ActionSpec().actions[int(rng.integers(3))])
            if g.motion_free(a,b):
                angle=math.atan2(math.sin(b[3]-a[3]),math.cos(b[3]-a[3]))
                for u in np.linspace(0,1,101):
                    p=a+u*(b-a);p[3]=a[3]+u*angle
                    self.assertTrue(body_free(p,*g.kernel_args()))


if __name__=='__main__':unittest.main()
