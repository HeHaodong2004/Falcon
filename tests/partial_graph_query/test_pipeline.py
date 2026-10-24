import copy
import math
import unittest
import numpy as np
from eventwm_graph.partial.map import PartialGraph,KNOWN_FREE,KNOWN_OCCUPIED
from eventwm_graph.partial.query import generate_queries,compile_query,physical_clearance,static_body_clearance
from eventwm_graph.partial.metrics import summarize
from scripts.collector.observation import ObservationConfig,LegalHumanTracker,HumanDetection


def room(zmin=-20,zmax=20):
    g=PartialGraph(.2,0.)
    g.update({'timestamp':0.,'robot_pose':[.1,0.,.1,0.],
        'static_updates':[{'cell':[x,z],'state':KNOWN_FREE,'height':0.,'timestamp':0.,'source':'synthetic_visible_square'}
                          for x in range(-20,21) for z in range(zmin,zmax+1)]})
    return g


def track(p=(1.8,0.,.1),velocity=(.2,0.,0.)):
    return {'track_id':'track_000000','position':list(p),'estimated_velocity':list(velocity),'velocity_valid':True,
            'visible_now':True,'valid':True,'first_seen_time':-.2,'last_seen_time':0.,'observation_age':0.,'num_observations':4}


class PipelineTests(unittest.TestCase):
    def test_no_unknown_goal_shortcut(self):
        g=room();goal=[8.,0.,0.]
        q,d=generate_queries(g,[],{'timestamp':0.,'robot_pose':[.1,0.,.1,0.]},goal)
        self.assertTrue(q)
        self.assertFalse(any(p.get('actual_goal_known_connected') for x in q for p in x['provenance']))
        self.assertTrue(all(g.point_free(p) for x in q for p in x['waypoints']))
        with self.assertRaises(ValueError):g.route([.1,0.,.1],goal)

    def test_social_bands_and_legal_motion_heading(self):
        g=room();human=track()
        q,d=generate_queries(g,[human],{'timestamp':0.,'robot_pose':[.1,0.,.1,0.]},[3.,0.,3.],64)
        social=[x for x in q if x['provenance'][0]['source']=='human']
        self.assertEqual({x['provenance'][0]['intended_band'] for x in social},{'safe','intrusion'})
        self.assertEqual({x['provenance'][0]['direction'] for x in social},{'FRONT','BACK','LEFT','RIGHT'})
        for x in social:
            p=x['provenance'][0]
            self.assertAlmostEqual(np.linalg.norm(np.asarray(x['waypoints'][0])-human['position']),p['nominal_radius_m'])
            self.assertTrue(physical_clearance(x['waypoints'][0],x['final_yaw'],[human]))
            self.assertEqual(p['heading_basis'],'observed_motion')

    def test_intrusion_is_soft_not_collision(self):
        h=track((0.,0.,0.))
        self.assertTrue(physical_clearance([.96,0.,0.],math.pi/2,[h]))
        self.assertFalse(physical_clearance([.96,0.,0.],0.,[h]))
        self.assertFalse(physical_clearance([.4,0.,0.],math.pi/2,[h]))

    def test_unknown_heading_fallback_and_stale_tracks(self):
        g=room();h=track();h['velocity_valid']=False;h['estimated_velocity']=None
        q,d=generate_queries(g,[h],{'timestamp':0.,'robot_pose':[.1,0.,.1,0.]},[3.,0.,3.],64)
        self.assertTrue(any(p.get('heading_basis')=='robot_relative_fallback' for c in d['spatial_candidates'] for p in [c['provenance']]))
        h['observation_age']=.6;h['last_seen_time']=-.6;h['visible_now']=False
        q,d=generate_queries(g,[h],{'timestamp':0.,'robot_pose':[.1,0.,.1,0.]},[3.,0.,3.])
        self.assertFalse(any(c['provenance']['source']=='human' for c in d['spatial_candidates']))

    def test_future_and_backward_times_rejected(self):
        g=room();h=track();h['last_seen_time']=1.
        with self.assertRaises(ValueError):generate_queries(g,[h],{'timestamp':0.,'robot_pose':[.1,0.,.1,0.]},[3.,0.,3.])
        with self.assertRaises(ValueError):g.update({'timestamp':-.1,'robot_pose':[.1,0.,.1,0.],'static_updates':[]})

    def test_narrow_corridor_turn_rejected_but_aligned_route_valid(self):
        g=room(-1,0);ego={'timestamp':0.,'robot_pose':[.1,0.,.1,0.]}
        good=compile_query(g,{'waypoints':[[1.1,0.,.1]],'final_yaw':0.},ego)
        bad=compile_query(g,{'waypoints':[[.1,0.,.1]],'final_yaw':math.pi/2},ego)
        self.assertTrue(good['ok'],good)
        self.assertFalse(bad['ok'])
        self.assertIn('turn_body_clearance',bad['reason'])

    def test_intermediate_yaw_is_free_and_terminal_turn_consumes_time(self):
        g=room();ego={'robot_pose':[.1,0.,.1,0.]}
        q={'waypoints':[[.1,0.,1.3],[1.3,0.,1.3]],'final_yaw':math.pi}
        r=compile_query(g,q,ego)
        self.assertTrue(r['ok'],r)
        self.assertGreater(r['terminal_turn_actions'],0)
        self.assertGreater(r['nominal_seconds'],0.)
        self.assertLess(r['final_yaw_error'],r['yaw_tolerance'])
        self.assertTrue(any(abs(p[3]-math.pi)>.5 for p in r['planned_poses'][1:]))
        self.assertNotIn('stop',r['actions'])

    def test_backtracking_and_identical_target_different_yaws(self):
        g=room();ego={'timestamp':0.,'robot_pose':[.1,0.,.1,0.]}
        q,d=generate_queries(g,[],ego,[3.,0.,0.],64)
        self.assertTrue(any(p.get('behind_robot') for x in q for p in x['provenance']))
        turns=[x for x in q if x['provenance'][0]['source']=='current_position']
        self.assertGreaterEqual(len(turns),2)
        self.assertGreater(len({x['final_yaw'] for x in turns}),1)

    def test_reconstruction_and_determinism(self):
        g=room();ego={'timestamp':0.,'robot_pose':[.1,0.,.1,0.]}
        a,_=generate_queries(g,[track()],ego,[3.,0.,3.])
        restored=PartialGraph.restore(g.snapshot())
        b,_=generate_queries(restored,[track()],ego,[3.,0.,3.])
        self.assertEqual([x['query_id'] for x in a],[x['query_id'] for x in b])

    def test_tracker_visibility_and_successor_replay(self):
        c=ObservationConfig(90,90,10);tracker=LegalHumanTracker(c)
        a=tracker.update(0.,[HumanDetection(0.,(1.,0.,0.))])
        b=tracker.update(.048,[])
        self.assertTrue(a[0]['visible_now']);self.assertFalse(b[0]['visible_now'])
        self.assertAlmostEqual(b[0]['observation_age'],.048)
        clone=copy.deepcopy(tracker)
        self.assertEqual(tracker.update(.096,[HumanDetection(.096,(1.1,0.,0.))]),clone.update(.096,[HumanDetection(.096,(1.1,0.,0.))]))

    def test_psc_full_trajectory_and_terminal_endpoint(self):
        labels=[]
        for t,d,coll in ((.048,1.2,False),(.096,.9,False),(.144,.7,True)):
            labels.append({'timestamp':t,'official_dmin_m':d,'euclidean_dmin_m':d,'official_center_distances_m':[d],
                           'psc_violation':d<1.,'human_mesh_contact':coll,'static_obstacle_contact':False})
        query={'waypoints':[[.1,0.,.1]],'final_yaw':0.,'compiled':{'position_tolerance':.155,'yaw_tolerance':.042}}
        s=summarize(labels,[[.1,0.,.1,0.]]*4,0.,['left']*3,'human_collision',query,[4.,0.,0.])
        self.assertAlmostEqual(s['psc'],1/3)
        self.assertEqual(s['violating_steps'],2)
        self.assertAlmostEqual(s['violation_duration_s'],.096)
        self.assertAlmostEqual(s['integrated_intrusion_severity_m_s'],.0192)
        self.assertTrue(s['human_collision_after_psc_violation'])
        self.assertFalse(s['query_complete'])


if __name__=='__main__':unittest.main()
