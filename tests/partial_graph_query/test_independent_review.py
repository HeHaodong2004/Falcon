"""Independent Stage 0 regressions, authored without changing implementation.

Run: python -m unittest discover -s tests/partial_graph_query -p test_independent_review.py
No historical held-out episode or production collector is opened by these tests.
"""
import copy
import math
import unittest
from collections import defaultdict, deque
from types import SimpleNamespace

import numpy as np

from eventwm_graph.partial.map import PartialGraph, KNOWN_FREE, KNOWN_OCCUPIED, UNKNOWN
from eventwm_graph.partial.query import (
    FORWARD_STEP, TURN_STEP, ROBOT_BOUNDS, HUMAN_ENVELOPE,
    compile_query, generate_queries, physical_clearance, wrap,
)
from eventwm_graph.partial.observer import GeometryObserver, in_frustum
from scripts.collector.observation import ObservationConfig


def observation(t, updates=(), pose=(.1, 0., .1, 0.)):
    return {
        'timestamp':float(t), 'robot_pose':list(pose), 'tracks':[],
        'static_updates':[
            {'cell':list(k), 'height':float(h), 'state':int(s), 'timestamp':float(t),
             'source':source}
            for k,s,h,source in updates],
    }


def square_graph(lo=-5, hi=22, resolution=.2, pose=(.1, 0., .1, 0.)):
    graph=PartialGraph(resolution,0.)
    graph.update(observation(1.,[
        ((x,z),KNOWN_FREE,0.,'independent_fixture_square_certificate')
        for x in range(lo,hi) for z in range(lo,hi)
    ],pose))
    return graph


def human(position=(2.1,0.,2.1), **changes):
    track={
        'track_id':'track_000000', 'position':list(position),
        'estimated_velocity':[1.,0.,0.], 'velocity_valid':True,
        'visible_now':True, 'valid':True, 'first_seen_time':.5,
        'last_seen_time':1., 'observation_age':0., 'num_observations':4,
    }
    track.update(changes)
    return track


class IndependentMapReview(unittest.TestCase):
    def test_observation_and_static_evidence_timestamps_are_causal(self):
        graph=PartialGraph(.2,0.)
        graph.update(observation(1.))
        with self.assertRaises(ValueError):graph.update(observation(1.))
        future=observation(2.,[((0,0),KNOWN_FREE,0.,'sensor')])
        future['static_updates'][0]['timestamp']=3.
        with self.assertRaises(ValueError):graph.update(future)

    def test_corner_grazing_segment_cannot_skip_unknown_cell(self):
        graph=PartialGraph(1.,0.)
        graph.update(observation(1.,[
            ((x,z),KNOWN_FREE,0.,'sensor')
            for x in range(4) for z in range(4) if (x,z)!=(1,0)
        ]))
        start=np.array([.1,0.,.1]);target=np.array([2.9,0.,2.89])
        self.assertIn((1,0),graph.segment_cells(start,target))
        self.assertFalse(graph.segment_free(start,target))

    def test_grid_boundary_requires_both_neighbor_cells(self):
        graph=PartialGraph(1.,0.)
        graph.update(observation(1.,[
            ((0,z),KNOWN_FREE,0.,'sensor') for z in range(3)
        ]))
        self.assertFalse(graph.segment_free([1.,0.,.1],[1.,0.,2.9]))

    def test_diagonal_cannot_cut_unknown_corner(self):
        graph=PartialGraph(.2,0.)
        graph.update(observation(1.,[
            ((0,0),KNOWN_FREE,0.,'sensor'), ((1,1),KNOWN_FREE,0.,'sensor'),
        ]))
        self.assertNotIn((1,1),dict(graph.adjacency[(0,0)]))
        with self.assertRaises(ValueError):graph.route([.1,0.,.1],[.3,0.,.3])

    def test_new_route_connects_only_after_its_observation(self):
        graph=PartialGraph(.2,0.)
        graph.update(observation(1.,[
            ((0,0),KNOWN_FREE,0.,'sensor'), ((2,0),KNOWN_FREE,0.,'sensor'),
        ]))
        with self.assertRaises(ValueError):graph.route([.1,0.,.1],[.5,0.,.1])
        graph.update(observation(2.,[((1,0),KNOWN_FREE,0.,'sensor')]))
        path=graph.route([.1,0.,.1],[.5,0.,.1])
        self.assertTrue(all(graph.segment_free(a,b) for a,b in zip(path,path[1:])))

    def test_occupied_wins_regardless_of_within_frame_order(self):
        for states in ((KNOWN_OCCUPIED,KNOWN_FREE),(KNOWN_FREE,KNOWN_OCCUPIED)):
            graph=PartialGraph(.2,0.)
            graph.update(observation(1.,[((0,0),state,0.,'sensor') for state in states]))
            self.assertEqual(graph.state((0,0)),KNOWN_OCCUPIED)

    def test_ego_support_does_not_fill_its_containing_square(self):
        graph=PartialGraph(.2,0.)
        graph.update(observation(1.,[((0,0),KNOWN_FREE,0.,'occupied_ego_base_support')]))
        self.assertTrue(graph.point_free([.1,0.,.1]))
        self.assertFalse(graph.point_free([.15,0.,.1]))
        self.assertEqual(graph.state((0,0)),UNKNOWN)
        self.assertEqual(graph.reachable([.1,0.,.1]),set())

    def test_new_occupancy_invalidates_already_published_global_edges(self):
        graph=PartialGraph(.2,0.)
        graph.update(observation(1.,[
            ((x,0),KNOWN_FREE,0.,'sensor') for x in range(20)
        ]))
        graph.rebuild_global([.1,0.,.1])
        self.assertTrue(graph.global_edges)
        graph.update(observation(2.,[((5,0),KNOWN_OCCUPIED,0.,'new_wall')]))
        self.assertTrue(all(
            graph.segment_free(a,b) for edge in graph.global_edges
            for a,b in zip(edge['path'],edge['path'][1:])
        ))
        with self.assertRaises(ValueError):graph.route([.1,0.,.1],[2.1,0.,.1])

    def test_changed_surface_height_invalidates_cached_adjacency(self):
        graph=PartialGraph(.2,0.)
        graph.update(observation(1.,[
            ((0,0),KNOWN_FREE,0.,'sensor'),((1,0),KNOWN_FREE,0.,'sensor'),
        ]))
        self.assertIn((1,0),dict(graph.adjacency[(0,0)]))
        revision=graph.revision
        graph.update(observation(2.,[((1,0),KNOWN_FREE,.3,'updated_surface')]))
        self.assertGreater(graph.revision,revision)
        self.assertNotIn((1,0),dict(graph.adjacency[(0,0)]))

    def test_changed_surface_height_invalidates_published_global_paths(self):
        graph=PartialGraph(.2,0.)
        graph.update(observation(1.,[
            ((x,0),KNOWN_FREE,0.,'sensor') for x in range(20)
        ]))
        graph.rebuild_global([.1,0.,.1])
        self.assertTrue(graph.global_edges)
        graph.update(observation(2.,[((5,0),KNOWN_FREE,.3,'updated_surface')]))
        self.assertTrue(all(
            graph.segment_free(a,b) for edge in graph.global_edges
            for a,b in zip(edge['path'],edge['path'][1:])
        ))

    def test_human_observation_does_not_modify_static_topology(self):
        graph=square_graph(-2,4)
        graph.rebuild_global([.1,0.,.1])
        cells=copy.deepcopy(graph.cells);adj=copy.deepcopy(graph.adjacency)
        row=observation(2.);row['tracks']=[human(last_seen_time=2.)]
        graph.update(row)
        self.assertEqual(graph.cells,cells)
        self.assertEqual(graph.adjacency,adj)

    def test_snapshot_restore_preserves_route_and_query_inputs(self):
        graph=square_graph(-3,15)
        graph.rebuild_global([.1,0.,.1])
        restored=PartialGraph.restore(graph.snapshot())
        np.testing.assert_equal(graph.route([.1,0.,.1],[2.1,0.,1.1]),
                                restored.route([.1,0.,.1],[2.1,0.,1.1]))
        self.assertEqual(graph.snapshot(),restored.snapshot())

    def test_sparse_global_graph_keeps_reachable_ego_component_connected(self):
        graph=square_graph(0,70,pose=(13.1,0.,13.1,0.))
        sparse=graph.rebuild_global([13.1,0.,13.1])
        self.assertEqual(len(graph.reachable([13.1,0.,13.1])),4900)
        adjacency=defaultdict(set)
        for edge in sparse['edges']:
            adjacency[edge['source']].add(edge['target'])
            adjacency[edge['target']].add(edge['source'])
            self.assertTrue(all(graph.segment_free(a,b) for a,b in zip(edge['path'],edge['path'][1:])))
        root='g_65_65';self.assertIn(root,graph.global_nodes)
        reachable={root};queue=deque([root])
        while queue:
            for node in adjacency[queue.popleft()]-reachable:
                reachable.add(node);queue.append(node)
        self.assertEqual(reachable,set(graph.global_nodes))


class IndependentQueryReview(unittest.TestCase):
    def test_future_human_timestamp_is_rejected(self):
        graph=square_graph(-2,6)
        with self.assertRaises(ValueError):
            generate_queries(graph,[human(last_seen_time=2.)],
                             {'robot_pose':[.1,0.,.1,0.],'timestamp':1.},[1.,0.,1.])

    def test_human_directions_and_bands_use_observed_motion_only(self):
        graph=square_graph();track=human();saved=copy.deepcopy(track)
        _,diagnostics=generate_queries(graph,[track],
            {'robot_pose':[.1,0.,.1,0.],'timestamp':1.},[4.1,0.,.1],limit=64)
        candidates=[c for c in diagnostics['spatial_candidates'] if c['provenance']['source']=='human']
        self.assertEqual(len(candidates),8)
        expected={'FRONT':np.array([1.,0.,0.]),'BACK':np.array([-1.,0.,0.]),
                  'LEFT':np.array([0.,0.,-1.]),'RIGHT':np.array([0.,0.,1.])}
        for candidate in candidates:
            provenance=candidate['provenance'];radius=provenance['nominal_radius_m']
            delta=np.asarray(candidate['position'])-track['position']
            np.testing.assert_allclose(delta,expected[provenance['direction']]*radius,atol=1e-12)
            self.assertEqual(provenance['heading_basis'],'observed_motion')
            self.assertEqual(provenance['intended_band'],'safe' if radius>1. else 'intrusion')
        self.assertEqual(track,saved)

    def test_intrusion_candidate_can_pass_with_tangential_robot_heading(self):
        track=human((0.,0.,0.))
        self.assertTrue(physical_clearance(np.array([.96,0.,0.]),math.pi/2,[track]))
        self.assertFalse(physical_clearance(np.array([.96,0.,0.]),math.pi,[track]))
        self.assertLess(.96,1.)

    def test_stale_track_clearance_covers_plausible_unobserved_motion(self):
        track=human((0.,0.,0.),estimated_velocity=None,velocity_valid=False,
                    visible_now=False,last_seen_time=.5,observation_age=.5)
        # A human can move .6 m in .5 s at a physically plausible 1.2 m/s.
        # The robot terminal rectangle at x=1.2 would overlap that possible
        # human disk. A .5 m/s fallback cannot certify current non-overlap.
        possible=human((.6,0.,0.))
        target=np.array([1.2,0.,0.]);yaw=math.pi/2
        self.assertFalse(physical_clearance(target,yaw,[possible]))
        self.assertFalse(physical_clearance(target,yaw,[track]))

    def test_unknown_heading_fallback_ignores_extra_oracle_heading_field(self):
        graph=square_graph();track=human(velocity_valid=False,num_observations=1,
                                        estimated_velocity=None)
        state={'robot_pose':[.1,0.,.1,0.],'timestamp':1.}
        a,da=generate_queries(PartialGraph.restore(graph.snapshot()),[track],state,[4.1,0.,.1])
        track['oracle_heading']=2.73
        b,db=generate_queries(PartialGraph.restore(graph.snapshot()),[track],state,[4.1,0.,.1])
        self.assertEqual(a,b);self.assertEqual(da,db)

    def test_same_waypoint_different_terminal_headings_produce_different_actions(self):
        graph=square_graph(-10,15);state={'robot_pose':[.1,0.,.1,0.]}
        a=compile_query(graph,{'waypoints':[[1.1,0.,.1]],'final_yaw':0.},state)
        b=compile_query(graph,{'waypoints':[[1.1,0.,.1]],'final_yaw':math.pi/2},state)
        self.assertTrue(a['ok']);self.assertTrue(b['ok'])
        self.assertEqual(a['actions'],b['actions'][:len(a['actions'])])
        self.assertGreater(b['terminal_turn_actions'],0)
        self.assertLessEqual(b['final_yaw_error'],b['yaw_tolerance'])
        self.assertGreater(b['nominal_seconds'],a['nominal_seconds'])
        self.assertNotIn('stop',b['actions'])

    def test_intermediate_motion_does_not_lock_terminal_orientation(self):
        graph=square_graph(-10,16)
        result=compile_query(graph,{'waypoints':[[1.1,0.,.1],[1.1,0.,-1.1]],'final_yaw':math.pi},
                             {'robot_pose':[.1,0.,.1,0.]})
        self.assertTrue(result['ok'],result)
        self.assertEqual(len(result['waypoint_arrival_action_indices']),2)
        self.assertEqual(result['actions'][0],'forward')
        self.assertTrue(any(abs(wrap(p[3]-math.pi))>.5 for p in result['planned_poses'][:-1]))
        self.assertLessEqual(result['final_yaw_error'],result['yaw_tolerance'])

    def test_narrow_corridor_turn_requires_full_robot_body_clearance(self):
        graph=PartialGraph(.2,0.)
        # Cylinder-center support for a .9 m wide corridor is only .4 m wide.
        # Spot fits longitudinally but its long body cannot rotate through pi/2.
        graph.update(observation(1.,[
            ((x,z),KNOWN_FREE,0.,'visible_native_cspace_square_and_clearance_halo')
            for x in range(-20,21) for z in (-1,0)
        ],pose=(0.,0.,0.,0.)))
        result=compile_query(graph,{'waypoints':[[0.,0.,0.]],'final_yaw':math.pi/2},
                             {'robot_pose':[0.,0.,0.,0.]})
        self.assertFalse(result['ok'])

    def test_backtracking_and_current_position_queries_are_retained(self):
        graph=square_graph(-5,14)
        queries,_=generate_queries(graph,[],{'robot_pose':[.1,0.,.1,0.],'timestamp':1.},[2.1,0.,.1],limit=64)
        self.assertTrue(any(q['provenance'][0].get('behind_robot',False) for q in queries))
        turns=[q for q in queries if q['provenance'][0]['source']=='current_position']
        self.assertGreaterEqual(len(turns),2)
        self.assertTrue(all(q['compiled']['actions'] and
                            all(a in ('left','right') for a in q['compiled']['actions']) for q in turns))


class IndependentObservationReview(unittest.TestCase):
    def test_other_human_mesh_blocks_visibility_but_target_body_is_not_self_occlusion(self):
        # The sensing boundary may inspect native hit IDs, but the blocked
        # actor is not emitted as a static obstacle or a detector observation.
        sensor=GeometryObserver.__new__(GeometryObserver)
        sensor.robot_ids={1};sensor.performance={'rays':0}
        hit=SimpleNamespace(object_id=201,ray_distance=1.)
        sensor.sim=SimpleNamespace(cast_ray=lambda *args,**kwargs:SimpleNamespace(hits=[hit]))
        origin=np.array([0.,1.,0.]);target=np.array([0.,1.,-2.])
        self.assertFalse(sensor.clear_ray(origin,target))
        self.assertTrue(sensor.clear_ray(origin,target,ignore={201}))

    def test_frustum_uses_full_rotation_range_and_vertical_field(self):
        config=ObservationConfig(90.,90.,10.)
        origin=np.zeros(3);rotation=np.eye(3)
        points=np.array([[0.,0.,-2.],[0.,0.,2.],[0.,3.,-2.],[0.,0.,-11.]])
        np.testing.assert_equal(in_frustum(points,origin,rotation,config),[True,False,False,False])
        # Rotation through pi around world y reverses the legal visible half-space.
        rotated=np.diag([-1.,1.,-1.])
        np.testing.assert_equal(in_frustum(points,origin,rotated,config),[False,True,False,False])

    def test_square_certificate_cannot_miss_inflated_obstacle_between_probes(self):
        # A .25 m inflated obstacle centered just beyond the upper cell edge
        # leaves a 6 mm circular cap inside this square. The nine star probes
        # miss it; an interior point is outside native support by 5 mm, larger
        # than the adapter's 4 mm horizontal projection allowance.
        class CircularSupport:
            @staticmethod
            def snap_point(p):
                q=np.array(p,dtype=float).copy()
                center=np.array([.55,.444]);delta=q[[0,2]]-center
                distance=np.linalg.norm(delta)
                if distance<.25:q[[0,2]]=center+delta*.25/distance
                return q
            @staticmethod
            def try_step_no_sliding(a,b):return CircularSupport.snap_point(b)
            @staticmethod
            def distance_to_closest_obstacle(p,radius):
                return min(radius,np.linalg.norm(np.asarray(p)[[0,2]]-[.55,.444])-.25)
        sensor=GeometryObserver.__new__(GeometryObserver)
        sensor.resolution=.2;sensor.cache={}
        sensor.sim=SimpleNamespace(pathfinder=CircularSupport())
        sensor.performance={'private_geometry_seconds':0.}
        p=np.array([.55,0.,.199])
        self.assertGreater(np.linalg.norm(CircularSupport.snap_point(p)-p),.004)
        self.assertIsNone(sensor._support((2,0),0.))


if __name__=='__main__':unittest.main()
