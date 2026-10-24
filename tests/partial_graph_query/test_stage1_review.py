"""Independent Stage 1 metric, legal-record, integrity, and resume regressions.

Fixtures are generated in temporary directories. No real episode, historical
dataset, production service, or training process is read or started.
"""
import copy
from dataclasses import asdict
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from eventwm_graph.config import digest, file_digest, read_json, write_json
from eventwm_graph.partial.map import PartialGraph, KNOWN_FREE, KNOWN_OCCUPIED
from eventwm_graph.partial.metrics import distances, contact_labels, summarize
from eventwm_graph.partial.query import compile_query, EXECUTOR_VERSION, TURN_STEP
from eventwm_graph.partial.observer import GeometryObserver
from eventwm_graph.partial.storage import seal_root, validate_root, online_input
from eventwm_graph.stage1.storage import read_gzip, write_gzip, scene_split
from scripts.collector.observation import ObservationConfig


def row(timestamp, pose, updates=()):
    return {'timestamp':float(timestamp),'robot_pose':list(pose),'tracks':[],
            'camera_origin':[pose[0]+.51,pose[1]+.75,pose[2]],
            'camera_rotation':np.eye(3).tolist(),'static_updates':list(updates),
            'observation_valid':True,'sensor_protocol':'independent_fixture'}


def step_label(timestamp, distances_m=(), human_hit=False, static_hit=False):
    ds=list(distances_m)
    return {'timestamp':float(timestamp),'official_center_distances_m':ds,
            'euclidean_center_distances_m':ds,'official_dmin_m':min(ds) if ds else None,
            'euclidean_dmin_m':min(ds) if ds else None,'psc_violation':any(d<1. for d in ds),
            'euclidean_ps_violation':any(d<1. for d in ds),'human_mesh_contact':bool(human_hit),
            'static_obstacle_contact':bool(static_hit),'static_obstacle_contact_witnesses':[],
            'benchmark_robot_scene_contact_count':int(static_hit),
            'episode_over':bool(human_hit),'official_human_collision':bool(human_hit)}


def fixture():
    pose=[.1,0.,.1,0.];graph=PartialGraph(.2,0.)
    updates=[{'cell':[x,z],'state':KNOWN_FREE,'height':0.,'timestamp':1.,
              'source':'independent_known_free_square'} for x in range(-12,13) for z in range(-12,13)]
    initial=row(1.,pose,updates);graph.update(initial)
    query={'waypoints':[[.1+25/120,0.,.1]],'final_yaw':0.,
           'provenance':[{'source':'goal'}],'decision_timestamp':1.,
           'query_semantics':'ordered_waypoints_plus_terminal_yaw',
           'static_traversability':'known_free_reachable','executor_version':EXECUTOR_VERSION}
    compiled=compile_query(graph,query,initial)
    if not compiled['ok']:raise AssertionError(compiled)
    query['compiled']=compiled;query['route_distance']=compiled['route_distance'];query['query_id']=digest(query)
    final_pose=compiled['planned_poses'][-1]
    successor_row=row(1.048,final_pose)
    after=PartialGraph.restore(graph.snapshot());delta=after.update(successor_row)
    labels=step_label(1.048)
    summary=summarize([labels],[pose,final_pose],1.,compiled['actions'],'query_complete',query,[2.,0.,.1])
    outcome={'query_id':query['query_id'],'node_ids':[0],'summary':summary,
             'successor':{'timestamp':1.048,'robot_pose':final_pose,'tracks':[],'graph_revision':after.revision},
             'executor_version':EXECUTOR_VERSION,'execution_engine':'independent_fixture'}
    tape=SimpleNamespace(
        times=[1.,1.048],truth=[[],[]],
        nodes=[{'node_id':0,'parent':None,'action':compiled['actions'][0],
                'observation':successor_row,'graph_delta':delta,'labels':labels,'future_index':1}])
    episode={'episode_id':'unit_root','scene_id':'independent_review_scene',
             'start_position':pose[:3],'start_rotation':[1.,0.,0.,0.],
             'goals':[{'position':[2.,0.,.1]}],
             'info':{'human_num':0,'graph_navmesh':'private_unit_fixture.navmesh'}}
    record={'history':[initial],'history_actions':[],'partial_graph':graph.snapshot(),
            'pointgoal':[2.,0.,.1],'sensor':asdict(ObservationConfig(90.,90.,10.)),
            'scene_id':'independent_review_scene','seed':991827,
            'private_initialization':episode,'source_map_manifest':'private_unit_fixture_manifest.json',
            'prefix_labels':[],'sampling':{'prefix_actions':[],'future_draws':1,'scenario':'natural'},
            'performance':{},'collection_config':{'schema':'partial_collection_config_1',
                'queries_per_root':1,'max_query_steps':120,'human_horizontal_motion_bound_m_s':3.,
                'pilot':True,'automatic_training':False}}
    return record,tape,[query],[outcome]


class IndependentOfficialMetricReview(unittest.TestCase):
    def fake_sim(self, positions, found=False, geodesic=1.7):
        return SimpleNamespace(
            num_articulated_agents=len(positions),
            get_agent_state=lambda i:SimpleNamespace(position=positions[i]),
            pathfinder=SimpleNamespace(find_path=lambda request:found),
            geodesic_distance=lambda a,b:geodesic)

    def test_official_float32_threshold_matches_installed_measure(self):
        import habitat_sim  # Installed import order, before Falcon registration.
        from falcon.additional_metric import PersonalSpaceCompliance
        positions=[np.zeros(3,dtype=np.float32),np.array([.28,0.,.96],dtype=np.float32)]
        sim=self.fake_sim(positions)
        task=SimpleNamespace(measurements=SimpleNamespace(
            check_measure_dependencies=lambda *args:None,
            measures={'num_steps':SimpleNamespace(get_metric=lambda:1)}))
        episode=SimpleNamespace(info={'human_num':1})
        measure=PersonalSpaceCompliance(sim=sim,config=SimpleNamespace(use_geo_distance=True))
        measure.reset_metric(episode=episode,task=task,observations={})
        measure.update_metric(episode=episode,task=task,observations={})
        official,diagnostic=distances(sim,1)
        self.assertEqual(official,[1.])
        self.assertEqual(measure.get_metric(),float(not any(d<1. for d in official)))
        self.assertLess(diagnostic[0],1.)  # Separate, higher-precision diagnostic.

    def test_official_geodesic_and_euclidean_diagnostic_are_separate(self):
        sim=self.fake_sim([np.zeros(3,dtype=np.float32),np.array([.4,0.,0.],dtype=np.float32)],
                          found=True,geodesic=1.7)
        official,diagnostic=distances(sim,1)
        self.assertEqual(official,[1.7]);self.assertAlmostEqual(diagnostic[0],.4,places=6)

    def test_disconnected_fallback_uses_full_xyz_center_distance(self):
        sim=self.fake_sim([np.zeros(3,dtype=np.float32),np.array([0.,1.2,0.],dtype=np.float32)])
        official,_=distances(sim,1)
        self.assertAlmostEqual(official[0],1.2,places=6)

    def test_zero_executed_steps_do_not_become_compliant_training_labels(self):
        record,_,queries,_=fixture();q=queries[0]
        summary=summarize([],[record['history'][0]['robot_pose']],1.,q['compiled']['actions'],
                          'route_invalidated',q,record['pointgoal'])
        self.assertIsNone(summary['psc']);self.assertFalse(summary['psc_valid'])
        self.assertEqual(summary['executed_steps'],0)

    def test_all_humans_and_terminal_endpoint_determine_psc(self):
        record,_,queries,_=fixture();q=queries[0]
        labels=[step_label(1.048,[1.3,1.1]),step_label(1.096,[1.3,.9]),
                step_label(1.144,[1.3,1.]),step_label(1.192,[1.3,.8],human_hit=True)]
        summary=summarize(labels,[record['history'][0]['robot_pose']]*5,1.,['left']*4,
                          'human_collision',q,record['pointgoal'])
        self.assertEqual(summary['psc'],.5);self.assertEqual(summary['violating_steps'],2)
        self.assertAlmostEqual(summary['minimum_official_distance_m'],.8)
        self.assertAlmostEqual(summary['first_psc_violation_time_s'],.096)
        self.assertAlmostEqual(summary['violation_duration_s'],.096)
        self.assertTrue(summary['human_collision_after_psc_violation'])
        self.assertFalse(summary['query_complete'])


class IndependentContactSupportReview(unittest.TestCase):
    def fake_runtime(self, link=10, height=.1, normal=(0.,1.,0.), hits=(), human_contact=False):
        import habitat_sim
        robot=SimpleNamespace(
            sim_obj=SimpleNamespace(object_id=100,get_link_name=lambda i:'link_'+str(i),link_object_ids={110:10}),
            params=SimpleNamespace(leg_joints=list(range(8,20))),base_pos=np.zeros(3))
        other=201 if human_contact else 300
        contact=SimpleNamespace(object_id_a=100,object_id_b=other,link_id_a=link,link_id_b=-1,
            # Native Magnum contact vectors expose float32 coordinates.
            position_on_a_in_ws=np.array([0.,height,0.],dtype=np.float32),
            position_on_b_in_ws=np.array([0.,height,0.],dtype=np.float32),
            contact_normal_on_b_in_ws=np.array(normal,dtype=float),contact_distance=-.001)
        human=SimpleNamespace(sim_obj=SimpleNamespace(object_id=201,link_object_ids={211:1}))
        sim=SimpleNamespace(perform_discrete_collision_detection=lambda:None,
            get_physics_contact_points=lambda:[contact],
            get_agent_data=lambda i:SimpleNamespace(articulated_agent=robot if i==0 else human),
            cast_ray=lambda *args,**kwargs:SimpleNamespace(hits=[SimpleNamespace(object_id=i) for i in hits]))
        return SimpleNamespace(sim=sim,count=1),robot

    def check_static_rule(self, runtime,robot):
        import habitat_sim
        from eventwm_graph.robotmap.body import BodyOracle
        from eventwm_graph.robotmap.config import TraversabilityConfig
        oracle=BodyOracle.__new__(BodyOracle)
        oracle.robot=robot;oracle.sim=runtime.sim
        oracle.geometry=SimpleNamespace(hs=habitat_sim)
        oracle.robot_config=SimpleNamespace(max_climb=.2,max_slope=45.)
        oracle.config=TraversabilityConfig()
        expected=not oracle.contact_records(np.zeros(3))[0]['support_contact']
        with patch('habitat.tasks.rearrange.utils.rearrange_collision',
                   return_value=(True,SimpleNamespace(robot_scene_colls=1))):
            actual=contact_labels(runtime)
        self.assertEqual(actual['static_obstacle_contact'],expected)
        self.assertEqual(actual['benchmark_robot_scene_contact_count'],1)
        return actual

    def test_support_floor_and_low_unblocked_foot_edge_match_body_oracle(self):
        for normal in ((0.,1.,0.),(1.,0.,0.)):
            rt,robot=self.fake_runtime(normal=normal)
            self.assertFalse(self.check_static_rule(rt,robot)['static_obstacle_contact'])

    def test_blocked_foot_edge_nonfoot_and_high_contact_are_obstacles(self):
        cases=[dict(normal=(1.,0.,0.),hits=(400,)),dict(link=8),dict(height=.3)]
        for case in cases:
            rt,robot=self.fake_runtime(**case)
            self.assertTrue(self.check_static_rule(rt,robot)['static_obstacle_contact'])

    def test_human_mesh_contact_does_not_become_static_contact(self):
        rt,_=self.fake_runtime(human_contact=True)
        with patch('habitat.tasks.rearrange.utils.rearrange_collision',
                   return_value=(True,SimpleNamespace(robot_scene_colls=1))):
            actual=contact_labels(rt)
        self.assertTrue(actual['human_mesh_contact']);self.assertFalse(actual['static_obstacle_contact'])


class IndependentStorageReview(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='partial_stage1_review_')
        self.base=Path(self.temporary.name)
        self.inputs=fixture()
        result=seal_root(self.base,*self.inputs,min_free_bytes=0)
        self.folder=Path(result['folder'])

    def tearDown(self):self.temporary.cleanup()

    def change_member(self,name,change):
        value=read_gzip(self.folder/name);change(value);write_gzip(self.folder/name,value)
        manifest=read_json(self.folder/'manifest.json')
        manifest['files'][name]=file_digest(self.folder/name)
        write_json(self.folder/'manifest.json',manifest)

    def test_valid_generated_fixture_and_whitelisted_online_files(self):
        self.assertTrue(validate_root(self.folder)['sealed'])
        from eventwm_graph.partial import storage
        original=storage.read_gzip;opened=[]
        def restricted(path):
            opened.append(Path(path).name)
            self.assertIn(Path(path).name,('legal_root.json.gz','queries.json.gz'))
            return original(path)
        with patch.object(storage,'read_gzip',restricted):model=online_input(self.folder)
        self.assertEqual(set(opened),{'legal_root.json.gz','queries.json.gz'})
        self.assertEqual(model['history_actions'],[])

    def test_resume_return_shape_contains_published_folder(self):
        resumed=seal_root(self.base,*self.inputs,min_free_bytes=0)
        self.assertEqual(Path(resumed['folder']),self.folder)

    def test_checksums_detect_an_interrupted_or_corrupt_member(self):
        with (self.folder/'future_observations.json.gz').open('ab') as stream:stream.write(b'corrupt')
        with self.assertRaises(ValueError):validate_root(self.folder)

    def test_future_human_cannot_enter_legal_root_history(self):
        def change(root):
            root['history'][0]['tracks']=[{'track_id':'track_0','position':[1.,0.,0.],
                'estimated_velocity':None,'velocity_valid':False,'visible_now':True,'valid':True,
                'first_seen_time':2.,'last_seen_time':2.,'observation_age':0.,'num_observations':1}]
        self.change_member('legal_root.json.gz',change)
        with self.assertRaises(ValueError):validate_root(self.folder)

    def test_unknown_global_node_cannot_be_smuggled_with_correct_legal_cells(self):
        def change(root):
            root['partial_graph']['global_nodes']['unseen']={
                'id':'unseen','cell':[999,999],'position':[199.9,0.,199.9],
                'kind':'junction','created_at':1.}
        self.change_member('legal_root.json.gz',change)
        with self.assertRaises(ValueError):validate_root(self.folder)

    def test_root_graph_timestamp_must_match_legal_decision_time(self):
        self.change_member('legal_root.json.gz',lambda root:root['partial_graph'].update(timestamp=9.))
        with self.assertRaises(ValueError):validate_root(self.folder)

    def test_summary_psc_must_recompute_from_step_evidence(self):
        self.change_member('outcomes.json.gz',lambda outcomes:outcomes[0]['summary'].update(psc=0.))
        with self.assertRaises(ValueError):validate_root(self.folder)

    def test_step_psc_flag_must_match_all_saved_official_distances(self):
        def change(labels):
            labels[0].update(official_center_distances_m=[.5],official_dmin_m=.5,
                             psc_violation=False)
        self.change_member('privileged_step_labels.json.gz',change)
        with self.assertRaises(ValueError):validate_root(self.folder)

    def test_nan_tape_timestamps_cannot_bypass_clock_checks(self):
        name='privileged_future.npz'
        np.savez_compressed(self.folder/name,timestamps=np.array([np.nan,np.nan]),positions=np.empty((2,0,3)))
        manifest=read_json(self.folder/'manifest.json');manifest['files'][name]=file_digest(self.folder/name)
        write_json(self.folder/'manifest.json',manifest)
        with self.assertRaises(ValueError):validate_root(self.folder)

    def test_successor_timestamp_must_match_final_observation(self):
        self.change_member('outcomes.json.gz',lambda outcomes:outcomes[0]['successor'].update(timestamp=9.))
        with self.assertRaises(ValueError):validate_root(self.folder)

    def test_scene_split_metadata_is_verified(self):
        manifest=read_json(self.folder/'manifest.json')
        correct={'train':'train','validation':'dev','test':'confirmation'}[scene_split(manifest['scene_id'])]
        manifest['split']='dev' if correct!='dev' else 'train'
        write_json(self.folder/'manifest.json',manifest)
        with self.assertRaises(ValueError):validate_root(self.folder)


class IndependentGateReview(unittest.TestCase):
    def test_empty_pass_gate_is_closed(self):
        from eventwm_graph.partial.collect import verify_gate
        with tempfile.TemporaryDirectory(prefix='partial_gate_review_') as temporary:
            path=Path(temporary)/'gate.json'
            write_json(path,{'stage':'stage0','status':'PASS','checks':{},'implementation':{}})
            with self.assertRaises(ValueError):verify_gate(path)


class IndependentPrivateCacheReview(unittest.TestCase):
    def runtime(self,scene,blocked=False):
        robot=SimpleNamespace(sim_obj=SimpleNamespace(object_id=1,link_object_ids={}))
        support=SimpleNamespace(
            snap_point=lambda p:np.full(3,np.nan) if blocked else np.asarray(p,dtype=float).copy(),
            try_step_no_sliding=lambda a,b:np.asarray(b,dtype=float).copy(),
            distance_to_closest_obstacle=lambda p,radius:float(radius))
        return SimpleNamespace(
            sim=SimpleNamespace(pathfinder=support,get_agent_data=lambda i:SimpleNamespace(articulated_agent=robot)),
            count=0,observation_config=ObservationConfig(90.,90.,10.),
            episode=SimpleNamespace(scene_id=scene,info={'graph_navmesh':scene+'.navmesh'}))

    def test_shared_private_support_cache_cannot_cross_scene_boundaries(self):
        cache={}
        first=GeometryObserver(self.runtime('independent_scene_a'),.2,cache)
        self.assertIsNotNone(first._support((0,0),0.))
        second=GeometryObserver(self.runtime('independent_scene_b',blocked=True),.2,cache)
        self.assertIsNone(second._support((0,0),0.))

    def test_shared_private_support_cache_is_resolution_specific(self):
        cache={};runtime=self.runtime('independent_scene_a')
        first=GeometryObserver(runtime,.2,cache);first._support((0,0),0.)
        second=GeometryObserver(runtime,.1,cache)
        center,_=second._support((0,0),0.)
        np.testing.assert_allclose(center,[.05,0.,.05])


class IndependentBranchControlReview(unittest.TestCase):
    def test_new_observation_invalidates_remaining_turn_without_losing_shared_prefix(self):
        self.check_new_observation_invalidates_remaining_action('left')

    def test_new_observation_invalidates_forward_body_sweep_before_endpoint(self):
        self.check_new_observation_invalidates_remaining_action('forward')

    def test_action_budget_preserves_completed_prefix_but_times_out_longer_query(self):
        self.check_new_observation_invalidates_remaining_action(
            'left', expected_reason='query_timeout', max_steps=1, obstruction=False)

    def check_new_observation_invalidates_remaining_action(
            self, second_action, expected_reason='route_invalidated', max_steps=3, obstruction=True):
        import habitat_sim
        import magnum as mn
        from eventwm_graph.partial.execution import GraphPrefixFuture
        from eventwm_graph.robotmap.execution import signed_yaw
        record,_,_,_=fixture();initial=record['history'][0]
        graph=PartialGraph.restore(record['partial_graph'])
        class Robot:
            def __init__(self):
                self.sim_obj=SimpleNamespace(
                    transformation=mn.Matrix4.translation(mn.Vector3(initial['robot_pose'][:3])),
                    joint_positions=np.zeros(1),joint_velocities=np.zeros(1))
            @property
            def base_pos(self):return np.asarray(self.sim_obj.transformation.translation)
            def update(self):pass
        robot=Robot()
        class Left:
            def step(self,**kwargs):
                yaw=signed_yaw(robot)+TURN_STEP
                rotation=mn.Matrix4.rotation_y(mn.Rad(yaw)).rotation()
                robot.sim_obj.transformation=mn.Matrix4.from_(rotation,mn.Vector3(robot.base_pos))
        class Forward:
            def step(self,**kwargs):
                yaw=signed_yaw(robot)
                robot.sim_obj.transformation.translation=mn.Vector3(
                    robot.base_pos+[25/120*math.cos(yaw),0.,-25/120*math.sin(yaw)])
        class Collision:
            _ever_collide=False
            def update_metric(self,**kwargs):pass
            def get_metric(self):return False
        collision=Collision()
        sim=SimpleNamespace(get_agent_data=lambda index:SimpleNamespace(articulated_agent=robot))
        task=SimpleNamespace(should_end=False,is_stop_called=False,actions={'left':Left(),'forward':Forward()},
            measurements=SimpleNamespace(measures={
                'human_collision':collision,
                'did_multi_agents_collide':SimpleNamespace(update_metric=lambda **kwargs:None)}))
        class Sensor:
            def state(self):return {}
            def restore(self,state):pass
            def observe(self,timestamp):
                pose=[*robot.base_pos.tolist(),signed_yaw(robot)]
                # For forward motion the newly occupied rear strip is swept at
                # the action start but lies behind the final body rectangle.
                # Checking only the endpoint misses this invalidated support.
                obstruction={'cell':[0,0] if second_action=='left' else [-3,0],
                             'state':KNOWN_OCCUPIED,'height':0.,
                             'timestamp':timestamp,'source':'newly_observed_test_wall'}
                return row(timestamp,pose,[obstruction] if add_obstruction else [])
        add_obstruction=obstruction
        tape=GraphPrefixFuture.__new__(GraphPrefixFuture)
        tape.sim=sim;tape.task=task;tape.sensor=Sensor();tape.sensor_root={}
        tape.graph_root=graph.snapshot();tape.root_observation=initial
        tape.robot_transform=mn.Matrix4(robot.sim_obj.transformation)
        tape.robot_joints=np.zeros(1);tape.robot_velocities=np.zeros(1);tape.gait_frame=0
        tape.times=[1.,1.048,1.096];tape.truth=[[],[],[]]
        tape.pointgoal=record['pointgoal'];tape.prefix_compliant=0;tape.prefix_steps=0
        tape.max_steps=max_steps;tape.nodes=[];tape.performance={}
        tape.rt=SimpleNamespace(action_names={'left':'left','forward':'forward'},episode=SimpleNamespace(),
            config=SimpleNamespace(habitat=SimpleNamespace(environment=SimpleNamespace(max_episode_steps=500))))
        queries=[]
        short_target=initial['robot_pose'][:3]
        long_target=short_target if second_action=='left' else (
            np.asarray(short_target)+[25/120*math.cos(TURN_STEP),0.,-25/120*math.sin(TURN_STEP)]).tolist()
        long_yaw=2*TURN_STEP if second_action=='left' else TURN_STEP
        for target,yaw in ((short_target,TURN_STEP),(long_target,long_yaw)):
            q={'waypoints':[target],'final_yaw':yaw,'executor_version':EXECUTOR_VERSION}
            q['compiled']=compile_query(graph,q,initial)
            self.assertTrue(q['compiled']['ok'],q['compiled'])
            q['query_id']=digest(q);queries.append(q)
        self.assertEqual(queries[1]['compiled']['actions'],['left',second_action])
        with patch('eventwm_graph.partial.execution.step_labels',side_effect=lambda rt,t:step_label(t)), \
                patch('falcon.additional_action.play_i',0):
            outcomes=tape.evaluate_many(queries)
        self.assertEqual(len(tape.nodes),1)
        self.assertEqual(outcomes[0]['node_ids'],outcomes[1]['node_ids'])
        self.assertEqual(outcomes[0]['summary']['termination'],'query_complete')
        self.assertEqual(outcomes[1]['summary']['termination'],expected_reason)
        self.assertEqual(outcomes[1]['summary']['executed_steps'],1)
        self.assertEqual(outcomes[1]['successor']['timestamp'],1.048)


if __name__=='__main__':unittest.main()
