"""Static swept-query and binary human-event contract regression tests."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from collector.core import LegalRoot,compile_motion,write,load_wm_input
from collector.static_feasibility import RobotGeometry,StaticConfig,is_static_feasible
from collector.v2 import QueryConfig,generate_queries_v2,InsufficientStaticQueries,human_event_labels,guard_output

ROOT=LegalRoot('root',(0.,0.,0.),0.,.05,1/120)
CFG=dict(directions=8,target_radius=1.5,speed=.7,yaw_speed=1.5,max_projection=.25,max_path_length=4.,human_blind=True)
GEO=RobotGeometry(.5)

class Map:
    def __init__(self,wall=False,clearance=5.):self.wall=wall;self.value=clearance;self.calls=0
    def path(self,start,target,cfg):
        self.calls+=1
        if self.calls<=2:return None
        return [list(start),list(target)]
    def clearance(self,p):return min(self.value,abs(p[0]-.7)) if self.wall else self.value
    def pose_free(self,p,yaw):return True
    def segment_free(self,a,b):return True


def query(end=(1.5,0,0)):
    return compile_motion(ROOT,[list(ROOT.position),list(end)],[1.5,0],0,0,1,CFG).data


def branch(native=False,disc=False,static=False,completed=False,reason='timeout'):
    q=query();q['static_precheck']={'static_feasible':True}
    duration=q['planned_duration'] if completed else .1
    b=dict(branch_id='b000',query_id=q['query_id'],planned_duration=q['planned_duration'],
        completed=completed,termination_reason=reason,controls=[dict(index=i,issued_command=c)
        for i,c in enumerate(q['executable_controls'])] if completed else [],truth=[
        dict(timestamp=t,robot={'position':[0,0,0]},humans=[{'position':[.5 if disc else 3,0,0]}],
             native_human_contact=native and t>0,native_static_contact=static and t>0) for t in [0,duration]])
    return b,q


class StaticTests(unittest.TestCase):
    def test_fixed_controls_cannot_disagree_with_checked_trajectory(self):
        q=query();q['executable_controls'][-1][0]*=2
        self.assertEqual(is_static_feasible(ROOT,q,Map(),GEO)['rejection_reason'],'fixed_controls_trajectory_mismatch')
    def test_empty_motion_cannot_skip_root_check(self):
        q=query();q['nominal_timed_trajectory']=q['nominal_timed_trajectory'][:1]
        self.assertFalse(is_static_feasible(ROOT,q,Map(),GEO)['static_feasible'])
    def test_free_target_and_entire_path(self):
        self.assertTrue(is_static_feasible(ROOT,query(),Map(),GEO)['static_feasible'])
    def test_free_endpoint_but_path_crosses_wall(self):
        r=is_static_feasible(ROOT,query(),Map(wall=True),GEO)
        self.assertFalse(r['static_feasible']);self.assertIsNotNone(r['failed_segment'])
    def test_target_in_static_geometry(self):
        self.assertFalse(is_static_feasible(ROOT,query((.7,0,0)),Map(wall=True),GEO)['static_feasible'])
    def test_radius_and_margin_are_respected(self):
        self.assertTrue(is_static_feasible(ROOT,query(),Map(clearance=.31),GEO,StaticConfig(safety_margin=.03))['static_feasible'])
        self.assertFalse(is_static_feasible(ROOT,query(),Map(clearance=.31),GEO,StaticConfig(safety_margin=.08))['static_feasible'])
        self.assertFalse(is_static_feasible(ROOT,query(),Map(clearance=.31),RobotGeometry(.7))['static_feasible'])
    def test_n_valid_not_n_attempts_and_wait_separate(self):
        qs,d=generate_queries_v2(ROOT,Map(),GEO,QueryConfig(movement_queries=8,max_attempts=64))
        self.assertEqual(d['accepted_movement_queries'],8);self.assertGreater(d['attempts'],8)
        self.assertEqual(len(qs),33);self.assertEqual(sum(q.data['direction']=='WAIT' for q in qs),1)
        self.assertTrue(all(q.data['static_precheck']['static_feasible'] for q in qs))
        ends=[x['path'][-1] for x in d['accepted_candidates']]
        self.assertTrue(all(np.linalg.norm(np.array(a)-b)>=.30 for i,a in enumerate(ends) for b in ends[i+1:]))
    def test_insufficient_does_not_return_wait_only(self):
        with self.assertRaises(InsufficientStaticQueries) as ctx:
            generate_queries_v2(ROOT,Map(clearance=0),GEO,QueryConfig(movement_queries=2,max_attempts=5))
        self.assertEqual(ctx.exception.diagnostics['accepted_movement_queries'],0)
    def test_near_duplicates_rejected_and_fail_explicitly(self):
        m=Map();m.path=lambda *a:[[0,0,0],[1,0,0]]
        with self.assertRaises(InsufficientStaticQueries) as ctx:
            generate_queries_v2(ROOT,m,GEO,QueryConfig(movement_queries=2,max_attempts=6))
        self.assertEqual(ctx.exception.diagnostics['accepted_movement_queries'],1)
        self.assertEqual(ctx.exception.diagnostics['candidate_rejections'][0]['rejection_reason'],'near_duplicate_endpoint')
    def test_turn_and_wait_root_pose_checked(self):
        m=Map();m.pose_free=lambda p,yaw:False
        q=compile_motion(ROOT,[[0,0,0]],[0,0],'WAIT',1,0,CFG)
        self.assertFalse(is_static_feasible(ROOT,q,m,GEO)['static_feasible'])
    def test_controller_clipped_path_rejected(self):
        m=Map();m.segment_free=lambda a,b:False
        self.assertEqual(is_static_feasible(ROOT,query(),m,GEO)['rejection_reason'],'controller_navmesh_clipping')

class LabelTests(unittest.TestCase):
    def test_early_native_or_disc_or_both_positive(self):
        for n,d in [(True,False),(False,True),(True,True)]:
            b,q=branch(native=n,disc=d,reason='robot_human_collision');l=human_event_labels(b,q)
            self.assertTrue(l['human_event_valid']);self.assertEqual(l['human_event_label'],1)
    def test_complete_clean_negative_and_zero_human_invariant(self):
        b,q=branch(completed=True,reason='query_completed');l=human_event_labels(b,q,human_count=0)
        self.assertTrue(l['human_event_valid']);self.assertEqual(l['human_event_label'],0)
        b,q=branch(native=True);l=human_event_labels(b,q,human_count=0)
        self.assertFalse(l['human_event_valid']);self.assertIsNone(l['human_event_label'])
    def test_static_overrides_human_even_after_accepted_precheck(self):
        for human in (False,True):
            b,q=branch(static=True,native=human);l=human_event_labels(b,q)
            self.assertFalse(l['human_event_valid']);self.assertIsNone(l['human_event_label'])
            self.assertTrue(l['static_feasibility_violation'])
    def test_premature_noncollision_invalid(self):
        for reason in ('timeout','execution_failure','environment_termination','collection_error'):
            b,q=branch(reason=reason);l=human_event_labels(b,q)
            self.assertFalse(l['human_event_valid']);self.assertIsNone(l['human_event_label'])
    def test_completed_flag_without_full_controls_invalid(self):
        b,q=branch(completed=True,reason='query_completed');b['controls']=b['controls'][:-1]
        self.assertFalse(human_event_labels(b,q)['human_event_valid'])

class BoundaryTests(unittest.TestCase):
    def test_legacy_path_and_symlink_alias_rejected(self):
        legacy=Path(__file__).resolve().parents[2]/'datasets/raw/eventwm_prod_v1'
        for p in (legacy,legacy/'new_root',legacy.resolve()):
            with self.assertRaises(ValueError):guard_output(p)
        with tempfile.TemporaryDirectory() as td:
            alias=Path(td)/'alias';alias.symlink_to(legacy.resolve(),target_is_directory=True)
            with self.assertRaises(ValueError):guard_output(alias/'new')
    def test_spot_collision_meshes_match_visual_bounds(self):
        # The native geometry adapter uses link scene bounds on this Habitat build.
        # Fail if URDF collision geometry stops being covered by those visual bounds.
        import xml.etree.ElementTree as ET
        urdf=Path(__file__).resolve().parents[2]/'data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf'
        def tree(n):return (n.tag,tuple(sorted(n.attrib.items())),tuple(tree(c) for c in n))
        def spec(n):return (tree(n.find('geometry')),n.find('origin').attrib if n.find('origin') is not None else {})
        for link in ET.parse(urdf).getroot().findall('link'):
            for collision in link.findall('collision'):
                self.assertIn(spec(collision),[spec(v) for v in link.findall('visual')],link.attrib['name'])

    def test_other_protocol_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            write(Path(td)/'manifest.json',{'schema_version':1})
            with self.assertRaises(ValueError):guard_output(td)
    def test_privileged_poison_never_reaches_model_loader(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);q=query();q['schema_version']=2
            history=[dict(timestamp=0,robot_position=[0,0,0],robot_yaw=0,previous_command=[0,0],
                human_tracks=[],tracking_available=True,sensor_keys=[])]
            write(p/'legal_observations/history.json',history);write(p/'queries/queries.json',[{'query':q}])
            clean=load_wm_input(p,q['query_id'])
            history[0].update(hidden_humans=[999],future_states=[999],human_goals=[999],gt_velocity=[999],controller_state=[999])
            q.update(human_event_label=1,static_precheck={'secret_future':[999]},future_humans=[999])
            q['robot_spec']['human_goal']=[999];q['generation_config']['hidden_gt']=[999]
            write(p/'legal_observations/history.json',history);write(p/'queries/queries.json',[{'query':q}])
            self.assertEqual(clean,load_wm_input(p,q['query_id']))

class LaunchAuditGateTests(unittest.TestCase):
    def test_static_contact_in_repeat_also_quarantines_and_stops(self):
        import argparse
        from types import SimpleNamespace
        from unittest.mock import patch
        import eventwm_collect
        from collector.v2 import PROTOCOL
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);b,q=branch(static=True,reason='static_collision');b['branch_id']='reverse_1'
            write(out/'roots/replay.json',{'protocol_version':PROTOCOL})
            write(out/'queries/queries.json',[{'query':q}])
            write(out/'audits/workers/reverse_1/branch.json',b)
            args=argparse.Namespace(output=out,storage_profile='production')
            with patch.object(eventwm_collect.subprocess,'run',return_value=SimpleNamespace(returncode=0)):
                with self.assertRaisesRegex(RuntimeError,'STATIC_FEASIBILITY_VIOLATION'):
                    eventwm_collect.launch(args,'reverse_1',q['query_id'])
            self.assertTrue((out/'labels/quarantined_branch.json').exists())

class RunnerTests(unittest.TestCase):
    def test_static_violation_stops_before_next_root(self):
        import argparse
        from types import SimpleNamespace
        from unittest.mock import patch
        import eventwm_v2
        from collector.core import read
        calls=[]
        def fail(cmd,**kwargs):
            dest=Path(cmd[cmd.index('--output')+1]);calls.append(dest)
            write(dest/'audits/STATIC_FEASIBILITY_VIOLATION.json',{'static_feasibility_violation':True})
            write(dest/'labels/quarantined_branch.json',{'static_feasibility_violation':True})
            return SimpleNamespace(returncode=1)
        with tempfile.TemporaryDirectory() as td:
            args=argparse.Namespace(output=Path(td)/'new',movement_queries=3,candidate_attempts=16,
                                    static_margin=.03,first_root_only=False)
            with patch.object(eventwm_v2,'provenance',return_value={}),patch.object(eventwm_v2.subprocess,'run',side_effect=fail):
                self.assertEqual(eventwm_v2.run(args),1)
            self.assertEqual(len(calls),1)
            self.assertEqual(read(args.output/'manifest.json')['state'],'stopped_static_violation')
            self.assertEqual(read(args.output/'smoke_summary.json')['unexpected_static_collisions'],1)

if __name__=='__main__':unittest.main()
