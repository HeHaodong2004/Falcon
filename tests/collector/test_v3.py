"""Three-class protocol and frozen-v2 boundary tests (no simulator required)."""
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from collector.core import read,write,load_wm_input
from collector.v3 import PROTOCOL,SOCIAL_METRIC,SocialOutcome,social_outcome_labels,guard_output,social_violation,generate_queries_v3
from collector.v2 import generate_queries_v2,QueryConfig
from test_v2 import branch,ROOT,Map,GEO


def fixture(distance=3.,native=False,static=False,completed=True,reason=None,count=1):
    b,q=branch(native=native,static=static,completed=completed,
               reason=reason or ('query_completed' if completed else 'robot_human_collision' if native else 'timeout'))
    q['protocol_version']=PROTOCOL
    for r in b['truth']:
        r['humans']=[{'position':[distance,0,0]} for _ in range(count)]
        r['social_metric']=dict(version=SOCIAL_METRIC,distances=[dict(human_slot=i+1,distance=distance,mode='geodesic') for i in range(count)])
    return b,q


class SocialLabels(unittest.TestCase):
    def label(self,**kwargs):
        b,q=fixture(**kwargs);return social_outcome_labels(b,q,human_count=kwargs.get('count',1))
    def test_clean_full(self):self.assertEqual(self.label()['social_outcome_class'],SocialOutcome.SAFE_SUCCESS)
    def test_social_full(self):self.assertEqual(self.label(distance=.8)['social_outcome_class'],SocialOutcome.SOCIAL_VIOLATION_SUCCESS)
    def test_direct(self):self.assertEqual(self.label(native=True)['social_outcome_class'],SocialOutcome.HUMAN_COLLISION)
    def test_social_then_physical_precedence(self):
        l=self.label(distance=.8,native=True)
        self.assertEqual(l['social_outcome_class'],2);self.assertTrue(l['social_distance_violation'])
        self.assertLess(l['first_social_violation_time'],l['first_human_collision_time'])
    def test_early_physical_valid(self):self.assertEqual(self.label(native=True,completed=False)['social_outcome_class'],2)
    def test_social_incomplete_invalid(self):self.assertIsNone(self.label(distance=.8,completed=False)['social_outcome_class'])
    def test_safe_incomplete_invalid(self):self.assertIsNone(self.label(completed=False)['social_outcome_class'])
    def test_static_invalid(self):
        l=self.label(static=True);self.assertFalse(l['social_outcome_valid']);self.assertTrue(l['static_feasibility_violation'])
    def test_static_human_invalid(self):self.assertIsNone(self.label(static=True,native=True)['social_outcome_class'])
    def test_zero_humans(self):
        self.assertEqual(self.label(count=0)['social_outcome_class'],0)
        self.assertFalse(self.label(count=0,native=True)['social_outcome_valid'])
        b,q=fixture(distance=.8);self.assertFalse(social_outcome_labels(b,q,0)['social_outcome_valid'])
    def test_threshold_strict(self):
        for d,expected in [(1.0,0),(1.00000001,0),(.99999999,1)]:
            self.assertEqual(self.label(distance=d)['social_outcome_class'],expected)
        for d in (float('nan'),float('inf'),-1):
            with self.assertRaises(ValueError):social_violation(d)
    def test_distance_duration_count(self):
        l=self.label(distance=.8)
        self.assertAlmostEqual(l['min_human_distance'],.8);self.assertAlmostEqual(l['min_social_clearance'],-.2)
        self.assertAlmostEqual(l['min_human_clearance'],.25)
        self.assertEqual(l['social_violation_count'],1);self.assertEqual(l['social_violation_duration'],l['observed_duration'])
    def test_research_disc_not_physical(self):
        l=self.label(distance=.5)
        self.assertTrue(l['research_disc_collision']);self.assertFalse(l['direct_human_collision'])
        self.assertEqual(l['social_outcome_class'],1)
    def test_missing_evidence_invalid_not_safe(self):
        b,q=fixture();del b['truth'][-1]['social_metric']
        self.assertFalse(social_outcome_labels(b,q,1)['social_outcome_valid'])
    def test_geodesic_not_euclidean_threshold(self):
        b,q=fixture(distance=.8)
        for r in b['truth']:r['social_metric']['distances'][0]['distance']=1.4
        self.assertEqual(social_outcome_labels(b,q,1)['social_outcome_class'],0)
    def test_euclidean_no_path(self):
        b,q=fixture(distance=.8)
        for r in b['truth']:r['social_metric']['distances'][0]['mode']='euclidean_no_path'
        self.assertEqual(social_outcome_labels(b,q,1)['social_outcome_class'],1)
    def test_no_binary_target(self):self.assertNotIn('human_event_label',self.label())


class V3Boundaries(unittest.TestCase):
    def test_no_privileged_input(self):
        b,q=fixture()
        h=[dict(timestamp=0,robot_position=[0,0,0],robot_yaw=0,previous_command=[0,0],human_tracks=[],tracking_available=True,sensor_keys=[])]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);write(p/'legal_observations/history.json',h);write(p/'queries/queries.json',[{'query':q}])
            clean=load_wm_input(p,q['query_id'])
            poison=dict(social_metric={'gt':999},social_outcome_class=2,future_human=[999],native_human_collision=True)
            h[0].update(poison);q.update(poison);q['robot_spec'].update(poison)
            write(p/'legal_observations/history.json',h);write(p/'queries/queries.json',[{'query':q}])
            self.assertEqual(clean,load_wm_input(p,q['query_id']))
    def test_legacy_guards_and_aliases(self):
        repo=Path(__file__).resolve().parents[2]
        for name in ('eventwm_prod_v1','eventwm_human_event_v2_future','eventwm_human_event_v2_smoke_03'):
            with self.assertRaises(ValueError):guard_output(repo/'datasets/raw'/name/'new')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);write(p/'old/manifest.json',{'protocol_version':'eventwm-human-event-v2'})
            before=(p/'old/manifest.json').read_bytes();(p/'alias').symlink_to(p/'old',target_is_directory=True)
            for d in (p/'old',p/'old/sub',p/'alias/sub'):
                with self.assertRaises(ValueError):guard_output(d)
            self.assertEqual(before,(p/'old/manifest.json').read_bytes())
            self.assertEqual(guard_output(p/'new'),p/'new')
    def test_v2_frozen_files_unchanged(self):
        repo=Path(__file__).resolve().parents[2]
        for name in ('scripts/collector/v2.py','scripts/collector/static_feasibility.py','scripts/collector/static_habitat.py','scripts/eventwm_v2.py'):
            old=subprocess.check_output(['git','show','36e2872:'+name],cwd=repo)
            # Static certification was repaired in a new source epoch. The
            # historical epoch stays frozen; label/protocol files stay current.
            source=Path('/home/haodong/workspace/projects/eventwm-falcon/production/eventwm-v4-science') if '/static_' in name else repo
            self.assertEqual(old,(source/name).read_bytes())
    def test_query_only_envelope_changes(self):
        v2,g2=generate_queries_v2(ROOT,Map(),GEO,QueryConfig(movement_queries=3))
        v3,g3=generate_queries_v3(ROOT,Map(),GEO,QueryConfig(movement_queries=3))
        for a,b in zip(v2,v3):
            a,b=a.data,b.data
            for k in ('schema_version','protocol_version','query_id'):a.pop(k);b.pop(k)
            self.assertEqual(a,b)
    def test_visual_default_off(self):
        import eventwm_v3
        args=eventwm_v3.parser().parse_args([])
        self.assertFalse(args.debug_visuals);self.assertEqual(args.movement_queries,8)
        self.assertNotIn('matplotlib',inspect.getsource(eventwm_v3))
    def test_benchmark_predicate_has_not_drifted(self):
        # Audited source guard: a changed benchmark requires an explicit protocol review.
        source=(Path(__file__).resolve().parents[2]/'falcon/additional_metric.py').read_text()
        psc=source.split('class PersonalSpaceCompliance(Measure):')[1].split('@registry.register_measure')[0]
        self.assertIn('if distance < 1.0:',psc)
        self.assertIn('distance = self._sim.geodesic_distance(robot_pos, human_position)',psc)
        self.assertIn('distance = np.linalg.norm(human_position - robot_pos, ord=2)',psc)
        self.assertIn('use_geo_distance: bool = True',source)


class V3CollectionGates(unittest.TestCase):
    def test_static_in_audit_branch_stops_v3_launch(self):
        from argparse import Namespace
        from types import SimpleNamespace
        from unittest.mock import patch
        import eventwm_collect
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);b,q=fixture(native=True,static=True);b['branch_id']='reverse_1'
            write(out/'roots/replay.json',dict(protocol_version=PROTOCOL))
            write(out/'queries/queries.json',[{'query':q}]);write(out/'audits/workers/reverse_1/branch.json',b)
            with patch.object(eventwm_collect.subprocess,'run',return_value=SimpleNamespace(returncode=0)):
                with self.assertRaisesRegex(RuntimeError,'STATIC_FEASIBILITY_VIOLATION'):
                    eventwm_collect.launch(Namespace(output=out,protocol='v3',storage_profile='production'),'reverse_1',q['query_id'])
            self.assertIsNone(read(out/'labels/quarantined_branch.json')['social_outcome_class'])
    def test_v3_root_failure_stops_next_root(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import eventwm_v3
        calls=[]
        def fail(cmd,**kwargs):
            dest=Path(cmd[cmd.index('--output')+1]);calls.append(dest)
            write(dest/'audits/STATIC_FEASIBILITY_VIOLATION.json',dict(static_feasibility_violation=True))
            return SimpleNamespace(returncode=1)
        with tempfile.TemporaryDirectory() as td:
            args=eventwm_v3.parser().parse_args(['--output',td+'/new','--movement-queries','3'])
            with patch.object(eventwm_v3,'provenance',return_value={}),patch.object(eventwm_v3.subprocess,'run',side_effect=fail):
                self.assertEqual(eventwm_v3.run(args),1)
            self.assertEqual(len(calls),1)
            self.assertEqual(read(args.output/'manifest.json')['roots'][0]['state'],'quarantined')
            self.assertFalse(read(next((args.output/'recipes').glob('*.json')))['debug_static_reference'])
            self.assertFalse((args.output/'debug_visuals').exists())
            self.assertEqual(read(args.output/'primary_index.json')['samples'],[])
    def test_debug_bounds_rejected_before_collection(self):
        import eventwm_v3
        for flag,value in [('--debug-visual-roots','0'),('--debug-visual-roots','5'),('--debug-visual-branches','7')]:
            with self.assertRaisesRegex(ValueError,'Debug limits'):
                eventwm_v3.run(eventwm_v3.parser().parse_args([flag,value]))
    def test_omitted_images_explicit_loader_contract(self):
        b,q=fixture()
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);write(out/'queries/queries.json',[{'query':q}]);write(out/'legal_observations/history.json',[])
            write(out/'legal_observations/image_storage.json',dict(retained=False))
            self.assertIsNone(load_wm_input(out,q['query_id'])['history_images'])
    def test_incomplete_invalid_even_if_disc_contact(self):
        b,q=fixture(distance=.5,completed=False,reason='environment_termination')
        l=social_outcome_labels(b,q,1)
        self.assertTrue(l['research_disc_collision']);self.assertFalse(l['social_outcome_valid'])
    def test_per_human_entries_and_union_duration(self):
        b,q=fixture(distance=.8,count=2)
        end=b['truth'][-1]['timestamp'];r=copy.deepcopy(b['truth'][0]);r['timestamp']=end/2
        r['social_metric']['distances'][0]['distance']=1.5
        b['truth'].insert(1,r)
        l=social_outcome_labels(b,q,2)
        self.assertEqual(l['social_violation_count'],3)
        self.assertEqual(l['social_violation_duration'],end)


class BinarySRContract(unittest.TestCase):
    def test_collision_termination_is_valid_negative(self):
        for native,static in [(True,False),(False,True),(True,True)]:
            b,q=fixture(native=native,static=static,completed=False,reason='static_collision' if static else 'robot_human_collision')
            label=social_outcome_labels(b,q,1)
            self.assertTrue(label['sr_valid']);self.assertEqual(label['sr_label'],0)
            self.assertEqual(label['static_feasibility_violation'],static)
    def test_safe_and_social_success_are_sr_positive(self):
        for distance in (3.,.8):
            b,q=fixture(distance=distance)
            self.assertEqual(social_outcome_labels(b,q,1)['sr_label'],1)
    def test_unrelated_early_termination_is_excluded(self):
        for reason in ('timeout','environment_termination','execution_failure','collection_error'):
            b,q=fixture(completed=False,reason=reason)
            label=social_outcome_labels(b,q,1)
            self.assertFalse(label['sr_valid']);self.assertIsNone(label['sr_label'])

if __name__=='__main__':unittest.main()
