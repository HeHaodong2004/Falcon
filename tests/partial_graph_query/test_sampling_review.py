"""Independent private spawn-law and campaign receipt regressions.

Only synthetic maps and temporary sealed roots are used. No native episode,
held-out episode content, production worker, or historical service is opened.
"""
import copy
import math
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from eventwm_graph.config import digest, file_digest, read_json, write_json
from eventwm_graph.partial import campaign, sampling
from eventwm_graph.partial.initialization import make_episode
from eventwm_graph.partial.storage import seal_root, online_input
from eventwm_graph.partial.dataset import supervised_example, sampling_metadata
from eventwm_graph.stage1.storage import scene_split
from test_stage1_review import fixture


class SyntheticNativeGeometry:
    """Deterministic finite rays, without a complete graph API or humans."""
    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=float)
        self.angles = np.linspace(-math.pi, math.pi, 32, endpoint=False)

    def locate(self, start, target):
        node = int(round(float(start[0])/10.))
        d = np.asarray(target)-start
        angle = math.atan2(-float(d[2]), float(d[0]))
        differences = np.abs(np.arctan2(np.sin(self.angles-angle), np.cos(self.angles-angle)))
        return node, int(differences.argmin())

    def try_step_no_sliding(self, start, target):
        start = np.asarray(start, dtype=float); node, direction = self.locate(start, target)
        d = np.asarray(target)-start; length = float(np.linalg.norm(d[[0,2]]))
        if length > 3.7:
            return start+d*(self.rows[node,direction,0]/length)
        result = np.asarray(target, dtype=float).copy()
        result[1] = start[1]+self.rows[node,direction,2]
        return result

    def distance_to_closest_obstacle(self, vantage, maximum):
        node = int(round(float(vantage[0])/10.))
        start = np.array([10.*node,0.,0.])
        _, direction = self.locate(start, vantage)
        return float(self.rows[node,direction,1])


def synthetic_map(rows):
    nodes = {'n'+str(i):{'position':[10.*i,0.,0.]} for i in range(len(rows))}
    return SimpleNamespace(nodes=nodes,
        valid_starts=lambda:[{'node_id':name} for name in sorted(nodes)],
        navigation=SimpleNamespace(pf=SyntheticNativeGeometry(rows)),
        metadata={'scene_id':'synthetic_sampling_review'},map_id='synthetic_sampling_map')


class IndependentSpawnLawReview(unittest.TestCase):
    def table(self):
        rows = np.empty((4,32,3)); rows[:,:,0] = 2.; rows[:,:,1] = .3; rows[:,:,2] = 0.
        rows[0,:,0] = 3.8                  # Always eligible.
        rows[1,:,0] = 3.8; rows[1,:,1] = .1 # No terminal viewpoint clearance.
        rows[2,:,0] = 3.8; rows[2,:,2] = .2 # Unsupported vertical transition.
        rows[3,:,0] = 3.49                 # Too little straight progress.
        return rows

    def test_uniform_zero_probability_keeps_original_node_draw_and_reads_no_geometry(self):
        h = synthetic_map(self.table())
        with patch.object(h.navigation.pf,'try_step_no_sliding',side_effect=AssertionError('private ray')):
            for seed in range(20):
                override, receipt = sampling.choose_start(h,seed,0.,{})
                rng = np.random.default_rng(int(digest({'seed':seed,'domain':'partial_root_prior_v1'})[:16],16))
                self.assertIsNone(override)
                self.assertEqual(receipt['selected_start_index'],int(rng.integers(4)))
                self.assertEqual(receipt['unconditional_selected_start_probability'],.25)
                self.assertEqual(receipt['uniform_reference_importance_ratio'],1.)

    def test_geometry_heading_tie_is_the_actual_initialization_tie(self):
        rows = self.table()
        rows[0,:,0] = 3.4; rows[0,[1,7,12,19],0] = [3.8,3.78,3.76,3.75]
        rows[0,:,1] = .1; rows[0,[1,12],1] = .3
        h = synthetic_map(rows); table = sampling.geometry_table(h)
        def fake_episode(hmap,node,seed,count):
            return SimpleNamespace(start_position=hmap.nodes[node]['position'],info={})
        with patch('eventwm_graph.partial.initialization.episode_for_map',side_effect=fake_episode):
            for seed in range(64):
                eligible = sampling.eligible_indices(table,seed)
                for index,start in enumerate(h.valid_starts()):
                    _, metadata = make_episode(h,start['node_id'],seed,0)
                    a = metadata['root_heading']; directions = np.linspace(-math.pi,math.pi,32,endpoint=False)
                    direction = int(np.argmin(np.abs(directions-a)))
                    row = table['rows'][index]
                    expected = row[:,0].max() >= 3.5 and row[direction,1] >= .2 and row[direction,2] <= .15
                    self.assertEqual(index in eligible,expected,(seed,index,a))

    def test_component_summed_probabilities_normalize_and_importance_weights_recover_uniform(self):
        h = synthetic_map(self.table()); cache = {}; seen = set(); p = .8
        for seed in range(256):
            override, receipt = sampling.choose_start(h,seed,p,cache)
            index = receipt['selected_start_index']; eligible = sampling.eligible_indices(cache['geometry_table'],seed)
            masses = [(1-p)/4+(p/len(eligible) if i in eligible else 0.) for i in range(4)]
            self.assertAlmostEqual(sum(masses),1.,places=15)
            self.assertAlmostEqual(receipt['unconditional_selected_start_probability'],masses[index],places=15)
            self.assertAlmostEqual(masses[index]*receipt['uniform_reference_importance_ratio'],.25,places=15)
            self.assertEqual(override,index if receipt['component']=='geometry_curriculum' else None)
            seen.add((receipt['component'],index in eligible))
        self.assertEqual(seen,{('geometry_curriculum',True),('uniform',True),('uniform',False)})

    def test_no_eligible_start_has_explicit_rejection_mass_and_no_fallback_bias(self):
        rows = self.table(); rows[:,:,0] = 1.; h = synthetic_map(rows); cache = {}; rejections = successes = 0
        for seed in range(64):
            try:
                override, receipt = sampling.choose_start(h,seed,.8,cache)
            except ValueError as error:
                self.assertEqual(str(error),'no_private_bootstrap_start'); rejections += 1
            else:
                self.assertIsNone(override); self.assertEqual(receipt['eligible_start_count'],0)
                self.assertAlmostEqual(receipt['unconditional_selected_start_probability'],.2/4)
                self.assertAlmostEqual(receipt['uniform_reference_importance_ratio'],5.)
                successes += 1
        self.assertGreater(rejections,0); self.assertGreater(successes,0)

    def test_same_seed_and_cached_geometry_have_identical_receipts(self):
        h = synthetic_map(self.table()); cache = {}
        first = sampling.choose_start(h,107,.8,cache)
        with patch('eventwm_graph.partial.sampling.geometry_table',side_effect=AssertionError('rebuild')):
            self.assertEqual(sampling.choose_start(h,107,.8,cache),first)
        self.assertEqual(sampling.choose_start(h,107,.8,{}),first)
        self.assertNotIn('seconds',first[1])

    def test_no_human_future_or_graph_query_api_is_consulted(self):
        h = synthetic_map(self.table())
        with patch('eventwm_graph.partial.query.generate_queries',side_effect=AssertionError('online query')), \
                patch('eventwm_graph.partial.observer.GeometryObserver',side_effect=AssertionError('observation')), \
                patch('eventwm_graph.stage1.runtime.NativeEpisode',side_effect=AssertionError('native future')):
            _, receipt = sampling.choose_start(h,1,.8,{})
        self.assertTrue(receipt['private_initialization_only']); self.assertFalse(receipt['uses_future'])

    def test_invalid_probabilities_cannot_remove_uniform_support(self):
        for p in (-.1,1.,1.1,float('nan')):
            with self.assertRaises(ValueError):sampling.choose_start(synthetic_map(self.table()),1,p,{})


class IndependentSamplingReceiptReview(unittest.TestCase):
    def setUp(self):
        campaign._WORKER_CACHE.clear(); self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup); self.addCleanup(campaign._WORKER_CACHE.clear)
        record,tape,queries,outcomes = fixture()
        self.record = seal_root(self.directory.name,record,tape,queries,outcomes,min_free_bytes=0)
        self.seed = record['seed']; self.scene = record['scene_id']
        self.private_manifest = Path(self.directory.name)/'synthetic_manifest.json'
        write_json(self.private_manifest,{'private':'synthetic native geometry fixture'})
        self.jobs = [dict(scene=self.scene,seed=self.seed,manifest=str(self.private_manifest))]
        self.config = dict(dataset=self.directory.name,stage0_gate='unit_stage0',stage1_gate='unit_stage1',
                           queries_per_root=1,max_query_steps=120,private_root_curriculum_probability=0.)
        self.h = synthetic_map(IndependentSpawnLawReview().table())

    def batch(self, **extra):
        with patch('eventwm_graph.benchmark.map.load_certified_map',return_value=self.h), \
                patch('eventwm_graph.partial.collect.collect_root',return_value=self.record), \
                patch('eventwm_graph.partial.report.distribution',return_value={}), \
                patch('eventwm_graph.partial.campaign.verify_gate'), patch('eventwm_graph.partial.campaign.os.nice'):
            return campaign.worker_batch(self.jobs,self.config)

    @property
    def receipt(self):return Path(self.directory.name)/'sampling'/(self.record['root_id']+'.json')

    def test_retry_has_identical_receipt_bytes_hash_and_node_override(self):
        first = self.batch(); before = self.receipt.read_bytes(); sha = file_digest(self.receipt)
        second = self.batch()
        self.assertEqual([r['status'] for r in first],['sealed'])
        self.assertEqual(first[0]['root_prior'],second[0]['root_prior'])
        self.assertEqual(first[0]['sampling_receipt_sha256'],sha)
        self.assertEqual(second[0]['sampling_receipt_sha256'],sha)
        self.assertEqual(self.receipt.read_bytes(),before)

    def test_failure_before_receipt_publication_preserves_root_then_retry_repairs_receipt(self):
        original = campaign.write_json
        def fail(path,value):
            if Path(path).parent.name=='sampling':raise OSError('injected receipt publication failure')
            return original(path,value)
        original_files = copy.deepcopy(self.record['files'])
        with patch('eventwm_graph.partial.campaign.write_json',side_effect=fail):first = self.batch()
        self.assertEqual([r['status'] for r in first],['error']); self.assertFalse(self.receipt.exists())
        self.assertEqual(read_json(Path(self.record['folder'])/'manifest.json')['files'],original_files)
        second = self.batch(); self.assertEqual([r['status'] for r in second],['sealed'])
        self.assertTrue(self.receipt.exists())

    def test_directory_fsync_failure_is_unacknowledged_and_stable_retry_succeeds(self):
        original = os.fsync
        def fail_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):raise OSError('injected receipt directory sync failure')
            return original(fd)
        with patch('eventwm_graph.partial.campaign.os.fsync',side_effect=fail_directory):first = self.batch()
        self.assertEqual([r['status'] for r in first],['error']); before = self.receipt.read_bytes()
        second = self.batch(); self.assertEqual([r['status'] for r in second],['sealed'])
        self.assertEqual(self.receipt.read_bytes(),before)

    def test_changed_existing_receipt_is_rejected_and_never_overwritten(self):
        self.batch(); altered = read_json(self.receipt); altered['curriculum_probability'] = .2
        write_json(self.receipt,altered); before = self.receipt.read_bytes()
        result = self.batch(); self.assertEqual([r['status'] for r in result],['error'])
        self.assertIn('Root sampling receipt changed',result[0]['error'])
        self.assertEqual(self.receipt.read_bytes(),before)

    def test_no_sealed_ack_is_returned_if_final_receipt_hashing_fails(self):
        def fail_receipt(path):
            if Path(path).parent.name=='sampling':raise OSError('injected receipt hash read failure')
            return file_digest(path)
        with patch('eventwm_graph.partial.campaign.file_digest',side_effect=fail_receipt):
            result = self.batch()
        self.assertEqual([r['status'] for r in result],['error'])
        self.assertTrue(self.receipt.exists())

    def test_sampling_sidecar_is_never_read_by_online_or_supervised_inputs(self):
        self.batch(); baseline = online_input(self.record['folder'])
        self.receipt.write_text('{ deliberately invalid private sampling sidecar')
        self.assertEqual(online_input(self.record['folder']),baseline)
        example = supervised_example(self.record['folder'])
        self.assertEqual(example['inputs'],baseline)
        self.assertNotIn('root_prior',example['inputs'])

    def test_offline_sampling_helper_is_optional_and_checks_root_identity(self):
        self.assertIsNone(sampling_metadata(self.record['folder']))
        self.batch(); receipt = sampling_metadata(self.record['folder'])
        self.assertEqual(receipt,read_json(self.receipt))
        self.assertEqual(receipt['sampler_sha256'],file_digest(Path(sampling.__file__)))
        self.assertEqual(receipt['private_geometry_manifest_sha256'],file_digest(self.private_manifest))
        for field,value in (('root_id','wrong_root'),('seed',self.seed+1),('scene','wrong_scene')):
            write_json(self.receipt,{**receipt,field:value})
            with self.assertRaisesRegex(ValueError,'does not identify this root'):
                sampling_metadata(self.record['folder'])

    def test_worker_geometry_prior_caches_are_separate_per_scene(self):
        scenes = ['hm3d_X7gTkoDHViv','hm3d_yZME6UR9dUN']
        h1 = self.h; h2 = synthetic_map(IndependentSpawnLawReview().table())
        jobs = [dict(scene=scenes[i%2],seed=self.seed+i,manifest=str(self.private_manifest)) for i in range(3)]
        cache_ids = {}; choose_start = sampling.choose_start
        def choose(h,seed,probability,cache):
            old = cache.setdefault('independent_bound_map',id(h)); self.assertEqual(old,id(h))
            cache_ids.setdefault(id(h),set()).add(id(cache))
            return choose_start(h,seed,probability,cache)
        def collected(h,output,seed,*args,**kwargs):
            return {**self.record,'root_id':digest({'synthetic_receipt_seed':seed})}
        with patch('eventwm_graph.benchmark.map.load_certified_map',side_effect=[h1,h2]) as loaded, \
                patch('eventwm_graph.partial.collect.collect_root',side_effect=collected), \
                patch('eventwm_graph.partial.sampling.choose_start',side_effect=choose), \
                patch('eventwm_graph.partial.report.distribution',return_value={}), \
                patch('eventwm_graph.partial.campaign.verify_gate'), patch('eventwm_graph.partial.campaign.os.nice'):
            result = campaign.worker_batch(jobs,self.config)
        self.assertEqual([r['status'] for r in result],['sealed']*3)
        self.assertEqual(loaded.call_count,2)
        self.assertEqual([len(v) for v in cache_ids.values()],[1,1])
        self.assertNotEqual(next(iter(cache_ids[id(h1)])),next(iter(cache_ids[id(h2)])))

    def test_held_out_roots_do_not_enter_development_distribution_reader(self):
        heldout = next('independent_sampling_heldout_'+str(i) for i in range(100)
                       if scene_split('independent_sampling_heldout_'+str(i))!='train')
        record = {**self.record,'scene_id':heldout,'split':'confirmation'}
        jobs = [dict(scene=heldout,seed=self.seed,manifest=str(self.private_manifest))]
        with patch('eventwm_graph.benchmark.map.load_certified_map',return_value=self.h), \
                patch('eventwm_graph.partial.collect.collect_root',return_value=record), \
                patch('eventwm_graph.partial.report.distribution') as distribution, \
                patch('eventwm_graph.partial.campaign.verify_gate'), patch('eventwm_graph.partial.campaign.os.nice'):
            campaign.worker_batch(jobs,self.config)
        distribution.assert_not_called()


if __name__=='__main__':unittest.main()
