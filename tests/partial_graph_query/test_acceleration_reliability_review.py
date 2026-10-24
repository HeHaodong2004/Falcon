"""Reviewer C: accelerated publication, validation, receipts and recovery faults.

All roots/journals are synthetic and temporary. No native episode, historical
dataset, service, worker process or trainer is opened or started.
"""
import copy
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from eventwm_graph.config import digest, file_digest, read_json, write_json
from eventwm_graph.partial import acceleration, campaign, collect, storage
from eventwm_graph.partial.fast_validation import validate_root as accelerated_validate
from eventwm_graph.partial.map import PartialGraph, KNOWN_FREE, KNOWN_OCCUPIED
from eventwm_graph.partial.metrics import summarize
from eventwm_graph.partial.query import compile_query, TURN_STEP, EXECUTOR_VERSION
from eventwm_graph.stage1.storage import read_gzip, write_gzip, scene_split
from test_stage1_review import fixture, row, step_label


def fork_fixture():
    """Three queries share one endpoint, then diverge in graph evidence."""
    record, _, _, _ = fixture()
    record['scene_id'] = next('acceleration_integrity_'+str(i) for i in range(100)
                              if scene_split('acceleration_integrity_'+str(i)) == 'train')
    initial = record['history'][-1]
    root = PartialGraph.restore(record['partial_graph'])
    queries, outcomes, nodes, prefixes = [], [], [], {}
    times = [1.+.048*i for i in range(3)]
    for yaw in (0., TURN_STEP, -TURN_STEP):
        query = dict(waypoints=[[.1+25/120, 0., .1]], final_yaw=yaw,
            provenance=[dict(source='goal')], decision_timestamp=1.,
            query_semantics='ordered_waypoints_plus_terminal_yaw',
            static_traversability='known_free_reachable', executor_version=EXECUTOR_VERSION)
        compiled = compile_query(root, query, initial)
        if not compiled['ok']:
            raise AssertionError(compiled)
        query.update(compiled=compiled, route_distance=compiled['route_distance'])
        query['query_id'] = digest(query)
        graph, ids, labels = PartialGraph.restore(root.snapshot()), [], []
        for i, action in enumerate(compiled['actions'], 1):
            prefix = tuple(compiled['actions'][:i])
            if prefix not in prefixes:
                key = [99, 99] if i == 1 else [100, 99]
                state = KNOWN_OCCUPIED if action == 'left' else KNOWN_FREE
                evidence = [dict(cell=key, state=state, height=0., timestamp=times[i],
                                 source='independent_visible_fork')]
                observation = row(times[i], compiled['planned_poses'][i], evidence)
                delta = graph.update(observation)
                nid = len(nodes)
                nodes.append(dict(node_id=nid, parent=ids[-1] if ids else None, action=action,
                    observation=observation, graph_delta=delta, labels=step_label(times[i]), future_index=i))
                prefixes[prefix] = nid
            else:
                nid = prefixes[prefix]
                graph.update(nodes[nid]['observation'])
            ids.append(nid); labels.append(nodes[nid]['labels'])
        poses = [initial['robot_pose'], *[nodes[n]['observation']['robot_pose'] for n in ids]]
        summary = summarize(labels, poses, 1., compiled['actions'], 'query_complete', query, record['pointgoal'])
        outcomes.append(dict(query_id=query['query_id'], node_ids=ids, summary=summary,
            successor=dict(timestamp=times[len(ids)], robot_pose=poses[-1], tracks=[], graph_revision=graph.revision),
            executor_version=EXECUTOR_VERSION, execution_engine='independent_reliability_fixture'))
        queries.append(query)
    record['collection_config']['queries_per_root'] = len(queries)
    return record, SimpleNamespace(times=times, truth=[[] for _ in times], nodes=nodes), queries, outcomes


def extended_versions():
    return {**storage.implementation(), 'acceleration/independent_fixture':digest('reviewer_c_source_1')}


def accelerated_sealer(versions=None, validator=accelerated_validate):
    versions = extended_versions() if versions is None else versions
    return acceleration.bind(storage.seal_root, implementation=lambda:dict(versions), validate_root=validator)


class AcceleratedPublicationIntegrityReview(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='acceleration_integrity_review_')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.inputs = fork_fixture()
        self.seal = accelerated_sealer()

    def publish(self):
        self.result = self.seal(self.base, *self.inputs, min_free_bytes=0)
        self.folder = Path(self.result['folder'])
        return self.result

    def mutate(self, name, change):
        value = read_gzip(self.folder/name)
        change(value)
        write_gzip(self.folder/name, value)
        manifest = read_json(self.folder/'manifest.json')
        manifest['files'][name] = file_digest(self.folder/name)
        write_json(self.folder/'manifest.json', manifest)

    def both_reject(self, message=None):
        for validator in (storage.validate_root, accelerated_validate):
            with self.subTest(validator=validator.__module__):
                if message:
                    with self.assertRaisesRegex((ValueError, IndexError), message):validator(self.folder)
                else:
                    with self.assertRaises((ValueError, IndexError)):validator(self.folder)

    def test_original_writer_code_and_manifest_identity_bind_every_acceleration_source(self):
        result = self.publish()
        self.assertIs(self.seal.__code__, storage.seal_root.__code__)
        self.assertIsNot(self.seal.__globals__, storage.seal_root.__globals__)
        self.assertEqual(result['version'], extended_versions())
        self.assertEqual(read_gzip(self.folder/'legal_root.json.gz')['version'], extended_versions())
        identity = read_gzip(self.folder/'privileged_initialization.json.gz')['identity']
        self.assertEqual(identity['version'], extended_versions())
        self.assertEqual(result['root_id'], digest(identity))
        self.assertEqual(storage.validate_root(self.folder), accelerated_validate(self.folder))
        second = accelerated_sealer({**extended_versions(), 'acceleration/independent_fixture':digest('source_2')})
        new = second(self.base, *self.inputs, min_free_bytes=0)
        self.assertNotEqual(result['root_id'], new['root_id'])
        self.assertEqual(result['query_ids'], new['query_ids'])
        self.assertEqual(result['files']['queries.json.gz'], new['files']['queries.json.gz'])

    def test_changed_graph_delta_is_rejected_after_checksums_are_resealed(self):
        self.publish()
        self.mutate('future_observations.json.gz', lambda nodes:nodes[2]['graph_delta'].update(revision=999))
        self.both_reject('graph_delta_not_reconstructable')

    def test_changed_successor_revision_is_rejected_after_checksums_are_resealed(self):
        self.publish()
        self.mutate('outcomes.json.gz', lambda outcomes:outcomes[1]['successor'].update(graph_revision=999))
        self.both_reject('successor_not_reconstructable')

    def test_corrupt_shared_parent_cannot_cross_fork_or_hide_as_valid_delta(self):
        self.publish()
        self.mutate('future_observations.json.gz', lambda nodes:nodes[2].update(parent=1))
        self.both_reject('invalid_branch_order')

    def test_future_track_labels_and_query_compilation_checks_are_not_skipped(self):
        changes = (
            ('privileged_step_labels.json.gz', lambda labels:labels[0].update(psc_violation=True)),
            ('future_observations.json.gz', lambda nodes:nodes[0]['observation'].update(
                tracks=[dict(position=[1., 0., 0.], first_seen_time=8., last_seen_time=8., observation_age=0.)])),
            ('queries.json.gz', lambda queries:queries[0]['compiled']['actions'].append('right')),
            ('outcomes.json.gz', lambda outcomes:outcomes[2]['summary'].update(executed_steps=999)),
        )
        for name, change in changes:
            with self.subTest(member=name):
                self.publish()
                original = (self.folder/name).read_bytes()
                original_manifest = (self.folder/'manifest.json').read_bytes()
                try:
                    self.mutate(name, change)
                    self.both_reject()
                finally:
                    (self.folder/name).write_bytes(original)
                    (self.folder/'manifest.json').write_bytes(original_manifest)

    def test_sha256_rejects_corrupt_payload_before_cached_parsing(self):
        self.publish()
        with (self.folder/'future_observations.json.gz').open('ab') as stream:stream.write(b'corrupt')
        self.both_reject('corrupt_root_member')

    def test_validation_failure_never_publishes_and_retains_failed_evidence(self):
        def reject(folder):
            raise ValueError('independent_validation_failure')
        failing = accelerated_sealer(validator=reject)
        with self.assertRaisesRegex(ValueError, 'independent_validation_failure'):
            failing(self.base, *self.inputs, min_free_bytes=0)
        staged = list(self.base.glob('roots/*/.partial_*/FAILED.json'))
        self.assertEqual(len(staged), 1)
        self.assertFalse([p for p in self.base.glob('roots/*/*') if not p.name.startswith('.')])
        published = self.publish()
        self.assertTrue(staged[0].exists())
        self.assertEqual(accelerated_validate(published['folder'])['root_id'], published['root_id'])

    def test_file_fsync_failure_never_publishes_or_returns_acknowledgement(self):
        original = os.fsync
        def fail_file(fd):
            if stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError('independent_file_sync_failure')
            return original(fd)
        with patch('eventwm_graph.partial.storage.os.fsync', side_effect=fail_file):
            with self.assertRaisesRegex(OSError, 'independent_file_sync_failure'):
                self.seal(self.base, *self.inputs, min_free_bytes=0)
        self.assertFalse([p for p in self.base.glob('roots/*/*') if not p.name.startswith('.')])
        self.assertTrue(list(self.base.glob('roots/*/.partial_*/FAILED.json')))
        self.publish()

    def test_atomic_rename_failure_preserves_staging_and_retry_identity(self):
        original = os.replace
        def fail_publication(source, target):
            if Path(source).is_dir() and Path(source).name.startswith('.partial_'):
                raise OSError('independent_atomic_publication_failure')
            return original(source, target)
        with patch('eventwm_graph.partial.storage.os.replace', side_effect=fail_publication):
            with self.assertRaisesRegex(OSError, 'independent_atomic_publication_failure'):
                self.seal(self.base, *self.inputs, min_free_bytes=0)
        failed = list(self.base.glob('roots/*/.partial_*/FAILED.json'))
        self.assertEqual(len(failed), 1)
        result = self.publish()
        before = (self.folder/'manifest.json').read_bytes()
        again = self.seal(self.base, *self.inputs, min_free_bytes=0)
        self.assertEqual(again['root_id'], result['root_id'])
        self.assertEqual(before, (self.folder/'manifest.json').read_bytes())
        self.assertTrue(failed[0].exists())

    def test_post_rename_sync_failure_preserves_recoverable_root_without_ack(self):
        original = os.fsync
        def fail_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError('independent_parent_sync_failure')
            return original(fd)
        with patch('eventwm_graph.partial.storage.os.fsync', side_effect=fail_directory):
            with self.assertRaisesRegex(OSError, 'independent_parent_sync_failure'):
                self.seal(self.base, *self.inputs, min_free_bytes=0)
        published = [p for p in self.base.glob('roots/*/*/manifest.json') if not p.parent.name.startswith('.')]
        self.assertEqual(len(published), 1)
        recovered = accelerated_validate(published[0].parent)
        again = self.publish()
        self.assertEqual(recovered['root_id'], again['root_id'])
        self.assertEqual(recovered['files'], again['files'])


class AccelerationSourceAndReceiptReview(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='acceleration_receipt_review_')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.record = accelerated_sealer()(self.base, *fork_fixture(), min_free_bytes=0)
        self.manifest_before = (Path(self.record['folder'])/'manifest.json').read_bytes()
        self.private = self.base/'private_geometry.json'; write_json(self.private, dict(private=True))
        self.stage0 = self.base/'stage0.json'; write_json(self.stage0, dict(stage='stage0', status='PASS'))
        self.stage1 = self.base/'stage1.json'; write_json(self.stage1, dict(stage='stage1', status='PASS'))
        self.scene = self.record['scene_id']; self.seed = self.record['seed']
        self.job = dict(seed=self.seed, scene=self.scene, manifest=str(self.private))
        self.configuration = dict(dataset=str(self.base), stage0_gate=str(self.stage0), stage1_gate=str(self.stage1),
                                  workers=28, automatic_training=False, queries_per_root=3, max_query_steps=120,
                                  private_root_curriculum_probability=.8, acceleration=dict(mode=acceleration.MODE))
        self.backend = SimpleNamespace(configuration=SimpleNamespace(scope=lambda:nullcontext()),
                                       collect_root=lambda *a, **k:self.record)

    @property
    def receipt(self):
        return self.base/'sampling'/(self.record['root_id']+'.json')

    def batch(self):
        with patch.object(acceleration, 'configure', return_value=self.backend), \
                patch.dict(acceleration._WORKER_CACHE, {}, clear=True), \
                patch('eventwm_graph.benchmark.map.load_certified_map', return_value=SimpleNamespace(metadata=dict(scene_id=self.scene))), \
                patch('eventwm_graph.partial.sampling.choose_start', return_value=(None, dict(law='synthetic_reviewer_c', selected_start_index=0))), \
                patch('eventwm_graph.partial.report.distribution', return_value={}), \
                patch('eventwm_graph.partial.acceleration.os.nice'):
            return acceleration.worker_batch([self.job], self.configuration)

    def assert_preserved(self):
        self.assertEqual((Path(self.record['folder'])/'manifest.json').read_bytes(), self.manifest_before)
        self.assertTrue(accelerated_validate(self.record['folder'])['sealed'])

    def test_receipt_publish_failure_keeps_root_unacknowledged_and_retry_repairs_receipt(self):
        with patch.object(acceleration, 'write_json', side_effect=OSError('independent_receipt_publication_failure')):
            result = self.batch()
        self.assertEqual([r['status'] for r in result], ['error'])
        self.assertFalse(self.receipt.exists()); self.assert_preserved()
        self.assertEqual([r['status'] for r in self.batch()], ['sealed'])

    def test_receipt_directory_sync_failure_is_unacknowledged_then_stable_retry(self):
        original = os.fsync
        def fail_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):raise OSError('independent_receipt_sync_failure')
            return original(fd)
        with patch.object(acceleration.os, 'fsync', side_effect=fail_directory):result = self.batch()
        self.assertEqual([r['status'] for r in result], ['error'])
        before = self.receipt.read_bytes(); self.assert_preserved()
        self.assertEqual([r['status'] for r in self.batch()], ['sealed'])
        self.assertEqual(self.receipt.read_bytes(), before)

    def test_final_receipt_hash_failure_never_returns_sealed_ack(self):
        def fail_receipt(path):
            if Path(path).parent.name == 'sampling':raise OSError('independent_receipt_hash_failure')
            return file_digest(path)
        with patch.object(acceleration, 'file_digest', side_effect=fail_receipt):result = self.batch()
        self.assertEqual([r['status'] for r in result], ['error'])
        self.assertTrue(self.receipt.exists()); self.assert_preserved()
        self.assertEqual([r['status'] for r in self.batch()], ['sealed'])

    def test_changed_receipt_is_rejected_without_overwriting_recovery_evidence(self):
        self.batch()
        receipt = read_json(self.receipt); receipt['seed'] += 1; write_json(self.receipt, receipt)
        altered = self.receipt.read_bytes(); result = self.batch()
        self.assertEqual([r['status'] for r in result], ['error'])
        self.assertIn('Root sampling receipt changed', result[0]['error'])
        self.assertEqual(self.receipt.read_bytes(), altered); self.assert_preserved()

    def backend_for_sources(self, identity):
        config = copy.deepcopy(self.configuration)
        with patch.object(collect, 'verify_gate'), patch.object(acceleration, 'source_identity', return_value=identity), \
                patch('eventwm_graph.partial.configuration_cache.ExactConfigurationCache', return_value=SimpleNamespace(scope=lambda:nullcontext())):
            return acceleration.Backend(config)

    def test_live_source_change_fails_before_any_root_publication(self):
        backend = self.backend_for_sources(extended_versions())
        sealer = backend.collect_root.__globals__['seal_root']
        self.assertIs(backend.collect_root.__code__, collect.collect_root.__code__)
        changed = {**extended_versions(), 'acceleration/independent_fixture':digest('unreviewed_source')}
        fresh = self.base/'source_change_must_not_publish'
        with patch.object(acceleration, 'source_identity', return_value=changed):
            with self.assertRaisesRegex(ValueError, 'acceleration_source_changed_during_collection'):
                sealer(fresh, *fork_fixture(), min_free_bytes=0)
        self.assertFalse(list(fresh.glob('roots/*/*/manifest.json')))

    def test_known_failed_persistent_runtime_is_rejected_before_backend_initialization(self):
        config = {**self.configuration, 'acceleration':dict(mode=acceleration.MODE, persistent_scene=True)}
        with patch.object(collect, 'verify_gate'), patch.object(acceleration, 'source_identity') as identity:
            with self.assertRaisesRegex(ValueError, 'persistent_scene_correctness_hold'):
                acceleration.Backend(config)
        identity.assert_not_called()

    def test_worker_reuse_rejects_changed_scientific_contract_but_allows_destination_change(self):
        backend = SimpleNamespace(contract=acceleration.execution_contract(self.configuration))
        with patch.object(acceleration, '_BACKEND', backend):
            self.assertIs(acceleration.configure({**self.configuration, 'dataset':'different_destination'}), backend)
            with self.assertRaisesRegex(ValueError, 'acceleration_worker_configuration_changed'):
                acceleration.configure({**self.configuration, 'queries_per_root':24})
            with self.assertRaisesRegex(ValueError, 'unknown_acceleration_options'):
                acceleration.configure({**self.configuration, 'acceleration':dict(mode=acceleration.MODE, skip_validation=True)})

    def test_scene_cache_binds_both_manifest_path_and_changed_content(self):
        other = self.base/'other_private_geometry.json'; write_json(other, dict(private=True, version=2))
        jobs = [self.job, {**self.job, 'seed':self.seed+1, 'manifest':str(other)},
                {**self.job, 'seed':self.seed+2, 'manifest':str(other)}]
        maps = [SimpleNamespace(metadata=dict(scene_id=self.scene)) for _ in jobs]
        records = {self.seed:self.record}
        for offset in (1, 2):
            values = list(fork_fixture()); values[0]['seed'] += offset
            records[self.seed+offset] = accelerated_sealer()(self.base, *values, min_free_bytes=0)
        seen = []
        def collected(hmap, output, seed, *args, **kwargs):
            seen.append(hmap)
            return records[seed]
        self.backend.collect_root = collected
        with patch.object(acceleration, 'configure', return_value=self.backend), \
                patch.dict(acceleration._WORKER_CACHE, {}, clear=True), \
                patch('eventwm_graph.benchmark.map.load_certified_map', side_effect=maps) as loader, \
                patch('eventwm_graph.partial.sampling.choose_start', return_value=(None, dict(law='synthetic_reviewer_c'))), \
                patch('eventwm_graph.partial.report.distribution', return_value={}), \
                patch('eventwm_graph.partial.acceleration.os.nice'):
            result = []
            for index, job in enumerate(jobs):
                if index == 2:write_json(other, dict(private=True, version=3))
                result.extend(acceleration.worker_batch([job], self.configuration))
        self.assertEqual([r['status'] for r in result], ['sealed']*3)
        self.assertEqual(loader.call_count, 3)
        self.assertEqual([id(h) for h in seen], [id(h) for h in maps])

    def test_loaded_geometry_scene_mismatch_cannot_be_cached_or_collected(self):
        self.backend.collect_root = Mock()
        with patch.object(acceleration, 'configure', return_value=self.backend), \
                patch.dict(acceleration._WORKER_CACHE, {}, clear=True), \
                patch('eventwm_graph.benchmark.map.load_certified_map', return_value=SimpleNamespace(metadata=dict(scene_id='wrong_scene'))), \
                patch('eventwm_graph.partial.acceleration.os.nice'):
            result = acceleration.worker_batch([self.job], self.configuration)
            self.assertFalse(acceleration._WORKER_CACHE)
        self.assertEqual([r['status'] for r in result], ['error'])
        self.assertIn('private_geometry_scene_identity_mismatch', result[0]['error'])
        self.backend.collect_root.assert_not_called()
        self.assertFalse(self.receipt.exists())

    def source_identity_fixture(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch('eventwm_graph.partial.native_batch.load_backend',
            return_value=(None, None, dict(extension_sha256=digest('review_native_binary')))))
        stack.enter_context(patch('eventwm_graph.partial.configuration_cache.configuration_sources',
            return_value={'review_configuration':digest('original_configuration')}))
        stack.enter_context(patch.object(acceleration, '_NUMERIC_IDENTITY',
            dict(version='review_numeric_version', files={'review_loaded_library':digest('original_numeric_binary')})))
        return stack

    def test_sampler_scheduler_and_codec_drift_change_producer_identity_without_editing_sources(self):
        targets = ('partial/sampling.py', 'partial/campaign.py', 'stage1/storage.py', 'config.py')
        with self.source_identity_fixture():
            baseline = acceleration.source_identity()
            for target in targets:
                with self.subTest(dependency=target):
                    path = (Path(acceleration.REPO)/'eventwm_graph'/target).resolve()
                    key = 'dependency/eventwm_graph/'+target
                    self.assertEqual(baseline[key], file_digest(path))
                    def modified_digest(candidate):
                        return digest('independent_unreviewed_dependency') if Path(candidate).resolve() == path else file_digest(candidate)
                    with patch.object(acceleration, 'file_digest', side_effect=modified_digest):
                        changed = acceleration.source_identity()
                    self.assertEqual({name for name in baseline if baseline[name] != changed[name]}, {key})

    def test_missing_pinned_sampler_fails_closed_instead_of_omitting_dependency(self):
        path = (Path(acceleration.REPO)/'eventwm_graph/partial/sampling.py').resolve()
        def missing(candidate):
            if Path(candidate).resolve() == path:raise FileNotFoundError('independent_missing_sampler')
            return file_digest(candidate)
        with self.source_identity_fixture(), patch.object(acceleration, 'file_digest', side_effect=missing):
            with self.assertRaisesRegex(FileNotFoundError, 'independent_missing_sampler'):
                acceleration.source_identity()

    def test_loaded_numeric_contract_is_explicitly_bound_to_every_producer_identity(self):
        with self.source_identity_fixture():
            baseline = acceleration.source_identity()
            acceleration._NUMERIC_IDENTITY['files']['review_loaded_library'] = digest('different_numeric_binary')
            changed = acceleration.source_identity()
        self.assertEqual({name for name in baseline if baseline[name] != changed[name]},
                         {'acceleration/numpy_numeric_contract'})


class AcceleratedJournalRecoveryReview(unittest.TestCase):
    def test_stopped_resume_requeues_interrupted_jobs_preserves_sealed_counts_and_starts_no_worker(self):
        with tempfile.TemporaryDirectory(prefix='acceleration_journal_review_') as directory:
            base = Path(directory); output = base/'campaign'; output.mkdir()
            db = campaign.database(output/'campaign.sqlite3')
            sealed = dict(outcomes=3, bytes=123, statistics={}, distributions={})
            db.executemany('INSERT INTO jobs VALUES(?,?,?,?,?)', [
                (100, 'train_scene', 'synthetic_manifest', 'sealed', json.dumps(sealed)),
                (101, 'train_scene', 'synthetic_manifest', 'running', None),
                (102, 'train_scene', 'synthetic_manifest', 'queued', None)])
            db.commit(); db.close()
            (output/'STOP').write_text('Independent stopped recovery fixture.\n')
            config = dict(output=str(output), dataset=str(base/'dataset'), automatic_training=False, workers=28,
                stage0_gate='unused0', stage1_gate='unused1', target_outcomes=1000, roots_per_batch=2,
                campaign_seed=20261021, seed_base=100, disk_reserve_bytes=0, memory_reserve_bytes=0,
                max_dataset_bytes=10**9, report_interval_s=20)
            configuration = base/'configuration.json'; write_json(configuration, config)

            class NoWorkers:
                def __init__(self, **kwargs):
                    if kwargs['max_workers'] != 28:raise AssertionError('worker_count_changed')
                def __enter__(self):return self
                def __exit__(self, *args):pass
                def submit(self, *args):raise AssertionError('STOP must prevent every worker submission')

            concurrent = SimpleNamespace(futures=SimpleNamespace(ProcessPoolExecutor=NoWorkers))
            signals = SimpleNamespace(SIGTERM=15, SIGINT=2, signal=lambda *args:None)
            run = acceleration.bind(campaign.run, verify_gate=lambda *args:None,
                ready_scenes=lambda:{'train_scene':'synthetic_manifest'}, available_memory=lambda:10**12,
                concurrent=concurrent, signal=signals, worker_batch=lambda *a:(_ for _ in ()).throw(AssertionError('worker_started')))
            with patch.object(campaign.os, 'cpu_count', return_value=32), redirect_stdout(io.StringIO()):
                result = run(configuration)
            self.assertEqual(result['status'], 'stopped')
            self.assertEqual((result['roots'], result['outcomes'], result['bytes']), (1, 3, 123))
            db = campaign.database(output/'campaign.sqlite3')
            try:
                self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
                self.assertEqual(db.execute('SELECT seed,state FROM jobs ORDER BY seed').fetchall(),
                                 [(100, 'sealed'), (101, 'queued'), (102, 'queued')])
                self.assertEqual(db.execute('SELECT result FROM jobs WHERE seed=100').fetchone()[0],
                                 json.dumps(sealed))
            finally:
                db.close()
            self.assertTrue((output/'STOP').exists())


class AcceleratedProductionGateReview(unittest.TestCase):
    """Typed synthetic proof fixtures never authorize or launch a collector."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='acceleration_gate_review_')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        gate0, gate1 = self.base/'old0.json', self.base/'old1.json'
        write_json(gate0, dict(stage='stage0', status='PASS'))
        write_json(gate1, dict(stage='stage1', status='PASS'))
        self.configuration = dict(workers=28, automatic_training=False, stage0_gate=str(gate0), stage1_gate=str(gate1),
            queries_per_root=24, max_query_steps=120, roots_per_batch=2,
            private_root_curriculum_probability=.8, safe_margin_m=.2, intrusion_radius_m=.96,
            campaign_seed=20261021, seed_base=100, scenes=['synthetic_train_a'],
            acceleration=dict(mode=acceleration.MODE, persistent_scene=False))
        self.versions = dict(accepted_core=digest('original'), acceleration=digest('candidate'))
        self.gate = dict(schema='stage01_acceleration_acceptance_1', status='READY_FOR_PRODUCTION',
            implementation=self.versions, execution_contract=acceleration.execution_contract(self.configuration),
            checks={key:True for key in ('differential', 'native_replay', 'complete_tests', 'scientific_review',
                'performance_review', 'reliability_review', 'full_pipeline_target')}, evidence={})
        self.documents = {}
        self.document('benchmark_plan', dict(workers=28, jobs=[dict(seed=100), dict(seed=101)]))
        plan_sha = self.gate['evidence']['benchmark_plan']['sha256']
        self.document('benchmark_configuration', copy.deepcopy(self.configuration))
        self.document('benchmark_source', dict(identity=dict(implementation=self.versions), plan_sha256=plan_sha))
        self.document('reference', dict(schema='stage01_full_pipeline_fixed28_benchmark_1', status='PASS', plan_sha256=plan_sha))
        benchmark = dict(schema='stage01_full_pipeline_fixed28_benchmark_1', status='PASS', backend='fast', workers=28,
            elapsed_seconds=100., validated_outcomes_per_second=18.5, sqlite_integrity='ok', planned_attempts=2,
            plan_sha256=plan_sha, statistics=dict(outcomes=1850, roots=1, attempts=dict(sealed=1, rejected=1)),
            results=[dict(seed=100, status='sealed', outcomes=1850), dict(seed=101, status='rejected')])
        self.document('full_pipeline_benchmark', benchmark)
        self.document('differential', dict(schema='stage01_exact_scientific_differential_1', status='PASS',
            candidate=copy.deepcopy(self.gate['evidence']['full_pipeline_benchmark']),
            reference=copy.deepcopy(self.gate['evidence']['reference']), attempts=2, sealed_roots=1,
            validated_outcomes=1850, errors=[], results=[dict(seed=i, passed=True) for i in (100, 101)]))
        for kind, schema in (
            ('native_replay', 'stage01_acceleration_native_replay_1'),
            ('complete_tests', 'stage01_acceleration_test_suite_1'),
            ('scientific_review', 'stage01_acceleration_scientific_review_1'),
            ('performance_review', 'stage01_acceleration_performance_review_1'),
            ('reliability_review', 'stage01_acceleration_reliability_review_1')):
            self.document(kind, dict(schema=schema, status='PASS', implementation=self.versions,
                branches=4, action_endpoints=32, repeats=2, scenes=['synthetic_a', 'synthetic_b'],
                tests=150, failures=0, errors=0))
        self.path = self.base/'acceptance.json'

    def document(self, kind, value):
        self.documents[kind] = value
        path = self.base/(kind+'.json')
        # Python's JSON reader accepts non-standard NaN/Infinity. The verifier
        # must reject them explicitly; a corrupted proof must fail closed.
        path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')
        self.gate['evidence'][kind] = dict(path=str(path), sha256=file_digest(path))

    def refresh_benchmark(self):
        self.document('full_pipeline_benchmark', self.documents['full_pipeline_benchmark'])
        differential = self.documents['differential']
        differential['candidate'] = copy.deepcopy(self.gate['evidence']['full_pipeline_benchmark'])
        self.document('differential', differential)

    def verify(self, configuration=None):
        write_json(self.path, self.gate)
        with patch.object(acceleration, 'source_identity', return_value=self.versions):
            return acceleration.verify_acceleration_gate(self.path, configuration or self.configuration)

    def test_complete_typed_source_and_configuration_bound_fixture_is_accepted(self):
        self.assertEqual(self.verify(), self.gate)

    def test_failed_target_status_cannot_be_overridden_by_true_flags(self):
        self.gate['status'] = 'PERFORMANCE_TARGET_NOT_MET'
        with self.assertRaisesRegex(ValueError, 'production_gate_closed'):self.verify()

    def test_arbitrary_nonempty_evidence_list_cannot_open_gate(self):
        self.gate['evidence'] = [self.gate['evidence']['scientific_review']]
        with self.assertRaisesRegex(ValueError, 'typed_evidence_missing'):self.verify()

    def test_missing_failed_or_stale_source_report_is_rejected(self):
        baseline = copy.deepcopy(self.documents['scientific_review'])
        for change in (dict(status='FAIL'), dict(implementation=dict(wrong='source')), dict(schema='unrelated_report')):
            with self.subTest(change=change):
                self.document('scientific_review', {**baseline, **change})
                with self.assertRaises(ValueError):self.verify()

    def test_changed_scientific_configuration_or_old_gate_content_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'validated_configuration_changed'):
            self.verify({**self.configuration, 'private_root_curriculum_probability':0.})
        old_gate = Path(self.configuration['stage0_gate'])
        write_json(old_gate, dict(stage='stage0', status='PASS', unreviewed=True))
        with self.assertRaisesRegex(ValueError, 'validated_configuration_changed'):self.verify()

    def test_nonfinite_or_nonpositive_benchmark_clock_cannot_prove_throughput(self):
        for elapsed in (float('nan'), float('inf'), 0., -1., True):
            with self.subTest(elapsed=elapsed):
                self.documents['full_pipeline_benchmark']['elapsed_seconds'] = elapsed
                self.refresh_benchmark()
                with self.assertRaisesRegex(ValueError, 'invalid_benchmark_clock'):self.verify()

    def test_rate_is_recomputed_instead_of_trusting_declared_throughput(self):
        self.documents['full_pipeline_benchmark']['elapsed_seconds'] = 200.
        self.refresh_benchmark()
        with self.assertRaisesRegex(ValueError, 'full_pipeline_target_not_proved'):self.verify()

    def test_older_benchmark_source_cannot_authorize_current_implementation(self):
        source = self.documents['benchmark_source']
        source['identity']['implementation'] = dict(accepted_core=digest('original'), acceleration=digest('older'))
        self.document('benchmark_source', source)
        with self.assertRaisesRegex(ValueError, 'benchmark_source_mismatch'):self.verify()

    def test_other_benchmark_configuration_cannot_authorize_this_configuration(self):
        config = self.documents['benchmark_configuration']; config['queries_per_root'] = 3
        self.document('benchmark_configuration', config)
        with self.assertRaisesRegex(ValueError, 'benchmark_configuration_mismatch'):self.verify()

    def test_duplicate_attempts_or_unfinished_jobs_cannot_inflate_rate(self):
        baseline = copy.deepcopy(self.documents['full_pipeline_benchmark'])
        for change in ('duplicate', 'running'):
            with self.subTest(change=change):
                self.documents['full_pipeline_benchmark'] = copy.deepcopy(baseline)
                if change == 'duplicate':
                    self.documents['full_pipeline_benchmark']['results'][1]['seed'] = 100
                else:
                    self.documents['full_pipeline_benchmark']['statistics']['attempts']['running'] = 1
                self.refresh_benchmark()
                with self.assertRaisesRegex(ValueError, 'full_pipeline_target_not_proved'):self.verify()

    def test_differential_must_name_exact_benchmark_and_reference_workload(self):
        old = copy.deepcopy(self.documents['differential'])
        old['candidate'] = copy.deepcopy(self.gate['evidence']['reference'])
        self.document('differential', old)
        with self.assertRaisesRegex(ValueError, 'differential_benchmark_mismatch'):self.verify()


class AcceleratedFrozenBundleReview(unittest.TestCase):
    """Freeze tiny committed fixtures; no native binary is imported or executed."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='acceleration_freezer_review_')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.repo = self.base/'repo'; self.repo.mkdir()
        self.output = self.base/'bundles'
        self.git('init', '-q')
        (self.repo/'README.md').write_text('initial unrelated documentation\n')
        self.unrelated_commit = self.commit('initial documentation')
        sources = {
            'eventwm_graph/config.py': 'from pathlib import Path\nREPO = Path(__file__).resolve().parents[1]\n',
            'eventwm_graph/partial/acceleration.py': 'candidate = True\n',
            'eventwm_graph/partial/fast_observer.py': 'observer = True\n',
            'eventwm_graph/partial/fast_graph.py': 'graph = True\n',
            'eventwm_graph/partial/fast_validation.py': 'validation = True\n',
            'eventwm_graph/partial/native_batch.cpp': '// exact native adapter fixture\n',
            'scripts/stage01_benchmark.py': '# benchmark fixture\n',
            'scripts/build_stage01_native_batch.py': '# builder fixture\n',
            'scripts/collector/observation.py': '# observation fixture\n',
            'scripts/collector/__init__.py': '',
            'configs/stage0_scenes.json': '{}\n',
            'falcon/additional_action.py': '# native action fixture\n',
            'falcon/additional_metric.py': '# native metric fixture\n',
        }
        for name, value in sources.items():
            path = self.repo/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value)
        self.source_commit = self.commit('committed source fixture')
        self.build = self.base/'build'; self.build.mkdir()
        extension = '_fixture_native_batch.so'
        (self.build/extension).write_bytes(b'never imported or executed\n')
        self.native = dict(extension=extension, extension_sha256=file_digest(self.build/extension))
        write_json(self.build/'build.json', self.native)
        path = Path(__file__).resolve().parents[2]/'scripts/stage01_acceleration_freeze.py'
        spec = importlib.util.spec_from_file_location('_independent_freezer_review', path)
        self.freezer = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.freezer)
        self.versions = {'accepted_core': digest('original'), 'acceleration': digest('fixture')}
        for context in (
                patch.object(self.freezer, 'ROOT', self.repo),
                patch.object(self.freezer, 'source_identity', return_value=self.versions),
                patch.object(self.freezer, 'load_backend', return_value=(None, None, self.native)),
                patch.dict(os.environ, {'EVENTWM_NATIVE_BATCH_BUILD': str(self.build)})):
            context.start(); self.addCleanup(context.stop)
        self.destination = self.freezer.freeze(self.output)
        self.original_manifest = (self.destination/'manifest.json').read_bytes()

    def git(self, *arguments):
        return subprocess.run(['git', *arguments], cwd=self.repo, check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit(self, message):
        self.git('add', '.')
        self.git('-c', 'user.name=Independent Reviewer C', '-c', 'user.email=review@example.invalid',
                 'commit', '-qm', message)
        return self.git('rev-parse', 'HEAD')

    def manifest(self, change):
        value = json.loads(self.original_manifest)
        change(value)
        path = self.destination/'manifest.json'; path.chmod(0o644); write_json(path, value)

    def test_valid_bundle_reuse_preserves_original_receipt_across_documentation_commit(self):
        self.assertEqual(self.freezer.freeze(self.output), self.destination)
        (self.repo/'README.md').write_text('later documentation only\n')
        later = self.commit('later documentation')
        self.assertNotEqual(later, self.source_commit)
        self.assertEqual(self.freezer.freeze(self.output), self.destination)
        self.assertEqual((self.destination/'manifest.json').read_bytes(), self.original_manifest)
        manifest = read_json(self.destination/'manifest.json')
        self.assertEqual(manifest['repository_commit'], self.source_commit)
        self.assertIn('native_batch/build.json', manifest['frozen_sources'])
        self.assertIn('eventwm_graph/partial/native_batch.cpp', manifest['frozen_sources'])
        self.assertNotEqual(manifest['original_sources']['eventwm_graph/config.py'],
                            manifest['frozen_sources']['eventwm_graph/config.py'])

    def test_manifest_identity_and_deleted_inventory_entry_fail_closed(self):
        mutations = (
            lambda m:m.update(source_id='different'),
            lambda m:m.update(implementation={'wrong':'source'}),
            lambda m:m['original_sources'].pop('eventwm_graph/partial/acceleration.py'),
            lambda m:m.update(repository_assets='/different/assets'),
            lambda m:m.update(external_sources={}),
            lambda m:m.update(native_build_environment='/different/build'),
            lambda m:m.update(automatic_training=True),
            lambda m:m.update(service_started=True),
            lambda m:m['frozen_sources'].pop('native_batch/build.json'),
            lambda m:m.update(unreviewed_metadata=True),
        )
        for index, change in enumerate(mutations):
            with self.subTest(change=index):
                self.manifest(change)
                with self.assertRaisesRegex(ValueError, 'manifest changed'):
                    self.freezer.freeze(self.output)

    def test_payloads_cannot_be_corrupted_then_rehashed_in_manifest(self):
        members = ('eventwm_graph/config.py', 'eventwm_graph/partial/native_batch.cpp',
                   'scripts/build_stage01_native_batch.py', 'native_batch/build.json',
                   'native_batch/'+self.native['extension'])
        for name in members:
            with self.subTest(member=name):
                path = self.destination/name; before = path.read_bytes(); path.chmod(0o644)
                path.write_bytes(before+b'corruption')
                try:
                    self.manifest(lambda m:m['frozen_sources'].update({name:file_digest(path)}))
                    with self.assertRaisesRegex(ValueError, 'manifest changed'):
                        self.freezer.freeze(self.output)
                finally:
                    path.write_bytes(before)
        (self.destination/'manifest.json').write_bytes(self.original_manifest)
        path = self.destination/'eventwm_graph/partial/acceleration.py'; path.chmod(0o644)
        path.write_text('changed without rehashing\n')
        with self.assertRaisesRegex(ValueError, 'payload changed'):self.freezer.freeze(self.output)

    def test_missing_extra_or_symlinked_physical_members_are_rejected(self):
        manifest_path = self.destination/'manifest.json'; manifest_path.chmod(0o644)
        path = self.destination/'eventwm_graph/partial/acceleration.py'
        before = path.read_bytes(); path.unlink()
        with self.assertRaisesRegex(ValueError, 'inventory changed'):self.freezer.freeze(self.output)
        path.write_bytes(before)
        extra = self.destination/'unbound.py'; extra.write_text('# unbound\n')
        with self.assertRaisesRegex(ValueError, 'inventory changed'):self.freezer.freeze(self.output)
        extra.unlink(); path.unlink(); path.symlink_to(self.repo/'eventwm_graph/partial/acceleration.py')
        with self.assertRaisesRegex(ValueError, 'symlinks'):self.freezer.freeze(self.output)

    def test_recorded_repository_commit_must_reproduce_original_sources(self):
        for commit in ('not-a-commit', self.unrelated_commit, '0'*40):
            with self.subTest(commit=commit):
                self.manifest(lambda m:m.update(repository_commit=commit))
                with self.assertRaisesRegex(ValueError, 'repository commit'):
                    self.freezer.freeze(self.output)


if __name__ == '__main__':
    unittest.main()
