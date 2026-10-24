"""Explicit acceleration bindings around the unchanged accepted collector.

The scientific collector and atomic writer retain their accepted bytecode. We
bind their private dependencies in new function-global dictionaries, rather
than altering legacy modules, acceptance hashes, checkpoints or source bundles.
All new dependencies are added to every accelerated root's version identity.
Production entry requires a separate source-bound acceleration acceptance file;
bounded benchmarks are explicit and never imply production authorization.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import FunctionType

from ..config import REPO, digest, file_digest, read_json, write_json
from . import campaign, collect, storage

MODE = 'exact_native_batch_live_prefix_v1'
_BACKEND = None
_WORKER_CACHE = {}
_NUMERIC_IDENTITY = None
SOURCES = ('acceleration.py', 'fast_observer.py', 'fast_execution.py',
           'native_batch.py', 'native_batch.cpp', 'configuration_cache.py',
           'fast_graph.py', 'fast_validation.py')


def bind(function, **dependencies):
    """New dependency bindings; original module and function are unchanged."""
    result = FunctionType(function.__code__, {**function.__globals__, **dependencies},
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def source_identity():
    global _NUMERIC_IDENTITY
    from .native_batch import load_backend, HABITAT_SHA256
    _, _, native = load_backend()
    directory = Path(__file__).parent
    result = storage.implementation()
    result.update({'acceleration/'+name: file_digest(directory/name) for name in SOURCES})
    result['acceleration/build_stage01_native_batch.py'] = file_digest(REPO/'scripts/build_stage01_native_batch.py')
    result['acceleration/habitat_native_binary'] = HABITAT_SHA256
    result['acceleration/native_batch_extension'] = native['extension_sha256']
    from .configuration_cache import configuration_sources
    result['acceleration/configuration_sources'] = digest(configuration_sources())
    # These original dependencies affect scheduling, sampling, serialization
    # and private geometry but were absent from the legacy version dictionary.
    # Frozen config.py only redirects assets to REPO; pin its logical original
    # here and its physical shim separately in the immutable bundle manifest.
    dependencies = ('partial/sampling.py', 'partial/campaign.py', 'partial/report.py',
                    'stage1/__init__.py', 'stage1/storage.py', 'benchmark/motion.py',
                    'benchmark/certify.py', 'benchmark/dependencies.py', 'native.py',
                    'map.py', 'local.py', 'surface.py', 'topology.py',
                    'robotmap/execution.py', 'robotmap/body.py', 'robotmap/config.py')
    for name in dependencies:
        path = directory.parent/name
        if not path.exists():
            path = REPO/'eventwm_graph'/name
        result['dependency/eventwm_graph/'+name] = file_digest(path)
    result['dependency/eventwm_graph/config.py'] = file_digest(REPO/'eventwm_graph/config.py')
    # Loaded numeric libraries are process-lifetime dependencies, as is the
    # already verified Habitat ABI. Hash them once, rather than rereading tens
    # of MB at every root publication. Their digests remain in every identity.
    if _NUMERIC_IDENTITY is None:
        import numpy as np
        import numpy.core._multiarray_umath as core
        import numpy.linalg.linalg as linalg
        numeric_paths = [Path(core.__file__), Path(linalg.__file__)]
        numeric_paths += sorted((Path(np.__file__).parent.parent/'numpy.libs').glob('*.so*'))
        _NUMERIC_IDENTITY = dict(version=np.__version__, files={
            str(path.resolve()): file_digest(path) for path in numeric_paths})
    result['acceleration/numpy_numeric_contract'] = digest(_NUMERIC_IDENTITY)
    return result


def execution_contract(configuration):
    """Bind every scientific/scheduling option, excluding only destinations."""
    options = configuration.get('acceleration', {})
    if set(options)-{'mode', 'persistent_scene', 'acceptance_gate'}:
        raise ValueError('unknown_acceleration_options')
    if options.get('mode') != MODE or options.get('persistent_scene', False) is not False:
        raise ValueError('unvalidated_acceleration_execution_mode')
    if configuration.get('workers') != 28 or configuration.get('automatic_training') is not False:
        raise ValueError('acceleration_requires_fixed28_collection_only')
    operational = {'output', 'dataset', 'report_interval_s', 'target_outcomes',
                   'stage0_gate', 'stage1_gate', 'acceleration'}
    return dict(configuration={k: v for k, v in configuration.items() if k not in operational},
                acceleration=dict(mode=MODE, persistent_scene=False),
                stage0_gate_sha256=file_digest(configuration['stage0_gate']),
                stage1_gate_sha256=file_digest(configuration['stage1_gate']))


class Backend:
    def __init__(self, configuration):
        from .fast_observer import FastGeometryObserver
        from .fast_execution import LiveGraphPrefixFuture
        from .fast_graph import IncrementalPartialGraph
        from .fast_validation import validate_root
        from .configuration_cache import ExactConfigurationCache
        from ..stage1.runtime import NativeEpisode
        # These are genuine original-source checks, before any dependency is
        # rebound. The existing accepted gates are neither edited nor bypassed.
        collect.verify_gate(configuration['stage0_gate'])
        collect.verify_gate(configuration['stage1_gate'], 'stage1')
        options = configuration.get('acceleration', {})
        if options.get('mode') != MODE:
            raise ValueError('unknown_acceleration_mode')
        if options.get('persistent_scene', False):
            raise ValueError('persistent_scene_correctness_hold_native_physics_clock')
        self.contract = execution_contract(configuration)
        self.versions = source_identity()
        self.configuration = ExactConfigurationCache()

        @contextmanager
        def runtime(hmap, episode, seed, visual=False):
            with self.configuration.scope(), NativeEpisode(hmap, episode, seed, visual=visual) as rt:
                rt.private_geometry_identity = (hmap.map_id, hmap.robot_configuration_id,
                                                hmap.contract['identity'])
                yield rt

        def versions():
            current = source_identity()
            if current != self.versions:
                raise ValueError('acceleration_source_changed_during_collection')
            return dict(self.versions)

        sealer = bind(storage.seal_root, implementation=versions, validate_root=validate_root)
        class Execution(LiveGraphPrefixFuture):
            evaluate_many = bind(LiveGraphPrefixFuture.evaluate_many, PartialGraph=IncrementalPartialGraph)
        self.collect_root = bind(collect.collect_root, GeometryObserver=FastGeometryObserver,
                                 GraphPrefixFuture=Execution, NativeEpisode=runtime,
                                 PartialGraph=IncrementalPartialGraph, seal_root=sealer)


def configure(configuration):
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = Backend(configuration)
    elif _BACKEND.contract != execution_contract(configuration):
        raise ValueError('acceleration_worker_configuration_changed')
    return _BACKEND


def worker_batch(jobs, configuration):
    """Same deterministic sampling and durable receipt order as the reference."""
    from ..benchmark.map import load_certified_map
    from .report import distribution
    from .sampling import choose_start
    from ..stage1.storage import scene_split
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet')
    try:
        os.nice(10)
    except OSError:
        pass
    backend = configure(configuration)
    results = []
    for job in jobs:
        began = time.time()
        try:
            scene = job['scene']
            cache_key = (scene, str(Path(job['manifest']).resolve()), file_digest(job['manifest']))
            if cache_key not in _WORKER_CACHE:
                if len(_WORKER_CACHE) >= 2:
                    _WORKER_CACHE.pop(next(iter(_WORKER_CACHE)))
                t = time.perf_counter()
                with backend.configuration.scope():
                    h = load_certified_map(job['manifest'])
                if h.metadata['scene_id'] != scene:
                    raise ValueError('private_geometry_scene_identity_mismatch')
                _WORKER_CACHE[cache_key] = (h, {}, time.perf_counter()-t, {})
            h, private_cache, cache_seconds, prior_cache = _WORKER_CACHE[cache_key]
            prior_started = time.perf_counter()
            node_override, prior = choose_start(h, job['seed'], configuration.get('private_root_curriculum_probability', 0.), prior_cache)
            prior_seconds = time.perf_counter()-prior_started
            record = backend.collect_root(h, configuration['dataset'], job['seed'], configuration['stage0_gate'],
                configuration['queries_per_root'], configuration['max_query_steps'], node_index=node_override,
                private_cache=private_cache, safe_margin=configuration.get('safe_margin_m', .2),
                intrusion_radius=configuration.get('intrusion_radius_m', .96))
            prior.update(root_id=record['root_id'], seed=job['seed'], scene=scene,
                         sampler_sha256=file_digest(Path(__file__).with_name('sampling.py')),
                         private_geometry_manifest_sha256=file_digest(job['manifest']))
            sampling_file = Path(configuration['dataset'])/'sampling'/(record['root_id']+'.json')
            if sampling_file.exists() and read_json(sampling_file) != prior:
                raise ValueError('Root sampling receipt changed')
            write_json(sampling_file, prior)
            with sampling_file.open('rb') as stream:
                os.fsync(stream.fileno())
            fd = os.open(sampling_file.parent, os.O_RDONLY); os.fsync(fd); os.close(fd)
            result = dict(seed=job['seed'], status='sealed', root_id=record['root_id'],
                folder=record['folder'], outcomes=record['queries'], statistics=record['statistics'],
                bytes=sum(p.stat().st_size for p in Path(record['folder']).iterdir() if p.is_file()),
                wall_seconds=time.time()-began, geometry_load_seconds=cache_seconds,
                performance=record['performance'], private_initialization_sampling_seconds=prior_seconds)
            if scene_split(scene) == 'train':
                result['distributions'] = distribution(record['folder'])
            result['root_prior'] = prior
            result['sampling_receipt_sha256'] = file_digest(sampling_file)
            results.append(result)
        except Exception as error:
            expected = isinstance(error, ValueError) and str(error) in campaign.EXPECTED_REJECTIONS
            results.append(dict(seed=job['seed'], status='rejected' if expected else 'error',
                                error=repr(error), wall_seconds=time.time()-began,
                                traceback=None if expected else traceback.format_exc()))
            if not expected:
                break
    return results


def verify_acceleration_gate(path, configuration):
    gate = read_json(path)
    required = {'differential', 'native_replay', 'complete_tests', 'scientific_review',
                'performance_review', 'reliability_review', 'full_pipeline_target'}
    if gate.get('schema') != 'stage01_acceleration_acceptance_1' or gate.get('status') != 'READY_FOR_PRODUCTION':
        raise ValueError('acceleration_production_gate_closed')
    if not required.issubset(gate.get('checks', {})) or any(gate['checks'][k] is not True for k in required):
        raise ValueError('acceleration_production_evidence_incomplete')
    if gate.get('implementation') != source_identity():
        raise ValueError('acceleration_validated_source_changed')
    if gate.get('execution_contract') != execution_contract(configuration):
        raise ValueError('acceleration_validated_configuration_changed')
    required_evidence = {
        'differential': 'stage01_exact_scientific_differential_1',
        'native_replay': 'stage01_acceleration_native_replay_1',
        'complete_tests': 'stage01_acceleration_test_suite_1',
        'scientific_review': 'stage01_acceleration_scientific_review_1',
        'performance_review': 'stage01_acceleration_performance_review_1',
        'reliability_review': 'stage01_acceleration_reliability_review_1',
        'full_pipeline_benchmark': 'stage01_full_pipeline_fixed28_benchmark_1',
    }
    evidence = gate.get('evidence', {})
    if not isinstance(evidence, dict) or not set(required_evidence).issubset(evidence):
        raise ValueError('acceleration_typed_evidence_missing')
    reports = {}
    for kind, schema in required_evidence.items():
        row = evidence[kind]
        if file_digest(row['path']) != row['sha256']:
            raise ValueError('acceleration_evidence_changed')
        report = reports[kind] = read_json(row['path'])
        if report.get('schema') != schema or report.get('status') != 'PASS':
            raise ValueError('acceleration_typed_evidence_failed:'+kind)
        if kind not in ('differential', 'full_pipeline_benchmark') and report.get('implementation') != gate['implementation']:
            raise ValueError('acceleration_evidence_source_mismatch:'+kind)
    differential = reports['differential']
    if (not differential.get('attempts') or not differential.get('sealed_roots') or differential.get('errors') or
            len(differential.get('results', [])) != differential['attempts'] or
            not all(r.get('passed') is True for r in differential['results'])):
        raise ValueError('acceleration_differential_incomplete')
    tests = reports['complete_tests']
    if tests.get('tests', 0) < 117 or tests.get('failures', 1) or tests.get('errors', 1):
        raise ValueError('acceleration_test_suite_incomplete')
    replay = reports['native_replay']
    if replay.get('branches', 0) <= 0 or replay.get('action_endpoints', 0) <= 0:
        raise ValueError('acceleration_native_replay_empty')
    benchmark = reports['full_pipeline_benchmark']
    stats = benchmark.get('statistics', {})
    attempts = stats.get('attempts', {})
    results = benchmark.get('results', [])
    elapsed = benchmark.get('elapsed_seconds', 0)
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('acceleration_invalid_benchmark_clock')
    if type(stats.get('outcomes')) is not int or stats['outcomes'] <= 0:
        raise ValueError('acceleration_invalid_outcome_count')
    rate = stats['outcomes']/elapsed
    if (benchmark.get('backend') != 'fast' or benchmark.get('workers') != 28 or
            stats.get('outcomes', 0)/elapsed < 18.5 or benchmark.get('sqlite_integrity') != 'ok' or
            attempts.get('running', 0) or attempts.get('error', 0) or attempts.get('queued', 0) or
            len(results) != benchmark.get('planned_attempts') or len(results) != differential['attempts'] or
            len({r['seed'] for r in results}) != len(results) or
            sum(r['status'] == 'sealed' for r in results) != stats.get('roots') or
            sum(r.get('outcomes', 0) for r in results if r['status'] == 'sealed') != stats.get('outcomes')):
        raise ValueError('acceleration_full_pipeline_target_not_proved')
    if (not math.isfinite(rate) or not math.isclose(benchmark.get('validated_outcomes_per_second', -1), rate, rel_tol=1e-12) or
            set(attempts)-{'sealed', 'rejected'} or sum(attempts.values()) != len(results)):
        raise ValueError('acceleration_benchmark_statistics_mismatch')
    bound = {}
    for kind in ('benchmark_source', 'benchmark_configuration', 'benchmark_plan'):
        if kind not in evidence:
            raise ValueError('acceleration_benchmark_source_evidence_missing')
        row = evidence[kind]
        if file_digest(row['path']) != row['sha256']:
            raise ValueError('acceleration_benchmark_source_evidence_changed')
        bound[kind] = read_json(row['path'])
    if bound['benchmark_source'].get('identity', {}).get('implementation') != gate['implementation']:
        raise ValueError('acceleration_benchmark_source_mismatch')
    if execution_contract(bound['benchmark_configuration']) != gate['execution_contract']:
        raise ValueError('acceleration_benchmark_configuration_mismatch')
    plan_sha = evidence['benchmark_plan']['sha256']
    if benchmark.get('plan_sha256') != plan_sha or bound['benchmark_source'].get('plan_sha256') != plan_sha:
        raise ValueError('acceleration_benchmark_plan_mismatch')
    if {r['seed'] for r in results} != {j['seed'] for j in bound['benchmark_plan'].get('jobs', [])}:
        raise ValueError('acceleration_benchmark_attempt_identity_mismatch')
    candidate = differential['candidate']
    if file_digest(candidate['path']) != candidate['sha256'] or candidate['sha256'] != evidence['full_pipeline_benchmark']['sha256']:
        raise ValueError('acceleration_differential_benchmark_mismatch')
    reference = differential['reference']
    if file_digest(reference['path']) != reference['sha256'] or read_json(reference['path']).get('plan_sha256') != plan_sha:
        raise ValueError('acceleration_reference_workload_mismatch')
    return gate


def run(configuration_path):
    cfg = read_json(configuration_path)
    verify_acceleration_gate(cfg['acceleration']['acceptance_gate'], cfg)
    configure(cfg)
    return bind(campaign.run, worker_batch=worker_batch)(configuration_path)
