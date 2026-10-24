"""Paired CPU systems benchmark; no training and no future/holdout payloads.

All three methods start from one fixed Repaired-model incoming belief. The
optimized/unoptimized structural comparison changes computation only; its
residual and successor laws are identical. The preserved old reference has
different declared semantics and is reported separately. Quantiles include the
whole public invocation (observation, forecast, all queries and successors).

Run only after coordinating the two-thread research allowance. This process
uses one low-priority CPU thread. ``--quick`` is a diagnostic, not a release
tail-latency study. No experiment runs on import or with --help.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import collections
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

# Capture before importing runtime modules so startup edits cannot produce a
# misleading disk-source hash for a different already-imported implementation.
EARLY_SOURCE_HASHES = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((REPO / 'eventwm_graph/l2_partial').glob('*.py'))}
EARLY_SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from eventwm_graph.l2_partial.repair_profile import (
    _sha, _metadata, _rss_kib, _percentiles, _cache_state, _clear_caches, _publish,
    _sources, _causal_input,
)
from eventwm_graph.l2_partial.neural import LearnedDynamics
from eventwm_graph.l2_partial.queries import NativeEventModel
from eventwm_graph.l2_partial.repair_model import RepairedGraphBeliefWorldModel
from eventwm_graph.l2_partial.structural_model import StructuralGraphBeliefWorldModel
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
from eventwm_graph.l2_partial.structural_budget import InferenceBudgetExceeded, UnsupportedConditioning
from eventwm_graph.l2_partial.structural_kernels import (
    warmup_kernels, clear_kernel_caches, kernel_cache_info, event_features_many_fast,
)
from l2_partial_repair_systems_stress import fixture as preserved_fixture, validate as validate_fixture

BASELINE_REVISION = 'd697d966'
METHODS = ('old_reference', 'structural_bound', 'structural_bound_unoptimized')
CACHE_MODES = ('fully_cold', 'graph_revalidation', 'warm')
SNAPSHOT = REPO / 'output/l2_partial_graph_wm_explore_20261021/snapshot'
MANIFEST_SHA256 = '92b80427e258db7f1ddef94a70e0d932dbe6d2ba834809ac322715f16599210c'
REPAIR = REPO / 'output/l2_partial_graph_wm_repair_20261021'
OUTPUT_ROOT = REPO / 'output/l2_partial_graph_wm_structural_20261021'
ASSETS = {
    'checkpoint': (REPO / 'output/l2_partial_graph_wm_explore_20261021/learned_posdiff/checkpoint.pt',
                   '57c91156101320e2be6a02aee23d214a0405959e823b73be3786b0287bd25c05'),
    'population': (REPO / 'output/l2_partial_graph_wm_explore_20261021/train.npy.population.json',
                   'b7df625ac62e0eae09656fa32532b72e28039af140fc7b2803e7866b8f1f539c'),
    'event_checkpoint': (REPAIR / 'native_events_v5_scene.json',
                         'feec91f6918963f618c8a44694fb210bcd92549c9a92e52fc394f2c0b6aa736b'),
    'optical': (REPAIR / 'observation_fit.json',
                'ace29a5160b4a8510ac96d3c7bfb3cbad0936fc809c2dd35bf302aaa874e8793'),
}
COLLECTOR_STATUS = Path('/research-user/workspace/projects/eventwm-falcon-acceleration-v2/output/'
    'stage01_acceleration_v2_20261021/collection_user_authorized_5s_20261021_1/status.json')


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def collector_metadata():
    result = {'timestamp_unix': time.time(), 'path': str(COLLECTOR_STATUS),
              'logical_cpu_count': os.cpu_count(), 'load_average': os.getloadavg()}
    try:
        raw = COLLECTOR_STATUS.read_bytes(); value = json.loads(raw)
        result.update(status_sha256=hashlib.sha256(raw).hexdigest(),
            status={k: value.get(k) for k in ('status', 'status_reason', 'outcomes', 'roots',
                'active_worker_count', 'collector_restart_count', 'supervisor_pid',
                'validated_outcomes_per_second_recent')})
    except (OSError, ValueError) as exc:
        result['read_error'] = type(exc).__name__ + ': ' + str(exc)
    return result


def audit_preserved_baseline():
    """Read-only source verification; never checks out or rewrites old files."""
    prefix = 'eventwm_graph/l2_partial/'
    names = subprocess.run(['git', 'ls-tree', '-r', '--name-only', BASELINE_REVISION, '--', prefix],
                           cwd=REPO, check=True, capture_output=True, text=True).stdout.splitlines()
    names = [n for n in names if n.endswith('.py')]
    names.append('scripts/l2_partial_repair_systems_stress.py')
    hashes = {}
    for name in names:
        blob = subprocess.run(['git', 'show', BASELINE_REVISION + ':' + name], cwd=REPO,
                              check=True, capture_output=True).stdout
        expected = hashlib.sha256(blob).hexdigest(); actual = _sha(REPO / name)
        if actual != expected:
            raise ValueError('preserved baseline source differs from ' + BASELINE_REVISION + ': ' + name)
        hashes[name] = expected
    assets = {}
    for name, (path, expected) in ASSETS.items():
        actual = _sha(path)
        if actual != expected:
            raise ValueError('frozen asset mismatch: ' + name)
        assets[name] = {'path': str(path), 'sha256': actual}
    if _sha(SNAPSHOT / 'manifest.json') != MANIFEST_SHA256:
        raise ValueError('frozen source snapshot manifest mismatch')
    return {'revision': BASELINE_REVISION, 'source_sha256': hashes,
            'assets': assets, 'manifest_sha256': MANIFEST_SHA256}


def output_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    # Physical resolution prevents symlink/.. escape into old or production data.
    if OUTPUT_ROOT.resolve() != OUTPUT_ROOT or not path.is_relative_to(OUTPUT_ROOT) or path == OUTPUT_ROOT:
        raise ValueError('output must be a NEW file inside structural research output')
    if path.exists() or Path(str(path) + '.jsonl').exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def selected_rows(limit, cohort=None):
    manifest = json.loads((SNAPSHOT / 'manifest.json').read_text())
    rows = [r for r in manifest['rows'] if r['split'] == 'dev']
    if cohort:
        raw = json.loads(Path(cohort).read_text())
        requested = raw.get('root_ids', raw.get('rows')) if isinstance(raw, dict) else raw
        if requested is None:
            raise ValueError('cohort needs root_ids or rows')
        ids = {r['root_id'] if isinstance(r, dict) else r for r in requested}
    else:
        # Same real development cohort as the preserved release, in manifest
        # order. This metadata file is not a new predictive evaluation payload.
        ids = set(json.loads((REPAIR / 'latency_v5_active.json').read_text())['cohort']['root_ids'])
    if not ids <= {r['root_id'] for r in rows}:
        raise ValueError('only original dev roots can enter this benchmark')
    rows = [r for r in rows if r['root_id'] in ids]
    if limit:
        rows = rows[:limit]
    return rows


def verify_legal_files(row):
    result = {}
    for name in ('legal_root.json.gz', 'queries.json.gz'):
        path = SNAPSHOT / 'roots' / row['root_id'] / name
        digest = _sha(path)
        if digest != row['files'][name]:
            raise ValueError('frozen legal payload changed: ' + str(path))
        result[name] = digest
    return result


def make_models():
    started = time.perf_counter()
    dynamics = LearnedDynamics.load(str(ASSETS['checkpoint'][0]))
    kwargs = {'dynamics': dynamics,
              'population_prior': json.loads(ASSETS['population'][0].read_text()),
              'event_model': NativeEventModel(json.loads(ASSETS['event_checkpoint'][0].read_text())['coefficients']),
              'observation_checkpoint': str(ASSETS['optical'][0]),
              'maneuver_rate': 2., 'reset_std': 2., 'maneuver_retention': 1.}
    if any(p.device.type != 'cpu' for p in dynamics.network.parameters()):
        raise ValueError('CPU-only benchmark requires CPU checkpoint')
    models = {'old_reference': RepairedGraphBeliefWorldModel(**kwargs),
              'structural_bound': StructuralGraphBeliefWorldModel(**kwargs, residual_policy='bound', optimized=True),
              'structural_bound_unoptimized': StructuralGraphBeliefWorldModel(**kwargs, residual_policy='bound', optimized=False)}
    if models['structural_bound'].model_version != models['structural_bound_unoptimized'].model_version:
        raise AssertionError('optimized switch changed probability-model version')
    return models, (time.perf_counter() - started) * 1000.


def _old_observation_key(row):
    detections = sorted([r for r in row.get('tracks', []) if r.get('visible_now') and r.get('valid', True)],
                        key=lambda r: tuple(r['position']))
    boxes = sorted(set(tuple(np.asarray(b, float).ravel()) for b in row.get('certified_visible_empty_boxes', [])))
    payload = {'timestamp': float(row['timestamp']), 'robot_pose': row['robot_pose'],
               'positions': [r['position'] for r in detections], 'updates': row.get('static_updates', []),
               'empty': boxes, 'trace': {}, 'sensor': row.get('human_sensor_available', True)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
        default=lambda x: np.asarray(x).tolist()).encode()).hexdigest(), detections, boxes


def convert_incoming(previous, model, actual, producer):
    """Isolated array copy + explicit model tag migration, outside timed calls."""
    hs = [h.fork(**{k: v.copy() for k, v in vars(h).items() if isinstance(v, np.ndarray)})
          for h in previous.hypotheses]
    b = previous.fork(hypotheses=hs, robot_pose=previous.robot_pose.copy(),
                      evidence=copy.deepcopy(previous.evidence), model_version=model.model_version)
    migrated = False
    if isinstance(model, StructuralGraphBeliefWorldModel):
        b = StructuralBeliefState.wrap(b)
        from eventwm_graph.l2_partial.hidden import HIDDEN_RELAXATION
        if (type(producer) is not RepairedGraphBeliefWorldModel or producer.kind != 'structured'
                or previous.model_version != producer.model_version or HIDDEN_RELAXATION != .7
                or any(h.n and (h.relaxation <= 0 or not np.isclose(h.relaxation, .7))
                       for h in previous.hypotheses if h.weight > 0)):
            raise ValueError('incoming belief lacks audited canonical transition-law provenance')
        # A verified code-law certificate, not simulator truth or a fitted
        # statistical parameter. The snapshot/model/source guards above bind
        # the producer and this migration is recorded beside each timing row.
        b = b.fork(evidence={**b.evidence,'residual_count_support':list(range(len(producer.population_prior))), 'transition_noise_contract': {
            'kind': 'independent_position_diffusion', 'variance_rate': .0025,
            'source': 'audited RepairedGraphBeliefWorldModel initialization under frozen structured config',
            'model_version': model.model_version}})
        key, detections, boxes = _old_observation_key(actual)
        if previous.evidence.get('last_observation_key') == key:
            # The baseline already assimilated this physical sensor event.
            # Translating its receipt avoids double conditioning one-row roots.
            t = float(actual['timestamp'])
            payload = {'timestamp': t, 'robot_pose': np.asarray(actual['robot_pose'], float).tolist(),
                       'positions': sorted(tuple(np.asarray(r['position'], float)) for r in detections),
                       'empty_boxes': boxes}
            signature = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                default=lambda x: np.asarray(x).tolist(), separators=(',', ':')).encode()).hexdigest()
            identity = str(actual.get('human_observation_id', 'endpoint:' + repr(t)))
            b = b.fork(evidence={**b.evidence, 'human_evidence_ledger': {identity: signature},
                                'human_evidence_timestamp': t})
            migrated = True
    # The receipt/model tag are metadata; every probabilistic/physical field is
    # bit-identical before the timed new observation.
    for a, e in zip(b.hypotheses, previous.hypotheses):
        for name, value in vars(e).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(getattr(a, name), value)
            else:
                assert getattr(a, name) == value
    assert b.unresolved_mass == previous.unresolved_mass
    assert (b.graph.namespace, b.graph.version, b.graph.depth) == (
        previous.graph.namespace, previous.graph.version, previous.graph.depth)
    assert b.graph.materialize() == previous.graph.materialize()
    return b, migrated


def clear_caches(model, mode):
    _clear_caches(model, footprints=mode == 'fully_cold')
    if mode == 'fully_cold':
        if hasattr(model, 'endpoint_template_cache'):
            model.endpoint_template_cache.clear()
        clear_kernel_caches()


def cache_state(model):
    endpoints = getattr(model, 'endpoint_template_cache', {})
    endpoint_bytes = sum(v.nbytes for templates in endpoints.values() for h in templates
                         for v in vars(h).values() if isinstance(v, np.ndarray))
    return {**_cache_state(model), **kernel_cache_info(),
            'endpoint_template_entries': len(endpoints), 'endpoint_template_array_bytes': endpoint_bytes}


def _hypothesis_values(h):
    return {k: v for k, v in vars(h).items() if not k.startswith('_')}


def _belief_values(b):
    value = {k: getattr(b, k) for k in ('timestamp', 'robot_pose', 'unresolved_mass',
                                       'terminal', 'model_version', 'query_cursor', 'terminal_event')}
    value['graph'] = (b.graph.namespace, b.graph.version, b.graph.depth)
    value['hypotheses'] = [_hypothesis_values(h) for h in b.hypotheses]
    # Diagnostic CDFs intentionally differ; compare actual posterior state and
    # consequential evidence, not timing or optional offline diagnostics.
    keys = ('pruned_mass', 'unsupported_observation', 'assignment_evidence_exact',
            'residual_bound_status', 'birth_probability', 'exact_detections',
            'negative_evidence_available', 'empty_evidence_quadrature')
    value['evidence'] = {k: b.evidence[k] for k in keys if k in b.evidence}
    c = getattr(b, 'carrier', None)
    value['carrier'] = None if c is None else {
        **{k: getattr(c, k) for k in ('times', 'positions', 'velocities', 'weights',
            'unresolved_mass', 'seed', 'index', 'observation_ledger')},
        'worlds': [_hypothesis_values(h) for h in c.worlds],
        'rng_state': getattr(c, 'rng_state', None)}
    return value


def _batch_values(batch):
    f = batch.shared_future
    result = {'shared': {k: getattr(f, k) for k in ('times', 'positions', 'velocities', 'weights',
                                                   'unresolved_mass', 'seed')}, 'predictions': []}
    result['shared']['worlds'] = [_hypothesis_values(h) for h in f.worlds]
    for p in batch.predictions:
        fields = ('query_id', 'collision_probability', 'psc_probability', 'collision_interval',
                  'psc_interval', 'duration_samples', 'duration_weights', 'progress_samples',
                  'time_to_conflict', 'blockage_probability', 'termination_probabilities',
                  'unresolved_uncertainty', 'evaluated_interval', 'query_complete',
                  'remaining_query_cursor', 'executor_version', 'model_version')
        row = {k: getattr(p, k) for k in fields}
        row['branches'] = [{'probability': br.probability, 'observation': br.observation,
                            'successor': _belief_values(br.successor)} for br in p.observation_branches]
        result['predictions'].append(row)
    return result


def compare_same_law(left, right, *, atol=2e-8, rtol=2e-9):
    """Strict structural correspondence plus declared numerical tolerances.

    No matching/reordering of components or worlds is allowed. A tiny numeric
    perturbation that changes a beam or sampled world is a diagnosed failure,
    not silently relabeled parity. String fingerprints are reported separately
    because they intentionally distinguish floating-point bytes.
    """
    stats = {'atol': atol, 'rtol': rtol, 'comparisons': 0, 'mismatches': [],
             'maximum_absolute_numeric_error': 0., 'maximum_normalized_error': 0.}
    def mismatch(path, message):
        if len(stats['mismatches']) < 30:
            stats['mismatches'].append({'path': path, 'message': message})
        stats['mismatch_count'] = stats.get('mismatch_count', 0) + 1
    def compare(a, b, path):
        stats['comparisons'] += 1
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                mismatch(path, 'mapping keys differ'); return
            for key in sorted(a):
                compare(a[key], b[key], path + '.' + str(key))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                mismatch(path, 'sequence length differs'); return
            for i, (aa, bb) in enumerate(zip(a, b)):
                compare(aa, bb, path + '[' + str(i) + ']')
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            aa = np.asarray(a); bb = np.asarray(b)
            if aa.shape != bb.shape:
                mismatch(path, 'array shapes differ'); return
            if not np.issubdtype(aa.dtype, np.number) or not np.issubdtype(bb.dtype, np.number):
                if not np.array_equal(aa, bb):
                    mismatch(path, 'non-numeric arrays differ')
                return
            if np.issubdtype(aa.dtype, np.integer) and np.issubdtype(bb.dtype, np.integer):
                if not np.array_equal(aa, bb):
                    mismatch(path, 'discrete integer arrays differ')
                return
            if not np.array_equal(np.isfinite(aa), np.isfinite(bb)) or not np.array_equal(np.isnan(aa), np.isnan(bb)):
                mismatch(path, 'finite/NaN support differs'); return
            finite = np.isfinite(aa)
            if not np.array_equal(aa[~finite & ~np.isnan(aa)], bb[~finite & ~np.isnan(bb)]):
                mismatch(path, 'infinite supports differ'); return
            if np.any(finite):
                delta = np.abs(aa[finite] - bb[finite])
                scale = atol + rtol * np.abs(bb[finite])
                absolute = float(np.max(delta)); relative = float(np.max(delta / scale))
                stats['maximum_absolute_numeric_error'] = max(stats['maximum_absolute_numeric_error'], absolute)
                stats['maximum_normalized_error'] = max(stats['maximum_normalized_error'], relative)
                if relative > 1:
                    mismatch(path, 'numeric array exceeds declared tolerance')
        elif isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            if a is None or b is None:
                mismatch(path, 'defined/undefined value differs'); return
            aa = float(a); bb = float(b)
            if not math.isfinite(aa) or not math.isfinite(bb):
                if aa != bb:
                    mismatch(path, 'nonfinite scalar differs')
                return
            delta = abs(aa - bb); relative = delta / (atol + rtol * abs(bb))
            stats['maximum_absolute_numeric_error'] = max(stats['maximum_absolute_numeric_error'], delta)
            stats['maximum_normalized_error'] = max(stats['maximum_normalized_error'], relative)
            if relative > 1:
                mismatch(path, 'numeric scalar exceeds declared tolerance')
        elif a != b:
            mismatch(path, 'discrete value differs')
    lb, lq = left; rb, rq = right
    compare(_belief_values(lb), _belief_values(rb), 'posterior')
    compare(_batch_values(lq), _batch_values(rq), 'batch')
    stats['exact_posterior_fingerprint_equal'] = lb.fingerprint() == rb.fingerprint()
    stats['passed'] = not stats.get('mismatch_count', 0)
    return stats


def invoke(model, previous, actual, queries, seed, budget_ms=None):
    model.timings.clear()
    wall = time.perf_counter(); cpu = time.process_time()
    posterior = batch = None; error = None
    try:
        kwargs = {'horizon': 3., 'num_worlds': 16, 'seed': seed}
        if budget_ms is not None:
            if not isinstance(model, StructuralGraphBeliefWorldModel):
                raise ValueError('legacy reference has no enforced end-to-end budget')
            kwargs['budget_ms'] = budget_ms
        posterior, batch = model.online_call(previous, actual, queries, **kwargs)
    except (InferenceBudgetExceeded, UnsupportedConditioning) as exc:
        error = {'type': type(exc).__name__, 'status': exc.status, 'message': str(exc),
                 'stage': getattr(exc, 'stage', None),
                 'pending_observation': getattr(exc, 'pending_observation', None)}
    elapsed = (time.perf_counter() - wall) * 1000.
    cpu_ms = (time.process_time() - cpu) * 1000.
    timings = dict(batch.timings_ms) if batch is not None else dict(model.timings)
    timings.update(outer_wall=elapsed, process_cpu=cpu_ms)
    return posterior, batch, timings, error


def describe_result(belief, batch):
    belief.validate()
    branch_error = 0.; successors = 0
    for p in batch.predictions:
        if p.observation_branches:
            branch_error = max(branch_error, abs(sum(b.probability for b in p.observation_branches) - 1.))
        for br in p.observation_branches:
            br.successor.validate(); successors += 1
        if abs(float(p.duration_weights.sum()) - 1.) > 1e-8:
            raise AssertionError('duration weights lost mass')
        dt = p.evaluated_interval[1] - p.evaluated_interval[0]
        if dt > 3. + 1e-8 or abs(dt / .048 - round(dt / .048)) > 1e-6:
            raise AssertionError('native endpoint/horizon changed')
    if branch_error > 1e-8:
        raise AssertionError('observation branch mass not conserved')
    return {'tracked_slots_max': max((h.n for h in belief.hypotheses), default=0),
            'hypotheses': len(belief.hypotheses), 'unresolved_mass': float(belief.unresolved_mass),
            'retained_mass': float(sum(h.weight for h in belief.hypotheses)),
            'worlds': len(batch.shared_future.weights),
            'sampled_people_max': batch.shared_future.positions.shape[2],
            'successors': successors, 'maximum_branch_mass_error': branch_error,
            'branches_per_refined': [len(p.observation_branches) for p in batch.predictions[:4]],
            'complete_queries': sum(bool(p.query_complete) for p in batch.predictions),
            'native_evaluated_endpoints': [int(round((p.evaluated_interval[1]-p.evaluated_interval[0])/.048))
                                         for p in batch.predictions],
            'graph_namespace': belief.graph.namespace,
            'graph_depth': belief.graph.depth}


def summarize(records):
    if not records:
        return {'calls': 0}
    keys = sorted({k for r in records for k in r['timings_ms']})
    return {'calls': len(records), 'statuses': dict(collections.Counter(r['status'] for r in records)),
            'roots': len({r.get('root_id') for r in records if r.get('root_id')}),
            'scenes': len({r.get('scene_id') for r in records if r.get('scene_id')}),
            'timings_ms': {k: _percentiles([float(r['timings_ms'][k]) for r in records if k in r['timings_ms']])
                           for k in keys}}


def grouped_summary(records, keys):
    groups = collections.defaultdict(list)
    for r in records:
        groups['/'.join(str(r.get(k, 'NA')) for k in keys)].append(r)
    return {key: summarize(rows) for key, rows in sorted(groups.items())}


def measure_case(models, previous, actual, queries, description, args, records, parity_records, log, case_index):
    fixed_fingerprint = previous.fingerprint()
    methods = list(args.methods)
    methods = methods[case_index % len(methods):] + methods[:case_index % len(methods)]
    for batch_size in args.batches:
        workload = (queries * math.ceil(batch_size / len(queries)))[:batch_size]
        representatives = {}
        for method_index, name in enumerate(methods):
            model = models[name]; incoming, migration = convert_incoming(previous, model, actual, models['old_reference'])
            incoming_fingerprint = incoming.fingerprint()
            modes = CACHE_MODES[(case_index + method_index) % 3:] + CACHE_MODES[:(case_index + method_index) % 3]
            for mode in modes:
                # Warm and revalidation blocks have an explicit untimed prime.
                # Revalidation retains exactly the warmed pose footprints.
                if mode != 'fully_cold':
                    p, b, _, error = invoke(model, incoming, actual, workload, args.seed)
                    if error:
                        raise RuntimeError('unbounded preparation failed: ' + repr(error))
                    del p, b
                repeat_count = args.warm_repeats if mode == 'warm' else args.cold_repeats
                if description['kind'] == 'synthetic' and mode == 'warm':
                    repeat_count = args.synthetic_warm_repeats
                for repetition in range(repeat_count):
                    if mode != 'warm':
                        clear_caches(model, mode)
                    before = cache_state(model)
                    p, batch, timing, error = invoke(model, incoming, actual, workload, args.seed)
                    row = {**description, 'method': name, 'cache_mode': mode, 'batch_size': batch_size,
                           'repetition': repetition, 'seed': args.seed, 'timings_ms': timing,
                           'status': error['status'] if error else 'COMPLETE', 'error': error,
                           'incoming_baseline_fingerprint': fixed_fingerprint,
                           'already_seen_receipt_translated': migration,
                           'transition_noise_contract': incoming.evidence.get('transition_noise_contract'),
                           'distinct_queries': len({q['query_id'] for q in workload}),
                           'cache_before': before, 'cache_after': cache_state(model)}
                    if error is None:
                        row.update(describe_result(p, batch))
                        if name.startswith('structural_bound') and name not in representatives:
                            representatives[name] = (p, batch)
                        if description['kind'] == 'synthetic':
                            checks = validate_fixture(p, batch, actual, description['synthetic_tracked'])
                            row['synthetic_checks'] = {k: v for k, v in checks.items() if k != 'observation_evidence'}
                    if incoming.fingerprint() != incoming_fingerprint or previous.fingerprint() != fixed_fingerprint:
                        raise AssertionError('timed operation mutated a shared incoming belief')
                    records.append(row)
                    log.write(json.dumps(row, default=json_default, allow_nan=False) + '\n'); log.flush()
                    del p, batch
        if all(name in representatives for name in ('structural_bound', 'structural_bound_unoptimized')):
            parity = {**description, 'batch_size': batch_size,
                      **compare_same_law(representatives['structural_bound'], representatives['structural_bound_unoptimized'])}
            parity_records.append(parity)
            log.write(json.dumps({'same_law_parity': parity}, default=json_default, allow_nan=False) + '\n'); log.flush()
            if not parity['passed']:
                raise AssertionError('optimized/unoptimized posterior/outcome parity failed; see preserved record')
        del representatives


def repeated_planner(models, cases, args, log):
    result = []
    if not args.rl_repeats or not args.rl_roots:
        return result
    ranked = sorted(cases, key=lambda c: (max((h.n for h in c[0].hypotheses), default=0), c[3]['graph_cells']))
    picks = []
    if ranked:
        for case in [ranked[-1], max(ranked, key=lambda c: c[3]['graph_cells']), ranked[len(ranked)//2]] + ranked:
            if case[3]['root_id'] not in {c[3]['root_id'] for c in picks}:
                picks.append(case)
            if len(picks) >= args.rl_roots:
                break
    for previous, actual, queries, description in picks:
        workload = (queries * math.ceil(24 / len(queries)))[:24]
        for name in args.methods:
            model = models[name]; incoming, _ = convert_incoming(previous, model, actual, models['old_reference'])
            belief, batch, _, error = invoke(model, incoming, actual, workload, args.seed)
            if error:
                raise RuntimeError('planner-loop preparation: ' + repr(error))
            del batch
            fingerprint = belief.fingerprint(); samples = []
            memory = [{'call': 0, 'rss_kib': _rss_kib(), **cache_state(model)}]
            for repetition in range(args.rl_repeats):
                p, batch, timing, error = invoke(model, belief, actual, workload, args.seed + repetition)
                unchanged = p is not None and p.fingerprint() == fingerprint
                row = {**description, 'method': name, 'cache_mode': 'repeated_planner', 'batch_size': 24,
                       'repetition': repetition, 'timings_ms': timing,
                       'status': error['status'] if error else 'COMPLETE', 'error': error,
                       'belief_unchanged': unchanged}
                if not unchanged:
                    raise AssertionError('repeated identical physical evidence changed belief')
                samples.append(row); del p, batch
                if (repetition + 1) % 10 == 0 or repetition + 1 == args.rl_repeats:
                    memory.append({'call': repetition + 1, 'rss_kib': _rss_kib(), **cache_state(model)})
            record = {'root_id': description['root_id'], 'method': name, 'records': samples,
                      'summary': summarize(samples), 'memory_samples': memory,
                      'rss_change_kib': memory[-1]['rss_kib'] - memory[0]['rss_kib'],
                      'semantics': 'same arrived belief/evidence; varied shared-world seed; no physical-time rollout'}
            result.append(record); log.write(json.dumps({'repeated_planner': record}, default=json_default) + '\n'); log.flush()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--roots', type=int)
    parser.add_argument('--warm-repeats', type=int)
    parser.add_argument('--cold-repeats', type=int, default=1)
    parser.add_argument('--synthetic-warm-repeats', type=int)
    parser.add_argument('--rl-repeats', type=int)
    parser.add_argument('--rl-roots', type=int, default=3)
    parser.add_argument('--cohort', help='dev-only metadata root list; default preserves old 60-root cohort')
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 4, 8, 16, 24])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--skip-real', action='store_true')
    parser.add_argument('--skip-synthetic', action='store_true')
    parser.add_argument('--seed', type=int, default=27)
    args = parser.parse_args()
    args.roots = args.roots if args.roots is not None else (2 if args.quick else 60)
    args.warm_repeats = args.warm_repeats if args.warm_repeats is not None else (2 if args.quick else 10)
    args.synthetic_warm_repeats = args.synthetic_warm_repeats if args.synthetic_warm_repeats is not None else (1 if args.quick else 5)
    args.rl_repeats = args.rl_repeats if args.rl_repeats is not None else (5 if args.quick else 100)
    if (args.roots < 0 or min(args.warm_repeats, args.cold_repeats, args.synthetic_warm_repeats) < 1
            or min(args.rl_repeats, args.rl_roots) < 0 or not args.methods
            or not args.batches or any(q not in (1, 4, 8, 16, 24) for q in args.batches)
            or len(set(args.methods)) != len(args.methods) or len(set(args.batches)) != len(args.batches)):
        raise ValueError('invalid benchmark settings')
    output = output_path(args.output)
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    provenance = audit_preserved_baseline()
    before = _metadata(); collector_before = collector_metadata(); source_before = _sources()
    script_sha = _sha(__file__)
    if source_before != EARLY_SOURCE_HASHES or script_sha != EARLY_SCRIPT_SHA256:
        raise RuntimeError('source changed during process startup; freeze before benchmarking')
    records = []; cases = []; initialization = []; repeated = []; parity_records = []; failure = None
    started = time.perf_counter()
    models = {}; model_load_ms = jit_ms = None; pools = []
    try:
        with threadpool_limits(limits=1), Path(str(output) + '.jsonl').open('x') as log:
            pools = threadpool_info()
            models, model_load_ms = make_models()
            jit = time.perf_counter(); warmup_kernels()
            # NumPy advanced indexing produces A-layout world tensors whereas
            # simple fixtures are C-layout. Compile both normal float64 paths,
            # including broadcast clocks, before any timed full invocation.
            p = np.zeros((16, 4, 2, 2)); r = np.zeros((4, 4, 4))
            event_features_many_fast(p, p, r, np.zeros(16), np.arange(4) * .048,
                                     np.broadcast_to(np.arange(4) * .048, (4, 4)))
            event_features_many_fast(p[:, ::2], p[:, ::2], r[:, :2].copy(), np.zeros(16),
                                     np.arange(2) * .048, np.broadcast_to(np.arange(2) * .048, (4, 2)))
            jit_ms = (time.perf_counter() - jit) * 1000.
            # Compile/disk-cache load is excluded and recorded; every measured
            # method uses the same initialized Python/Torch/Numba process.
            rows = [] if args.skip_real else selected_rows(args.roots, args.cohort)
            for index, row in enumerate(rows):
                checksums = verify_legal_files(row)
                previous, actual, _, queries, init_ms, history_length = _causal_input(
                    models['old_reference'], SNAPSHOT, row)
                description = {'kind': 'real_dev', 'root_id': row['root_id'], 'scene_id': row['scene_id'],
                    'graph_cells': len(previous.graph.materialize()), 'incoming_graph_depth': previous.graph.depth,
                    'incoming_tracked_max': max((h.n for h in previous.hypotheses), default=0),
                    'incoming_unresolved_mass': previous.unresolved_mass,
                    'legal_visible_detections': sum(bool(t.get('visible_now') and t.get('valid', True)) for t in actual.get('tracks', [])),
                    'incoming_history_length': history_length, 'elapsed_observation_time': actual['timestamp']-previous.timestamp,
                    'legal_payload_sha256': checksums}
                cases.append((previous, actual, queries, description))
                initialization.append({'root_id': row['root_id'], 'baseline_prefix_initialization_ms': init_ms})
                measure_case(models, previous, actual, queries, description, args, records, parity_records, log, index)
                print(json.dumps({'real_roots_finished': index + 1, 'real_roots': len(rows),
                                  'calls_finished': len(records)}), flush=True)
            repeated = repeated_planner(models, cases, args, log)
            if not args.skip_synthetic:
                for index, (n, certificate) in enumerate(((6, False), (12, False), (6, True), (12, True))):
                    previous, actual, queries = preserved_fixture(models['old_reference'], n, certificate)
                    description = {'kind': 'synthetic', 'root_id': None, 'scene_id': None,
                        'synthetic_tracked': n, 'explicit_certificate': certificate,
                        'graph_cells': len(previous.graph.materialize()), 'incoming_graph_depth': previous.graph.depth,
                        'incoming_tracked_max': n, 'incoming_unresolved_mass': previous.unresolved_mass,
                        'legal_visible_detections': len(actual['tracks']), 'elapsed_observation_time': .048,
                        'fixture_source': 'unchanged d697d966 scripts/l2_partial_repair_systems_stress.py:fixture'}
                    measure_case(models, previous, actual, queries, description, args, records, parity_records, log, index)
                    print(json.dumps({'synthetic_cases_finished': index + 1, 'calls_finished': len(records)}), flush=True)
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    source_after = _sources(); collector_after = collector_metadata()
    changed = source_before != source_after or script_sha != _sha(__file__)
    result = {'schema': 'partial_graph_structural_paired_benchmark_1', 'settings': vars(args),
              'baseline_provenance': provenance, 'sources_before': source_before, 'sources_after': source_after,
              'source_changed_during_measurement': changed, 'script_sha256': script_sha,
              'model_versions': {k: m.model_version for k, m in models.items()},
              'collector_before': collector_before, 'collector_after': collector_after,
              'metadata_before': before, 'metadata_after': _metadata(),
              'resource_policy': {'device': 'cpu', 'torch_threads': torch.get_num_threads(),
                  'torch_interop_threads': torch.get_num_interop_threads(), 'threadpools': pools,
                  'nice': 19, 'io_class': 'idle', 'research_job_threads': 1,
                  'coexistence_claim': 'status/throughput observational only; no causal collector-only control'},
              'model_load_ms': model_load_ms, 'jit_or_disk_cache_startup_ms': jit_ms,
              'initialization': initialization, 'records': records, 'repeated_planner': repeated,
              'same_law_parity': parity_records,
              'all_same_law_comparisons_passed': all(p['passed'] for p in parity_records) if parity_records else None,
              'summary': grouped_summary(records, ('kind', 'method', 'batch_size', 'cache_mode')),
              'density_summary': grouped_summary(records, ('kind', 'method', 'batch_size', 'cache_mode', 'incoming_tracked_max')),
              'synthetic_summary': grouped_summary([r for r in records if r['kind']=='synthetic'],
                  ('method', 'synthetic_tracked', 'explicit_certificate', 'batch_size', 'cache_mode')),
              'failure': failure, 'elapsed_seconds': time.perf_counter() - started,
              'workload': {'physical_horizon_s': 3., 'worlds': 16, 'refined_queries': 'min(4,Q)',
                  'candidate_batches': args.batches, 'preserved_old_comparison_batch': 24,
                  'incoming_beliefs': 'one preserved baseline legal prefix; isolated copies with model tag migration',
                  'old_reference': 'different old residual/successor semantics; never claimed a pure kernel ablation',
                  'optimized_ablation': 'structural_bound versus structural_bound_unoptimized; identical probability semantics',
                  'fully_cold': 'clear support results, pose footprints, endpoint templates and Sobol buffers; initialized model/JIT',
                  'graph_revalidation': 'retain footprints; clear support results; actual legal graph delta still processed',
                  'warm': 'same fixed causal incoming belief and new observation; caches primed',
                  'timed_interval': 'complete public online_call including necessary validation, fingerprints, copies and successors',
                  'timing_excludes': 'data reading, prefix initialization, checkpoint/JIT loading, result validation, report writing',
                  'branch_reduction': 'none imposed by benchmark'},
              'limitations': ['No privileged truth, native outcomes, future observations or fresh/holdout payload is opened.',
                  'Paired callers share frozen incoming beliefs, not independently learned histories.',
                  'Kernel speedup must compare both structural variants; old-model comparison also changes semantics.',
                  'Repeated candidate rows preserve the old computational workload, not route diversity.',
                  'Report unresolved mass and materialized hypotheses alongside latency; vacuous outputs are not success.',
                  'Warm/cold empirical P95/P99 use correlated repeated calls; quick runs cannot establish tail guarantees.',
                  'Repeated planner loops hold physical time fixed; they are not long physical observation sequences.',
                  'Cold means initialized inference with empty dynamic caches, not checkpoint or JIT startup.',
                  'Any runtime/source change during the run invalidates release timing; partial evidence is preserved.']}
    # Convert NumPy scalars before atomic no-overwrite publication.
    result = json.loads(json.dumps(result, default=json_default, allow_nan=False))
    _publish(output, result)
    print(json.dumps({'output': str(output), 'calls': len(records), 'failure': failure,
                      'source_changed': changed, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    if failure or changed:
        raise RuntimeError('benchmark incomplete or source changed; preserved report is diagnostic only')


if __name__ == '__main__':
    main()
