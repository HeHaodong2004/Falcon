"""Bounded validation prototype; no runtime package edits or dataset access.

``validate_batched`` preserves the reference checks and adds hidden-covariance
PSD validation by default. Set ``hidden_psd=False`` for the historical validator
contract. The executable probe compares both contracts on valid/invalid fixed
fixtures, then times every component of the unchanged 288-component synthetic
certificate posterior. No hypothesis, population alternative, or check is
sampled or dropped. This module does not install or monkeypatch the prototype.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

import numpy as np


def validate_batched(belief, *, hidden_psd=True):
    """Validate all existing invariants using packed numerical checks.

    This returns the original belief and mutates nothing. Ragged arrays are
    packed only inside this call: there is no stale validity certificate or
    reliance on the caller keeping mutable arrays unchanged. Slot uniqueness
    retains the exact reference set check; covariance eigenvalues are batched.
    As in the reference, eigvalsh uses the lower triangle and threshold -1e-7;
    this is a PSD check, not a new symmetry requirement.
    """
    hs = belief.hypotheses
    weights = np.array([h.weight for h in hs])
    assert np.isfinite(weights).all() and np.all(weights >= 0)
    assert 0 <= belief.unresolved_mass <= 1
    assert abs(weights.sum() + belief.unresolved_mass - 1) < 1e-7

    means, covariances, seen = [], [], []
    hidden_means, hidden_covariances = [], []
    pmf_groups = {}
    for h in hs:
        assert h.mean.shape == (h.n, 4)
        assert h.covariance.shape == (h.n, 4, 4)
        assert len(set(h.slot_ids.tolist())) == h.n
        means.append(h.mean)
        covariances.append(h.covariance)
        seen.append(np.asarray(h.last_seen).reshape(-1))
        # Group equal shapes so each per-row sum has the same reduction shape
        # as the original h.hidden_pmf.sum(), without per-hypothesis dispatch.
        pmf_groups.setdefault((h.hidden_pmf.shape, h.hidden_pmf.dtype.str), []).append(h)
        if h.hidden_mean is not None:
            assert h.hidden_mean.ndim == 2 and h.hidden_mean.shape[1] == 4
            assert h.hidden_covariance.shape == (len(h.hidden_mean), 4, 4)
            if h.hidden_exchangeable:
                assert len(h.hidden_mean) == 1
            hidden_means.append(h.hidden_mean)
            hidden_covariances.append(h.hidden_covariance)

    for _, group in pmf_groups.items():
        pmfs = np.stack([h.hidden_pmf for h in group])
        assert np.all(pmfs >= 0)
        axes = tuple(range(1, pmfs.ndim))
        sums = pmfs.sum(axis=axes) if axes else pmfs
        assert np.all(np.abs(sums - 1) < 1e-8)
        private = np.array([h.hidden_mean is not None and not h.hidden_exchangeable
                            for h in group], dtype=bool)
        if np.any(private):
            rows = pmfs[private].reshape(int(private.sum()), -1)
            assert np.all(np.count_nonzero(rows > 1e-10, axis=1) == 1)
            counts = np.array([len(h.hidden_mean) for h, take in zip(group, private) if take])
            assert np.array_equal(counts, np.argmax(rows, axis=1))

    if hs:
        assert np.isfinite(np.concatenate(means, axis=0)).all()
        tracked = np.concatenate(covariances, axis=0)
        assert np.isfinite(tracked).all()
        assert np.all(np.concatenate(seen) <= belief.timestamp + 1e-7)
    else:
        tracked = np.empty((0, 4, 4))
    if hidden_means:
        assert np.isfinite(np.concatenate(hidden_means, axis=0)).all()
        hidden = np.concatenate(hidden_covariances, axis=0)
        assert np.isfinite(hidden).all()
    else:
        hidden = np.empty((0, 4, 4))
    # Preserve each source dtype at numerical tolerance boundaries. A mixed
    # float32/float64 stack would otherwise change the reference eigensolver's
    # precision; similarly, PMF groups above preserve sum precision.
    covariance_groups = {}
    for value in covariances + (hidden_covariances if hidden_psd else []):
        if len(value):
            covariance_groups.setdefault(value.dtype.str, []).append(value)
    for group in covariance_groups.values():
        assert np.linalg.eigvalsh(np.concatenate(group, axis=0)).min() > -1e-7

    # Preserve the entire StructuralBeliefState carrier contract as well. It
    # normally has only 16 components, so keep exact array equality checks.
    c = getattr(belief, 'carrier', None)
    if c is not None:
        if (abs(belief.timestamp - c.times[c.index]) > 1e-8
                or abs(belief.unresolved_mass - c.unresolved_mass) > 1e-10
                or len(hs) != len(c.weights)):
            raise ValueError('belief/carrier clock, count or mass mismatch')
        for w, h in enumerate(hs):
            if abs(h.weight - c.weights[w]) > 1e-10 or not np.array_equal(h.slot_ids, c.worlds[w].slot_ids):
                raise ValueError('belief/carrier population or weight mismatch')
            if (not np.array_equal(h.mean[:, :2], c.positions[w, c.index, :h.n])
                    or not np.array_equal(h.mean[:, 2:], c.velocities[w, c.index, :h.n])):
                raise ValueError('belief/carrier physical endpoint mismatch')
    return belief


def reference_hidden_psd(belief):
    """The full existing validator, plus the explicitly stronger hidden check."""
    belief.validate()
    for h in belief.hypotheses:
        if h.hidden_mean is not None and len(h.hidden_mean):
            assert np.linalg.eigvalsh(h.hidden_covariance).min() > -1e-7
    return belief


def clone(belief):
    hs = [h.fork(**{key: value.copy() for key, value in vars(h).items()
                    if isinstance(value, np.ndarray)}) for h in belief.hypotheses]
    return belief.fork(hypotheses=hs, robot_pose=belief.robot_pose.copy(),
                       evidence=copy.deepcopy(belief.evidence))


def base_fixture():
    from eventwm_graph.l2_partial.state import BeliefState, Hypothesis, ObservedGraph
    graph = ObservedGraph('validation-fixture', 1, .25, 0., {})
    h = Hypothesis(1., np.array([[0., 1., .1, .2], [2., 3., -.1, 0.]]),
        np.tile(np.diag([.2, .1, .3, .4]), (2, 1, 1)), np.zeros(2),
        np.arange(2), np.array([.25, .5, .25]),
        hidden_mean=np.zeros((1, 4)), hidden_covariance=np.eye(4)[None])
    return BeliefState(graph, [h], 1., np.zeros(4))


def carrier_fixture():
    from eventwm_graph.l2_partial.structural_state import StructuralBeliefState, WorldCarrier
    b = base_fixture()
    hs = [b.hypotheses[0].fork(weight=.5, hidden_pmf=np.array([0., 1.])) for _ in range(2)]
    positions = np.zeros((2, 2, 3, 2))
    velocities = np.zeros_like(positions)
    for w, h in enumerate(hs):
        positions[w, :, :h.n] = h.mean[:, :2]
        velocities[w, :, :h.n] = h.mean[:, 2:]
    c = WorldCarrier(np.array([1., 1.048]), positions, velocities, tuple(hs),
                     np.array([.5, .5]), 0., 73, 'validation-fixture-origin', 0)
    return StructuralBeliefState.wrap(b.fork(hypotheses=hs), carrier=c)


def fixed_cases():
    """Deterministic valid/invalid edge cases; no simulator or dataset input."""
    base = base_fixture()
    cases = []

    def add(name, mutate=None, *, legacy=True, strengthened=None, source=None):
        b = clone(base if source is None else source)
        if mutate is not None:
            mutate(b)
        cases.append((name, b, legacy, legacy if strengthened is None else strengthened))

    def attr(name, value):
        return lambda b: setattr(b.hypotheses[0], name, value)

    add('ordinary_valid')
    add('zero_weight_hypothesis_valid', lambda b: (setattr(b, 'unresolved_mass', 1.), setattr(b.hypotheses[0], 'weight', 0.)))
    add('all_unresolved_empty', lambda b: (setattr(b, 'hypotheses', []), setattr(b, 'unresolved_mass', 1.)))
    add('singular_covariance', attr('covariance', np.zeros((2, 4, 4))))
    tiny = np.zeros((2, 4, 4)); tiny[:, 0, 0] = -5e-8
    add('allowed_negative_roundoff', attr('covariance', tiny))
    add('hidden_zero_count_private', lambda b: (
        setattr(b.hypotheses[0], 'hidden_pmf', np.array([1.])),
        setattr(b.hypotheses[0], 'hidden_exchangeable', False),
        setattr(b.hypotheses[0], 'hidden_mean', np.empty((0, 4))),
        setattr(b.hypotheses[0], 'hidden_covariance', np.empty((0, 4, 4)))))
    ragged = clone(base)
    h = ragged.hypotheses[0]
    ragged.hypotheses = [h.fork(weight=.4), h.fork(weight=.6, mean=np.empty((0, 4)),
        covariance=np.empty((0, 4, 4)), last_seen=np.empty(0), slot_ids=np.empty(0, int),
        hidden_pmf=np.array([1.]), hidden_mean=None, hidden_covariance=None)]
    add('ragged_zero_and_two_tracked', source=ragged)
    mixed = clone(base)
    h = mixed.hypotheses[0]
    mixed.hypotheses = [h.fork(weight=.5, hidden_pmf=np.array([.1, .1, .8], dtype=np.float32)),
                        h.fork(weight=.5)]
    add('mixed_pmf_dtype_preserves_sum_precision', source=mixed)
    mixed = clone(base)
    h = mixed.hypotheses[0]
    mixed.hypotheses = [h.fork(weight=.5, covariance=h.covariance.astype(np.float32),
                              hidden_covariance=h.hidden_covariance.astype(np.float32)),
                        h.fork(weight=.5)]
    add('mixed_covariance_dtype', source=mixed)
    add('negative_weight', attr('weight', -1.), legacy=False)
    add('nan_weight', attr('weight', float('nan')), legacy=False)
    add('infinite_weight', attr('weight', float('inf')), legacy=False)
    add('invalid_total_mass', attr('weight', .8), legacy=False)
    add('negative_unresolved', lambda b: setattr(b, 'unresolved_mass', -.1), legacy=False)
    add('nan_unresolved', lambda b: setattr(b, 'unresolved_mass', float('nan')), legacy=False)
    add('unresolved_above_one', lambda b: setattr(b, 'unresolved_mass', 1.1), legacy=False)
    add('wrong_mean_shape', attr('mean', np.zeros((2, 3))), legacy=False)
    add('wrong_covariance_shape', attr('covariance', np.zeros((2, 3, 3))), legacy=False)
    add('duplicate_slot_ids', attr('slot_ids', np.array([4, 4])), legacy=False)
    add('wrong_slot_count', attr('slot_ids', np.array([4])), legacy=False)
    nan_mean = base.hypotheses[0].mean.copy(); nan_mean[0, 0] = np.nan
    add('nan_mean', attr('mean', nan_mean), legacy=False)
    inf_cov = base.hypotheses[0].covariance.copy(); inf_cov[0, 0, 0] = np.inf
    add('infinite_covariance', attr('covariance', inf_cov), legacy=False)
    add('negative_hidden_pmf', attr('hidden_pmf', np.array([1.1, -.1])), legacy=False)
    add('nonnormalized_hidden_pmf', attr('hidden_pmf', np.array([.4, .4])), legacy=False)
    add('nan_hidden_pmf', attr('hidden_pmf', np.array([np.nan, 1.])), legacy=False)
    add('future_last_seen', attr('last_seen', np.array([0., 1.01])), legacy=False)
    add('nan_last_seen', attr('last_seen', np.array([0., np.nan])), legacy=False)
    neg_cov = base.hypotheses[0].covariance.copy(); neg_cov[1, 2, 2] = -.01
    add('negative_tracked_psd', attr('covariance', neg_cov), legacy=False)
    add('wrong_hidden_mean_shape', attr('hidden_mean', np.zeros((1, 3))), legacy=False)
    add('wrong_hidden_covariance_shape', attr('hidden_covariance', np.zeros((1, 3, 3))), legacy=False)
    add('nan_hidden_mean', attr('hidden_mean', np.full((1, 4), np.nan)), legacy=False)
    add('nan_hidden_covariance', attr('hidden_covariance', np.full((1, 4, 4), np.nan)), legacy=False)
    add('exchangeable_needs_one_template', lambda b: (
        setattr(b.hypotheses[0], 'hidden_mean', np.zeros((2, 4))),
        setattr(b.hypotheses[0], 'hidden_covariance', np.tile(np.eye(4), (2, 1, 1)))), legacy=False)
    add('private_needs_fixed_count', attr('hidden_exchangeable', False), legacy=False)
    add('private_count_mismatch', lambda b: (
        setattr(b.hypotheses[0], 'hidden_exchangeable', False),
        setattr(b.hypotheses[0], 'hidden_pmf', np.array([0., 0., 1.]))), legacy=False)
    neg_hidden = np.eye(4)[None]; neg_hidden[0, 3, 3] = -.01
    add('negative_hidden_psd_new_guard', attr('hidden_covariance', neg_hidden), legacy=True, strengthened=False)
    carrier = carrier_fixture()
    add('finite_carrier_valid', source=carrier)
    add('carrier_timestamp_mismatch', lambda b: setattr(b, 'timestamp', 1.01), source=carrier, legacy=False)
    add('carrier_weight_mismatch', lambda b: (
        setattr(b.hypotheses[0], 'weight', .6), setattr(b.hypotheses[1], 'weight', .4)), source=carrier, legacy=False)
    add('carrier_slot_mismatch', lambda b: setattr(b.hypotheses[0], 'slot_ids', np.array([4, 5])), source=carrier, legacy=False)
    add('carrier_position_mismatch', lambda b: b.hypotheses[0].mean.__setitem__((0, 0), 12.), source=carrier, legacy=False)
    add('carrier_velocity_mismatch', lambda b: b.hypotheses[0].mean.__setitem__((0, 2), 12.), source=carrier, legacy=False)
    return cases


def verdict(fn, b):
    before = b.fingerprint()
    try:
        if fn(b) is not b:
            raise RuntimeError('validator did not return its input')
        accepted, error = True, None
    except (AssertionError, ValueError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
        accepted, error = False, type(exc).__name__
    if before != b.fingerprint():
        raise AssertionError('validator modified the input')
    return {'accepted': accepted, 'exception': error}


def quantiles(values):
    q = np.percentile(values, [50, 95, 99])
    return dict(zip(('p50', 'p95', 'p99'), map(float, q)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=31)
    parser.add_argument('--seconds', type=float, default=45.)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 51:
        raise ValueError('bounded probe requires 5..51 repeats')
    from l2_partial_structural_benchmark import (
        _sources, make_models, audit_preserved_baseline, preserved_fixture,
        convert_incoming, collector_metadata, output_path, json_default)
    import torch
    from threadpoolctl import threadpool_limits
    output = output_path(args.output)
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    source = _sources()
    before = collector_metadata()
    provenance = audit_preserved_baseline()
    cases, timings = [], []
    failure = None
    start = time.perf_counter()

    def expired(signum, frame):
        raise TimeoutError('bounded validation prototype deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    try:
        with threadpool_limits(limits=1):
            for name, b, expected_legacy, expected_strengthened in fixed_cases():
                result = {'name': name,
                    'expected_legacy': expected_legacy, 'expected_strengthened': expected_strengthened,
                    'original': verdict(lambda x: x.validate(), b),
                    'batched_original_contract': verdict(lambda x: validate_batched(x, hidden_psd=False), b),
                    'strengthened_reference': verdict(reference_hidden_psd, b),
                    'batched_strengthened': verdict(validate_batched, b)}
                if (result['original']['accepted'] != expected_legacy
                        or result['batched_original_contract']['accepted'] != expected_legacy
                        or result['strengthened_reference']['accepted'] != expected_strengthened
                        or result['batched_strengthened']['accepted'] != expected_strengthened):
                    raise AssertionError('fixed-case parity failure: ' + repr(result))
                cases.append(result)
            models, load_ms = make_models()
            for n in (6, 12):
                original, row, _ = preserved_fixture(models['old_reference'], n, True)
                incoming, _ = convert_incoming(original, models['structural_bound'], row,
                                                models['old_reference'])
                b = models['structural_bound'].observe(incoming, row, {})
                assert len(b.hypotheses) == 288
                support = sorted({h.n + int(k) for h in b.hypotheses
                                  for k in np.flatnonzero(h.hidden_pmf > 0)})
                assert support == [n, n + 2, n + 4]
                fingerprint = b.fingerprint()
                variants = {'original': lambda: b.validate(),
                    'strengthened_reference': lambda: reference_hidden_psd(b),
                    'batched_original_contract': lambda: validate_batched(b, hidden_psd=False),
                    'batched_strengthened': lambda: validate_batched(b)}
                for fn in variants.values():
                    fn()
                raw = {key: [] for key in variants}
                keys = list(variants)
                for repeat in range(args.repeats):
                    for key in keys[repeat % len(keys):] + keys[:repeat % len(keys)]:
                        wall = time.perf_counter(); cpu = time.process_time()
                        variants[key]()
                        raw[key].append({'wall_ms': (time.perf_counter() - wall) * 1000,
                                         'cpu_ms': (time.process_time() - cpu) * 1000})
                assert b.fingerprint() == fingerprint and len(b.hypotheses) == 288
                timings.append({'synthetic_tracked': n, 'explicit_certificate': True,
                    'components': len(b.hypotheses), 'population_support': support,
                    'unresolved_mass': b.unresolved_mass, 'unchanged_fingerprint': fingerprint,
                    'summaries': {key: {'wall_ms': quantiles([r['wall_ms'] for r in rows]),
                        'cpu_ms': quantiles([r['cpu_ms'] for r in rows])} for key, rows in raw.items()},
                    'raw_timings': raw})
    except Exception as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
    after_source = _sources()
    report = {'schema': 'structural_validation_prototype_1',
        'claim': 'synthetic validator microbenchmark only; not full-path latency or predictive validation',
        'source_sha256_before': source, 'source_sha256_after': after_source,
        'sources_unchanged': source == after_source,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'stronger_guard': 'hidden covariance PSD is additional to the original validator; compatibility tested separately',
        'provenance': provenance, 'collector_before': before, 'collector_after': collector_metadata(),
        'repeats': args.repeats, 'fixed_cases': cases, 'synthetic_timings': timings,
        'failure': failure, 'elapsed_seconds': time.perf_counter() - start}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False, default=json_default)
    print(json.dumps({'output': str(output), 'fixed_cases': len(cases), 'failure': failure,
                      'sources_unchanged': report['sources_unchanged'],
                      'timings': [{k: v for k, v in r.items() if k != 'raw_timings'} for r in timings]}))
    if failure:
        raise RuntimeError(failure)
    if not report['sources_unchanged']:
        raise RuntimeError('runtime sources changed during the prototype probe')


if __name__ == '__main__':
    main()
