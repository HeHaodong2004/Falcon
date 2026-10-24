"""Operator parity tests: reference code remains an independent implementation."""
import numpy as np
import unittest
import itertools

from eventwm_graph.l2_partial.hidden import hypothesis_emissions, missed_hidden_moments, prepare_prior
from eventwm_graph.l2_partial.observation import EndpointObservationLaw
from eventwm_graph.l2_partial.queries import event_features_many
from eventwm_graph.l2_partial.repair_filter import condition_all_empty
from eventwm_graph.l2_partial.state import Hypothesis
from eventwm_graph.l2_partial.structural_kernels import (
    ObservationWorkspace, clear_kernel_caches, condition_all_empty_fast,
    event_features_many_fast, kernel_cache_info, missed_hidden_moments_fast,
    _sobol_normals,
)


def parametrize(name, values):
    def decorate(function):
        parameters = getattr(function, '_parameters', [])
        function._parameters = parameters + [(name, values)]
        return function
    return decorate


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if not name.startswith('test_') or not callable(function):
            continue
        parameters = getattr(function, '_parameters', [])
        for values in itertools.product(*(p[1] for p in parameters)):
            kwargs = dict(zip([p[0] for p in parameters], values))
            def run(function=function, kwargs=kwargs):
                function(**kwargs)
            label = name + (' ' + repr(kwargs) if kwargs else '')
            suite.addTest(unittest.FunctionTestCase(run, description=label))
    return suite


def hypothesis(n=3, seed=17, weight=.9, private=False, mixture=False, scale=1.):
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=(n, 4))
    a = rng.normal(size=(n, 4, 4)) * scale
    cov = a @ a.swapaxes(-1, -2) + np.eye(4)[None] * .01
    hm, hc = prepare_prior(np.array([.2, 0., -.3, .1]))
    pmf = np.array([.2, .3, .5])
    if private:
        hm = np.repeat(hm, 2, axis=0); hm[1] += [3., 2., .5, -.2]
        hc = np.repeat(hc, 2, axis=0); pmf = np.array([0., 0., 1.])
    h = Hypothesis(weight, mean, cov, np.zeros(n), np.arange(n), pmf,
                   hidden_mean=hm, hidden_covariance=hc, hidden_exchangeable=not private,
                   visibility_mode=seed % 2)
    if mixture:
        mm = np.repeat(mean[:, None], 2, axis=1); mm[:, 1, 2:] += [.4, -.8]
        cc = np.repeat(cov[:, None], 2, axis=1); cc[:, 1, 2:, 2:] += np.eye(2) * .5
        hmm = np.repeat(hm[:, None], 2, axis=1); hmm[:, 1, 2:] += [-1., .2]
        hcc = np.repeat(hc[:, None], 2, axis=1); hcc[:, 1, 2:, 2:] += np.eye(2)
        h = h.fork(maneuver_mean=mm, maneuver_covariance=cc, maneuver_probability=np.array([.8, .2]),
                   hidden_maneuver_mean=hmm, hidden_maneuver_covariance=hcc,
                   hidden_maneuver_probability=np.array([.8, .2]))
    return h


def same_nested(a, b, atol=2e-11):
    if isinstance(a, tuple):
        assert len(a) == len(b)
        for aa, bb in zip(a, b):
            same_nested(aa, bb, atol)
    else:
        np.testing.assert_allclose(a, b, atol=atol, rtol=2e-12)


@parametrize('private', [False, True])
@parametrize('boxes', [[], [([-1., -1.], [1., 1.])],
                                   [([-1., -1.], [1., 1.]), ([2., -2.], [3., .3])]])
def test_certificate_complete_weighted_posterior_parity(private, boxes):
    hs = [hypothesis(private=private, weight=.45), hypothesis(seed=8, private=private, weight=.45)]
    snapshots = [(h.mean.copy(), h.covariance.copy(), h.hidden_pmf.copy()) for h in hs]
    expected, eu, ed = condition_all_empty(hs, boxes, .1)
    actual, au, ad = condition_all_empty_fast(hs, boxes, .1, cdf_diagnostic=True)
    assert len(actual) == len(expected)
    assert abs(au - eu) < 1e-14
    assert abs(sum(h.weight for h in actual) + au - 1.) < 2e-14
    assert ad['empty_quadrature_points'] == ed['empty_quadrature_points']
    np.testing.assert_allclose(ad['empty_evidence_quadrature'], ed['empty_evidence_quadrature'], atol=1e-14)
    np.testing.assert_allclose(ad['empty_evidence_cdf'], ed['empty_evidence_cdf'], atol=2e-10)
    for a, e in zip(actual, expected):
        assert abs(a.weight - e.weight) < 2e-14
        for field in ('mean', 'covariance', 'hidden_mean', 'hidden_covariance', 'hidden_pmf'):
            np.testing.assert_allclose(getattr(a, field), getattr(e, field), atol=2e-11, rtol=1e-11)
        assert not a.hidden_exchangeable
        assert len(a.hidden_mean) == int(a.expected_hidden)
        xy = np.concatenate([a.mean[:, :2], a.hidden_mean[:, :2]])
        for lo, hi in boxes:
            assert not np.any(np.all((xy >= lo) & (xy <= hi), axis=1))
    for h, before in zip(hs, snapshots):
        same_nested((h.mean, h.covariance, h.hidden_pmf), before, atol=0.)


def test_certificate_diagnostic_off_and_unsupported_dirac():
    h = hypothesis(n=1).fork(mean=np.zeros((1, 4)), covariance=np.zeros((1, 4, 4)),
                             hidden_pmf=np.array([1.]))
    _, u, d = condition_all_empty_fast([h], [([-1., -1.], [1., 1.])], .1)
    assert u == 1.
    assert d['empty_evidence_cdf'] is None
    assert d['empty_evidence_absolute_error'] is None
    assert d['empty_cdf_diagnostic_status'] == 'NOT_COMPUTED'
    _, u, d = condition_all_empty_fast([h.fork(weight=1.)], [([-1., -1.], [1., 1.])], 0.)
    assert u == 1. and d['unsupported_empty']


def test_certificate_mixed_dirac_rank_one_and_no_people():
    h = hypothesis(n=3).fork(mean=np.array([[3., 4., 1., 2.], [2., 1., 2., 1.], [-2., -2., 0., 1.]]),
                            covariance=np.stack([np.diag([0., 0., 1., 1.]),
                                                 np.diag([1., 0., 1., 1.]), np.eye(4)]),
                            hidden_pmf=np.array([1.]))
    boxes = [([-.2, -.2], [.2, .2])]
    a, u, _ = condition_all_empty_fast([h], boxes, .1)
    e, eu, _ = condition_all_empty([h], boxes, .1)
    assert len(a) == len(e) and abs(u - eu) < 1e-14
    for aa, ee in zip(a, e):
        same_nested((aa.mean, aa.covariance), (ee.mean, ee.covariance))
    h = hypothesis(n=0).fork(hidden_pmf=np.array([1.]))
    a, u, _ = condition_all_empty_fast([h], boxes, .1)
    assert a[0].n == 0 and a[0].hidden_mean.shape == (0, 4)
    assert abs(a[0].weight + u - 1.) < 1e-14


def test_sobol_cache_is_immutable_bounded_and_reproducible():
    clear_kernel_caches()
    a = _sobol_normals(2, 256); b = _sobol_normals(2, 256)
    assert a is b and not a.flags.writeable
    retained = a.copy()
    for n in range(1, 25):
        _sobol_normals(n, 256)
    info = kernel_cache_info()
    assert info['sobol_entries'] <= info['sobol_max_entries']
    assert info['sobol_bytes'] <= info['sobol_max_bytes']
    clear_kernel_caches()
    np.testing.assert_array_equal(_sobol_normals(2, 256), retained)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'power of two'):
        condition_all_empty_fast([], [], 1., points=17)


def test_sobol_shared_cache_operations_are_serialized_across_callers():
    from concurrent.futures import ThreadPoolExecutor
    clear_kernel_caches()
    def caller(i):
        for j in range(6):
            if (i + j) % 4 == 0:
                clear_kernel_caches()
            value = _sobol_normals(1 + (i + j) % 5, 16)
            assert not value.flags.writeable and np.isfinite(value).all()
            info = kernel_cache_info()
            assert info['sobol_entries'] <= info['sobol_max_entries']
            assert info['sobol_bytes'] <= info['sobol_max_bytes']
        return True
    # At most one generator holds the lock; waiting callers consume no CPU.
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert all(pool.map(caller, range(3)))


@parametrize('order', [9, 15])
@parametrize('regime', [0, 1])
def test_gh_weighted_moments_preserve_full_rank_line_and_dirac(order, regime):
    h = hypothesis(n=8)
    h.covariance[0] = np.diag([0., 0., 1., 1.])
    h.covariance[1] = np.diag([1., 0., .3, .2])
    pose = np.array([1., 0., -2., .7]); law = EndpointObservationLaw()
    same_nested(missed_hidden_moments_fast(h.mean, h.covariance, pose, regime, law, order),
                missed_hidden_moments(h.mean, h.covariance, pose, regime, law, order))


class ConstantLaw(EndpointObservationLaw):
    def __init__(self, probability):
        self.probability = probability
    def detection(self, p, pose, regime):
        return np.full(np.asarray(p).shape[:-1], self.probability)


@parametrize('p', [0., .33, 1.])
def test_gh_constant_detection_and_zero_mass(p):
    h = hypothesis(n=4); pose = np.zeros(4); law = ConstantLaw(p)
    same_nested(missed_hidden_moments_fast(h.mean, h.covariance, pose, 1, law),
                missed_hidden_moments(h.mean, h.covariance, pose, 1, law))


@parametrize('detection_count', [0, 1, 5])
def test_workspace_matches_per_hypothesis_mixed_population_laws(detection_count):
    hs = [hypothesis(n=0, seed=6), hypothesis(n=1, scale=.03),
          hypothesis(n=4, private=True, mixture=True), hypothesis(n=2, seed=8, mixture=True)]
    # Identical hidden state with a different tracked population tests content
    # deduplication without equating different hypotheses or their weights.
    hs.append(hs[2].fork(weight=.04, mean=hs[2].mean + .1))
    pose = np.array([.2, 0., -1., .3]); law = EndpointObservationLaw()
    z = np.random.default_rng(9).normal(size=(detection_count, 2))
    work = ObservationWorkspace(hs, z, pose, law)
    for i, h in enumerate(hs):
        same_nested(work.tracked_misses[i], law.missed_moments(h, pose))
        same_nested(work.hidden_emissions[i], hypothesis_emissions(h, z, pose, law))
        if detection_count:
            cm, cv = work.hidden_emissions[i][3]
            lp = work.hidden_emissions[i][4]
            for m, d in zip(*np.nonzero(np.isfinite(lp))):
                np.testing.assert_array_equal(cm[m, d, :2], z[d])
                np.testing.assert_array_equal(cv[m, d, :2, :], 0.)


@parametrize('q', [1, 4, 8, 16, 24])
@parametrize('n', [0, 1, 6, 12])
def test_fused_events_keep_every_query_world_native_endpoint(q, n):
    rng = np.random.default_rng(73); W, T = 16, 63
    p = rng.normal(size=(W, T, n, 2)); v = rng.normal(size=p.shape)
    if n:
        p[0, :, -1] = np.nan; v[0, :, -1] = np.nan
    r = rng.normal(size=(q, T, 4)); hidden = np.arange(W) % 4
    age = np.linspace(.1, 7., T); elapsed = np.arange(T)[None] * .048 + np.arange(q)[:, None] * .048
    e = event_features_many(p, v, r, hidden, age, elapsed)
    a = event_features_many_fast(p, v, r, hidden, age, elapsed)
    assert a.shape == (q, W, T, 12) and a.dtype == np.float32
    np.testing.assert_allclose(a, e, atol=1e-7, rtol=1e-7)


def test_fused_event_ties_masking_no_clipping_and_scalar_age():
    p = np.array([[[[1., 0.], [-1., 0.]], [[np.nan, 0.], [200., 0.]],
                   [[0., np.nan], [3., 4.]]]])
    v = np.array([[[[2., 0.], [8., 0.]], [[np.nan, np.nan], [3., 4.]],
                   [[1., 0.], [6., 8.]]]])
    r = np.zeros((2, 3, 4)); r[1, :, 3] = np.pi / 2
    e = event_features_many(p, v, r, np.array([2]), 7., np.arange(3) * .048)
    a = event_features_many_fast(p, v, r, np.array([2]), 7., np.arange(3) * .048)
    np.testing.assert_allclose(a, e, atol=1e-7, rtol=1e-7)
    assert a[0, 0, 0, 8] == np.float32(2 / 3)  # First nearest tie.
    assert a[0, 0, 1, 8] == 0.  # NaN padding uses reference distance 100.


def test_fused_event_noncontiguous_inputs():
    rng = np.random.default_rng(3)
    p = rng.normal(size=(3, 10, 8, 2))[:, ::2, ::2]
    v = rng.normal(size=(3, 10, 8, 2))[:, ::2, ::2]
    r = rng.normal(size=(4, 10, 4))[:, ::2]
    np.testing.assert_allclose(event_features_many_fast(p, v, r, np.zeros(3), 0., np.arange(5) * .048),
                               event_features_many(p, v, r, np.zeros(3), 0., np.arange(5) * .048),
                               atol=1e-7, rtol=1e-7)


def test_fused_event_float32_infinite_velocity_matches_dtype_replacement():
    p = np.ones((1, 2, 1, 2), dtype=np.float32)
    v = np.full_like(p, np.inf); r = np.zeros((1, 2, 4), dtype=np.float32)
    e = event_features_many(p, v, r, np.zeros(1), 0., np.arange(2) * .048)
    a = event_features_many_fast(p, v, r, np.zeros(1), 0., np.arange(2) * .048)
    np.testing.assert_allclose(a, e, atol=1e-7, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
