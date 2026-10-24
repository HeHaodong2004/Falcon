"""Small independent regression oracles for exact performance transformations.

No data or holdout access.  The scalar native-clock implementation is retained
as an executable reference for batching; endpoint caches must not permit a
consumer of one successor to corrupt another possible world's belief.
"""
import unittest

import numpy as np

from eventwm_graph.l2_partial.state import BeliefState, Hypothesis
from eventwm_graph.l2_partial.repair_model import RepairedGraphBeliefWorldModel
from test_observation_repair import ConstantLaw, ZeroEvents, _belief, _graph, _observation, _wm


ARRAY_FIELDS = ('mean', 'covariance', 'intent', 'diffusion', 'mode', 'hidden_pmf',
                'maneuver_mean', 'maneuver_covariance', 'maneuver_probability',
                'hidden_mean', 'hidden_covariance', 'hidden_maneuver_mean',
                'hidden_maneuver_covariance', 'hidden_maneuver_probability')


def _random_hypothesis(rng, n, hidden, rate, weight):
    mean = rng.normal(size=(n, 4))
    factor = rng.normal(size=(n, 4, 4)) * .2
    covariance = factor @ factor.transpose(0, 2, 1)
    pmf = np.zeros(hidden + 1); pmf[hidden] = 1.
    hm = rng.normal(size=(hidden, 4))
    factor = rng.normal(size=(hidden, 4, 4)) * .4
    hc = factor @ factor.transpose(0, 2, 1)
    return Hypothesis(weight, mean, covariance, np.zeros(n), np.arange(n), pmf,
                      intent=rng.normal(size=(n, 2)), diffusion=rng.uniform(.01, .3, n),
                      mode=np.zeros(n, int), relaxation=rate, height=np.zeros(n),
                      hidden_mean=hm, hidden_covariance=hc, hidden_exchangeable=False)


class BatchTransitionTests(unittest.TestCase):
    def assertSameHypothesis(self, expected, actual):
        self.assertEqual(expected.weight, actual.weight)
        self.assertAlmostEqual(expected.maneuver_until, actual.maneuver_until, places=13)
        for name in ARRAY_FIELDS:
            x, y = getattr(expected, name), getattr(actual, name)
            if x is None:
                self.assertIsNone(y, name)
            else:
                np.testing.assert_allclose(x, y, rtol=2e-12, atol=2e-12, err_msg=name)

    def test_batch_equals_scalar_for_mixed_population_phases_and_retention(self):
        rng = np.random.default_rng(412)
        for retention in (0., 1.):
            wm = RepairedGraphBeliefWorldModel(kind='memory_cv', maneuver_retention=retention)
            base = [_random_hypothesis(rng, n, k, r, 1 / 12)
                    for n in (0, 1, 3) for k in (0, 2) for r in (0., .7)]
            # Include absent hidden priors, iid hidden priors, and private states.
            base[0] = base[0].fork(hidden_mean=None, hidden_covariance=None)
            base[3] = base[3].fork(hidden_mean=base[3].hidden_mean[:1],
                                  hidden_covariance=base[3].hidden_covariance[:1],
                                  hidden_exchangeable=True)
            for start in (0., .013, .024, .048, .109):
                phased = [wm.advance_hypothesis(h, start, 0.) if i % 2 and start else h
                          for i, h in enumerate(base)]
                snapshots = [{name: None if getattr(h, name) is None else getattr(h, name).copy()
                              for name in ARRAY_FIELDS} for h in phased]
                for dt in (0., .007, .048, .123, .96):
                    with self.subTest(retention=retention, start=start, dt=dt):
                        actual = wm.advance_hypotheses(phased, dt, start)
                        expected = [wm.advance_hypothesis(h, dt, start) for h in phased]
                        for x, y in zip(expected, actual):
                            self.assertSameHypothesis(x, y)
                        for h, snapshot in zip(phased, snapshots):
                            for name, x in snapshot.items():
                                if x is not None:
                                    np.testing.assert_array_equal(x, getattr(h, name), err_msg=name)

    def test_batch_preserves_partial_interval_composition(self):
        wm = RepairedGraphBeliefWorldModel(kind='memory_cv')
        rng = np.random.default_rng(71)
        hs = [_random_hypothesis(rng, n, 2, r, .25)
              for n in (1, 3) for r in (0., .7)]
        direct = wm.advance_hypotheses(hs, .181, .013)
        recursive = wm.advance_hypotheses(hs, .023, .013)
        recursive = wm.advance_hypotheses(recursive, .158, .036)
        for expected, actual in zip(direct, recursive):
            self.assertSameHypothesis(expected, actual)

    def test_roundoff_sized_increment_matches_scalar_noop(self):
        wm = _wm()
        h = _belief().hypotheses[0]
        for start in (0., 1e6):
            for dt in (1e-12, 2e-12, 1e-11, 1e-10, 1e-9):
                with self.subTest(start=start, dt=dt):
                    self.assertSameHypothesis(wm.advance_hypothesis(h, dt, start),
                                              wm.advance_hypotheses([h], dt, start)[0])

    def test_unified_sampling_retains_every_person_and_seed_replays(self):
        rng = np.random.default_rng(91)
        hs = [_random_hypothesis(rng, n, k, .7, .25)
              for n, k in ((0, 2), (1, 0), (2, 3), (3, 1))]
        b = BeliefState(_graph(), hs, .013, np.zeros(4)).validate()
        wm = RepairedGraphBeliefWorldModel(kind='memory_cv')
        before = b.fingerprint()
        a = wm.predict_shared(b, .12, num_worlds=16, seed=511)
        c = wm.predict_shared(b, .12, num_worlds=16, seed=511)
        self.assertEqual(before, b.fingerprint())
        np.testing.assert_array_equal(a.positions, c.positions)
        np.testing.assert_array_equal(a.velocities, c.velocities)
        for w, h in enumerate(a.worlds):
            count = h.n + int(round(h.expected_hidden))
            self.assertTrue(np.isfinite(a.positions[w, :, :count]).all())
            self.assertTrue(np.isnan(a.positions[w, :, count:]).all())
            self.assertEqual(len(h.hidden_mean), int(round(h.expected_hidden)))

    def test_unified_sample_transition_moments_match_analytic_law(self):
        # Independent numerical oracle for the revised sampler, not merely
        # equality of two analytic paths. Four thousand worlds take <1 CPU sec.
        wm = RepairedGraphBeliefWorldModel(kind='memory_cv')
        h = _random_hypothesis(np.random.default_rng(992), 1, 1, 0., 1.)
        h = h.fork(covariance=np.zeros((1, 4, 4)), hidden_covariance=np.zeros((1, 4, 4)))
        expected = wm.advance_hypothesis(h, .096, 0.)
        count = 4096
        positions, velocities, _ = wm.sample_maneuver_paths(
            [h] * count, np.array([0., .048, .096]), 0., np.random.default_rng(321), 1)
        samples = np.concatenate((positions[:, -1], velocities[:, -1]), axis=-1)
        means = np.concatenate((expected.mean, expected.hidden_mean))
        covariances = np.concatenate((expected.covariance, expected.hidden_covariance))
        for i in range(2):
            error = samples[:, i] - means[i]
            mean_se = np.sqrt(np.diag(covariances[i]) / count)
            self.assertTrue(np.all(np.abs(error.mean(0)) < 6 * mean_se + 1e-12))
            products = error[:, :, None] * error[:, None, :]
            covariance_se = products.std(0, ddof=1) / np.sqrt(count)
            self.assertTrue(np.all(np.abs(products.mean(0) - covariances[i]) <
                                   6 * covariance_se + 1e-12))


class EndpointCacheTests(unittest.TestCase):
    def _future(self):
        wm = _wm()
        b = _belief()
        for h in b.hypotheses:
            h.hidden_pmf = np.array([0., 1.])
            h.hidden_mean = np.array([[5., 2., 0., 0.]])
            h.hidden_covariance = np.zeros((1, 4, 4))
        f = wm.predict_shared(b, .096, seed=9)
        return wm, b, f

    def test_cached_arrays_cannot_corrupt_siblings_or_templates(self):
        wm, _, f = self._future()
        names = ('mean', 'covariance', 'intent', 'diffusion', 'mode', 'hidden_pmf',
                 'last_seen', 'slot_ids', 'height', 'hidden_mean', 'hidden_covariance')
        for name in names:
            with self.subTest(field=name):
                a = wm.successor_at(f, 2, f.weights, np.zeros(4))
                sibling = wm.successor_at(f, 2, f.weights, np.ones(4))
                before = sibling.fingerprint()
                value = getattr(a.hypotheses[0], name)
                if value is None or not value.size:
                    continue
                try:
                    value.flat[0] += 1
                except (ValueError, TypeError):
                    pass  # Immutable shared arrays satisfy the contract.
                self.assertEqual(before, sibling.fingerprint(), name)
                fresh = wm.successor_at(f, 2, f.weights, np.ones(4))
                self.assertEqual(before, fresh.fingerprint(), name)

    def test_branch_weights_refinement_and_api_updates_leave_cache_unchanged(self):
        wm, _, f = self._future()
        first = wm.successor_at(f, 1, f.weights, np.zeros(4))
        fingerprint = first.fingerprint()
        weights = f.weights * np.linspace(.1, .9, len(f.worlds))
        refinement = (np.full(len(f.worlds), 2.), np.zeros(len(f.worlds)))
        second = wm.successor_at(f, 1, weights, np.ones(4), refinement=refinement)
        self.assertNotEqual(first.fingerprint(), second.fingerprint())
        wm.propagate(second, second.timestamp + .023)
        wm.observe(second, _observation(second.timestamp + .048, pose=[1., 1., 1., 1.]))
        self.assertEqual(fingerprint, first.fingerprint())
        fresh = wm.successor_at(f, 1, f.weights, np.zeros(4))
        self.assertEqual(fingerprint, fresh.fingerprint())
        self.assertEqual(len(f.endpoint_cache), 1)
        later = wm.successor_at(f, 2, f.weights, np.zeros(4))
        self.assertEqual(len(f.endpoint_cache), 2)
        self.assertAlmostEqual(later.timestamp - first.timestamp, .048)
        for h in fresh.hypotheses:
            self.assertEqual(h.coarse_observation_time, -1.)


if __name__ == '__main__':
    unittest.main()
