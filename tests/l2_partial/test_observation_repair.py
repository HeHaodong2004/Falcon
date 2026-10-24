"""Independent analytical oracles for the bounded scientific repair.

These tests use no dataset, tracker identity, privileged population cap, or
holdout. Constant-position/constant-emission cases isolate probability and
endpoint composition from forecasting accuracy. The diagnostic function also
measures, rather than hides, error in approximate missing-detection integrals.
"""
import itertools
import json
import sys
import unittest

import numpy as np

from eventwm_graph.l2_partial.hidden import (
    advance_hidden_samples, exchangeable_count_update,
    first_detection_moments, hidden_subset_likelihoods, hypothesis_emissions,
    missed_hidden_moments, prepare_prior, propagate_hidden_gaussian,
    sample_hidden_states,
)
from eventwm_graph.l2_partial.observation import EndpointObservationLaw
from eventwm_graph.l2_partial.maneuver import moment_match
from eventwm_graph.l2_partial.repair_filter import set_evidence
from eventwm_graph.l2_partial.repair_model import RepairedGraphBeliefWorldModel
from eventwm_graph.l2_partial.state import BeliefState, Hypothesis, ObservedGraph, EXECUTOR_VERSION


class ZeroEvents:
    def hazards(self, features):
        return np.zeros(features.shape[:-1] + (3,))


class ConstantLaw(EndpointObservationLaw):
    regime_prior = (.35, .65)
    in_view = (.2, .8)
    # Kept to expose the old, inconsistent hidden-first-report law if used.
    hidden_detection = (.1, .1)

    def detection(self, positions, robot_pose, regime):
        p = np.asarray(positions)
        regime = np.asarray(regime, dtype=int)
        pd = np.asarray(self.in_view)[regime]
        if regime.ndim:
            pd = pd.reshape(regime.shape + (1,) * (p.ndim - 1 - regime.ndim))
        return np.where(np.isfinite(p[..., 0]), pd, 0.)


class HalfSpaceLaw(ConstantLaw):
    """Two discrete spatial supports admit an exact coarse-vs-exact oracle."""
    def detection(self, positions, robot_pose, regime):
        p = np.asarray(positions)
        return np.where(np.isfinite(p[..., 0]), np.where(p[..., 0] > 0, .8, .2), 0.)


class RetiredHiddenFirstReportLaw(HalfSpaceLaw):
    """Executed regression witness for the superseded nonspatial hidden law."""
    def no_detection(self, future, indices, poses, return_parts=False):
        result = np.ones(len(future.worlds))
        before = result.copy(); last = result.copy()
        for w, h in enumerate(future.worlds):
            pd = self.detection(future.positions[w, indices[1:], :h.n], poses[1:], h.visibility_mode)
            per_endpoint = np.log1p(-pd).sum(axis=-1) + h.expected_hidden * np.log(.9)
            result[w] = np.exp(per_endpoint.sum())
            before[w] = np.exp(per_endpoint[:-1].sum())
            if len(per_endpoint):
                last[w] = np.exp(per_endpoint[-1])
        return (result, before, last) if return_parts else result


class SyntheticNearConflict:
    """Known event oracle on native endpoint features, not a learned native-label result."""
    def hazards(self, features):
        result = np.zeros(features.shape[:-1] + (3,))
        result[..., 0] = .2 * (features[..., 2] > .5)
        return result


def _graph():
    return ObservedGraph('endpoint-probability-oracle', 1, .2, 0.,
                         {(i, j): (1, 0.) for i in range(-8, 9) for j in range(-8, 9)})


def _hypothesis(weight=1., regime=0, position=(3., 0.), hidden=0):
    mean = np.asarray([[*position, 0., 0.]])
    pmf = np.zeros(hidden + 1)
    pmf[hidden] = 1.
    return Hypothesis(weight, mean, np.zeros((1, 4, 4)), np.zeros(1), np.zeros(1, int), pmf,
                      intent=np.zeros((1, 2)), diffusion=np.zeros(1), mode=np.zeros(1, int),
                      height=np.zeros(1), visibility_mode=regime)


def _belief(start=0.):
    hs = [_hypothesis(p, r) for r, p in enumerate(ConstantLaw.regime_prior)]
    return BeliefState(_graph(), hs, start, np.zeros(4)).validate()


def _query(steps=2):
    poses = [[0., 0., 0., i * 10 / 120] for i in range(steps + 1)]
    return {'query_id': 'endpoint-oracle-turn', 'executor_version': EXECUTOR_VERSION,
            'compiled': {'planned_poses': poses, 'actions': ['left'] * steps, 'graph_revision': 1},
            'waypoints': [poses[-1][:3]], 'final_yaw': poses[-1][3]}


def _observation(timestamp, detected=False, pose=None, position=(3., 0.), **extras):
    return {'timestamp': timestamp, 'robot_pose': [0., 0., 0., 0.] if pose is None else pose,
            'static_updates': [], 'tracks': ([{'position': [position[0], 0., position[1]],
             'last_seen_time': timestamp, 'visible_now': True, 'valid': True, 'track_id': 999}] if detected else []),
            **extras}


def _wm(law=None, maneuver_rate=0.):
    wm = RepairedGraphBeliefWorldModel(kind='memory_cv', max_hypotheses=32,
                                      event_model=ZeroEvents(), maneuver_rate=maneuver_rate,
                                      observation_law=law or ConstantLaw(),
                                      successor_regularization=(0., 0.))
    # This oracle isolates exact endpoint probabilities. Production uses a
    # declared nonzero successor regularization measured in separate tests.
    return wm


def _branches(wm, belief, query, horizon):
    f = wm.predict_shared(belief, horizon, seed=17)
    p = wm.evaluate_queries(belief, [query], f).predictions[0]
    return {int(br.observation['kind'] == 'survived_detection'): br
            for br in p.observation_branches if br.observation['kind'].startswith('survived_')}


def _regimes(belief):
    result = np.zeros(2)
    for h in belief.hypotheses:
        result[h.visibility_mode] += h.weight
    return result


def _exhaustive_set(assign, miss, pmf, ph, birth):
    """Independent labeled-person enumeration, then sum unknown counts."""
    N, D = assign.shape
    evidence = 0.
    for H, prior in enumerate(pmf):
        for people in itertools.permutations(range(N + H), D):
            w = float(prior)
            used = set(people)
            for j, i in enumerate(people):
                w *= assign[i, j] if i < N else ph * birth[j]
            for i in range(N):
                if i not in used:
                    w *= miss[i]
            w *= (1 - ph) ** (H - sum(i >= N for i in people))
            evidence += w
    return evidence


def quadrature_diagnostics():
    """Actual integral errors; broad cases deliberately have no pass label."""
    law = EndpointObservationLaw()
    rows = []
    for variance in (.001, .01, .1, 1., 4., 16.):
        h = _hypothesis(regime=1, position=(2., 1.8))
        h = h.fork(covariance=np.diag([variance, variance, .25, .25])[None])
        mass, mean, covariance = law.missed_moments(h, np.zeros(4))
        ref, rm, rc = missed_hidden_moments(h.mean, h.covariance, np.zeros(4), 1, law, order=151)
        finer, _, _ = missed_hidden_moments(h.mean, h.covariance, np.zeros(4), 1, law, order=251)
        # Preserve the original 4D axis rule as an executable failure witness.
        eigenvalues, eigenvectors = np.linalg.eigh(h.covariance)
        factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :]
        offsets = np.concatenate([np.zeros((1, 1, 4)), np.sqrt(8) * factor.transpose(0, 2, 1),
                                  -np.sqrt(8) * factor.transpose(0, 2, 1)], axis=1)
        points = h.mean[:, None] + offsets
        weights = np.r_[.5, np.full(8, .0625)] * (1 - law.detection(points[..., :2], np.zeros(4), 1))
        legacy_mass = weights.sum(1)
        rows.append({'position_variance': variance, 'repaired_position_gh_mass': float(mass[0]),
                     'repaired_order_per_axis': 9 if variance <= .25 else 15,
                     'retired_nine_point_mass': float(legacy_mass[0]),
                     'retired_absolute_mass_error': float(abs(legacy_mass[0] - ref[0])),
                     'reference_151x151_mass': float(ref[0]),
                     'reference_251x251_mass': float(finer[0]),
                     'reference_order151_to251_difference': float(abs(ref[0] - finer[0])),
                     'absolute_mass_error': float(abs(mass[0] - ref[0])),
                     'mean_l2_error': float(np.linalg.norm(mean - rm)),
                     'covariance_frobenius_error': float(np.linalg.norm(covariance - rc))})
    m, c = prepare_prior(np.zeros(4))
    coarse, _, _ = missed_hidden_moments(m, c, np.zeros(4), 1, law)
    fine, _, _ = missed_hidden_moments(m, c, np.zeros(4), 1, law, order=151)
    return {'tracked_miss_approximation': rows,
            'hidden_initial_miss': {'order15': float(coarse[0]), 'order151': float(fine[0]),
                                    'absolute_error': float(abs(coarse[0] - fine[0]))}}


def hidden_closure_diagnostic(law=None):
    """Detection/location marginalization and one subsequent risk prediction.

    One hidden person lies in one of two separated supports with equal prior
    mass. Endpoint detection probabilities are .8 and .2, giving posterior
    near-support probability .8 and next detection probability .68. A known
    next-endpoint event has hazard .2 in the near support and zero in the far
    support. Dynamics are stochastic but cannot cross these separated supports
    in this finite seeded oracle; this tests bounded code, not a general
    integration-error guarantee.
    """
    wm, q = _wm(law or HalfSpaceLaw()), _query()
    hs = []
    for x in (.3, -3.):
        h = Hypothesis(.5, np.empty((0, 4)), np.empty((0, 4, 4)), np.empty(0), np.empty(0, int),
                       np.array([0., 1.]), intent=np.empty((0, 2)), diffusion=np.empty(0),
                       mode=np.empty(0, int), relaxation=.7, visibility_mode=0,
                       hidden_mean=np.array([[x, 0., 0., 0.]]), hidden_covariance=np.zeros((1, 4, 4)),
                       hidden_exchangeable=False)
        hs.append(h)
    b = BeliefState(_graph(), hs, 0., np.zeros(4)).validate()
    first = _branches(wm, b, q, .048)[1]
    second = _branches(wm, first.successor, q, .048)[1]
    near_posterior = sum(h.weight * (h.hidden_mean[:, 0] > 0).sum() for h in first.successor.hypotheses)
    prior_at_endpoint = wm.propagate(b, .048).fork(robot_pose=np.asarray(q['compiled']['planned_poses'][1]),
        query_cursor={'query_id': q['query_id'], 'action_index': 1})
    actual = [wm.observe(prior_at_endpoint, _observation(.048, True, q['compiled']['planned_poses'][1], (x, 0.)))
              for x in (.3, -3.)]
    exact_future_detection = np.array([_branches(wm, a, q, .048)[1].probability for a in actual])
    wm.event_model = SyntheticNearConflict()
    coarse_risk = wm.continue_rollout(first.successor, q, horizon=.048, seed=19).collision_probability
    exact_risks = np.array([wm.continue_rollout(a, q, horizon=.048, seed=19).collision_probability for a in actual])
    total_count_error = max(abs(h.n + h.expected_hidden - 1.) for a in actual for h in a.hypotheses)
    return {'first_detection_probability': first.probability,
            'coarse_near_posterior': near_posterior,
            'coarse_next_detection': second.probability,
            'marginalized_exact_next_detection': float(np.array([.8, .2]) @ exact_future_detection),
            'coarse_next_conflict_probability': coarse_risk,
            'marginalized_exact_next_conflict_probability': float(np.array([.8, .2]) @ exact_risks),
            'exact_count_error': total_count_error,
            'exact_remaining_hidden': [sum(h.weight * h.expected_hidden for h in a.hypotheses) for a in actual],
            'exact_unresolved': [a.unresolved_mass for a in actual]}


class SetLikelihoodOracleTests(unittest.TestCase):
    def test_set_evidence_matches_exhaustive_assignments_and_counts(self):
        rng = np.random.default_rng(9301)
        for N in range(4):
            for D in range(4):
                a, miss, birth = rng.uniform(.01, .9, (N, D)), rng.uniform(.1, .9, N), rng.uniform(.01, 1., D)
                pmf = rng.uniform(.1, 1., 4); pmf /= pmf.sum()
                for ph in (0., .17, 1.):
                    with self.subTest(N=N, D=D, ph=ph):
                        expected = _exhaustive_set(a, miss, pmf, ph, birth)
                        self.assertAlmostEqual(set_evidence(a, miss, pmf, ph, birth), expected, places=12)

    def test_private_hidden_subset_likelihood_is_injective(self):
        rng = np.random.default_rng(193)
        a = rng.uniform(.01, .9, (3, 3)); miss = rng.uniform(.1, .9, 3)
        got = hidden_subset_likelihoods(a, miss)
        for mask, value in enumerate(got):
            selected = [j for j in range(3) if mask & (1 << j)]
            reference = _exhaustive_set(a[:, selected], miss, np.array([1.]), 0., np.ones(len(selected)))
            self.assertAlmostEqual(value, reference, places=12)

    def test_exchangeable_falling_factorial_and_remaining_count(self):
        pmf = np.array([.1, .2, .3, .4]); first = np.array([.07, .09]); miss = .6
        evidence, posterior = exchangeable_count_update(pmf, miss, first)
        expected = np.array([pmf[2] * 2, pmf[3] * 6 * miss, 0., 0.]) * np.prod(first)
        self.assertAlmostEqual(evidence, expected.sum())
        np.testing.assert_allclose(posterior, expected / expected.sum(), atol=1e-14)
        evidence0, p0 = exchangeable_count_update([1.], miss, first)
        self.assertEqual(evidence0, 0.)
        np.testing.assert_array_equal(p0, [0.])

    def test_detection_permutation_does_not_change_evidence(self):
        a = np.array([[.1, .2, .3], [.4, .1, .2]])
        birth = np.array([.01, .2, .1]); pmf = np.array([.1, .3, .6]); miss = np.array([.3, .4])
        reference = set_evidence(a, miss, pmf, .1, birth)
        for order in itertools.permutations(range(3)):
            self.assertAlmostEqual(set_evidence(a[:, order], miss, pmf, .1, birth[list(order)]), reference)

    def test_beam_omission_is_measured_against_complete_evidence(self):
        wm = _wm(); wm.max_hypotheses = 1
        h = _hypothesis()
        h = h.fork(mean=np.array([[-.2, 0., 0., 0.], [.2, 0., 0., 0.]]),
                   covariance=np.tile(np.diag([1., 1., .1, .1]), (2, 1, 1)),
                   last_seen=np.zeros(2), slot_ids=np.arange(2), intent=np.zeros((2, 2)),
                   diffusion=np.zeros(2), mode=np.zeros(2, int), height=np.zeros(2))
        b = BeliefState(_graph(), [h], 0., np.zeros(4))
        posterior = wm.observe(b, _observation(0., True, position=(0., 0.)))
        self.assertAlmostEqual(posterior.unresolved_mass, .5, places=12)
        self.assertAlmostEqual(posterior.evidence['pruned_mass'], .5, places=12)
        self.assertAlmostEqual(posterior.hypotheses[0].weight, .5, places=12)
        self.assertEqual(posterior.hypotheses[0].n, 2)


class EndpointCompositionTests(unittest.TestCase):
    def test_exact_endpoint_refines_coarse_event_without_double_evidence(self):
        wm, b, q = _wm(), _belief(), _query()
        alpha = np.asarray(ConstantLaw.regime_prior); pd = np.asarray(ConstantLaw.in_view)
        coarse = _branches(wm, b, q, .096)
        for any_detection, endpoint_detected in ((0, 0), (1, 0), (1, 1)):
            posterior = wm.observe(coarse[any_detection].successor,
                                  _observation(.096, endpoint_detected, q['compiled']['planned_poses'][2]))
            if endpoint_detected:
                raw = alpha * pd
            elif any_detection:
                raw = alpha * pd * (1 - pd)
            else:
                raw = alpha * (1 - pd)**2
            with self.subTest(coarse=any_detection, endpoint=endpoint_detected):
                np.testing.assert_allclose(_regimes(posterior), raw / raw.sum(), atol=1e-12)
                posterior.validate()
        impossible = wm.observe(coarse[0].successor,
                                _observation(.096, True, q['compiled']['planned_poses'][2]))
        self.assertEqual(impossible.unresolved_mass, 1.)

    def test_retained_maneuver_positive_and_missed_events_normalize(self):
        # Exact two-atom oracle: the Gaussian closure of the mixture is NOT its
        # prior measure. Positive and negative sensing must use the same one.
        mm = np.array([[[-3., 0., 0., 0.], [3., 0., 0., 0.]]])
        cc = np.zeros((1, 2, 4, 4)); pp = np.array([[.5, .5]])
        mean, cov = moment_match(mm, cc, pp)
        h = _hypothesis().fork(mean=mean, covariance=cov, maneuver_mean=mm,
                              maneuver_covariance=cc, maneuver_probability=pp)
        law = HalfSpaceLaw()
        negative_mass = law.missed_moments(h, np.zeros(4))[0][0]
        positive_mass = np.dot(pp[0], law.detection(mm[0, :, :2], np.zeros(4), 0))
        self.assertAlmostEqual(negative_mass + positive_mass, 1., places=12)

    def test_all_two_endpoint_histories_and_posteriors(self):
        alpha = np.asarray(ConstantLaw.regime_prior); pd = np.asarray(ConstantLaw.in_view)
        for start in (0., .013):
            wm, b, q = _wm(), _belief(start), _query()
            direct = _branches(wm, b, q, .096)
            first = _branches(wm, b, q, .048)
            history_mass = 0.; detected_mass = np.zeros(2)
            for d1, d2 in itertools.product((0, 1), repeat=2):
                branch1 = first[d1]
                branch2 = _branches(wm, branch1.successor, q, .048)[d2]
                raw = alpha * (pd if d1 else 1 - pd) * (pd if d2 else 1 - pd)
                joint = branch1.probability * branch2.probability
                with self.subTest(start=start, history=(d1, d2)):
                    self.assertAlmostEqual(joint, raw.sum(), places=12)
                    np.testing.assert_allclose(_regimes(branch2.successor), raw / raw.sum(), atol=1e-12)
                    branch2.successor.validate()
                    self.assertEqual(branch2.successor.timestamp, start + .096)
                    actual = wm.observe(b, _observation(start + .048, d1, q['compiled']['planned_poses'][1]))
                    actual = wm.observe(actual, _observation(start + .096, d2, q['compiled']['planned_poses'][2]))
                    np.testing.assert_allclose(_regimes(actual), raw / raw.sum(), atol=1e-12)
                history_mass += joint
                if d1 or d2:
                    detected_mass += joint * _regimes(branch2.successor)
            self.assertAlmostEqual(history_mass, 1., places=12)
            self.assertAlmostEqual(direct[0].probability, np.dot(alpha, (1 - pd)**2), places=12)
            self.assertAlmostEqual(direct[1].probability, detected_mass.sum(), places=12)
            np.testing.assert_allclose(_regimes(direct[1].successor), detected_mass / detected_mass.sum(), atol=1e-12)
            recovered = sum(br.probability * _regimes(br.successor) for br in direct.values())
            np.testing.assert_allclose(recovered, alpha, atol=1e-12)

    def test_maneuver_clock_is_independent_of_propagation_segmentation(self):
        wm = _wm(maneuver_rate=2.)
        b = _belief(.013)
        b = b.fork(hypotheses=[h.fork(mean=h.mean + [0., 0., 1., -.2],
                                     covariance=np.diag([.1, .2, .4, .5])[None],
                                     diffusion=np.array([.2])) for h in b.hypotheses])
        direct = wm.propagate(b, .193)
        sequential = b
        for t in (.021, .047, .048, .049, .095, .096, .138, .177, .193):
            sequential = wm.propagate(sequential, t)
        for expected, actual in zip(direct.hypotheses, sequential.hypotheses):
            np.testing.assert_allclose(actual.mean, expected.mean, atol=2e-12)
            np.testing.assert_allclose(actual.covariance, expected.covariance, atol=2e-12)
            np.testing.assert_allclose(actual.maneuver_probability, expected.maneuver_probability, atol=2e-12)
            self.assertEqual(actual.maneuver_until, expected.maneuver_until)

    def test_unavailable_sensor_does_not_supply_missed_detection_evidence(self):
        wm, b = _wm(), _belief()
        c = wm.observe(b, _observation(.048, human_sensor_available=False))
        np.testing.assert_allclose(_regimes(c), _regimes(b), atol=1e-14)
        self.assertEqual(sum(h.weight * h.n for h in c.hypotheses), 1.)


class HiddenGaussianOracleTests(unittest.TestCase):
    def test_hidden_retained_components_share_positive_and_missed_measure(self):
        mm = np.array([[[-3., 0., -1., 0.], [3., 0., 1., 0.]]])
        cc = np.zeros((1, 2, 4, 4)); pp = np.array([[.5, .5]])
        mean, covariance = moment_match(mm, cc, pp)
        h = _hypothesis().fork(hidden_mean=mean, hidden_covariance=covariance,
                              hidden_maneuver_mean=mm, hidden_maneuver_covariance=cc,
                              hidden_maneuver_probability=pp)
        miss, postmean, postcov, (detmean, detcov), logjoint = hypothesis_emissions(
            h, [[-3., 0.], [3., 0.]], np.zeros(4), HalfSpaceLaw())
        self.assertAlmostEqual(float(miss[0] + np.exp(logjoint[0]).sum()), 1., places=12)
        self.assertAlmostEqual(miss[0], .5, places=12)
        np.testing.assert_allclose(postmean[0], [-1.8, 0., -.6, 0.], atol=1e-12)
        self.assertAlmostEqual(postcov[0, 0, 0], 5.76, places=12)
        np.testing.assert_allclose(detmean[0], mm[0], atol=1e-12)
        np.testing.assert_allclose(detcov, np.zeros_like(detcov), atol=1e-12)

    def test_hidden_forecast_initial_samples_preserve_retained_components(self):
        mm = np.array([[[-3., 0., 0., 0.], [3., 0., 0., 0.]]])
        cc = np.zeros((1, 2, 4, 4)); pp = np.array([[.5, .5]])
        mean, covariance = moment_match(mm, cc, pp)
        h = Hypothesis(1., np.empty((0, 4)), np.empty((0, 4, 4)), np.empty(0), np.empty(0, int),
                       np.array([0., 1.]), intent=np.empty((0, 2)), diffusion=np.empty(0),
                       mode=np.empty(0, int), relaxation=.7, visibility_mode=0,
                       hidden_mean=mean, hidden_covariance=covariance,
                       hidden_maneuver_mean=mm, hidden_maneuver_covariance=cc,
                       hidden_maneuver_probability=pp, maneuver_until=.048)
        b = BeliefState(_graph(), [h], .024, np.zeros(4)).validate()
        f = _wm(HalfSpaceLaw()).predict_shared(b, .024, seed=72)
        self.assertTrue(np.all(np.isin(f.positions[:, 0, 0, 0], [-3., 3.])))
        self.assertEqual(len(np.unique(f.positions[:, 0, 0, 0])), 2)

    def test_certified_empty_excludes_private_hidden_point(self):
        hs = []
        for count in (0, 1):
            pmf = np.zeros(count + 1); pmf[count] = 1.
            hs.append(Hypothesis(.5, np.empty((0, 4)), np.empty((0, 4, 4)), np.empty(0), np.empty(0, int),
                pmf, intent=np.empty((0, 2)), diffusion=np.empty(0), mode=np.empty(0, int),
                relaxation=.7, visibility_mode=0,
                hidden_mean=np.array([[.3, 0., 0., 0.]]) if count else np.empty((0, 4)),
                hidden_covariance=np.zeros((count, 4, 4)), hidden_exchangeable=False))
        b = BeliefState(_graph(), hs, 0., np.zeros(4)).validate()
        posterior = _wm(HalfSpaceLaw()).observe(b, _observation(0.,
            certified_visible_empty_boxes=[[[-1., -1.], [1., 1.]]]))
        self.assertEqual(sum(h.weight * h.expected_hidden for h in posterior.hypotheses), 0.)
        self.assertAlmostEqual(sum(h.weight for h in posterior.hypotheses), 1.)
        posterior.validate()

    def test_private_hidden_coarse_and_exact_updates_have_same_future_law(self):
        d = hidden_closure_diagnostic()
        self.assertAlmostEqual(d['first_detection_probability'], .5, places=12)
        self.assertAlmostEqual(d['coarse_near_posterior'], .8, places=12)
        self.assertAlmostEqual(d['coarse_next_detection'], .68, places=12)
        self.assertAlmostEqual(d['coarse_next_detection'], d['marginalized_exact_next_detection'], places=12)
        self.assertAlmostEqual(d['coarse_next_conflict_probability'], .16, places=12)
        self.assertAlmostEqual(d['coarse_next_conflict_probability'], d['marginalized_exact_next_conflict_probability'], places=12)
        self.assertEqual(d['exact_count_error'], 0.)
        np.testing.assert_allclose(d['exact_remaining_hidden'], [0., 0.], atol=1e-12)
        np.testing.assert_allclose(d['exact_unresolved'], [0., 0.], atol=1e-12)

    def test_retired_hidden_first_report_law_has_recorded_closure_failure(self):
        d = hidden_closure_diagnostic(RetiredHiddenFirstReportLaw())
        self.assertAlmostEqual(d['first_detection_probability'], .1, places=12)
        self.assertAlmostEqual(d['coarse_next_detection'], .1, places=12)
        self.assertAlmostEqual(d['marginalized_exact_next_detection'], .68, places=12)
        self.assertGreater(abs(d['coarse_next_conflict_probability'] - d['marginalized_exact_next_conflict_probability']), .05)

    def test_hidden_prior_is_persistent_and_ou_composes(self):
        m, c = prepare_prior([2., 9., -3., 1.])
        np.testing.assert_array_equal(m, [[2., -3., 0., 0.]])
        np.testing.assert_array_equal(np.diag(c[0]), [100., 100., 4., 4.])
        a, b = propagate_hidden_gaussian(m, c, .13)
        a, b = propagate_hidden_gaussian(a, b, .27)
        x, y = propagate_hidden_gaussian(m, c, .4)
        np.testing.assert_allclose(a, x, atol=1e-12)
        np.testing.assert_allclose(b, y, atol=1e-12)

    def test_exact_hidden_position_and_conditional_velocity(self):
        m = np.zeros((1, 4)); c = np.eye(4)[None]; c[0, 0, 2] = c[0, 2, 0] = .5
        (post, cov), logp = first_detection_moments(m, c, [[2., 0.]], np.zeros(4), 0, ConstantLaw())
        np.testing.assert_allclose(post[0, 0], [2., 0., 1., 0.], atol=1e-12)
        self.assertAlmostEqual(cov[0, 0, 2, 2], .75)
        np.testing.assert_array_equal(cov[0, 0, :2], np.zeros((2, 4)))
        self.assertAlmostEqual(logp[0, 0], np.log(.2) - np.log(2 * np.pi) - 2.)

    def test_constant_miss_preserves_full_gaussian_moments(self):
        m, c = prepare_prior(np.zeros(4)); c[0, 0, 2] = c[0, 2, 0] = .5
        miss, post, cov = missed_hidden_moments(m, c, np.zeros(4), 0, ConstantLaw())
        np.testing.assert_allclose(miss, [.8], atol=1e-14)
        np.testing.assert_allclose(post, m, atol=1e-12)
        np.testing.assert_allclose(cov, c, atol=1e-12)

    def test_sampled_hidden_kernel_matches_analytic_moments(self):
        rng = np.random.default_rng(92)
        m = np.array([[1., 2., .4, -.3]]); c = np.diag([.1, .2, .4, .5])[None]
        x = sample_hidden_states(m, c, 12000, rng)
        x = advance_hidden_samples(x, .31, rng)
        em, ec = propagate_hidden_gaussian(m, c, .31)
        np.testing.assert_allclose(x.mean(0), em[0], atol=.025)
        np.testing.assert_allclose(np.cov(x, rowvar=False), ec[0], atol=.025)

    def test_narrow_track_miss_integral_against_high_order_reference(self):
        h = _hypothesis(regime=1, position=(2., 1.8))
        h = h.fork(covariance=np.diag([.001, .001, .25, .25])[None])
        law = EndpointObservationLaw()
        mass, _, _ = law.missed_moments(h, np.zeros(4))
        ref, _, _ = missed_hidden_moments(h.mean, h.covariance, np.zeros(4), 1, law, order=61)
        self.assertLess(abs(mass[0] - ref[0]), 1e-5)


if __name__ == '__main__':
    if '--diagnostics' in sys.argv:
        print(json.dumps({**quadrature_diagnostics(), 'hidden_closure': hidden_closure_diagnostic(),
                          'retired_hidden_law_failure': hidden_closure_diagnostic(RetiredHiddenFirstReportLaw())}, indent=2))
    else:
        unittest.main()
