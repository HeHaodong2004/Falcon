"""Explicit normalized assumed-density channel alongside the preserved bounds.

Beam projection is a declared approximation of a posterior, not evidence that
the omitted physical states have disappeared. The ledger measures each local
retained fraction; its product is a support diagnostic, NOT a global TV bound.
Forecast worlds are iid draws from ALL represented count/association support.
"""
import copy
from dataclasses import replace
import math
import time
import json
from pathlib import Path
import numpy as np

from eventwm_graph.l2_partial.structural_model import StructuralGraphBeliefWorldModel, _serialized, _digest
from eventwm_graph.l2_partial.structural_state import immutable_array, freeze_hypothesis, seal_future
from eventwm_graph.l2_partial.state import SharedFuture
from eventwm_graph.l2_partial.model import GraphBeliefWorldModel
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState


APPROXIMATION = 'normalized-beam-Gaussian-GH-1; iid-full-count-and-intention-shared-worlds-2'


class NormalizedGraphWorldModel(StructuralGraphBeliefWorldModel):
    """Persistent population model; point probabilities are explicitly approximate."""
    def __init__(self, *args, **kwargs):
        if kwargs.get('residual_policy', 'bound') != 'bound':
            raise ValueError('legacy residual likelihood is not a normalized physical law')
        super().__init__(*args, **kwargs)
        self.model_version = 'l2-normalized-1-' + _digest({'base': self.model_version, 'law': APPROXIMATION})[:16]

    def history_filter(self):
        # Same legal, network-independent history estimator in training/inference.
        # Network intentions are installed once after replay, then persist.
        return NormalizedGraphWorldModel(kind='structured', max_hypotheses=self.max_hypotheses,
            population_prior=self.population_prior, observation_law=self.observation_law,
            maneuver_rate=self.maneuver_rate, reset_std=self.reset_std,
            maneuver_retention=self.maneuver_retention, optimized=self.optimized)

    def _initialize_history(self, legal_history, observed_graph):
        if not legal_history:
            raise ValueError('empty legal history')
        self._initial_legal_pose=np.asarray(legal_history[0]['robot_pose'],float)
        b=GraphBeliefWorldModel.initialize(self,legal_history,observed_graph)
        return StructuralBeliefState.wrap(b).fork(model_version=self.model_version,
            evidence={**b.evidence,'belief_role':'actual_continuous'})

    @_serialized
    def initialize(self,legal_history,observed_graph):
        # Forecast sampling can draw categorical intentions directly. Expanding
        # and pruning joint modes before sampling needlessly changes marginals.
        return self.prepare(self._initialize_history(legal_history,observed_graph)).validate()

    def _project(self, belief, operation):
        mass = math.fsum(h.weight for h in belief.hypotheses)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError('normalized approximation has zero represented support; no fabricated recovery')
        dropped = max(0., 1.-mass)
        ledger = dict(belief.evidence.get('normalized_approximation', {}))
        oldlog = float(ledger.get('log_retention_product', 0.))
        count = int(ledger.get('projection_count', 0))
        if dropped > 1e-14:
            oldlog += math.log(mass); count += 1
        diag = dict(version=APPROXIMATION, probability_status='NORMALIZED_APPROXIMATE_NOT_A_BOUND',
            operation=operation, retained_fraction=mass, omitted_fraction=dropped,
            log_retention_product=oldlog, retention_product=math.exp(oldlog),
            projection_count=count, maximum_local_omission=max(dropped, float(ledger.get('maximum_local_omission', 0.))),
            interpretation='local beam support diagnostic; no bound on accumulated projection or likelihood error')
        b = belief.fork(hypotheses=[h.fork(weight=float(h.weight/mass)) for h in belief.hypotheses],
            unresolved_mass=0., evidence={**belief.evidence, 'normalized_approximation': diag,
            'probability_certificate_scope':'NONE: normalized assumed-density approximation',
            'posterior_status':'NORMALIZED_APPROXIMATE'})
        if getattr(b, 'carrier', None) is not None:
            c = b.carrier
            b = b.fork(carrier=replace(c, weights=immutable_array(c.weights/mass), unresolved_mass=0.))
        return b

    @_serialized
    def propagate(self, belief, timestamp):
        if belief.terminal:
            return super().propagate(belief, timestamp)
        return self._project(super().propagate(belief, timestamp), 'propagate')

    @_serialized
    def observe(self, belief, actual_observation, actual_execution_trace=None):
        if belief.terminal:
            return super().observe(belief, actual_observation, actual_execution_trace)
        if (belief.evidence.get('external_event_exact_refinement_unsupported') and
                abs(float(actual_observation['timestamp'])-float(belief.evidence.get('external_event_timestamp',belief.timestamp)))<1e-8 and
                actual_observation.get('human_sensor_available',True) and actual_observation.get('observation_valid',True)):
            from eventwm_graph.l2_partial.structural_budget import UnsupportedConditioning
            raise UnsupportedConditioning('same-endpoint exact refinement of a region/masked event requires a joint emission bridge')
        return self._project(super().observe(belief, actual_observation, actual_execution_trace), 'observe')

    @_serialized
    def predict_shared(self, belief, horizon, num_worlds=16, seed=0, times=None):
        self._check_belief(belief)
        if belief.unresolved_mass!=0. or abs(math.fsum(h.weight for h in belief.hypotheses)-1.)>1e-10:
            raise ValueError('normalized forecasting requires unit represented mass and no unresolved bound mass; explicit channel conversion required')
        if (not math.isfinite(horizon) or not 0 <= horizon <= 30 or isinstance(num_worlds, bool)
                or int(num_worlds) != num_worlds or not 1 <= num_worlds <= 256):
            raise ValueError('invalid forecast budget')
        if getattr(belief, 'carrier', None) is not None and not belief.terminal:
            return super().predict_shared(belief, horizon, num_worlds, seed, times)
        started = time.perf_counter()
        b = self.prepare(belief)
        rel = np.asarray(times if times is not None else np.unique(np.r_[0., np.arange(1,int(horizon/.048)+1)*.048,horizon]),float)
        if (rel.ndim != 1 or not len(rel) or not np.isfinite(rel).all() or rel[0] != 0
                or np.any(np.diff(rel) <= 0) or abs(rel[-1]-horizon)>1e-8):
            raise ValueError('invalid forecast clock')
        if belief.terminal:rel=np.array([0.])
        rng = np.random.default_rng(seed)
        weights = np.array([h.weight for h in b.hypotheses]); weights /= weights.sum()
        indices = rng.choice(len(weights), size=int(num_worlds), p=weights)
        worlds = []
        for index in indices:
            h = b.hypotheses[index]
            if h.intent is None:
                h=h.fork(intent=h.mean[:,2:].copy(),diffusion=np.zeros(h.n),mode=np.zeros(h.n,int))
            mean = self.sample_initial(h, rng)
            if h.intent.ndim == 3:
                modes = (np.cumsum(h.mode,axis=1)<rng.random()).sum(1).clip(0,2)
                ii=np.arange(h.n); intent=h.intent[ii,modes]; diffusion=h.diffusion[ii,modes]
            else: modes, intent, diffusion = h.mode, h.intent, h.diffusion
            count = int(rng.choice(len(h.hidden_pmf),p=h.hidden_pmf))
            pmf = np.zeros_like(h.hidden_pmf); pmf[count]=1.
            worlds.append(h.fork(weight=1./num_worlds, mean=mean, covariance=np.zeros_like(h.covariance),
                intent=intent.copy(),diffusion=diffusion.copy(),mode=modes.copy(),hidden_pmf=pmf))
        positions, velocities, worlds = self.sample_maneuver_paths(worlds,rel,b.timestamp,rng,0)
        f = SharedFuture(b.timestamp+rel, positions, velocities, np.full(num_worlds,1./num_worlds),
            [freeze_hypothesis(h) for h in worlds], b, 0., seed, belief.fingerprint())
        f.continuous_source = belief
        f.rng_state = copy.deepcopy(rng.bit_generator.state)
        f.generation_source_fingerprint = f.source.fingerprint()
        f.approximation_status = APPROXIMATION
        f.world_source_indices = tuple(int(i) for i in indices)
        self.timings['forecast'] = (time.perf_counter()-started)*1000
        return seal_future(f)

    @_serialized
    def evaluate_queries(self,*args,**kwargs):
        result=super().evaluate_queries(*args,**kwargs)
        for p in result.predictions:
            p.diagnostics={**p.diagnostics,
                'risk_semantics':'MONTE_CARLO_POINT_APPROXIMATION; inherited fixed-follower hazard surrogate, not refitted/calibrated here',
                'probability_bound_scope':'NONE: legacy interval fields contain finite empirical point masses, NOT confidence or conservative bounds',
                'sampling_law':APPROXIMATION,'native_event_acceptance':'SECONDARY_UNVALIDATED_AFTER_MOTION_LAW_CHANGE'}
            for branch in p.observation_branches:
                branch.observation={**branch.observation,
                    'probability_semantics':'normalized finite empirical conditional measure; Monte Carlo approximation of continuous law',
                    'probability_interval_status':'LEGACY_FIELD_POINT_MASS_NOT_A_BOUND'}
        return result


class KinematicGraphWorldModel(NormalizedGraphWorldModel):
    """Frozen CV diffusion comparator; same legal history/population estimator."""
    def __init__(self,*args,cv_diffusion=.64,**kwargs):
        self.cv_diffusion=float(cv_diffusion)
        if not math.isfinite(self.cv_diffusion) or self.cv_diffusion<0:raise ValueError('invalid CV diffusion')
        super().__init__(*args,**kwargs)
        if self.kind not in ('memory_cv','visible_cv'):raise ValueError('CV diffusion comparator requires a kinematic kind')
        self.model_version+='-cv-'+_digest(self.cv_diffusion)[:8]

    def prepare(self,belief):
        b=super().prepare(belief)
        if b.terminal:return b
        return b.fork(hypotheses=[h.fork(diffusion=np.full(h.n,self.cv_diffusion)) for h in b.hypotheses])


def load_model(checkpoint=None, population=None, observation=None, motion=None, kind='structured',cv_diffusion=None):
    from eventwm_graph.l2_partial.neural import LearnedDynamics
    from eventwm_graph.l2_partial.queries import NativeEventModel
    native_path=Path('docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints/native_events_v5_scene.json')
    event=NativeEventModel(json.loads(native_path.read_text())['coefficients'])
    cls=NormalizedGraphWorldModel if cv_diffusion is None else KinematicGraphWorldModel
    extra={} if cv_diffusion is None else {'cv_diffusion':cv_diffusion}
    return cls(kind=kind,
        dynamics=LearnedDynamics.load(checkpoint) if checkpoint else None,
        population_prior=population, observation_checkpoint=observation,event_model=event,
        **extra,**(motion or {}))
