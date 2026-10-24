"""Versioned, uncertainty-preserving L2 interface and a small fixed consumer.

This adapter owns small immutable result buffers and borrows the sealed human
future and observed graph. It neither completes a map nor implements a planner.
Actual beliefs are never replaced by an imagined conditional successor.
"""
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Tuple
import hashlib
import time

import numpy as np

from .state import EXECUTOR_VERSION
from .structural_state import immutable_array, StructuralObservedGraph
from .structural_budget import ACTIVE_DEADLINE, Deadline, check_budget, InferenceBudgetExceeded
from .structural_queries import query_fingerprint

PACKET_VERSION = 'l2-planning-packet-1'
EXOGENEITY_VERSION = 'falcon-campaign-exogenous-human-futures-1'
PROBABILITY_SCOPE = ('local omitted-mass accounting under retained event/optical surrogate; '
                     'Gaussian projection, quadrature and native calibration error are not bounded')


class CandidateStatus(str, Enum):
    COMPLETE = 'NOMINAL_PLAN_COVERED'
    CENSORED = 'HORIZON_CENSORED'
    EXECUTOR_LIMIT = 'EXECUTOR_ACTION_LIMIT'
    UNSUPPORTED = 'UNSUPPORTED_OBSERVED_GEOMETRY'
    ABSORBED = 'TERMINAL_ABSORBED'


class BranchStatus(str, Enum):
    AVAILABLE = 'FINITE_CONDITIONAL_BRANCHES'
    DEFERRED = 'NOT_REQUESTED'
    NO_EXPOSURE = 'NO_NEW_OBSERVATION_EXPOSURE'


@dataclass(frozen=True)
class BeliefReference:
    fingerprint: str
    timestamp: float
    role: str
    model_version: str


@dataclass(frozen=True)
class SharedHumanFuture:
    """Sealed zero-copy numerical references, aligned across every candidate."""
    reference: str
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    weights: np.ndarray
    unresolved_mass: float
    seed: int
    exogeneity_version: str = EXOGENEITY_VERSION


@dataclass(frozen=True)
class Distribution:
    """Lower submeasure plus an unidentified remainder on an explicit range."""
    values: np.ndarray
    lower_weights: np.ndarray
    unresolved_mass: float
    unresolved_range: Tuple[float, float]
    units: str

    @property
    def mean_interval(self):
        represented = float(self.values @ self.lower_weights)
        return (represented + self.unresolved_mass*self.unresolved_range[0],
                represented + self.unresolved_mass*self.unresolved_range[1])


@dataclass(frozen=True)
class QueryGeometry:
    waypoints: Tuple[Tuple[float, float, float], ...]
    requested_final_yaw: float
    planned_poses: np.ndarray
    start_action_index: int
    evaluated_action_index: int
    evaluated_survival_endpoint: Tuple[float, float, float, float]
    planned_duration_s: float
    proposal_provenance: object


@dataclass(frozen=True)
class SuccessorHandle:
    """Borrowed read-only state; resolve only for supported model operations.

    Carrier tapes/graph storage are shared and immutable. The legacy belief
    object is borrowed, not deep-frozen; callers must not mutate it. Exact
    continuous observations go to the separately retained actual belief.
    """
    kind: str
    timestamp: float
    terminal: bool
    unresolved_mass: float
    can_continue_reference: bool
    can_select_new_query: bool
    continuous_refinement_supported: bool
    _belief: object = field(repr=False, compare=False)

    def resolve(self):
        return self._belief


@dataclass(frozen=True)
class CoarseObservation:
    kind: str
    lower_mass: float
    probability_interval: Tuple[float, float]
    evaluated_interval: Tuple[float, float]
    observation_law_version: str
    successor: SuccessorHandle
    certified_human_empty: bool = False
    static_graph_delta_supported: bool = False


@dataclass(frozen=True)
class QueryConsequence:
    query_id: str
    geometry: QueryGeometry
    evaluated_interval: Tuple[float, float]
    status: CandidateStatus
    nominal_plan_covered: bool
    remaining_cursor: Mapping
    collision_risk_interval: Tuple[float, float]
    social_risk_interval: Tuple[float, float]
    duration: Distribution
    route_progress: Distribution
    goal_progress: Optional[Distribution]
    termination_lower_masses: Mapping
    termination_intervals: Mapping
    first_event: Mapping
    observation_branches: Tuple[CoarseObservation, ...]
    branch_status: BranchStatus
    represented_mass: float
    unresolved_mass: float
    per_world_collision: np.ndarray
    per_world_social: np.ndarray
    executor_version: str
    approximation_status: str = PROBABILITY_SCOPE


@dataclass(frozen=True)
class PredictionCapabilities:
    actual_continuous_filter: str = 'exact centers; approximate Gaussian/GH population filter'
    imagined_successors: str = 'finite conditional joint paths at stored native endpoints'
    exact_imagined_continuous_refinement: bool = False
    future_static_revelation: bool = False
    native_event_calibration_certified: bool = False
    general_human_empty_certificates: bool = False
    execution_delay_recovery_model: bool = False
    future_social_l4_executor: bool = False
    stage1_mid_query_interruption: bool = False
    cross_query_native_event_random_coupling: bool = False


@dataclass(frozen=True)
class Stage0DecisionContext:
    """Small arrived topology, separately bound to L2's cspace support version."""
    graph_version: Tuple[str,int]
    stage0_revision: int
    timestamp: float
    global_nodes: Mapping
    global_edges: Tuple
    pointgoal: Tuple[float,float,float]
    unknown_continuation: str = 'unsupported static revelation; defer to learned value'


@dataclass(frozen=True)
class L2PlanningPacket:
    version: str
    belief_reference: BeliefReference
    graph_reference: StructuralObservedGraph
    graph_version: Tuple[str, int]
    absolute_timestamp: float
    model_version: str
    executor_version: str
    shared_future: SharedHumanFuture
    candidates: Tuple[QueryConsequence, ...]
    capabilities: PredictionCapabilities
    uncertainty_diagnostics: Mapping
    computation_status: str
    timings_ms: Mapping
    stage0_context: Optional[Stage0DecisionContext] = None


def _frozen(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _frozen(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    if isinstance(value, np.ndarray):
        return immutable_array(value)
    return value


def packet_from_batch(belief, queries, batch, *, pointgoal=None, stage0_snapshot=None):
    """Wrap a matching public evaluation; do not re-evaluate or redraw humans."""
    started = time.perf_counter()
    future = batch.shared_future
    if len(queries) != len(batch.predictions) or future.input_fingerprint != belief.fingerprint():
        raise ValueError('planning packet requires the evaluated belief and candidate rows')
    if future.source.model_version != belief.model_version:
        raise ValueError('planning packet model version mismatch')
    if not isinstance(belief.graph, StructuralObservedGraph):
        raise ValueError('planning packet requires an immutable structural observed graph')
    for array in (future.times, future.positions, future.velocities, future.weights):
        if array.flags.writeable:
            raise ValueError('planning packet requires a sealed shared future')
    goal = None if pointgoal is None else np.asarray(pointgoal, float)
    if goal is not None and (goal.shape != (3,) or not np.isfinite(goal).all()):
        raise ValueError('PointGoal must have three finite coordinates')
    candidates = []
    for query, p in zip(queries, batch.predictions):
        check_budget('packet:candidate')
        if query['query_id'] != p.query_id or p.executor_version != EXECUTOR_VERSION:
            raise ValueError('candidate identity/executor mismatch')
        allposes = np.asarray(query['compiled']['planned_poses'], float)
        if query_fingerprint(query) != p.diagnostics['query_geometry_fingerprint']:
            raise ValueError('query geometry changed after evaluation')
        end = int(p.remaining_query_cursor['action_index'])
        steps = int(round((p.evaluated_interval[1]-p.evaluated_interval[0])/.048))
        start = end-steps
        poses = allposes[start:end+1]
        geometry = QueryGeometry(tuple(tuple(map(float, v)) for v in query['waypoints']),
            float(query['final_yaw']), immutable_array(allposes), start, end,
            tuple(map(float, poses[-1])), (len(allposes)-1)*.048,
            _frozen(query.get('provenance',{})))
        supported = bool(p.diagnostics['static_support_checked'])
        status = (CandidateStatus.ABSORBED if belief.terminal else
                  CandidateStatus.UNSUPPORTED if not supported else
                  CandidateStatus.COMPLETE if p.query_complete else
                  CandidateStatus.EXECUTOR_LIMIT if end >= 120 else CandidateStatus.CENSORED)
        u = float(p.duration_weights[-1])
        weights = immutable_array(p.duration_weights[:-1])
        # The final legacy bin is epistemic, not a physical sample.
        duration = Distribution(immutable_array(p.duration_samples[:-1]), weights, u,
                                (0., float(p.evaluated_interval[1]-p.evaluated_interval[0])), 'seconds')
        travel = Distribution(immutable_array(p.progress_samples[:-1]), weights, u,
                              (0., float(np.max(p.progress_samples, initial=0.))), 'route meters')
        goal_dist = None
        if goal is not None:
            progress = np.linalg.norm(poses[0,:3]-goal)-np.linalg.norm(poses[:,:3]-goal, axis=1)
            values = np.r_[progress[1:], progress[-1]]
            goal_dist = Distribution(immutable_array(values), weights, u,
                                    (float(progress.min()), float(progress.max())), 'signed Euclidean PointGoal meters')
        branches = []
        for branch in p.observation_branches:
            kind = branch.observation['kind']
            if kind == 'unresolved_physical_outcome':
                continue  # epistemic accounting never becomes a physical successor
            b = branch.successor
            handle = SuccessorHandle(kind, b.timestamp, b.terminal, b.unresolved_mass,
                not b.terminal and status == CandidateStatus.CENSORED,
                not b.terminal and status in (CandidateStatus.COMPLETE,CandidateStatus.EXECUTOR_LIMIT,CandidateStatus.UNSUPPORTED),
                getattr(b, 'carrier', None) is None and not b.terminal, b)
            branches.append(CoarseObservation(kind, float(branch.probability),
                tuple(branch.observation.get('probability_interval', (branch.probability, branch.probability))),
                tuple(p.evaluated_interval), p.diagnostics.get('observation_process_version', 'unknown'), handle))
        deterministic = belief.terminal or not supported or steps == 0
        survivor_cause = ('query_complete' if p.query_complete else
                          'query_timeout' if status == CandidateStatus.EXECUTOR_LIMIT else 'horizon_censored')
        possible_causes = {'human_collision',survivor_cause}
        termination = {k: ((float(v), float(v)) if deterministic or k not in possible_causes else
                            (float(v), min(1., float(v)+u)))
                       for k, v in p.termination_probabilities.items() if k != 'unresolved'}
        branch_status = (BranchStatus.DEFERRED if not p.diagnostics['refined'] else
                         BranchStatus.NO_EXPOSURE if steps == 0 else BranchStatus.AVAILABLE)
        candidates.append(QueryConsequence(p.query_id, geometry, tuple(p.evaluated_interval), status,
            bool(p.query_complete), _frozen(dict(p.remaining_query_cursor)), tuple(p.collision_interval),
            tuple(p.psc_interval), duration, travel, goal_dist, _frozen({k:v for k,v in p.termination_probabilities.items() if k!='unresolved'}),
            _frozen(termination), _frozen(dict(p.time_to_conflict)), tuple(branches), branch_status,
            float(future.weights.sum()), float(future.unresolved_mass), p.world_collision_probability,
            p.world_social_probability, p.executor_version))
    clock_id = hashlib.blake2b(future.times.tobytes(),digest_size=8).hexdigest()
    shared = SharedHumanFuture(future.input_fingerprint+':'+str(future.seed)+':'+str(len(future.weights))+':'+clock_id,
        future.times, future.positions, future.velocities, future.weights, float(future.unresolved_mass), future.seed)
    diagnostics = {
        'actual_filter_unresolved_mass': belief.unresolved_mass,
        'forecast_unresolved_mass': future.unresolved_mass,
        'forecast_omitted_component_mass': max(0., future.unresolved_mass-belief.unresolved_mass),
        'filter_mass_accounting': belief.evidence.get('mass_accounting', {}),
        'residual_allocation': 'one coupled simplex across mutually exclusive physical branches',
        'probability_scope': PROBABILITY_SCOPE,
        'sensor_protocol': 'ideal_frustum_observed_cspace_dynamic_occlusion_1; predictive optical law is a surrogate',
        'goal_progress_semantics': 'Euclidean signed progress; no unknown-space geodesic or navigation success claim',
    }
    timings = {**batch.timings_ms, 'packet': (time.perf_counter()-started)*1000.}
    check_budget('packet:complete')
    packet = L2PlanningPacket(PACKET_VERSION,
        BeliefReference(future.input_fingerprint, belief.timestamp,
                        belief.evidence.get('belief_role','finite_conditional' if getattr(belief, 'carrier', None) else 'continuous_source'), belief.model_version),
        belief.graph, (belief.graph.namespace, belief.graph.version), belief.timestamp,
        belief.model_version, EXECUTOR_VERSION, shared, tuple(candidates), PredictionCapabilities(),
        _frozen(diagnostics), 'COMPLETE', _frozen(timings))
    if stage0_snapshot is not None:
        packet = replace(packet,stage0_context=bind_stage0_context(stage0_snapshot,packet,pointgoal=pointgoal))
    return packet


def bind_stage0_context(snapshot,packet,*,pointgoal):
    """Join only a matching arrived Stage 0 snapshot, never future graph deltas.

    This optional join scans the supplied support once. It is unnecessary for
    repeated L2 predictions when L2.5 already retains a version-bound context.
    """
    if pointgoal is None or abs(float(snapshot['timestamp'])-packet.absolute_timestamp)>1e-8:
        raise ValueError('Stage 0 context must belong to the arrived decision clock')
    cells={tuple(c['cell']):(int(c['state']),float(c['height'])) for c in snapshot['cells']}
    if (float(snapshot['resolution'])!=packet.graph_reference.resolution or
            cells!=packet.graph_reference.materialize()):
        raise ValueError('Stage 0 topology and L2 observed support disagree')
    goal=np.asarray(pointgoal,float)
    if goal.shape!=(3,) or not np.isfinite(goal).all():raise ValueError('finite PointGoal required')
    return Stage0DecisionContext(packet.graph_version,int(snapshot['revision']),packet.absolute_timestamp,
        _frozen(snapshot.get('global_nodes',{})),_frozen(snapshot.get('global_edges',[])),tuple(map(float,goal)))


def compile_observed_queries(snapshot,proposals,robot_pose):
    """Use the existing Stage 1 compiler on arrived support only."""
    from ..partial.map import PartialGraph
    from ..partial.query import compile_query
    graph=PartialGraph.restore(snapshot)
    result=[]
    for proposal in proposals:
        compiled=compile_query(graph,proposal,{'robot_pose':robot_pose},max_actions=180)
        if not compiled.get('ok'):
            raise ValueError('saved legal proposal no longer compiles: '+compiled.get('reason','unknown'))
        result.append({**proposal,'compiled':compiled})
    return result


@dataclass(frozen=True)
class PlanningCallResult:
    status: str
    actual_belief: object
    packet: Optional[L2PlanningPacket]
    pending_observation: bool
    elapsed_ms: float
    timeout_stage: Optional[str] = None


def online_planning_call(model, actual_belief, observation, queries, *, pointgoal=None,
                         horizon=3., num_worlds=16, seed=0, branch_indices=None,
                         budget_ms=None, actual_execution_trace=None):
    """Atomic actual-update + forecast + selected branches + packet publication."""
    if getattr(actual_belief, 'carrier', None) is not None or actual_belief.evidence.get('belief_role','').startswith('imagined'):
        raise ValueError('actual execution must retain its continuous belief, not an imagined carrier')
    start = time.perf_counter()
    outer = ACTIVE_DEADLINE.get()
    requested = None if budget_ms is None else Deadline.create(budget_ms, pending_observation=True)
    deadline = requested
    if outer is not None and (deadline is None or outer.start+outer.budget_ms/1000. <= deadline.start+deadline.budget_ms/1000.):
        deadline = outer
    token = ACTIVE_DEADLINE.set(deadline)
    acquired = False
    try:
        if deadline is None:
            model._instance_lock.acquire(); acquired=True
        else:
            remaining=max(0.,deadline.budget_ms/1000.-(time.perf_counter()-deadline.start))
            acquired=model._instance_lock.acquire(timeout=remaining)
            check_budget('planning:instance_lock')
        check_budget('planning:entry')
        model.timings = {}
        updated = model.observe(actual_belief, observation, actual_execution_trace)
        future = model.predict_shared(updated, horizon, num_worlds, seed)
        chosen = tuple(range(min(4,len(queries)))) if branch_indices is None else tuple(branch_indices)
        batch = model.evaluate_queries(updated, queries, future,
            budget_ms=1e12 if budget_ms is None else budget_ms, refined_query_indices=chosen)
        batch.timings_ms = {**model.timings, **batch.timings_ms}
        packet = packet_from_batch(updated, queries, batch, pointgoal=pointgoal)
        check_budget('planning:publish')
        return PlanningCallResult('COMPLETE', updated, packet, False, (time.perf_counter()-start)*1000.)
    except InferenceBudgetExceeded as exc:
        return PlanningCallResult('PENDING_OBSERVATION', actual_belief, None, True,
                                  (time.perf_counter()-start)*1000., exc.stage)
    finally:
        if acquired:model._instance_lock.release()
        ACTIVE_DEADLINE.reset(token)


FEATURE_NAMES = ('endpoint_x', 'endpoint_z', 'endpoint_yaw', 'duration_low', 'duration_high',
    'route_progress_low', 'route_progress_high', 'goal_progress_low', 'goal_progress_high',
    'contact_low', 'contact_high', 'social_low', 'social_high', 'represented_mass',
    'unresolved_mass', 'nominal_plan_covered', 'known_support', 'branches_computed')


@dataclass(frozen=True)
class GraphDecisionFeatures:
    graph_reference: StructuralObservedGraph
    candidate_ids: Tuple[str, ...]
    candidate_features: np.ndarray
    feature_names: Tuple[str, ...]
    decision_supported: np.ndarray
    shared_future_reference: str
    stage0_context: Optional[Stage0DecisionContext]


def graph_decision_features(packet):
    """Candidate rows attach to the existing graph; no new executable edges."""
    rows, mask = [], []
    for c in packet.candidates:
        g = c.geometry.evaluated_survival_endpoint
        goal = (0.,0.) if c.goal_progress is None else c.goal_progress.mean_interval
        known = c.status not in (CandidateStatus.UNSUPPORTED, CandidateStatus.ABSORBED)
        rows.append((g[0], g[2], g[3], *c.duration.mean_interval, *c.route_progress.mean_interval,
            *goal, *c.collision_risk_interval, *c.social_risk_interval, c.represented_mass,
            c.unresolved_mass, c.nominal_plan_covered, known, c.branch_status != BranchStatus.DEFERRED))
        mask.append(known and c.nominal_plan_covered)
    return GraphDecisionFeatures(packet.graph_reference, tuple(c.query_id for c in packet.candidates),
        immutable_array(np.asarray(rows,float).reshape(-1,len(FEATURE_NAMES))), FEATURE_NAMES,
        immutable_array(np.asarray(mask,bool)), packet.shared_future.reference,packet.stage0_context)


def pessimistic_scores(packet):
    """Frozen reference consumer; interval rule, not calibrated expected utility."""
    return np.asarray([(-np.inf if c.status != CandidateStatus.COMPLETE or c.goal_progress is None else
        c.goal_progress.mean_interval[0]-5*c.collision_risk_interval[1]
        -c.social_risk_interval[1]-.1*c.duration.mean_interval[1]) for c in packet.candidates])
