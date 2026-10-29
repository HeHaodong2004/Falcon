"""World Model Sandbox Core (Layer 2.5, first component).

    sb = WorldModelSandbox.from_release()                       # Layer 2 (Stage 1 hwm_lite_e12 + Stage 2 v17_a)
    result = sb.evaluate(state, legal, [Candidate(q1, 'orca', 'native'), Candidate(q2, 'dwa', 'inertia'), ...])

Per decision (one legal ``DecisionState`` at time t + the executor's legal state at the same instant):

1. Stage 1 predicts ONE joint human bank (40 worlds x 3 s), shared by every candidate;
2. every candidate Query is executed in imagination by ITS planner + executor + dynamics family against one nominal
   future of that bank, on the frozen legal map (``imagine.py``) -> an imagined trajectory;
3. Stage 2 predicts, in one batched call, each trajectory's hypothetical map increment and complete hypothetical
   future partial graph; Layer 2 scores each trajectory against all bank worlds (collision / social risk);
4. the outcomes are returned as structured, hypothetical records (``CandidateOutcome`` / ``SandboxResult``) for the
   future Graph Compiler.

Legality: inputs are the legal decision state and the legal executor state (both built from the arrived
observations only); no simulator, navmesh, scene mesh, privileged Oracle information or future observation enters.
Predicted maps and future graphs are flagged hypothetical, are never merged into either legal state, and are rejected
by ``DecisionState`` as inputs of a later decision.
"""
from dataclasses import dataclass, field, asdict
import hashlib
import math
import time
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from eventwm_graph.layer2.state import CandidateTrajectory, DecisionState, CLOCK_TOLERANCE
from eventwm_graph.querygen.dynamics import STEP_SECONDS
from .imagine import NominalHumans, imagine, new_execution, frozen_map, failed_rollout, dynamics as resolve_dynamics
from . import accel

VERSION = 'wm_sandbox_core_1'
DECISION_INTERVAL = 16          # querygen OracleConfig.horizon: steps per movement decision (0.768 s)


class SandboxError(ValueError):
    """The decision context or a candidate violates the sandbox contract."""


@dataclass(frozen=True)
class Candidate:
    """One Query to imagine, with ITS executing planner and dynamics family (heterogeneous per candidate).

    ``query``: a querygen ``Query`` (any kind the executor accepts) or None with ``execution`` set (continue the
    running execution: a deep copy is imagined).  ``horizon``: imagined steps (None = the configuration's rule).
    ``level``: the robot's current speed level (stateful dynamics families; 0 for native / instant robots).
    """
    query: Any = None
    planner: str = 'orca'
    dynamics: Any = 'native'
    horizon: Optional[int] = None
    execution: Any = None
    level: int = 0
    planner_kw: Optional[dict] = None
    query_id: Optional[str] = None

    def describe(self):
        q = self.query if self.query is not None else getattr(self.execution, 'query', None)
        d = q.describe() if q is not None else {}
        return dict(d, planner=self.planner, dynamics=getattr(resolve_dynamics(self.dynamics), 'name', str(self.dynamics)),
                    continued=self.execution is not None and self.query is None, query_id=self.query_id)


@dataclass
class SandboxConfig:
    horizon: Any = 'interval'        # 'interval' = querygen horizon_for (16 steps per movement decision, query
                                     # durations for wait / creep, 8 for STOP); or an int number of steps
    interval: int = DECISION_INTERVAL
    lookahead: int = 0               # extra imagined steps continuing the query after its interval
    max_steps: int = 120             # hard cap on imagined steps per candidate
    nominal: str = 'mean'            # nominal human future the imagined planner perceives (NominalHumans modes)
    worlds: Optional[int] = None     # Stage 1 worlds (None = release default, 40)
    seed: Optional[int] = None       # Stage 1 sampler seed (None = release default)
    materialize: bool = True         # Stage 2 complete hypothetical future graphs
    maps: bool = True                # run Stage 2 (False: imagined trajectories + human risk only)
    memoize: bool = True             # exact per-decision caches on the frozen map (accel.py; bit-identical rollouts)
    prefetch_maps: bool = False      # warm Stage 2's legal-map context (graph compile, range voxel map) on a worker
                                     # thread while Stage 1 and the rollouts run (Layer 2 outputs are cache-independent)
    continuation: int = 0            # Layer 3 option: imagined steps after each candidate's interval under the default
                                     # policy (ImaginedRollout.cont; not given to Layer 2).  0 = off (the released core)

    def steps_for(self, cand):
        if cand.query is None and cand.execution is None:
            raise SandboxError('a candidate needs a query or a running execution')
        if cand.horizon is not None:
            n = int(cand.horizon)
        elif self.horizon == 'interval':
            from eventwm_graph.querygen.oracle import horizon_for
            q = cand.query if cand.query is not None else cand.execution.query
            n = horizon_for(q, _Interval(self.interval))+int(self.lookahead)
        else:
            n = int(self.horizon)+int(self.lookahead)
        return max(0, min(n, int(self.max_steps)))


RECORD_SCHEMA = 'wm_sandbox_record_1'


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _num(v):
    """Float, or NaN for an undefined statistic (e.g. minimum distance when no person exists in any world)."""
    return float('nan') if v is None else float(v)


class _Interval:
    def __init__(self, horizon):
        self.horizon = horizon


@dataclass(frozen=True)
class CandidateOutcome:
    """Everything the sandbox imagined for one candidate (HYPOTHETICAL)."""
    index: int
    candidate: dict                  # descriptor: query kind / label / via / meta, planner, dynamics
    rollout: Any                     # imagine.ImaginedRollout
    map: Optional[dict]              # Stage 2 increment (predict_segments contract) or None
    humans: Optional[dict]           # Layer 2 joint reductions against the shared bank
    hypothetical: bool = True

    @property
    def trajectory(self):
        return self.rollout.trajectory()

    @property
    def future_graph(self):
        return None if self.map is None else self.map.get('future_partial_graph')

    def summary(self):
        r = self.rollout
        out = dict(index=self.index, label=self.candidate.get('label'), kind=self.candidate.get('kind'),
                   planner=self.candidate['planner'], dynamics=self.candidate['dynamics'], steps=r.steps,
                   ended_by=r.ended_by, stop=r.stop, status=r.status, reason=r.reason, goal_distance=r.goal_distance,
                   progress=r.progress, path_length=r.path_length, blocked_steps=r.blocked_steps,
                   nominal_clearance=r.nominal_clearance, error=r.error)
        if self.humans is not None:
            h = self.humans
            out.update(p_collision=float(h['p_collision']), p_social=float(h['p_social']),
                       expected_social_seconds=float(h['expected_social_duration']),
                       min_distance_q10=_num(h['minimum_quantiles'][0]), min_distance_median=_num(h['minimum_quantiles'][1]),
                       risk_steps=int(h['steps']), risk_horizon_truncated=bool(h['horizon_truncated']))
        if self.map is not None:
            s = self.map['states']; p = self.map['probabilities'].astype(np.float64)
            out.update(candidate_cells=int(len(s)), predicted_free=int((s == 1).sum()), predicted_blocked=int((s == 2).sum()),
                       expected_free_uncalibrated=float(p[:, 1].sum()), expected_blocked_uncalibrated=float(p[:, 2].sum()))
        return out


@dataclass(frozen=True)
class SandboxResult:
    timestamp: float                 # decision time t
    robot_pose: np.ndarray
    stream: str
    humans: Any                      # shared Stage 1 bank (HumanWorlds; times relative to t)
    nominal: Any                     # NominalHumans the imagined planners perceived (shared)
    candidates: Tuple[CandidateOutcome, ...]
    layer2: Any                      # Layer2Prediction (None when no candidate)
    profile: dict
    provenance: dict
    hypothetical: bool = True

    def summaries(self):
        return [c.summary() for c in self.candidates]

    def record(self, maps=False):
        """JSON-ready structured record (schema ``RECORD_SCHEMA``) for logging and the future Graph Compiler.

        Per candidate: descriptor, imagined trajectory [t, x, y, z, yaw] rows and commands, outcome summary, human
        risk; with ``maps`` also the predicted increment (cells, states, probabilities).  The shared human futures are
        summarised by their track keys and the bank clock.  Everything is flagged hypothetical."""
        cands = []
        for c in self.candidates:
            r = c.rollout
            item = dict(index=c.index, candidate=_jsonable(c.candidate), summary=_jsonable(c.summary()),
                        trajectory=r.trajectory().round(6).tolist(), commands=[list(x) if isinstance(x, tuple) else x for x in r.commands])
            if c.humans is not None:
                item['risk'] = {k: _jsonable(c.humans[k]) for k in ('p_collision', 'p_social', 'expected_social_duration',
                                                                     'minimum_quantiles', 'steps', 'horizon_truncated') if k in c.humans}
            if maps and c.map is not None:
                item['map'] = dict(cells=c.map['candidate_cells'][c.map['states'] > 0].tolist(),
                                   states=c.map['states'][c.map['states'] > 0].tolist(),
                                   probabilities=c.map['probabilities'][c.map['states'] > 0].round(4).tolist())
            cands.append(item)
        return dict(schema=RECORD_SCHEMA, hypothetical=True, timestamp=self.timestamp, robot_pose=self.robot_pose.tolist(),
                    stream=self.stream, human_tracks=list(getattr(self.humans, 'track_keys', ())),
                    nominal_tracks=list(self.nominal.ids), candidates=cands, profile=_jsonable(self.profile),
                    provenance=_jsonable(self.provenance))

    def human_futures(self, steps=None):
        """The shared bank on the planner clock (Layer2Prediction.human_futures)."""
        if self.layer2 is None:
            raise SandboxError('no Layer 2 prediction (no candidates)')
        return self.layer2.human_futures(steps)


class WorldModelSandbox:
    """Batched imagined evaluation of K candidate Queries at one legal decision state."""

    def __init__(self, layer2, config=None):
        self.l2 = layer2
        self.config = config or SandboxConfig()
        self.last_profile = {}
        self.last_cache = {}
        self._prefetch = None
        accel.install()

    @classmethod
    def from_release(cls, config=None, **layer2_kw):
        from eventwm_graph.layer2.model import Layer2WorldModel
        return cls(Layer2WorldModel.from_release(**layer2_kw), config)

    def close(self):
        if self._prefetch is not None:
            self._prefetch.shutdown(wait=True)
            self._prefetch = None
        self.l2.close()

    def reset(self, stream=None):
        self.l2.reset(stream)

    # ------------------------------------------------------------------ checks
    @staticmethod
    def check_context(state, legal):
        if not isinstance(state, DecisionState):
            raise SandboxError('evaluate needs a layer2 DecisionState')
        if abs(float(legal.time)-state.timestamp) > CLOCK_TOLERANCE:
            raise SandboxError('executor legal state (t=%.6f) and decision state (t=%.6f) differ in time'
                               % (float(legal.time), state.timestamp))
        if not np.allclose(np.asarray(legal.pose, np.float64), state.robot_pose, rtol=0, atol=1e-7):
            raise SandboxError('executor legal state and decision state differ in the robot pose')
        if getattr(legal.map, 'timestamp', None) is not None and abs(float(legal.map.timestamp)-state.timestamp) > CLOCK_TOLERANCE:
            raise SandboxError('executor legal map is not at the decision time')

    def _execution(self, cand):
        dyn = resolve_dynamics(cand.dynamics)
        if cand.query is None:
            if cand.execution is None:
                raise SandboxError('a candidate needs a query or a running execution')
            ex = cand.execution.copy()
            if ex.status != 'running':
                raise SandboxError('cannot continue a finished execution')
            if ex.dyn.name != dyn.name:
                raise SandboxError('a continued execution keeps its dynamics family')
            ex.decision_boundary()
            return ex, dyn
        if getattr(cand.query, 'kind', None) == 'continue':
            raise SandboxError("'continue' candidates pass the running execution (query=None, execution=...)")
        try:
            return new_execution(cand.query, cand.planner, dyn, cand.planner_kw), dyn
        except Exception as exc:                        # this candidate's executor cannot start: recorded, not raised
            return exc, dyn

    # ------------------------------------------------------------------ evaluate
    def imagine_all(self, legal, candidates, humans, memoize=None):
        """Imagined rollouts of every candidate (shared nominal humans, one frozen map alias, exact per-decision
        caches). Failures stay per candidate."""
        memo = self.config.memoize if memoize is None else memoize
        gmap = frozen_map(legal.map)
        cache = accel.FrozenMapCache() if memo else None
        out = []
        with accel.active(cache):
            for c in candidates:
                steps = self.config.steps_for(c)
                ex, dyn = self._execution(c)            # contract violations of the call raise SandboxError
                if isinstance(ex, Exception):
                    out.append(failed_rollout(legal, '%s: %s' % (type(ex).__name__, ex)))
                    continue
                cq = None
                if self.config.continuation:
                    from .episode import default_query
                    cq = self._cont_query = getattr(self, '_cont_query', None) or default_query()
                out.append(imagine(legal, ex, dyn, humans, steps, level=int(c.level), gmap=gmap,
                                   continuation=self.config.continuation, cont_query=cq, planner=c.planner))
        self.last_cache = cache.summary() if cache is not None else {}
        return out

    def _layer2_predict(self, state, trajs):
        """Stage 2 (one batched call) + the joint human reductions; without maps only the human side."""
        cfg = self.config
        if cfg.maps:
            return self.l2.predict(state, trajs, materialize=cfg.materialize, worlds=cfg.worlds, seed=cfg.seed)
        return _humans_only(self.l2, state, trajs, cfg)

    def evaluate(self, state, legal, candidates: Sequence[Candidate]):
        cfg = self.config
        tick = time.perf_counter()
        self.check_context(state, legal)
        cands = list(candidates)
        steps = [cfg.steps_for(c) for c in cands]
        warm = None
        if cfg.maps and cfg.prefetch_maps and cands:
            if self._prefetch is None:
                from concurrent.futures import ThreadPoolExecutor
                self._prefetch = ThreadPoolExecutor(1, thread_name_prefix='sandbox-stage2-prefetch')
            stay = CandidateTrajectory(state.start_pose[None, :].copy(), query_id='__prefetch__')
            warm = self._prefetch.submit(self.l2.predict_maps, state, [stay], False)
        # 1. shared Stage 1 bank (cached by Layer 2 for this state object: the batched predict below reuses it)
        b0 = time.perf_counter()
        bank = self.l2.predict_humans(state, worlds=cfg.worlds, seed=cfg.seed)
        b1 = time.perf_counter()
        humans = NominalHumans(legal.tracks, legal.time, max(steps, default=0)+int(cfg.continuation), bank=bank,
                               mode=cfg.nominal)
        b2 = time.perf_counter()
        # 2. imagined executions
        rollouts = self.imagine_all(legal, cands, humans)
        b3 = time.perf_counter()
        # 3. Stage 2 (one batched call) + joint human reductions
        trajs = [CandidateTrajectory(r.trajectory(), query_id=c.query_id, metadata=dict(index=i))
                 for i, (c, r) in enumerate(zip(cands, rollouts))]
        pred = None
        w0 = time.perf_counter()
        if warm is not None:
            warm.result()                       # the warm-up call's own outputs are discarded
        wait_ms = (time.perf_counter()-w0)*1000
        if trajs:
            pred = self._layer2_predict(state, trajs)
        b4 = time.perf_counter()
        outs = []
        for i, (c, r) in enumerate(zip(cands, rollouts)):
            cp = pred.candidates[i] if pred is not None else None
            outs.append(CandidateOutcome(i, c.describe(), r, cp.map if (cp is not None and cfg.maps) else None,
                                         cp.humans if cp is not None else None))
        done = time.perf_counter()
        l2p = getattr(self.l2, 'last_profile', {}) if pred is not None else {}
        self.last_profile = dict(candidates=len(cands), total_ms=(done-tick)*1000, stage1_ms=(b1-b0)*1000,
                                 nominal_ms=(b2-b1)*1000, imagine_ms=(b3-b2)*1000, layer2_ms=(b4-b3)*1000,
                                 assemble_ms=(done-b4)*1000, imagined_steps=int(sum(r.steps for r in rollouts)),
                                 prefetch_wait_ms=wait_ms if warm is not None else None,
                                 stage2_ms=l2p.get('stage2_ms'), interaction_ms=l2p.get('interaction_ms'),
                                 per_candidate_imagine_ms=[r.planner_ms for r in rollouts])
        prov = dict(version=VERSION, hypothetical=True, privileged_runtime_inputs=False,
                    decision_state=state.identity(), layer2=None if pred is None else pred.provenance,
                    config=asdict(cfg), nominal_humans=dict(mode=humans.mode, tracks=list(humans.ids)),
                    legal_map='frozen at the decision time (no predicted cells)',
                    kinematics='nominal family kinematics (Falcon base-velocity integration, no navmesh)')
        return SandboxResult(state.timestamp, state.robot_pose, state.stream, bank, humans, tuple(outs), pred,
                             self.last_profile, prov)


def _humans_only(l2, state, trajs, cfg):
    """Layer 2 human-side prediction without Stage 2 (maps=False): same reductions, no map increments."""
    from eventwm_graph.layer2.model import robot_steps, CandidatePrediction, Layer2Prediction
    from eventwm_graph.layer2.state import normalise_candidates
    cands = normalise_candidates(trajs, state)
    paths = [c[0] for c in cands]
    step = l2.humans.step_seconds
    robots, lengths, rel, truncated = robot_steps(paths, state.timestamp, step, l2.humans.max_steps)
    bank, human_ms, profile, red, interaction_ms = l2._humans_side(state, cfg.worlds, cfg.seed, robots, lengths, True)
    out = []
    for k, (p, qid, meta) in enumerate(cands):
        h = {key: value[k] for key, value in red.items()}
        h.update(steps=int(lengths[k]), horizon_seconds=float(lengths[k]*step), horizon_truncated=bool(truncated[k]))
        out.append(CandidatePrediction(k, qid, p, {}, h, meta))
    l2.last_profile = dict(candidates=len(paths), stage1_ms=human_ms, stage2_ms=0., interaction_ms=interaction_ms)
    return Layer2Prediction(state.timestamp, state.robot_pose, state.stream, bank, tuple(out), rel, l2.last_profile,
                            dict(hypothetical=True, stage2=None))
