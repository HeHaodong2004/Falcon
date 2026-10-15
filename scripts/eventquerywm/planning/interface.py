"""Legal candidate selection. Outcomes have no place in this interface."""
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence
import math
import numpy as np


@dataclass(frozen=True)
class Candidate:
    query_id: str
    path_id: str
    kind: str                 # WAIT, SLOW, MOVE; RECOVERY when implemented
    nominal_time: float
    movement: float
    trajectory: tuple         # nominal [time,x,y,z,yaw], never executed truth
    static_feasible: bool
    wait_duration: float = 0.
    speed_scale: float = 1.
    controls: tuple = ()       # optional native commands from canonical compiler
    control_dt: float = .05

    def __post_init__(self):
        a = np.asarray(self.trajectory, dtype=float)
        if self.kind not in ('WAIT','SLOW','MOVE','RECOVERY') or not isinstance(self.trajectory,tuple) or any(not isinstance(row,tuple) for row in self.trajectory):
            raise ValueError('Expected immutable nominal candidate and declared kind')
        if (a.ndim != 2 or a.shape[1] != 5 or len(a) < 2 or not np.isfinite(a).all()
                or a[0,0] != 0 or np.any(np.diff(a[:,0]) <= 0)
                or not math.isclose(a[-1,0], self.nominal_time, abs_tol=1e-5)
                or not np.isfinite([self.nominal_time,self.movement,self.wait_duration,self.speed_scale]).all()
                or min(self.movement,self.wait_duration,self.speed_scale) < 0):
            raise ValueError('Invalid nominal candidate geometry/clock')
        if self.controls:
            c=np.asarray(self.controls,dtype=float)
            if (not isinstance(self.controls,tuple) or any(not isinstance(row,tuple) for row in self.controls)
                    or c.shape!=(len(a)-1,2) or not np.isfinite(c).all() or self.control_dt<=0
                    or not np.allclose(np.diff(a[:,0]),self.control_dt,atol=1e-6)):
                raise ValueError('Native command/nominal clock mismatch')


@dataclass(frozen=True)
class Risk:
    collision: Optional[float] = None
    social_success: Optional[float] = None
    safe_success: Optional[float] = None
    exposure: Optional[float] = None

    def __post_init__(self):
        for p in (self.collision,self.social_success,self.safe_success):
            if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
                raise ValueError('Invalid probability')
        if self.collision is not None and self.social_success is not None:
            if self.collision+self.social_success > 1+1e-6:
                raise ValueError('Exclusive outcome probabilities disagree')
        if all(p is not None for p in (self.collision,self.social_success,self.safe_success)):
            if not math.isclose(self.collision+self.social_success+self.safe_success,1,abs_tol=1e-6):
                raise ValueError('Outcome probabilities must normalize')
        if self.exposure is not None and (not math.isfinite(self.exposure) or self.exposure < 0):
            raise ValueError('Invalid ordinal exposure')


@dataclass(frozen=True)
class Preference:
    mode: str = 'cost'
    lambda_c: float = 10.
    lambda_s: float = 5.
    execution_segment_s: float = .25

    def __post_init__(self):
        if self.mode not in ('cost','fastest','collision','safe','exposure'):
            raise ValueError('Unknown decision mode')
        if (not np.isfinite([self.lambda_c,self.lambda_s,self.execution_segment_s]).all()
                or min(self.lambda_c,self.lambda_s) < 0 or self.execution_segment_s <= 0):
            raise ValueError('Invalid preference')


def decision_score(candidate, risk, preference):
    if preference.mode == 'fastest':return candidate.nominal_time
    if preference.mode == 'collision':
        if risk.collision is None:raise ValueError('Need physical probability')
        return risk.collision
    if preference.mode == 'safe':
        if risk.safe_success is None:raise ValueError('Need categorical safe probability')
        return -risk.safe_success
    if preference.mode == 'exposure':
        if risk.exposure is None:raise ValueError('Need ordinal exposure')
        return risk.exposure
    if risk.collision is None or (preference.lambda_s and risk.social_success is None):
        raise ValueError('Expected calibrated probabilities, not ordinal exposure')
    return candidate.nominal_time+preference.lambda_c*risk.collision+preference.lambda_s*(risk.social_success or 0.)


def select_candidate(candidates, risks, preference):
    if len(candidates) != len(risks):raise ValueError('Candidate/risk mismatch')
    eligible = [i for i,c in enumerate(candidates) if c.static_feasible]
    if not eligible:raise ValueError('No statically feasible candidate; caller must supply certified WAIT/recovery')
    # Declared deterministic tie handling, never tie-break using outcomes.
    i = min(eligible,key=lambda j:(decision_score(candidates[j],risks[j],preference),
                                  candidates[j].nominal_time,candidates[j].query_id))
    return candidates[i]


class CandidateGenerator(Protocol):
    def generate(self, legal_history, static_map, navigation_goal) -> Sequence[Candidate]: ...


class RiskEstimator(Protocol):
    def predict(self, legal_history, candidates) -> Sequence[Risk]: ...


class Planner:
    """Execute a prefix externally, receive arrived observations, then call again.

    Static-map provider certifies robot-footprint feasibility. The generator
    defines goal-directed paths; the risk model alone cannot make WAIT progress.
    """
    def __init__(self, generator, estimator, preference=Preference()):
        self.generator,self.estimator,self.preference = generator,estimator,preference

    def __call__(self, legal_history, static_map, navigation_goal):
        candidates = tuple(self.generator.generate(legal_history,static_map,navigation_goal))
        if len({c.query_id for c in candidates}) != len(candidates):raise ValueError('Duplicate candidate')
        risks = tuple(self.estimator.predict(legal_history,candidates))
        return select_candidate(candidates,risks,self.preference)

    def action_prefix(self, selected_query):
        """Nominal pose reference; executor must supply native controller semantics."""
        a=np.asarray(selected_query.trajectory,dtype=float)
        end=min(self.preference.execution_segment_s,selected_query.nominal_time)
        inside=a[a[:,0]<end]
        last=np.array([end]+[np.interp(end,a[:,0],a[:,k]) for k in range(1,4)]+
                      [np.interp(end,a[:,0],np.unwrap(a[:,4]))])
        return np.vstack([inside,last])

    def execution_controls(self, selected_query):
        if not selected_query.controls:raise ValueError('Executor adapter has not provided native commands')
        steps=max(1,math.ceil(self.preference.execution_segment_s/selected_query.control_dt-1e-9))
        return selected_query.controls[:steps]
