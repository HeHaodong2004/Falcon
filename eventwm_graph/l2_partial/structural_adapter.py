"""Typed uncertainty-preserving integration surface; no L2.5 or RL policy."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .state import QueryPrediction, QueryPredictionBatch


@dataclass(frozen=True)
class BranchSummary:
    kind: str
    probability_interval: Tuple[float,float]
    posterior_unresolved_mass: float
    posterior_expected_count_interval: Tuple[float,Optional[float]]
    supports_continuous_refinement: bool
    terminal: bool


@dataclass(frozen=True)
class QuerySummary:
    query_id: str
    collision_probability_interval: Tuple[float,float]
    psc_probability_interval: Tuple[float,float]
    evaluated_duration_mean_interval: Tuple[float,float]
    evaluated_progress_mean_interval: Tuple[float,float]
    evaluated_absolute_interval: Tuple[float,float]
    native_plan_completed_within_interval: bool
    remaining_cursor: Dict
    observation_options: Tuple[BranchSummary,...]
    model_version: str
    executor_version: str
    probability_scope: str


def summarize_prediction(prediction: QueryPrediction) -> QuerySummary:
    p=prediction
    # The final distribution bin explicitly denotes an unidentified remainder.
    u=float(p.duration_weights[-1])
    known=p.duration_weights[:-1]
    duration=float(known@p.duration_samples[:-1])
    progress=float(known@p.progress_samples[:-1])
    branches=[]
    for branch in p.observation_branches:
        b=branch.successor
        lower=float(sum(h.weight*(h.n+h.expected_hidden) for h in b.hypotheses))
        support=b.evidence.get('residual_count_support')
        upper=lower if b.unresolved_mass==0 else (None if not support else lower+b.unresolved_mass*max(support))
        interval=branch.observation.get('probability_interval')
        if interval is None:
            interval=(branch.probability,branch.probability)
        branches.append(BranchSummary(branch.observation['kind'],tuple(float(x) for x in interval),
            b.unresolved_mass,(lower,upper),getattr(b,'carrier',None) is None and b.unresolved_mass<1,b.terminal))
    return QuerySummary(p.query_id,p.collision_interval,p.psc_interval,
        (duration,duration+u*max(0.,p.evaluated_interval[1]-p.evaluated_interval[0])),
        (progress,progress+u*float(np.max(p.progress_samples,initial=0.))),
        p.evaluated_interval,p.query_complete,dict(p.remaining_query_cursor),tuple(branches),
        p.model_version,p.executor_version,
        p.diagnostics.get('probability_bound_scope','retained surrogate law; statistical calibration not guaranteed'))


def summarize_batch(batch: QueryPredictionBatch) -> List[QuerySummary]:
    return [summarize_prediction(p) for p in batch.predictions]
