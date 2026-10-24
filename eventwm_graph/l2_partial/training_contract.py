"""Bounded train/inference compatibility, not a large-data training pipeline.

The differentiable loss is a teacher-forced ONE native interval position
likelihood under the same jump/OU/noise kernel as structural inference. It is
not an OU-only displacement fit or a claim of a long-horizon joint likelihood.
Event bounds are scoring diagnostics, never a proper point-probability target.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from .maneuver import reset_probability
from .probability import transition_matrices

MOTION_LOSS_VERSION = 'native-jump-ou-one-step-position-nll-1'
EVENT_SCORE_VERSION = 'native-competing-first-event-residual-bounds-1'
SHARD_VERSION = 'observation_limited_query_conditioned_root_1'


@dataclass(frozen=True)
class MotionContract:
    native_dt: float = .048
    relaxation: float = .7
    maneuver_rate: float = 2.
    jump_std: float = 2.
    retention: float = 1.
    independent_position_variance_rate: float = .0025
    version: str = MOTION_LOSS_VERSION


def native_step_position_nll(mean, covariance, targets, diffusion, logits, next_position,
                             *, start_timestamp, contract=MotionContract()):
    """Per-person loss; persistent intention is marginalized once within this step.

    Shapes: mean[N,4], covariance[N,4,4], target[N,K,2], diffusion/logits[N,K],
    next_position[N,2]. Teacher-forced starts must lie on the absolute native
    grid. Off-grid or multi-step fitting requires the retained interval jump
    state / persistent mode; those objectives are explicitly not implemented.
    """
    c = contract
    if (c.version != MOTION_LOSS_VERSION or c.native_dt != .048 or
            not math.isfinite(start_timestamp) or
            abs(start_timestamp/c.native_dt-round(start_timestamp/c.native_dt)) > 1e-7):
        raise ValueError('one-step loss requires the declared native-grid start')
    if (not 0 <= c.retention <= 1 or c.jump_std < 0 or c.relaxation < 0 or
            c.independent_position_variance_rate != .0025):
        raise ValueError('incompatible motion contract')
    n, k, _ = targets.shape
    if (mean.shape != (n,4) or covariance.shape != (n,4,4) or
            diffusion.shape != (n,k) or logits.shape != (n,k) or next_position.shape != (n,2)):
        raise ValueError('incompatible training batch shapes')
    if any(not bool(torch.isfinite(x).all()) for x in (mean,covariance,targets,diffusion,logits,next_position)) or bool((diffusion < 0).any()):
        raise ValueError('finite nonnegative diffusion and finite training state required')
    F, Q, a, e = transition_matrices(c.native_dt,c.relaxation)
    F = mean.new_tensor(F); Q = mean.new_tensor(Q)
    branches, densities = [], []
    p = reset_probability(c.maneuver_rate,c.native_dt)
    for jump, probability in ((False,1-p),(True,p)):
        if probability == 0:
            continue
        transform = torch.eye(4,dtype=mean.dtype,device=mean.device)
        if jump:
            transform[2,2] = transform[3,3] = c.retention
        m = mean @ transform.T
        P = transform @ covariance @ transform.T
        if jump:
            P = P + mean.new_tensor(np.diag([0.,0.,c.jump_std**2,c.jump_std**2]))
        mu = (m @ F.T)[:,None,:2] + (c.native_dt-a)*targets
        cov = (F @ P @ F.T)[:,None,:2,:2] + diffusion[...,None,None]*Q[:2,:2]
        if c.relaxation > 0:
            cov = cov + torch.eye(2,dtype=mean.dtype,device=mean.device)*(c.independent_position_variance_rate*c.native_dt)
        L = torch.linalg.cholesky(cov)
        delta = next_position[:,None,:]-mu
        solve = torch.cholesky_solve(delta[...,None],L)[...,0]
        logp = -math.log(2*math.pi)-torch.log(torch.diagonal(L,dim1=-2,dim2=-1)).sum(-1)-.5*(delta*solve).sum(-1)
        branches.append(math.log(probability)); densities.append(logp)
    joint = torch.stack(densities,dim=-1) + mean.new_tensor(branches)
    joint = torch.logsumexp(joint,dim=-1) + torch.log_softmax(logits,dim=-1)
    return -torch.logsumexp(joint,dim=-1)


def root_balanced_mean(losses, root_indices):
    """Composite marginal objective; siblings are not independent human futures."""
    roots = torch.unique(root_indices)
    if not len(roots):
        raise ValueError('empty training batch')
    return torch.stack([losses[root_indices == r].mean() for r in roots]).mean()


@dataclass(frozen=True)
class NativeEventTarget:
    planned_steps: int
    observed_steps: int
    first_event_index: object = None  # zero-based native endpoint
    cause: object = None
    censoring: str = 'complete'
    metric_version: str = 'falcon_psc_geodesic_fallback_and_mesh_contact_1'


def first_event_likelihood_interval(hazards, world_weights, unresolved_mass, target):
    """Likelihood bounds for mutually exclusive native termination causes.

    hazards[W,planned_steps,C] is a normalized per-endpoint competing hazard
    law, not independent collision/static/episode Bernoulli outcomes. Prefix
    survival is right-censored at observed_steps; no post-contact exposure.
    PSC is a separate first social event channel, not a termination cause.
    """
    h = np.asarray(hazards,float); w = np.asarray(world_weights,float); u = float(unresolved_mass)
    allowed_censoring={'complete','administrative','collision_stop','static_stop','episode_limit','environment_termination'}
    if (target.censoring not in allowed_censoring or
            any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) for v in (target.planned_steps,target.observed_steps))):
        raise ValueError('declared integer exposure and censoring vocabulary required')
    if (h.ndim != 3 or h.shape[:2] != (len(w),target.planned_steps) or
            not 0 <= target.observed_steps <= target.planned_steps or
            not np.isfinite(h).all() or np.any(h < 0) or np.any(h.sum(-1)>1) or
            not np.isfinite(w).all() or np.any(w<0) or not 0<=u<=1 or abs(w.sum()+u-1)>1e-9):
        raise ValueError('invalid competing-event exposure/support')
    if target.metric_version != 'falcon_psc_geodesic_fallback_and_mesh_contact_1':
        raise ValueError('unrecognized native metric version')
    survival = 1-h.sum(-1)
    if target.first_event_index is None:
        if target.cause is not None or target.censoring in ('collision_stop','static_stop') or (target.censoring == 'complete' and target.observed_steps != target.planned_steps):
            raise ValueError('incomplete followup requires explicit censoring')
        likelihood = np.prod(survival[:,:target.observed_steps],axis=1)
    else:
        j = target.first_event_index
        if (isinstance(j,bool) or not isinstance(j,(int,np.integer)) or
                not 0 <= j < target.observed_steps or isinstance(target.cause,bool) or not isinstance(target.cause,(int,np.integer)) or
                not 0 <= target.cause < h.shape[2] or target.observed_steps != j+1):
            raise ValueError('first-event target must stop at its event endpoint')
        likelihood = np.prod(survival[:,:j],axis=1)*h[:,j,target.cause]
    low = float(w@likelihood)
    return low, min(1.,low+u)


def negative_log_score_interval(probability_interval):
    """A diagnostic interval, not a proper point likelihood to optimize."""
    lo,hi = probability_interval
    if not 0<=lo<=hi<=1:
        raise ValueError('invalid probability interval')
    return (-math.log(hi) if hi else math.inf, -math.log(lo) if lo else math.inf)


def require_point_training_law(*, residual_semantics, normalized_latent_law_version=None):
    """Prevent accidental reuse of legacy .005 residual-hazard fitting."""
    if residual_semantics != 'fully_specified_normalized' or not normalized_latent_law_version:
        raise ValueError('structural risk intervals/upper endpoints are not point training probabilities; declare a normalized train-time latent law')
    return normalized_latent_law_version


def iter_versioned_root_shards(manifest, *, split):
    """Read-only manifest iterator; no payload/corpus preprocessing or truth reads."""
    if split not in ('train','dev'):
        raise ValueError('bounded smoke interface excludes protected/confirmation payloads')
    for row in manifest['rows']:
        if row['split'] != split:
            continue
        if row['schema'] != SHARD_VERSION:
            raise ValueError('unsupported Stage 1 shard schema')
        versions=row.get('versions',row.get('version'))
        if versions is None or row.get('sealed',True) is not True or row.get('independent_futures',1)!=1:
            raise ValueError('sealed versioned root with one shared human future required')
        yield {'root_id':row['root_id'], 'scene_id':row['scene_id'],
               'query_ids':tuple(row['query_ids']), 'versions':versions,
               'files':row['files'], 'independent_human_futures':1,
               'grouping':'scene/root; siblings share future'}
