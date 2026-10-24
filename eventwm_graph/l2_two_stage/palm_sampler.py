"""Exact Palm proposal for a legally compulsory residual human witness.

If a terminal detection's current reachable disk is disjoint from every
possible neural-source support disk, any feasible world MUST have a PPP atom
in its bounding rectangle. A distinguished rectangle-conditioned PPP atom,
uniformly shuffled with ordinary PPP marks, size-biases by the number m_R of
PPP atoms in that rectangle. Acceptance1/m_R removes that bias exactly.
The model law, neural weights and physical support are unchanged.
"""
from dataclasses import dataclass
import math
import numpy as np
from .types import immutable_array

VERSION='exact_residual_terminal_rectangle_Palm_1'


@dataclass(frozen=True)
class RequiredRectangle:
    event:int
    timestamp:float
    radius:float
    lo:np.ndarray
    hi:np.ndarray
    minimum_source_gap:float
    reason:str='terminal_witness_disk_disjoint_from_all_possible_source_current_support'


def required_rectangle(graph,raw_last,age):
    """A necessary-event certificate derived ONLY from actual legal history.

    Conservative roundoff guards only forgo proposal optimization; they never
    remove a model hypothesis or broaden the physical feasibility predicate.
    """
    graph.validate_certificate();raw_last=np.asarray(raw_last,np.float64);age=np.asarray(age,np.float64)
    if raw_last.shape!=(len(age),2) or not np.isfinite(raw_last).all() or not np.isfinite(age).all() or np.any(age<0):
        raise ValueError('invalid source support coordinates')
    radii=3.*age;choices=[];eps=np.finfo(np.float64).eps
    for u in np.flatnonzero(np.diff(graph.indptr)==0):
        elapsed=graph.timestamp-float(graph.times[u])
        if elapsed<=0:continue
        radius=3.*elapsed;center=graph.xy[u]
        gap=math.inf;safe=True
        for point,source_radius in zip(raw_last,radii):
            distance=math.hypot(float(point[0]-center[0]),float(point[1]-center[1]))
            guard=1e-12+64*eps*(1+abs(center).sum()+abs(point).sum()+radius+source_radius)
            separation=distance-radius-source_radius
            if separation<=guard:safe=False;break
            gap=min(gap,separation)
        if safe:choices.append((elapsed,int(u),radius,gap))
    if not choices:return None
    _,u,radius,gap=min(choices)
    center=graph.xy[u]
    lo=np.nextafter(center-radius,-np.inf);hi=np.nextafter(center+radius,np.inf)
    return RequiredRectangle(u,float(graph.times[u]),radius,immutable_array(lo),immutable_array(hi),gap)


def add_anchor(batch,rectangle,rng):
    """Return marked PPP arrays with one distinguished atom, random slot order."""
    worlds=len(batch.mode);components,points=rectangle.sample(worlds,rng)
    counts=batch.hidden_counts+1;maximum=int(counts.max())
    comp=np.zeros((worlds,maximum),np.int64);xy=np.zeros((worlds,maximum,2),np.float64)
    for k in range(worlds):
        before=int(batch.hidden_counts[k]);size=before+1
        cs=np.r_[batch.hidden_components[k,:before],components[k]]
        ps=np.concatenate((batch.hidden_positions[k,:before],points[k:k+1]))
        permutation=rng.permutation(size)
        comp[k,:size]=cs[permutation];xy[k,:size]=ps[permutation]
    return counts,comp,xy


def acceptance_weights(batch,lo,hi):
    mask=np.arange(batch.hidden_positions.shape[1])[None]<batch.hidden_counts[:,None]
    inside=((batch.hidden_positions>=lo)&(batch.hidden_positions<=hi)).all(-1)&mask
    counts=inside.sum(-1)
    if np.any(counts<1):raise RuntimeError('Palm proposal lost its distinguished rectangle atom')
    return 1./counts
