"""Full-support association using retained sampled motion as a SOFT guide.

The neural current proposal's initial coefficients are already scored, including
absent proposal episodes. Given those latent draws, these guide probabilities
have no further theta dependence. No identity/existence edge is forced.
"""
import math
import numpy as np
from numba import njit
from .association import (_endpoints,_terminal_edges,_first_choices,
    _bitset_lineages,LineageWitness,LINEAGE_KERNEL)
from .types import immutable_array

GUIDANCE='sampled_initial_motion_soft_last_event_1; sigma=.05+.1dt; uniform=.1'


@njit(cache=True)
def _override(times,xy,now,indptr,indices,base_geometry,terminal,terminal_geometry,
              events,source_positions,uniforms,uniform_mass,first,columns,logaux):
    worlds,n,m=terminal.shape
    for k in range(worlds):
        for s in range(len(events)):
            u=events[s];age=now-times[u]
            if age<=1e-7:continue
            vx=(source_positions[k,s,0]-xy[u,0])/age
            vz=(source_positions[k,s,1]-xy[u,1])/age
            past=indptr[u+1]-indptr[u];degree=past;old_sum=0.
            logs=np.full(past+m,-np.inf);maximum=-np.inf
            for at in range(past):
                edge=indptr[u]+at;v=indices[edge];dt=times[v]-times[u]
                dx=xy[v,0]-xy[u,0]-dt*vx;dz=xy[v,1]-xy[u,1]-dt*vz
                logg=-.5*(dx*dx+dz*dz)/(.05+.1*dt)**2
                logs[at]=logg;maximum=max(maximum,logg);old_sum+=base_geometry[edge]
            for j in range(m):
                if not terminal[k,u,j]:continue
                degree+=1;dt=age
                # terminal_geometry is only the previous geometric law; actual
                # endpoints are passed separately below through appended xy.
                dx=source_positions[k,len(events)+j,0]-xy[u,0]-dt*vx
                dz=source_positions[k,len(events)+j,1]-xy[u,1]-dt*vz
                logg=-.5*(dx*dx+dz*dz)/(.05+.1*dt)**2
                logs[past+j]=logg;maximum=max(maximum,logg);old_sum+=terminal_geometry[k,u,j]
            if not degree:continue
            weights=np.exp(logs-maximum);normalizer=weights.sum()
            old_at=first[k,u]
            old_g=(base_geometry[indptr[u]+old_at] if old_at<past else terminal_geometry[k,u,old_at-past])
            old_probability=uniform_mass/degree+(1.-uniform_mass)*old_g/old_sum
            cdf=0.;chosen=-1;probability=0.
            for at in range(past+m):
                if at>=past and not terminal[k,u,at-past]:continue
                value=uniform_mass/degree+(1.-uniform_mass)*weights[at]/normalizer
                cdf+=value;chosen=at;probability=value
                if uniforms[k,u]<cdf:break
            first[k,u]=chosen
            columns[k,u]=indices[indptr[u]+chosen] if chosen<past else n+chosen-past
            logaux[k]+=math.log(probability)-math.log(old_probability)


def lineage(graph,endpoints,present,source_events,source_positions,rng,uniform_mass=.1,source_first=False):
    """Same complete matcher and full edge support; only normalized hints differ."""
    graph.validate_certificate();points,mask=_endpoints(endpoints,present)
    events=np.asarray(source_events,np.int64);guides=np.asarray(source_positions,np.float64)
    if (events.ndim!=1 or len(set(events.tolist()))!=len(events) or np.any(events<0)
            or np.any(events>=graph.detections) or guides.shape!=(len(points),len(events),2)
            or not np.isfinite(guides).all() or not 0<float(uniform_mass)<=1):
        raise ValueError('invalid sampled legal-motion guide')
    terminal,geometry=_terminal_edges(graph.times,graph.xy,graph.timestamp,points,mask,True)
    worlds=len(points);n=graph.detections;uniforms=rng.random((worlds,n))
    order=np.array([rng.permutation(n) for _ in range(worlds)],np.int64).reshape(worlds,n)
    if source_first:
        priority=np.zeros(n,bool);priority[events]=True
        for k in range(worlds):
            perm=order[k].copy();order[k]=np.r_[perm[priority[perm]],perm[~priority[perm]]]
    first,columns,logaux=_first_choices(graph.indptr,graph.indices,graph.geometry,terminal,geometry,uniforms,float(uniform_mass))
    _override(graph.times,graph.xy,graph.timestamp,graph.indptr,graph.indices,graph.geometry,
        terminal,geometry,events,np.concatenate((guides,points),axis=1),uniforms,float(uniform_mass),first,columns,logaux)
    valid,matching,event_person,seen=_bitset_lineages(graph.past_bits,terminal,columns,order)
    if not valid.all():raise ValueError('infeasible population has no guided lineage')
    if np.any(seen&~mask):raise RuntimeError('guided lineage attached to absent endpoint')
    result=object.__new__(LineageWitness)
    logaux-=(math.lgamma(len(events)+1)+math.lgamma(n-len(events)+1) if source_first else math.lgamma(n+1))
    for key,value in dict(matching=matching,event_person=event_person,physical_seen=seen,
            row_order=order,first_edge=columns,auxiliary_log_probability=logaux).items():
        object.__setattr__(result,key,immutable_array(value))
    object.__setattr__(result,'kernel',LINEAGE_KERNEL+'; '+GUIDANCE+'; source_first='+str(bool(source_first)))
    return result
