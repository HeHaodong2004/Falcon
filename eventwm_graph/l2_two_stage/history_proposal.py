"""Legal last-observation spatial basis; features, never fixed identities.

Every stale neural proposal episode supplies a legal observation basis atom.
Its label never forces a physical identity or existence. All history association
hypotheses remain in the population conditioning/association support.
A positive global Gaussian-mixture fraction preserves the original full support.
"""
import numpy as np


def tail_basis(record,graph):
    from .conditioned import source_coordinates
    from .positive_history import PositiveSetHistory
    from .speed import bound_numpy
    # Independent of the arbitrary cached maximum matching and of how the
    # actual prefix was partitioned into append calls. Raw and incremental
    # inference must define exactly the same probability law.
    history=record.get('positive_history')
    if history is None:
        history=PositiveSetHistory('','','',len(graph.frame_timestamps)-1,
            graph.frame_timestamps,graph.positive_offsets,graph.positive_xyz,
            graph.observation_valid,graph.human_sensor_available)
    raw_last,_=source_coordinates(record,history)
    stale=np.asarray(record['age'])>1e-7;age=np.asarray(record['age'],float)[stale]
    velocity=np.asarray(record['velocity'],float)[stale]
    mean=raw_last[stale]+bound_numpy(velocity,3.)*age[:,None]-record['robotxy']
    # This scale is a declared proposal parameter, not a calibrated confidence
    # interval. Final P is retrained/scored after exact history conditioning.
    scale=np.repeat((.25+.5*age)[:,None],2,axis=1)
    return mean,scale,velocity


def augment_numpy(parameters,record,graph,fraction):
    if not 0<=fraction<1:raise ValueError('global spatial support must remain positive')
    if fraction==0:return parameters
    mean,scale,velocity=tail_basis(record,graph)
    from scipy.special import logsumexp
    p=dict(parameters);count=len(mean)
    for key in ('spatial_mean','spatial_scale','spatial_logits','spatial_velocity','spatial_velocity_scale'):
        p[key]=np.asarray(p[key],np.float64)
    if not len(mean):return p
    p['spatial_mean']=np.concatenate((p['spatial_mean'],mean))
    p['spatial_scale']=np.concatenate((p['spatial_scale'],scale))
    p['spatial_logits']=np.r_[p['spatial_logits']-logsumexp(p['spatial_logits'])+np.log1p(-fraction),np.full(count,np.log(fraction/count))]
    p['spatial_velocity']=np.concatenate((p['spatial_velocity'],velocity))
    p['spatial_velocity_scale']=np.r_[p['spatial_velocity_scale'],np.full(count,1.04847)]
    return p


def augment_torch(parameters,records,graphs,fraction):
    if not 0<=fraction<1:raise ValueError('global spatial support must remain positive')
    if fraction==0:return parameters
    import torch
    from torch.nn import functional as F
    bases=[tail_basis(r,g) for r,g in zip(records,graphs)]
    extra=max(len(mean) for mean,_,_ in bases)
    p=dict(parameters);b=len(records);device=p['spatial_mean'].device;dtype=torch.float64
    for key in ('spatial_mean','spatial_scale','spatial_logits','spatial_velocity','spatial_velocity_scale'):
        p[key]=p[key].double()
    if not extra:return p
    mu=np.zeros((b,extra,2));sigma=np.ones((b,extra,2));vel=np.zeros((b,extra,2));logits=np.full((b,extra),-np.inf)
    fractions=[]
    for i,(mean,scale,velocity) in enumerate(bases):
        n=len(mean);fractions.append(fraction if n else 0.)
        mu[i,:n]=mean;sigma[i,:n]=scale
        vel[i,:n]=velocity
        if n:logits[i,:n]=np.log(fraction/n)
    tensor=lambda a:torch.as_tensor(a,device=device,dtype=dtype)
    p['spatial_mean']=torch.cat((p['spatial_mean'],tensor(mu)),1)
    p['spatial_scale']=torch.cat((p['spatial_scale'],tensor(sigma)),1)
    p['spatial_logits']=torch.cat((F.log_softmax(p['spatial_logits'],-1)+tensor(np.log1p(-np.array(fractions)))[:,None],tensor(logits)),1)
    p['spatial_velocity']=torch.cat((p['spatial_velocity'],tensor(vel)),1)
    p['spatial_velocity_scale']=torch.cat((p['spatial_velocity_scale'],torch.full((b,extra),1.04847,device=device,dtype=dtype)),1)
    return p
