"""Small frozen-encoder experiments and root-balanced ranking supervision.

Labels determine supervised pairs only. They never enter event-head features.
This module shares FactorizedEventWM.event_outputs with ordinary inference.
"""
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F
from ..sr_metrics import metrics
from .evaluation import root_interval


def frozen_feature_signature(model):
    """Reject features from a different binary/world encoder or query contract."""
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('outcome_head.') or name=='social_prior_logit':continue
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    config=asdict(model.config)
    digest.update(json.dumps({k:config[k] for k in
        ('variant','sr_size','interaction_samples','interaction_horizon')},sort_keys=True).encode())
    return digest.hexdigest()


def conditional_social_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    keep = y < 2
    if not keep.any():
        return dict(samples=0)
    q = p[keep,1]/np.clip(p[keep,:2].sum(1),1e-12,None)
    result = metrics((y[keep] == 0).astype(int),1-q)
    return dict(samples=int(keep.sum()),prevalence=float((y[keep] == 1).mean()),
        nll=result['bce'],ap=result['collision_pr_auc_average_precision'],
        roc_auc=result['collision_roc_auc'],brier=result['brier_score'],ece=result['collision_ece'],
        mean_probability=float(q.mean()),
        reliability=[dict(bin=k,count=int(((q*15).astype(int).clip(0,14)==k).sum()),
            probability=float(q[(q*15).astype(int).clip(0,14)==k].mean()),
            frequency=float((y[keep][(q*15).astype(int).clip(0,14)==k]==1).mean()))
            for k in range(15) if ((q*15).astype(int).clip(0,14)==k).any()])


def matched_banks(metadata, movement):
    groups = defaultdict(list)
    for i,m in enumerate(metadata):
        groups[m.get('group_id',m['root_id'])].append(i)
    return {g:[i for i in ix if movement[i] > .1 and movement[i] >= .95*max(movement[ix])]
            for g,ix in groups.items()}


def social_training_pairs(metadata, y, valid, movement):
    """One root is one sampling unit, irrespective of correlated pair count."""
    if any(m.get('role','train') != 'train' or m['split'] != 'train' for m in metadata):
        raise ValueError('Ranking supervision requires optimization TRAIN roots only')
    byroot = defaultdict(list)
    for ix in matched_banks(metadata,np.asarray(movement)).values():
        for a in ix:
            for b in ix:
                if valid[a] and valid[b] and y[a] == 0 and y[b] == 1:
                    byroot[metadata[a]['root_id']].append((a,b))
    return dict(byroot)


def sample_root_pairs(pairs, generator, roots_per_step=16):
    roots = list(pairs)
    if not roots:return torch.empty((0,2),dtype=torch.long)
    # Uniform root sampling followed by uniform pair sampling within that root.
    chosen = torch.randperm(len(roots),generator=generator)[:roots_per_step]
    return torch.tensor([pairs[roots[j]][int(torch.randint(len(pairs[roots[j]]),(1,),generator=generator))]
                         for j in chosen.tolist()],dtype=torch.long)


def social_pair_loss(social_logits, pairs):
    if not len(pairs):return social_logits.sum()*0
    # First index is SAFE_SUCCESS; second is SOCIAL_VIOLATION_SUCCESS.
    return F.softplus(social_logits[pairs[:,0]]-social_logits[pairs[:,1]]).mean()


def root_losses(metadata, values, mask=None):
    mask = np.ones(len(values),dtype=bool) if mask is None else mask
    grouped = defaultdict(list)
    for m,v,keep in zip(metadata,values,mask):
        if keep:grouped[m['root_id']].append(float(v))
    return {r:float(np.mean(v)) for r,v in grouped.items()}


def paired_root_delta(metadata, a, b, mask=None, replicates=2000):
    return root_interval(list(root_losses(metadata,np.asarray(a)-np.asarray(b),mask).values()),
                         replicates=replicates)


def ranking_roots(metadata,y,score,movement,task):
    y,score = np.asarray(y),np.asarray(score)
    good,bad = {'social':(y==0,y==1),'physical':(y!=2,y==2),'safe':(y==0,y!=0)}[task]
    grouped = defaultdict(list)
    for ix in matched_banks(metadata,np.asarray(movement)).values():
        for a in ix:
            for b in ix:
                if good[a] and bad[b]:
                    grouped[metadata[a]['root_id']].append(float(score[a]>score[b])+.5*float(score[a]==score[b]))
    return {r:float(np.mean(v)) for r,v in grouped.items()}


def paired_rank_delta(metadata,y,a,b,movement,task):
    aa,bb = (ranking_roots(metadata,y,s,movement,task) for s in (a,b))
    if aa.keys() != bb.keys():raise ValueError('Different informative roots')
    return root_interval([aa[r]-bb[r] for r in aa],replicates=2000)
