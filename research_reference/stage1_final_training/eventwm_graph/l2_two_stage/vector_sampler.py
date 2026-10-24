"""Vectorized iid current proposals; construct objects only for accepted worlds.

Padding marks are independent unused auxiliary noise, not extra people. Count
tables are prepared once per fixed law. No geometry, count or probability tail
is removed to improve latency.
"""
from dataclasses import dataclass
import numpy as np
from .count import certain_sources
from .speed import bound_numpy


def cdf(logits):
    logits=np.asarray(logits,np.float64);p=np.exp(logits-logits.max());p/=p.sum()
    answer=np.cumsum(p);answer/=answer[-1]
    return answer


@dataclass
class CurrentBatch:
    mode: np.ndarray
    source_present: np.ndarray
    initial: np.ndarray
    source_positions: np.ndarray
    hidden_counts: np.ndarray
    hidden_components: np.ndarray
    hidden_positions: np.ndarray
    positions: np.ndarray
    present: np.ndarray

    def objects(self,indices):
        from .conditioned import CurrentProposal
        rows=[]
        for i in indices:
            i=int(i);u=int(self.hidden_counts[i])
            rows.append(CurrentProposal(int(self.mode[i]),self.source_present[i].copy(),self.initial[i].copy(),
                self.hidden_components[i,:u].copy(),self.hidden_positions[i,:u].copy(),self.source_positions[i].copy()))
        return rows


class PreparedCurrentSampler:
    def __init__(self,features,parameters,raw_last,minimum,unique=True,total=None,graph=None):
        from .fast_count import PreparedCountSampler
        self.features=features;self.parameters=parameters;self.raw_last=np.asarray(raw_last,np.float64)
        self.n=len(features['keys']);self.mode_cdf=cdf(parameters['mode_logits'])
        self.spatial_cdf=cdf(parameters['spatial_logits'])
        certain=certain_sources(features['age'],features['minimum_count'],unique,False)
        logits=parameters['presence_logits'][:self.n];rate=float(parameters['rate'])
        self.rectangle=None;self.required_rectangle=None;self.count_offset=0
        self.log_normalizer_multiplier=0.;base_minimum=int(minimum)
        if graph is not None:
            from .palm_sampler import required_rectangle
            from .gaussian_rectangle import prepare_rectangle
            from .count import suffix_tails
            certificate=required_rectangle(graph,self.raw_last,features['age'])
            if certificate is not None:
                if rate<=0:raise RuntimeError('mandatory residual human has zero PPP support')
                rectangle=prepare_rectangle(parameters,features['robotxy'],certificate.lo,certificate.hi)
                # The proposal is useful only when its known density multiplier
                # is below one. This decision changes computation, never Q|A.
                old_normalizer=suffix_tails(logits,features['age'],rate,int(minimum),certain)[0][0,int(minimum)]
                reduced=max(0,int(minimum)-1)
                new_normalizer=suffix_tails(logits,features['age'],rate,reduced,certain)[0][0,reduced]
                multiplier=float(np.log(rate)+rectangle.log_mass+new_normalizer-old_normalizer)
                if not np.isfinite(multiplier):raise RuntimeError('nonfinite Palm proposal normalizer')
                if multiplier<0:
                    self.rectangle=rectangle;self.required_rectangle=certificate;self.count_offset=1
                    self.log_normalizer_multiplier=multiplier;base_minimum=reduced
        self.count=PreparedCountSampler(logits,features['age'],rate,base_minimum,certain=certain,
            total=None if total is None else int(total)-self.count_offset)

    def accept(self,batch,graph,rng):
        feasible=graph.feasible(batch.positions,batch.present)
        if self.rectangle is None:return feasible
        from .palm_sampler import acceptance_weights
        # Auxiliary Bernoulli thinning removes the Palm size bias exactly.
        return feasible&(rng.random(len(feasible))<acceptance_weights(batch,self.required_rectangle.lo,self.required_rectangle.hi))

    def draw(self,worlds,rng):
        if worlds<1:raise ValueError('positive iid proposal batch required')
        p=self.parameters;f=self.features;n=self.n
        mode=np.searchsorted(self.mode_cdf,rng.random(worlds),side='right')
        present,hidden_counts=self.count.sample(worlds,rng)
        initial=(p['means'][:n][np.arange(n)[None],mode[:,None],0]+
            p['scales'][:n][np.arange(n)[None],mode[:,None],0]*rng.normal(size=(worlds,n,2)))
        source=self.raw_last[None]+bound_numpy(f['velocity'][None]+initial,3.)*f['age'][None,:,None]
        maximum=int(np.max(hidden_counts,initial=0))
        component=np.searchsorted(self.spatial_cdf,rng.random((worlds,maximum)),side='right')
        hidden=f['robotxy']+p['spatial_mean'][component]+p['spatial_scale'][component]*rng.normal(size=(worlds,maximum,2))
        if self.rectangle is not None:
            from .palm_sampler import add_anchor
            raw=CurrentBatch(mode,present,initial,source,hidden_counts,component,hidden,None,None)
            hidden_counts,component,hidden=add_anchor(raw,self.rectangle,rng)
            maximum=hidden.shape[1]
        hidden_mask=np.arange(maximum)[None]<hidden_counts[:,None]
        positions=np.concatenate((source,hidden),axis=1)
        all_present=np.concatenate((present,hidden_mask),axis=1)
        return CurrentBatch(mode,present,initial,source,hidden_counts,component,hidden,positions,all_present)
