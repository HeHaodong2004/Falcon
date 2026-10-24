"""Persistent legal-history kinematic comparators, no architecture search."""
import numpy as np
from eventwm_graph.l2_partial.probability import transition_matrices
from .model import IntermittentWorldModel, sealed
from .dynamics import root_psd


class MemoryCV(IntermittentWorldModel):
    def __init__(self,diffusion=.16,**kwargs):
        self.cv_diffusion=float(diffusion)
        if not np.isfinite(self.cv_diffusion) or self.cv_diffusion<0:raise ValueError('invalid CV diffusion')
        super().__init__(**kwargs);self.model_version+='-memory-cv-'+str(self.cv_diffusion)
        self.stochastic_version+='-memory-cv-'+str(self.cv_diffusion)

    def from_legacy(self,b,seed=None):
        result=super().from_legacy(b,seed);cs=[]
        # Use the SAME frozen legal Gaussian-mixture bootstrap, with no neural
        # initial Stop reset. This retains the strongest legal velocity history.
        for c in result.components:
            h=b.hypotheses[0]
            # origin samples refer to stratified bootstrap choices, recovered by
            # identical seeded RNG decisions through a temporary nonstopped law.
            cs.append(c)
        rng=np.random.default_rng(self.seed if seed is None else seed)
        ws=np.array([h.weight for h in b.hypotheses]);ix=np.searchsorted(np.cumsum(ws/ws.sum()),(np.arange(self.max_components)+rng.random())/self.max_components)
        for j,(c,hi) in enumerate(zip(cs,ix)):
            h=b.hypotheses[hi];m=h.mean.copy();P=h.covariance.copy()
            if h.maneuver_mean is not None:
                pick=(rng.random((h.n,1))>np.cumsum(h.maneuver_probability,axis=-1)).sum(-1)
                m=h.maneuver_mean[np.arange(h.n),pick].copy();P=h.maneuver_covariance[np.arange(h.n),pick].copy()
            rng.random((h.n,1));rng.random(h.n);rng.integers(0,2**63)
            cs[j]=sealed(c.fork(mean=m,covariance=P,stopped=np.zeros(c.n,bool)))
        return result.fork(components=tuple(cs))

    def evolve(self,c,start,end,rng=None):
        F,Q,a,e=transition_matrices(end-start,0.,self.cv_diffusion)
        mean=c.mean@F.T;P=F@c.covariance@F.T+Q
        if rng is not None:
            mean+=np.einsum('ij,nj->ni',root_psd(Q),rng.normal(size=c.mean.shape));P=np.zeros_like(P)
        hm=c.hidden_mean;hc=c.hidden_covariance
        if hm is not None:hm=F@hm;hc=F@hc@F.T
        return c.fork(mean=mean,covariance=P,next_switch=np.full(c.n,end+1e10),stopped=np.zeros(c.n,bool),hidden_mean=hm,hidden_covariance=hc)

    def _birth(self,c,z,t,slot,hp,graph,offset):
        h=super()._birth(c,z,t,slot,hp,graph,offset);P=h.covariance.copy();P[-1,2:,2:]=np.eye(2)*.64
        return h.fork(covariance=P,stopped=np.r_[c.stopped,False])
