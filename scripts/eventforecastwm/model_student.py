"""Persistent scene-scale Student-t residual uncertainty.

One variance V~InverseGamma(alpha,beta) is shared across the entire scene path.
Conditional on V, the existing linear Gaussian residual process has covariance
V times its base covariance. Integrating V gives an exact joint multivariate-t
likelihood, including masked observations. V persists across sampled segments.
Actual observations update both the Gaussian state and the scale posterior.
"""
from dataclasses import dataclass,replace
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from eventforecastwm.model import ForecastWorldModel,SceneForecast,SceneBelief,sample_components
from eventforecastwm.model_age import AgeAwareForecastWorldModel


@dataclass
class StudentBelief:
    physical: SceneBelief
    concentration: torch.Tensor          # B,S inverse-gamma shape
    rate: torch.Tensor                   # B,S inverse-gamma scale

    @property
    def core(self):return self.physical.core

    @property
    def covariance(self):
        return self.physical.covariance*(self.rate/(self.concentration-1))[:,:,None,None,None]

    def detach(self):return StudentBelief(self.physical.detach(),self.concentration.detach(),self.rate.detach())


@dataclass
class StudentSampleState:
    gaussian_state: torch.Tensor
    variance: torch.Tensor               # B,S,draws; one persistent scene scale


@dataclass
class StudentForecast(SceneForecast):
    concentration: torch.Tensor
    rate: torch.Tensor
    decay_rate: object = None

    @property
    def position_variance(self):
        return (self.covariance[...,0,0]+self.readout_scale.square())*(self.rate/(self.concentration-1))[:,:,None,None]

    def sample(self,draws=4,generator=None,initial_state=None):
        b,s,t,n,_=self.positions.shape
        if initial_state is None:
            alpha=self.concentration[:,:,None].expand(b,s,draws)
            # Existing PyTorch primitive supports explicit RNG and implicit
            # reparameterized gradients. No new package or environment change.
            variance=self.rate[:,:,None]/torch._standard_gamma(alpha,generator=generator).clamp_min(1e-12)
            raw_state=None
        else:
            if not isinstance(initial_state,StudentSampleState):raise ValueError('Persistent Student sample state required')
            variance=initial_state.variance;raw_state=initial_state.gaussian_state
            if variance.shape!=(b,s,draws):raise ValueError('Student scenario identities/draw count changed')
        if self.decay_rate is None:
            paths,weights,state=sample_components(self,draws,generator,initial_state=raw_state,return_state=True)
        else:
            from eventforecastwm.model_damped import sample_components as damped_samples
            paths,weights,state=damped_samples(self,draws,generator,raw_state)
        means=self.positions[:,:,None].expand(b,s,draws,t,n,2).reshape_as(paths)
        paths=means+(paths-means)*variance.reshape(b,s*draws,1,1,1).sqrt()
        return paths,weights,StudentSampleState(state,variance)


class StudentWorldModel(ForecastWorldModel):
    def __init__(self,variant='student_scene_v1',age_aware=False,damped=False,mean_variant=None,**kwargs):
        if variant!='student_scene_v1':raise ValueError(variant)
        kwargs.setdefault('noise','integrated');super().__init__(**kwargs)
        if self.noise!='integrated':raise ValueError('Student residuals require the coherent state process')
        self.raw_df=nn.Parameter(torch.tensor(math.log(math.expm1(4.))))
        self.age_aware=age_aware;self.recipe.update(variant=variant,age_aware=age_aware)
        self.damped=damped
        if damped:
            self.raw_decay_rate=nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
            self.recipe['damped']=True
        from eventforecastwm.mean_adapters import attach
        attach(self,mean_variant)

    @property
    def degrees_of_freedom(self):return 2.05+F.softplus(self.raw_df)

    @property
    def decay_rate(self):return F.softplus(self.raw_decay_rate).clamp(.02,10.)

    def initial_covariance_for(self,core):
        if self.damped:
            from eventforecastwm.model_damped import DampedForecastWorldModel
            return DampedForecastWorldModel.initial_covariance_for(self,core)
        if self.age_aware:return AgeAwareForecastWorldModel.initial_covariance_for(self,core)
        p=core.positions.new_zeros(*core.positions.shape[:3],2,2)
        p[...,0,0]=self.log_initial_position.exp().clamp(.01,3.).square()
        p[...,1,1]=self.log_initial_velocity.exp().clamp(.01,3.).square()
        return p

    def encode_history(self,*args,**kwargs):
        from eventforecastwm.mean_adapters import encode
        physical=encode(self,super().encode_history(*args,**kwargs));physical.covariance=self.initial_covariance_for(physical.core)
        alpha=(self.degrees_of_freedom/2).expand_as(physical.core.scenario_weights)
        return StudentBelief(physical,alpha,alpha)

    def imagine(self,belief,candidate_query=None,horizon_s=3.):
        if self.damped:
            from eventforecastwm.model_damped import DampedForecastWorldModel
            f=DampedForecastWorldModel.imagine(self,belief.physical,candidate_query,horizon_s)
        else:f=super().imagine(belief.physical,candidate_query,horizon_s)
        values=dict(f.__dict__);values['next_belief']=StudentBelief(f.next_belief,belief.concentration,belief.rate)
        return StudentForecast(**values,concentration=belief.concentration,rate=belief.rate)

    def observe(self,belief,observation):
        updated=super().observe(belief.physical,observation) # Validates actual arrivals, IDs and clock first.
        c=belief.core;mask=observation['observed'].to(c.clock.device).bool()
        acquired=observation['acquisition_times'].to(c.clock.device)
        fresh=mask&(acquired>c.last_observed_times+1e-7)
        lag=(c.clock[:,None]-acquired).to(c.positions);H=torch.stack([torch.ones_like(lag),-lag],-1)[:,None]
        cov=belief.physical.covariance;mean=belief.physical.residual_mean
        innovation=observation['positions'].to(c.positions)[:,None]-(c.positions-lag[:,None,:,None]*c.velocities)-(H[...,None]*mean).sum(-2)
        variance=((H*(cov@H[...,None]).squeeze(-1)).sum(-1)+self.log_readout.exp().clamp(.01,5.).square()).clamp_min(1e-8)
        quadratic=(innovation.square()/variance[...,None]*fresh[:,None,:,None]).sum((2,3))
        logdet=(2*variance.log()*fresh[:,None]).sum(-1);count=2*fresh.sum(-1)[:,None]
        logp=scale_log_density(quadratic,logdet,count,belief.concentration,belief.rate)
        weights=(c.scenario_weights.clamp_min(1e-12).log()+logp).softmax(-1)
        updated.core=replace(updated.core,scenario_weights=weights)
        return StudentBelief(updated,belief.concentration+count/2,belief.rate+quadratic/2)


def scale_log_density(quadratic,logdet,count,alpha,beta):
    return (torch.lgamma(alpha+count/2)-torch.lgamma(alpha)-count/2*math.log(2*math.pi)
            -.5*logdet+alpha*beta.log()-(alpha+count/2)*(beta+quadratic/2).log())


def trajectory_log_probability(f,truth,mask):
    mu=f.positions;b,s,t,n,_=mu.shape;state=mu.new_zeros(b,s,n,2,2);cov=f.initial_covariance
    if f.decay_rate is None:
        transition=mu.new_tensor([[1.,f.dt],[0.,1.]]);g=mu.new_tensor([.5*f.dt**2,f.dt])
        process_covariance=f.acceleration_scale.square()*g[:,None]*g[None,:]
    else:
        from eventforecastwm.model_damped import process
        transition,process_covariance=process(f.decay_rate,f.acceleration_scale,f.dt)
    quadratic=mu.new_zeros(b,s);logdet=mu.new_zeros(b,s)
    for j in range(t):
        state=transition@state;cov=transition@cov@transition.T+process_covariance
        innovation=truth[:,None,j]-mu[:,:,j]-state[...,0,:];variance=(cov[...,0,0]+f.readout_scale.square()).clamp_min(1e-8)
        valid=mask[:,None,j];quadratic+=(innovation.square()/variance[...,None]*valid[...,None]).sum((2,3))
        logdet+=(2*variance.log()*valid).sum(-1)
        gain=cov[..., :,0]/variance[...,None]
        state=state+gain[...,None]*innovation[...,None,:]*valid[...,None,None]
        cov=cov-gain[..., :,None]*cov[...,0,None,:]*valid[...,None,None]
    count=2*mask.sum((1,2));component=scale_log_density(quadratic,logdet,count[:,None],f.concentration,f.rate)
    return torch.logsumexp(component+f.weights.clamp_min(1e-12).log(),1),count


def energy_score(f,truth,mask,draws=1,generator=None):
    a,w,_=f.sample(draws,generator);b,_,_=f.sample(draws,generator)
    m=mask[...,None].to(a.dtype);count=m.sum((1,2,3))*2;scale=count.clamp_min(1)
    first=(((a-truth[:,None])*m[:,None]).square().sum((2,3,4))/scale[:,None]+1e-10).sqrt()
    pair=(((a[:,:,None]-b[:,None])*m[:,None,None]).square().sum((3,4,5))/scale[:,None,None]+1e-10).sqrt()
    return (w*first).sum(1)-.5*(w[:,:,None]*w[:,None]*pair).sum((1,2)),count>0


@torch.no_grad()
def marginal_cdf(f,locations):
    from scipy.special import stdtr
    mean=f.positions
    scale=((f.covariance[...,0,0]+f.readout_scale.square())*(f.rate/f.concentration)[:,:,None,None]).sqrt()[...,None]
    if locations.ndim==6:mean=mean[...,None];scale=scale[...,None]
    z=(locations-mean)/scale;df=(2*f.concentration).reshape(*f.concentration.shape,*([1]*(z.ndim-2)))
    result=stdtr(df.detach().cpu().numpy(),z.detach().cpu().numpy())
    return torch.as_tensor(np.asarray(result),device=z.device,dtype=z.dtype)
