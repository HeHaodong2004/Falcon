"""Coherent integrated Ornstein-Uhlenbeck residuals around recurrent means.

dv = -lambda*v*dt + q_c*dW, dp=v*dt, q_c=q*sqrt(.25).
The exact continuous-time F/Q permits both canonical future transitions and
fractional observation ages. Mean reversion applies to uncertain residual
velocity, not to the learned physical mean. No trajectory clipping is used.
"""
from dataclasses import dataclass,replace
import math
import torch
from eventforecastwm.model import ForecastWorldModel,SceneBelief,SceneForecast


def process(rate,scale,duration):
    # Small matrices in float64 avoid cancellation in the integrated covariance.
    dtype=rate.dtype;rate=rate.double();q=scale.double();t=torch.as_tensor(duration,device=rate.device,dtype=torch.float64)
    e=torch.exp(-rate*t);one=-torch.expm1(-rate*t);two=-torch.expm1(-2*rate*t)
    f=torch.zeros(*t.shape,2,2,device=t.device,dtype=t.dtype)
    f[...,0,0]=1;f[...,0,1]=one/rate;f[...,1,1]=e
    spectral=q.square()*.25
    covariance=torch.zeros_like(f)
    covariance[...,0,0]=spectral/rate.square()*(t-2*one/rate+two/(2*rate))
    covariance[...,0,1]=covariance[...,1,0]=spectral*one.square()/(2*rate.square())
    covariance[...,1,1]=spectral*two/(2*rate)
    return f.to(dtype),covariance.to(dtype)


@dataclass
class DampedForecast(SceneForecast):
    decay_rate: torch.Tensor

    def sample(self,draws=4,generator=None,initial_state=None):
        return sample_components(self,draws,generator,initial_state)


class DampedForecastWorldModel(ForecastWorldModel):
    def __init__(self,variant='damped_velocity_v1',age_aware=False,mean_variant=None,**kwargs):
        if variant!='damped_velocity_v1':raise ValueError(variant)
        kwargs.setdefault('noise','integrated');super().__init__(**kwargs)
        if self.noise!='integrated':raise ValueError('Damped model requires integrated trajectory uncertainty')
        self.raw_decay_rate=torch.nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
        self.recipe.update(variant=variant,age_aware=age_aware)
        self.age_aware=age_aware
        from eventforecastwm.mean_adapters import attach
        attach(self,mean_variant)

    @property
    def decay_rate(self):return torch.nn.functional.softplus(self.raw_decay_rate).clamp(.02,10.)

    def initial_covariance_for(self,core):
        shape=core.positions.shape[:3];cov=core.positions.new_zeros(*shape,2,2)
        cov[...,0,0]=self.log_initial_position.exp().clamp(.01,3.).square()
        cov[...,1,1]=self.log_initial_velocity.exp().clamp(.01,3.).square()
        if self.age_aware:
            age=core.age[:,None].expand(shape).clamp_min(0.)
            f,q=process(self.decay_rate,self.log_acceleration.exp().clamp(.01,5.),age)
            cov=f@cov@f.transpose(-1,-2)+q
        return cov

    def encode_history(self,*args,**kwargs):
        from eventforecastwm.mean_adapters import encode
        belief=encode(self,super().encode_history(*args,**kwargs));belief.covariance=self.initial_covariance_for(belief.core)
        return belief

    def imagine(self,belief,candidate_query=None,horizon_s=3.):
        steps=round(horizon_s/.25)
        if steps<1 or abs(steps*.25-horizon_s)>1e-7:raise ValueError('Canonical .25s boundary required')
        current=belief.core;cov=belief.covariance;residual=belief.residual_mean
        q=self.log_acceleration.exp().clamp(.01,5.);r=self.log_readout.exp().clamp(.01,5.)
        f,Q=process(self.decay_rate,q,.25);positions=[];velocities=[];covariances=[]
        for _ in range(steps):
            current=self.core._step(current);cov=f@cov@f.T+Q;residual=f@residual
            positions.append(current.positions+residual[...,0,:]);velocities.append(current.velocities+residual[...,1,:]);covariances.append(cov)
        successor=SceneBelief(current,cov,residual)
        return DampedForecast(successor,torch.stack(positions,2),torch.stack(velocities,2),
            current.scenario_weights,torch.stack(covariances,2),belief.covariance,belief.residual_mean,
            q,r,self.noise,.25,current.valid,
            belief.core.clock[:,None]+torch.arange(1,steps+1,device=current.positions.device)*.25,self.decay_rate)


def trajectory_log_probability(forecast,truth,mask):
    mu=forecast.positions;b,s,t,n,_=mu.shape;residual=mu.new_zeros(b,s,n,2,2);cov=forecast.initial_covariance
    f,Q=process(forecast.decay_rate,forecast.acceleration_scale,forecast.dt)
    r2=forecast.readout_scale.square();logp=mu.new_zeros(b,s)
    for j in range(t):
        residual=f@residual;cov=f@cov@f.T+Q
        innovation=truth[:,None,j]-mu[:,:,j]-residual[...,0,:]
        variance=(cov[...,0,0]+r2).clamp_min(1e-8);valid=mask[:,None,j]
        logp+=((-.5*innovation.square()/variance[...,None]-.5*torch.log(2*math.pi*variance)[...,None])*valid[...,None]).sum((2,3))
        gain=cov[..., :,0]/variance[...,None]
        residual=residual+gain[...,None]*innovation[...,None,:]*valid[...,None,None]
        cov=cov-gain[..., :,None]*cov[...,0,None,:]*valid[...,None,None]
    count=mask.sum((1,2))*2
    return torch.logsumexp(forecast.weights.clamp_min(1e-12).log()+logp,1),count


def sample_components(forecast,draws=1,generator=None,initial_state=None):
    mu=forecast.positions;b,s,t,n,_=mu.shape
    rand=lambda shape:torch.randn(shape,device=mu.device,dtype=mu.dtype,generator=generator)
    if initial_state is not None:
        if initial_state.shape!=(b,s,draws,n,2,2):raise ValueError('Persistent sample identity/dimension mismatch')
        state=initial_state
    else:
        chol=torch.linalg.cholesky(forecast.initial_covariance+1e-8*torch.eye(2,device=mu.device,dtype=mu.dtype))
        state=chol[:,:,None]@rand((b,s,draws,n,2,2))
    f,Q=process(forecast.decay_rate,forecast.acceleration_scale,forecast.dt);noise=torch.linalg.cholesky(Q)
    values=[]
    for j in range(t):
        state=f@state+noise@rand((b,s,draws,n,2,2))
        values.append(mu[:,:,j,None]+state[...,0,:]+forecast.readout_scale*rand((b,s,draws,n,2)))
    positions=torch.stack(values,3).reshape(b,s*draws,t,n,2)
    weights=(forecast.weights[:,:,None]/draws).expand(b,s,draws).reshape(b,s*draws)
    return positions,weights,state


def energy_score(forecast,truth,mask,draws=1,generator=None):
    a,w,_=sample_components(forecast,draws,generator);b,_,_=sample_components(forecast,draws,generator)
    m=mask[...,None].to(a.dtype);count=m.sum((1,2,3))*2;scale=count.clamp_min(1)
    first=(((a-truth[:,None])*m[:,None]).square().sum((2,3,4))/scale[:,None]+1e-10).sqrt()
    pair=(((a[:,:,None]-b[:,None])*m[:,None,None]).square().sum((3,4,5))/scale[:,None,None]+1e-10).sqrt()
    return (w*first).sum(1)-.5*(w[:,:,None]*w[:,None]*pair).sum((1,2)),count>0
