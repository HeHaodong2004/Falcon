"""Persistent per-agent variance categories under a shared physical scene mode.

p(Y|h) = sum_z pi_z(h) product_n sum_l rho_nl(h) N(Y_n;mu_zn,K_zn*v_l).
The variance category for each agent persists across time. Human motion modes
remain joint. Parameters and categorical weights depend only on arrived legal
history; targets appear exclusively in likelihood computation.
"""
from dataclasses import dataclass,replace
import math
import torch
from torch import nn
from eventforecastwm.model_damped import DampedForecastWorldModel,DampedForecast,process,sample_components
from eventforecastwm.model_scale_mixture import scale_log_density


@dataclass
class AgentScaleBelief:
    physical: object
    agent_scale_probabilities: torch.Tensor  # B,S,N,L
    scale_values: torch.Tensor              # L
    legal_context: torch.Tensor             # B,N,8, fixed until re-encoding

    @property
    def core(self):return self.physical.core

    @property
    def covariance(self):
        mean=(self.agent_scale_probabilities*self.scale_values).sum(-1)
        return self.physical.covariance*mean[...,None,None]

    def detach(self):
        return AgentScaleBelief(self.physical.detach(),self.agent_scale_probabilities.detach(),
            self.scale_values.detach(),self.legal_context.detach())


@dataclass
class AgentScaleSampleState:
    gaussian_state: torch.Tensor
    variance: torch.Tensor


@dataclass
class AgentScaleForecast(DampedForecast):
    agent_scale_probabilities: torch.Tensor
    scale_values: torch.Tensor
    legal_context: torch.Tensor

    @property
    def position_variance(self):
        mean=(self.agent_scale_probabilities*self.scale_values).sum(-1)
        return (self.covariance[...,0,0]+self.readout_scale.square())*mean[:,:,None]

    def sample(self,draws=4,generator=None,initial_state=None):
        b,s,t,n,_=self.positions.shape
        if initial_state is None:
            index=torch.multinomial(self.agent_scale_probabilities.reshape(b*s*n,-1),draws,replacement=True,generator=generator)
            variance=self.scale_values[index].reshape(b,s,n,draws).permute(0,1,3,2);raw=None
        else:
            if not isinstance(initial_state,AgentScaleSampleState) or initial_state.variance.shape!=(b,s,draws,n):
                raise ValueError('Persistent per-agent scale identities required')
            variance=initial_state.variance;raw=initial_state.gaussian_state
        paths,weights,state=sample_components(self,draws,generator,raw)
        mean=self.positions[:,:,None].expand(b,s,draws,t,n,2).reshape_as(paths)
        paths=mean+(paths-mean)*variance.reshape(b,s*draws,1,n,1).sqrt()
        return paths,weights,AgentScaleSampleState(state,variance)


def legal_agent_context(core):
    valid=core.valid;age=core.age;weights=core.scenario_weights[:,:,None,None]
    velocity=(core.velocities*weights).sum(1);speed=velocity.norm(dim=-1)/2.
    spread=((core.velocities-velocity[:,None]).square().sum(-1)*core.scenario_weights[:,:,None]).sum(1)
    position=(core.positions*weights).sum(1);distance=torch.cdist(position,position)
    n=position.shape[1];eligible=valid[:,None]&~torch.eye(n,device=position.device,dtype=torch.bool)[None]
    nearest=distance.masked_fill(~eligible,8.).amin(-1).clamp_max(8.)/4.
    count=torch.log1p(valid.sum(-1).to(age))[:,None].expand_as(age)/3.
    features=torch.stack([(age/5.).clamp(0,2),(age>.25).to(age),(age>1.).to(age),
        core.velocity_valid.to(age),speed,spread,nearest,count],-1)
    return features*valid[...,None]


class AgentScaleWorldModel(DampedForecastWorldModel):
    def __init__(self,variant='agent_scene_scale_v1',scale_std=(.1,.25,.5,1.,2.,4.,8.),conditional=True,**kwargs):
        if variant!='agent_scene_scale_v1':raise ValueError(variant)
        super().__init__(**kwargs);self.register_buffer('scale_values',torch.tensor(scale_std).square())
        self.scale_logits=nn.Parameter(torch.zeros(len(scale_std)))
        self.scale_context=nn.Linear(8,len(scale_std));nn.init.zeros_(self.scale_context.weight);nn.init.zeros_(self.scale_context.bias)
        self.conditional=conditional
        if not conditional:
            for p in self.scale_context.parameters():p.requires_grad_(False)
        self.recipe.update(variant=variant,scale_std=list(scale_std),conditional=conditional)

    def probabilities(self,context,modes):
        logits=self.scale_logits+(self.scale_context(context) if self.conditional else 0.)
        if logits.ndim==1:logits=logits.expand(*context.shape[:2],-1)
        return logits.softmax(-1)[:,None].expand(-1,modes,-1,-1)

    def encode_history(self,*args,**kwargs):
        physical=super().encode_history(*args,**kwargs);context=legal_agent_context(physical.core)
        return AgentScaleBelief(physical,self.probabilities(context,physical.core.scenario_weights.shape[1]),self.scale_values,context)

    def imagine(self,belief,candidate_query=None,horizon_s=3.):
        f=super().imagine(belief.physical,candidate_query,horizon_s);values=dict(f.__dict__)
        values['next_belief']=AgentScaleBelief(f.next_belief,belief.agent_scale_probabilities,belief.scale_values,belief.legal_context)
        return AgentScaleForecast(**values,agent_scale_probabilities=belief.agent_scale_probabilities,
            scale_values=belief.scale_values,legal_context=belief.legal_context)

    def observe(self,belief,observation):
        updated=super().observe(belief.physical,observation);c=belief.core
        acquired=observation['acquisition_times'].to(c.clock.device)
        fresh=observation['observed'].to(c.clock.device).bool()&(acquired>c.last_observed_times+1e-7)
        lag=(c.clock[:,None]-acquired).to(c.positions);H=torch.stack([torch.ones_like(lag),-lag],-1)[:,None]
        cov=belief.physical.covariance;mean=belief.physical.residual_mean
        error=observation['positions'].to(c.positions)[:,None]-(c.positions-lag[:,None,:,None]*c.velocities)-(H[...,None]*mean).sum(-2)
        variance=((H*(cov@H[...,None]).squeeze(-1)).sum(-1)+self.log_readout.exp().clamp(.01,5.).square()).clamp_min(1e-8)
        q=(error.square()/variance[...,None]).sum(-1)*fresh[:,None]
        logdet=2*variance.log()*fresh[:,None];count=2*fresh[:,None]
        lp=scale_log_density(q,logdet,count,belief.scale_values)+belief.agent_scale_probabilities.clamp_min(1e-12).log()
        weights=(c.scenario_weights.clamp_min(1e-12).log()+lp.logsumexp(-1).sum(-1)).softmax(-1)
        updated.core=replace(updated.core,scenario_weights=weights)
        return AgentScaleBelief(updated,lp.softmax(-1),belief.scale_values,belief.legal_context)


def component_log_densities(f,truth,mask):
    mu=f.positions;b,s,t,n,_=mu.shape;state=mu.new_zeros(b,s,n,2,2);cov=f.initial_covariance
    F,Q=process(f.decay_rate,f.acceleration_scale,f.dt);quadratic=mu.new_zeros(b,s,n);logdet=mu.new_zeros(b,s,n)
    for j in range(t):
        state=F@state;cov=F@cov@F.T+Q
        error=truth[:,None,j]-mu[:,:,j]-state[...,0,:];variance=(cov[...,0,0]+f.readout_scale.square()).clamp_min(1e-8)
        valid=mask[:,None,j];quadratic+=(error.square()/variance[...,None]).sum(-1)*valid
        logdet+=2*variance.log()*valid;gain=cov[..., :,0]/variance[...,None]
        state=state+gain[...,None]*error[...,None,:]*valid[...,None,None]
        cov=cov-gain[..., :,None]*cov[...,0,None,:]*valid[...,None,None]
    count=2*mask.sum(1)
    return scale_log_density(quadratic,logdet,count[:,None],f.scale_values),count


def trajectory_log_probability(f,truth,mask):
    density,count=component_log_densities(f,truth,mask)
    component=(density+f.agent_scale_probabilities.clamp_min(1e-12).log()).logsumexp(-1).sum(-1)
    return (component+f.weights.clamp_min(1e-12).log()).logsumexp(-1),count.sum(-1)


def energy_score(f,truth,mask,draws=1,generator=None):
    a,w,_=f.sample(draws,generator);b,_,_=f.sample(draws,generator)
    m=mask[...,None].to(a.dtype);count=2*m.sum((1,2,3));scale=count.clamp_min(1)
    first=(((a-truth[:,None])*m[:,None]).square().sum((2,3,4))/scale[:,None]+1e-10).sqrt()
    pair=(((a[:,:,None]-b[:,None])*m[:,None,None]).square().sum((3,4,5))/scale[:,None,None]+1e-10).sqrt()
    return (w*first).sum(1)-.5*(w[:,:,None]*w[:,None]*pair).sum((1,2)),count>0


def marginal_cdf(f,locations):
    if hasattr(f,'branch_probabilities'):
        from eventforecastwm.performance_v2.regime import marginal_cdf as regime
        return regime(f,locations)
    result=0.;base=f.covariance[...,0,0]+f.readout_scale.square()
    for l,v in enumerate(f.scale_values):
        mean=f.positions;sigma=(base*v).sqrt()[...,None];probability=f.agent_scale_probabilities[...,l][:,:,None,:,None]
        if locations.ndim==6:mean=mean[...,None];sigma=sigma[...,None];probability=probability[...,None]
        result=result+probability*.5*(1+torch.erf((locations-mean)/(sigma*math.sqrt(2))))
    return result


def occupancy(f,extent=8.,size=32):
    if hasattr(f,'branch_probabilities'):
        from eventforecastwm.performance_v2.regime import occupancy as regime
        return regime(f,extent,size)
    p=f.positions;edges=torch.linspace(-extent,extent,size+1,device=p.device,dtype=p.dtype)
    base=f.covariance[...,0,0]+f.readout_scale.square();per_agent=0.
    for l,v in enumerate(f.scale_values):
        sigma=(base*v).sqrt();cdf=.5*(1+torch.erf((edges-p[...,None])/(sigma[...,None,None]*math.sqrt(2))))
        mass=(cdf[...,1:]-cdf[...,:-1]).clamp_min(0.)
        per_agent=per_agent+mass[...,1,:,None]*mass[...,0,None,:]*f.agent_scale_probabilities[...,l][:,:,None,:,None,None]
    per_agent=per_agent*f.valid[:,None,None,:,None,None]
    return ((1-(1-per_agent).prod(3))*f.weights[:,:,None,None,None]).sum(1)
