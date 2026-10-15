"""Legal-context conditional probabilities for a persistent scene variance.

The recurrent means, OU law and seven variance values are unchanged. A small
linear head predicts their probabilities once from legal initial-belief
features. Imagination persists these probabilities; real observations update
them by Bayes' rule, without rerunning the history encoder or this head.
"""
from dataclasses import dataclass
import torch
from torch import nn
from eventforecastwm.model_scale_mixture import ScaleMixtureWorldModel,ScaleBelief,ScaleForecast,scale_log_density
from eventforecastwm.model_damped import process


def legal_context(core):
    valid=core.valid;count=valid.sum(-1).clamp_min(1);age=(core.age/5.).clamp(0,2)
    speed=(core.velocities.norm(dim=-1)*core.scenario_weights[:,:,None]).sum(1)/2.
    mean=lambda x:(x*valid).sum(-1)/count
    avg=mean(speed)
    features=torch.stack([mean(age),(age*valid).max(-1).values,mean((core.age>.25).to(age)),
        mean(core.velocity_valid.to(age)),avg,mean((speed-avg[:,None]).square()),
        torch.log1p(valid.sum(-1).to(age))/3.,mean((core.age>1.).to(age))],-1)
    return features


@dataclass
class ConditionalBelief(ScaleBelief):
    legal_context: torch.Tensor

    def detach(self):
        return ConditionalBelief(self.physical.detach(),self.scale_probabilities.detach(),
            self.scale_values.detach(),self.legal_context.detach())


@dataclass
class ConditionalForecast(ScaleForecast):
    legal_context: torch.Tensor


class ConditionalScaleWorldModel(ScaleMixtureWorldModel):
    def __init__(self,variant='conditional_scene_scale_v1',**kwargs):
        if variant!='conditional_scene_scale_v1':raise ValueError(variant)
        super().__init__(**kwargs)
        self.scale_context=nn.Linear(8,len(self.scale_values))
        nn.init.zeros_(self.scale_context.weight);nn.init.zeros_(self.scale_context.bias)
        self.recipe['variant']=variant

    def probabilities(self,context,modes):
        return (self.scale_logits+self.scale_context(context)).softmax(-1)[:,None].expand(-1,modes,-1)

    def encode_history(self,*args,**kwargs):
        belief=super().encode_history(*args,**kwargs);context=legal_context(belief.core)
        return ConditionalBelief(belief.physical,self.probabilities(context,belief.core.scenario_weights.shape[1]),
            belief.scale_values,context)

    def imagine(self,belief,candidate_query=None,horizon_s=3.):
        f=super().imagine(belief,candidate_query,horizon_s);v=dict(f.__dict__);n=f.next_belief
        v['next_belief']=ConditionalBelief(n.physical,n.scale_probabilities,n.scale_values,belief.legal_context)
        return ConditionalForecast(**v,legal_context=belief.legal_context)

    def observe(self,belief,observation):
        updated=super().observe(belief,observation)
        return ConditionalBelief(updated.physical,updated.scale_probabilities,updated.scale_values,belief.legal_context)


def component_log_densities(f,truth,mask):
    """Sufficient Gaussian likelihood terms; no conditional mixture weights."""
    mu=f.positions;b,s,t,n,_=mu.shape;state=mu.new_zeros(b,s,n,2,2);cov=f.initial_covariance
    F,Q=process(f.decay_rate,f.acceleration_scale,f.dt);quadratic=mu.new_zeros(b,s);logdet=mu.new_zeros(b,s)
    for j in range(t):
        state=F@state;cov=F@cov@F.T+Q
        error=truth[:,None,j]-mu[:,:,j]-state[...,0,:]
        variance=(cov[...,0,0]+f.readout_scale.square()).clamp_min(1e-8);valid=mask[:,None,j]
        quadratic+=(error.square()/variance[...,None]*valid[...,None]).sum((2,3))
        logdet+=(2*variance.log()*valid).sum(-1);gain=cov[..., :,0]/variance[...,None]
        state=state+gain[...,None]*error[...,None,:]*valid[...,None,None]
        cov=cov-gain[..., :,None]*cov[...,0,None,:]*valid[...,None,None]
    count=2*mask.sum((1,2))
    return scale_log_density(quadratic,logdet,count[:,None],f.scale_values),count
