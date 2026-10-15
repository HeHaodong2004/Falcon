"""Finite persistent scene-scale mixture around coherent OU trajectory means.

One categorical variance multiplier is shared by all agents, coordinates and
times in a scene. Unlike an inverse-gamma scale law, its mixing probabilities
are free parameters. This is an explicit joint probability model, not smoothing
of an occupancy raster. Calibration changes no learned physical mean.
"""
from dataclasses import dataclass,replace
import math
import torch
from eventforecastwm.model import SceneForecast
from eventforecastwm.model_damped import DampedForecastWorldModel,process,sample_components


@dataclass
class ScaleBelief:
    physical: object
    scale_probabilities: torch.Tensor       # B,S,L conditional on scene mode
    scale_values: torch.Tensor              # L variance multipliers

    @property
    def core(self):return self.physical.core

    @property
    def covariance(self):
        mean=(self.scale_probabilities*self.scale_values).sum(-1)
        return self.physical.covariance*mean[:,:,None,None,None]

    def detach(self):return ScaleBelief(self.physical.detach(),self.scale_probabilities.detach(),self.scale_values.detach())


@dataclass
class ScaleSampleState:
    gaussian_state: torch.Tensor
    variance: torch.Tensor


@dataclass
class ScaleForecast(SceneForecast):
    decay_rate: torch.Tensor
    scale_probabilities: torch.Tensor
    scale_values: torch.Tensor

    @property
    def position_variance(self):
        return (self.covariance[...,0,0]+self.readout_scale.square())*(self.scale_probabilities*self.scale_values).sum(-1)[:,:,None,None]

    def sample(self,draws=4,generator=None,initial_state=None):
        b,s,t,n,_=self.positions.shape
        if initial_state is None:
            ix=torch.multinomial(self.scale_probabilities.reshape(b*s,-1),draws,replacement=True,generator=generator)
            variance=self.scale_values[ix].reshape(b,s,draws);raw=None
        else:
            if not isinstance(initial_state,ScaleSampleState) or initial_state.variance.shape!=(b,s,draws):raise ValueError('Persistent scale identities required')
            variance=initial_state.variance;raw=initial_state.gaussian_state
        paths,weights,state=sample_components(self,draws,generator,raw)
        mean=self.positions[:,:,None].expand(b,s,draws,t,n,2).reshape_as(paths)
        paths=mean+(paths-mean)*variance.reshape(b,s*draws,1,1,1).sqrt()
        return paths,weights,ScaleSampleState(state,variance)


def scale_log_density(quadratic,logdet,count,values):
    return -.5*(quadratic[...,None]/values+logdet[...,None]+count[...,None]*(math.log(2*math.pi)+values.log()))


class ScaleMixtureWorldModel(DampedForecastWorldModel):
    def __init__(self,variant='finite_scene_scale_v1',scale_std=(.1,.25,.5,1.,2.,4.,8.),**kwargs):
        if variant!='finite_scene_scale_v1':raise ValueError(variant)
        super().__init__(**kwargs)
        self.register_buffer('scale_values',torch.tensor(scale_std).square())
        self.scale_logits=torch.nn.Parameter(torch.zeros(len(scale_std)))
        self.recipe.update(variant=variant,scale_std=list(scale_std))

    def encode_history(self,*args,**kwargs):
        physical=super().encode_history(*args,**kwargs);b,s=physical.core.scenario_weights.shape
        return ScaleBelief(physical,self.scale_logits.softmax(-1).expand(b,s,-1),self.scale_values)

    def imagine(self,belief,candidate_query=None,horizon_s=3.):
        f=super().imagine(belief.physical,candidate_query,horizon_s);values=dict(f.__dict__)
        values['next_belief']=ScaleBelief(f.next_belief,belief.scale_probabilities,belief.scale_values)
        return ScaleForecast(**values,scale_probabilities=belief.scale_probabilities,scale_values=belief.scale_values)

    def observe(self,belief,observation):
        updated=super().observe(belief.physical,observation)
        c=belief.core;acquired=observation['acquisition_times'].to(c.clock.device)
        fresh=observation['observed'].to(c.clock.device).bool()&(acquired>c.last_observed_times+1e-7)
        lag=(c.clock[:,None]-acquired).to(c.positions);H=torch.stack([torch.ones_like(lag),-lag],-1)[:,None]
        cov=belief.physical.covariance;mean=belief.physical.residual_mean
        error=observation['positions'].to(c.positions)[:,None]-(c.positions-lag[:,None,:,None]*c.velocities)-(H[...,None]*mean).sum(-2)
        variance=((H*(cov@H[...,None]).squeeze(-1)).sum(-1)+self.log_readout.exp().clamp(.01,5.).square()).clamp_min(1e-8)
        q=(error.square()/variance[...,None]*fresh[:,None,:,None]).sum((2,3))
        logdet=(2*variance.log()*fresh[:,None]).sum(-1);count=2*fresh.sum(-1)[:,None]
        lp=scale_log_density(q,logdet,count,belief.scale_values)+belief.scale_probabilities.clamp_min(1e-12).log()
        joint=lp+c.scenario_weights.clamp_min(1e-12).log()[...,None]
        joint=(joint-joint.flatten(1).logsumexp(1)[:,None,None]).exp()
        updated.core=replace(updated.core,scenario_weights=joint.sum(-1))
        return ScaleBelief(updated,lp.softmax(-1),belief.scale_values)


def trajectory_log_probability(f,truth,mask):
    mu=f.positions;b,s,t,n,_=mu.shape;state=mu.new_zeros(b,s,n,2,2);cov=f.initial_covariance
    F,Q=process(f.decay_rate,f.acceleration_scale,f.dt);quadratic=mu.new_zeros(b,s);logdet=mu.new_zeros(b,s)
    for j in range(t):
        state=F@state;cov=F@cov@F.T+Q
        innovation=truth[:,None,j]-mu[:,:,j]-state[...,0,:];variance=(cov[...,0,0]+f.readout_scale.square()).clamp_min(1e-8)
        valid=mask[:,None,j];quadratic+=(innovation.square()/variance[...,None]*valid[...,None]).sum((2,3))
        logdet+=(2*variance.log()*valid).sum(-1);gain=cov[..., :,0]/variance[...,None]
        state=state+gain[...,None]*innovation[...,None,:]*valid[...,None,None]
        cov=cov-gain[..., :,None]*cov[...,0,None,:]*valid[...,None,None]
    count=2*mask.sum((1,2));component=scale_log_density(quadratic,logdet,count[:,None],f.scale_values)
    joint=component+f.scale_probabilities.clamp_min(1e-12).log()+f.weights.clamp_min(1e-12).log()[...,None]
    return joint.flatten(1).logsumexp(1),count


def energy_score(f,truth,mask,draws=1,generator=None):
    # Proper Monte Carlo score for evaluation. Discrete scale draws do not give
    # pathwise gradients for their probabilities, so profile this law by NLL.
    a,w,_=f.sample(draws,generator);b,_,_=f.sample(draws,generator)
    m=mask[...,None].to(a.dtype);count=2*m.sum((1,2,3));scale=count.clamp_min(1)
    first=(((a-truth[:,None])*m[:,None]).square().sum((2,3,4))/scale[:,None]+1e-10).sqrt()
    pair=(((a[:,:,None]-b[:,None])*m[:,None,None]).square().sum((3,4,5))/scale[:,None,None]+1e-10).sqrt()
    return (w*first).sum(1)-.5*(w[:,:,None]*w[:,None]*pair).sum((1,2)),count>0


def marginal_cdf(f,locations):
    result=0.;base=f.covariance[...,0,0]+f.readout_scale.square()
    for l,v in enumerate(f.scale_values):
        mean=f.positions;sigma=(base*v).sqrt()[...,None];probability=f.scale_probabilities[...,l,None,None,None]
        if locations.ndim==6:mean=mean[...,None];sigma=sigma[...,None];probability=probability[...,None]
        result=result+probability*.5*(1+torch.erf((locations-mean)/(sigma*math.sqrt(2))))
    return result


def occupancy(f,extent=8.,size=32):
    p=f.positions;edges=torch.linspace(-extent,extent,size+1,device=p.device,dtype=p.dtype)
    base=f.covariance[...,0,0]+f.readout_scale.square();result=0.
    for l,v in enumerate(f.scale_values):
        sigma=(base*v).sqrt();cdf=.5*(1+torch.erf((edges-p[...,None])/(sigma[...,None,None]*math.sqrt(2))))
        mass=(cdf[...,1:]-cdf[...,:-1]).clamp_min(0.)
        per_agent=mass[...,1,:,None]*mass[...,0,None,:]*f.valid[:,None,None,:,None,None]
        result=result+((1-(1-per_agent).prod(3))*(f.weights*f.scale_probabilities[...,l])[:,:,None,None,None]).sum(1)
    return result
