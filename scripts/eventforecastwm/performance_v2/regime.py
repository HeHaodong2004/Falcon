"""Persistent learned/stopped agent regimes under a shared scene motion mode.

The stopped branch anchors at the last legal observation. Its OU residual mean
and covariance propagate and assimilate arrivals; no identity is invented. The
physical branch keeps the incumbent recurrent transition and correction. Branch
and variance probabilities receive Bayesian observation likelihood updates.
"""
from dataclasses import dataclass,fields,replace
import math
import torch
from torch import nn

from eventforecastwm.model_agent_scale import AgentScaleWorldModel,AgentScaleForecast,component_log_densities
from eventforecastwm.model_damped import process
from eventforecastwm.model_scale_mixture import scale_log_density


@dataclass
class RegimeBelief:
    moving: object
    anchor: torch.Tensor                  # B,S,N,2, fixed legal anchor
    stopped_residual_mean: torch.Tensor    # B,S,N,2,2, state axis then x/y
    branch_probabilities: torch.Tensor     # B,S,N,2, persistent conditional on z
    stopped_scale_probabilities: torch.Tensor

    @property
    def core(self):return self.moving.core  # learned branch; shared identity/clock metadata

    @property
    def covariance(self):
        # Within-branch residual covariance. Branch spread is separately explicit.
        rho=self.branch_probabilities[...,0,None]*self.moving.agent_scale_probabilities
        rho=rho+self.branch_probabilities[...,1,None]*self.stopped_scale_probabilities
        return self.moving.physical.covariance*(rho*self.moving.scale_values).sum(-1)[...,None,None]

    def detach(self):
        return RegimeBelief(self.moving.detach(),self.anchor.detach(),self.stopped_residual_mean.detach(),
                           self.branch_probabilities.detach(),self.stopped_scale_probabilities.detach())


@dataclass
class RegimeSampleState:
    gaussian_state: torch.Tensor
    variance: torch.Tensor
    branch: torch.Tensor


@dataclass
class RegimeForecast(AgentScaleForecast):
    moving_positions: torch.Tensor
    moving_velocities: torch.Tensor
    stopped_positions: torch.Tensor
    stopped_velocities: torch.Tensor
    branch_probabilities: torch.Tensor
    stopped_scale_probabilities: torch.Tensor

    @property
    def position_variance(self):
        p=self.branch_probabilities
        rho=p[...,0,None]*self.agent_scale_probabilities+p[...,1,None]*self.stopped_scale_probabilities
        noise=(self.covariance[...,0,0]+self.readout_scale.square())*(rho*self.scale_values).sum(-1)[:,:,None]
        spread=(p[:,:,None,:,0]*(self.moving_positions-self.positions).square().mean(-1)+
                p[:,:,None,:,1]*(self.stopped_positions-self.positions).square().mean(-1))
        return noise+spread

    def sample(self,draws=4,generator=None,initial_state=None):
        b,s,t,n,_=self.positions.shape
        if initial_state is None:
            rho=torch.stack([self.agent_scale_probabilities,self.stopped_scale_probabilities],-2)
            probability=(rho*self.branch_probabilities[...,None]).flatten(-2)
            index=torch.multinomial(probability.reshape(b*s*n,-1),draws,replacement=True,generator=generator)
            index=index.reshape(b,s,n,draws).permute(0,1,3,2)
            branch=torch.div(index,len(self.scale_values),rounding_mode='floor');variance=self.scale_values[index%len(self.scale_values)]
            eye=torch.eye(2,device=self.positions.device,dtype=self.positions.dtype)
            chol=torch.linalg.cholesky(self.initial_covariance+1e-8*eye)
            state=chol[:,:,None]@torch.randn(b,s,draws,n,2,2,device=self.positions.device,dtype=self.positions.dtype,generator=generator)
        else:
            if not isinstance(initial_state,RegimeSampleState) or initial_state.branch.shape!=(b,s,draws,n):
                raise ValueError('Persistent regime/variance/residual identities required')
            state,variance,branch=initial_state.gaussian_state,initial_state.variance,initial_state.branch
        transition,Q=process(self.decay_rate,self.acceleration_scale,self.dt);cholQ=torch.linalg.cholesky(Q)
        rand=lambda shape:torch.randn(shape,device=self.positions.device,dtype=self.positions.dtype,generator=generator)
        values=[]
        for j in range(t):
            state=transition@state+cholQ@rand((b,s,draws,n,2,2))
            mean=torch.where(branch[...,None].bool(),self.stopped_positions[:,:,j,None],self.moving_positions[:,:,j,None])
            values.append(mean+(state[...,0,:]+self.readout_scale*rand((b,s,draws,n,2)))*variance.sqrt()[...,None])
        paths=torch.stack(values,3).reshape(b,s*draws,t,n,2)
        weights=(self.weights[:,:,None]/draws).expand(b,s,draws).reshape(b,s*draws)
        return paths,weights,RegimeSampleState(state,variance,branch)


def branches(f):
    values={field.name:getattr(f,field.name) for field in fields(AgentScaleForecast)}
    moving=AgentScaleForecast(**dict(values,positions=f.moving_positions,velocities=f.moving_velocities))
    stopped=AgentScaleForecast(**dict(values,positions=f.stopped_positions,velocities=f.stopped_velocities,
        agent_scale_probabilities=f.stopped_scale_probabilities))
    return moving,stopped


def agent_density(f,truth,mask):
    branch_scores=[];count=None
    for branch in branches(f):
        density,count=component_log_densities(branch,truth,mask)
        branch_scores.append((density+branch.agent_scale_probabilities.log()).logsumexp(-1))
    agent=(torch.stack(branch_scores,-1)+f.branch_probabilities.log()).logsumexp(-1)
    return agent,count


def component_density(f,truth,mask):
    agent,count=agent_density(f,truth,mask)
    return (agent*(count>0)[:,None]).sum(-1),count.sum(-1)


def trajectory_log_probability(f,truth,mask):
    density,count=component_density(f,truth,mask)
    return (density+f.weights.log()).logsumexp(-1),count


def marginal_cdf(f,locations):
    from eventforecastwm.model_agent_scale import marginal_cdf as cdf
    result=0.
    for index,branch in enumerate(branches(f)):
        value=cdf(branch,locations);probability=f.branch_probabilities[...,index][:,:,None,:,None]
        if value.ndim==6:probability=probability[...,None]
        result=result+probability*value
    return result


def occupancy(f,extent=8.,size=32):
    mass=0.
    for index,branch in enumerate(branches(f)):
        edges=torch.linspace(-extent,extent,size+1,device=f.positions.device,dtype=f.positions.dtype)
        base=branch.covariance[...,0,0]+branch.readout_scale.square();agent=0.
        for category,value in enumerate(branch.scale_values):
            sigma=(base*value).sqrt()
            cdf=.5*(1+torch.erf((edges-branch.positions[...,None])/(sigma[...,None,None]*math.sqrt(2))))
            axis=(cdf[...,1:]-cdf[...,:-1]).clamp_min(0.)
            agent=agent+axis[...,1,:,None]*axis[...,0,None,:]*branch.agent_scale_probabilities[...,category][:,:,None,:,None,None]
        mass=mass+agent*branch.valid[:,None,None,:,None,None]*f.branch_probabilities[...,index][:,:,None,:,None,None]
    return ((1-(1-mass).prod(3))*f.weights[:,:,None,None,None]).sum(1)


class RegimeWorldModel(nn.Module):
    def __init__(self,physical_recipe,stop_weight=.5,variant='stopped_agent_scale_v2'):
        super().__init__()
        if variant!='stopped_agent_scale_v2' or physical_recipe.get('variant')!='agent_scene_scale_v1':
            raise ValueError('Calibrated per-agent Gaussian variance parent required')
        if not 0<=stop_weight<=1:raise ValueError('Invalid prior stop dilution')
        self.physical=AgentScaleWorldModel(**physical_recipe);self.stop_head=nn.Linear(8,2)
        self.stop_weight=stop_weight
        self.recipe=dict(variant=variant,physical_recipe=physical_recipe,stop_weight=stop_weight)

    @property
    def core(self):return self.physical.core

    @property
    def noise(self):return self.physical.noise

    @property
    def raw_decay_rate(self):return self.physical.raw_decay_rate

    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    regime_gate_parameters=sum(p.numel() for p in self.stop_head.parameters()),recipe=self.recipe)

    def encode_history(self,*args,**kwargs):
        moving=self.physical.encode_history(*args,**kwargs)
        stop=self.stop_head(moving.legal_context).softmax(-1)[...,1]*self.stop_weight
        probability=torch.stack([1-stop,stop],-1)[:,None].expand(-1,moving.core.positions.shape[1],-1,-1)
        anchor=moving.core.last_observed_positions[:,None].expand_as(moving.core.positions)
        return RegimeBelief(moving,anchor,torch.zeros_like(moving.physical.residual_mean),probability,moving.agent_scale_probabilities)

    def imagine(self,belief,candidate_query=None,horizon_s=3.):
        forecast=self.physical.imagine(belief.moving,candidate_query,horizon_s)
        transition,_=process(forecast.decay_rate,forecast.acceleration_scale,forecast.dt)
        residual=belief.stopped_residual_mean;positions=[];velocities=[]
        for _ in range(forecast.positions.shape[2]):
            residual=transition@residual
            positions.append(belief.anchor+residual[...,0,:]);velocities.append(residual[...,1,:])
        stopped_positions=torch.stack(positions,2);stopped_velocities=torch.stack(velocities,2)
        p=belief.branch_probabilities[:,:,None]
        successor=RegimeBelief(forecast.next_belief,belief.anchor,residual,belief.branch_probabilities,belief.stopped_scale_probabilities)
        values={field.name:getattr(forecast,field.name) for field in fields(AgentScaleForecast)}
        values.update(next_belief=successor,positions=forecast.positions*p[...,0,None]+stopped_positions*p[...,1,None],
                      velocities=forecast.velocities*p[...,0,None]+stopped_velocities*p[...,1,None])
        return RegimeForecast(**values,moving_positions=forecast.positions,moving_velocities=forecast.velocities,
            stopped_positions=stopped_positions,stopped_velocities=stopped_velocities,
            branch_probabilities=belief.branch_probabilities,stopped_scale_probabilities=belief.stopped_scale_probabilities)

    def observe(self,belief,observation):
        # Parent validates source, identity, clock, finite values and <=.25s lag.
        updated=self.physical.observe(belief.moving,observation);c=belief.core
        acquired=observation['acquisition_times'].to(c.clock.device)
        fresh=observation['observed'].to(c.clock.device).bool()&(acquired>c.last_observed_times+1e-7)
        lag=(c.clock[:,None]-acquired).to(c.positions);H=torch.stack([torch.ones_like(lag),-lag],-1)[:,None]
        covariance=belief.moving.physical.covariance
        ph=(covariance@H[...,None]).squeeze(-1)
        variance=((H*ph).sum(-1)+self.physical.log_readout.exp().clamp(.01,5.).square()).clamp_min(1e-8)
        measurement=observation['positions'].to(c.positions)[:,None]
        error_moving=measurement-(c.positions-lag[:,None,:,None]*c.velocities)-(H[...,None]*belief.moving.physical.residual_mean).sum(-2)
        error_stopped=measurement-belief.anchor-(H[...,None]*belief.stopped_residual_mean).sum(-2)
        def density(error,rho):
            q=(error.square()/variance[...,None]).sum(-1)*fresh[:,None]
            return scale_log_density(q,2*variance.log()*fresh[:,None],2*fresh[:,None],belief.moving.scale_values)+rho.log()
        moving_lp=density(error_moving,belief.moving.agent_scale_probabilities)
        stopped_lp=density(error_stopped,belief.stopped_scale_probabilities)
        joint=torch.stack([moving_lp,stopped_lp],-2)+belief.branch_probabilities.log()[...,None]
        agent=joint.logsumexp((-1,-2));weights=(c.scenario_weights.log()+agent.sum(-1)).softmax(-1)
        updated.physical.core=replace(updated.core,scenario_weights=weights)
        stopped_mean=belief.stopped_residual_mean+(ph/variance[...,None])[...,None]*error_stopped[...,None,:]*fresh[:,None,:,None,None]
        return RegimeBelief(updated,belief.anchor,stopped_mean,joint.logsumexp(-1).softmax(-1),stopped_lp.softmax(-1))
