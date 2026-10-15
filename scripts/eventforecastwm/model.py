"""Query-independent recurrent mixture of coherent trajectory distributions.

For integrated noise, each scene mode has a deterministic recurrent mean and
an independent linear Gaussian position/velocity residual for each human/axis:
u[t+1] = F u[t] + [dt²/2,dt] a[t], a~N(0,q²). Native body-center readouts add
N(0,r²) jitter. The shared categorical mode induces joint inter-agent dependence.
Covariance, residual mean and physical clock persist across imagination calls.
This is a structured residual distribution, not diffusion or an interactive policy.
"""
from dataclasses import dataclass,replace
import math
import torch
from torch import nn
from eventbeliefwm.model import BeliefWorldModel,BeliefConfig,Belief


@dataclass
class SceneBelief:
    core: Belief
    covariance: torch.Tensor             # B,S,N,2,2; state axes=(position,velocity)
    residual_mean: torch.Tensor          # B,S,N,2,2; state axis then spatial axis

    def detach(self):
        return SceneBelief(self.core.detach(),self.covariance.detach(),self.residual_mean.detach())


@dataclass
class SceneForecast:
    next_belief: SceneBelief
    positions: torch.Tensor              # B,S,T,N,2 component means
    velocities: torch.Tensor
    weights: torch.Tensor
    covariance: torch.Tensor             # B,S,T,N,2,2
    initial_covariance: torch.Tensor
    initial_residual_mean: torch.Tensor
    acceleration_scale: torch.Tensor
    readout_scale: torch.Tensor
    noise_kind: str
    dt: float
    valid: torch.Tensor
    times: torch.Tensor

    @property
    def position_variance(self):
        return self.covariance[...,0,0]+self.readout_scale.square()

    @property
    def validity(self):
        return dict(observed_track_support=self.valid,within_trained_horizon=self.times<=3.+1e-6,
            joint_probability_defined=True,identity_existence_calibrated=False,
            anonymous_joint_trajectories_supported=False,calibration_measured=False)

    def sample(self,draws=4,generator=None,initial_state=None):
        """Return paths, weights and persistent residuals for segmented sampling.

        Pass the returned residuals and the SAME generator to the next forecast's
        sample call. After a real observation, draw fresh posterior residuals.
        """
        return sample_components(self,draws,generator,initial_state=initial_state,return_state=True)


class ForecastWorldModel(nn.Module):
    def __init__(self,kind='stochastic',width=128,modes=6,noise='iid',motion_window=.75,interaction_layers=2):
        super().__init__()
        self.recipe=dict(kind=kind,width=width,modes=modes,noise=noise,motion_window=motion_window,
                         interaction_layers=interaction_layers)
        self.core=BeliefWorldModel(BeliefConfig(kind=kind,width=width,modes=modes,
            motion_window_s=motion_window,interaction_layers=interaction_layers))
        self.noise=noise
        if noise not in ('iid','integrated'):raise ValueError(noise)
        self.log_initial_position=nn.Parameter(torch.tensor(math.log(.12)))
        self.log_initial_velocity=nn.Parameter(torch.tensor(math.log(.5)))
        self.log_acceleration=nn.Parameter(torch.tensor(math.log(.8)))
        self.log_readout=nn.Parameter(torch.tensor(math.log(.08 if noise=='integrated' else .5)))
        # Old event/detection and original observation-correction nuisance heads
        # do not participate in this stage's physical training objective.
        for name,p in self.core.named_parameters():
            if name.startswith(('correction.','anonymous_head.')) or name in (
                'log_position_scale','log_contact_temperature','contact_offset','detection_logit'):
                p.requires_grad_(False)

    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
            trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),recipe=self.recipe)

    def encode_history(self,legal_history,static_map=None,sensor_config=None,robot_geometry=None,**kwargs):
        b=self.core.encode_history(legal_history,static_map,sensor_config,robot_geometry,**kwargs)
        shape=b.positions.shape[:3]
        cov=b.positions.new_zeros(*shape,2,2)
        if self.noise=='integrated':
            cov[...,0,0]=self.log_initial_position.exp().clamp(.01,3).square()
            cov[...,1,1]=self.log_initial_velocity.exp().clamp(.01,3).square()
        return SceneBelief(b,cov,b.positions.new_zeros(*shape,2,2))

    def imagine(self,belief,candidate_query=None,horizon_s=3.):
        # The optional argument is accepted for API compatibility, never read.
        steps=round(horizon_s/.25)
        if steps<1 or abs(steps*.25-horizon_s)>1e-7:raise ValueError('Canonical.25s boundaries required')
        current=belief.core;cov=belief.covariance;residual=belief.residual_mean
        f=cov.new_tensor([[1.,.25],[0.,1.]]);g=cov.new_tensor([.5*.25**2,.25])
        q=self.log_acceleration.exp().clamp(.01,5.)
        r=self.log_readout.exp().clamp(.01,5.)
        positions=[];velocities=[];covariances=[]
        for _ in range(steps):
            current=self.core._step(current)
            if self.noise=='integrated':
                cov=f@cov@f.T+q.square()*g[:,None]*g[None,:]
                residual=f@residual
            positions.append(current.positions+residual[...,0,:])
            velocities.append(current.velocities+residual[...,1,:]);covariances.append(cov)
        successor=SceneBelief(current,cov,residual)
        return SceneForecast(successor,torch.stack(positions,2),torch.stack(velocities,2),
            current.scenario_weights,torch.stack(covariances,2),belief.covariance,belief.residual_mean,
            q,r,self.noise,.25,current.valid,
            belief.core.clock[:,None]+torch.arange(1,steps+1,device=current.positions.device)*.25)

    def observe(self,belief,actual_arrived_observation):
        """Gaussian correction of existing legal identities at current acquisition.

        This update never consumes predicted detections. Sub-tick acquisition
        lag uses a local constant-velocity observation operator. It is an
        explicitly approximate delayed update, limited to one canonical tick;
        changed-ID association is a separate layer.
        """
        obs=actual_arrived_observation;c=belief.core
        if obs.get('source','actual_arrived')!='actual_arrived':raise ValueError('Actual observations only')
        acquired=obs['acquisition_times'].to(c.clock.device)
        mask=obs['observed'].to(c.clock.device).bool()
        time=torch.as_tensor(obs['timestamp_s'],device=c.clock.device).expand_as(c.clock)
        if not torch.allclose(time,c.clock,atol=1e-5,rtol=0):raise ValueError('Observation clock differs from belief')
        lag=time[:,None]-acquired
        if (mask & ((lag < -1e-5)|(lag > .25001))).any():raise ValueError('Future or more-than-one-tick delayed observation')
        measurement=obs['positions'].to(c.positions)
        if measurement.shape!=c.last_observed_positions.shape or not torch.isfinite(measurement[mask]).all():
            raise ValueError('Finite positions for existing identities required')
        if 'track_ids' in obs and not torch.equal(obs['track_ids'].to(c.track_ids),c.track_ids):
            raise ValueError('Changed-ID association required before Gaussian correction')
        if (mask & ~c.valid).any():raise ValueError('Birth needs legal history assimilation')
        fresh=mask & (acquired>c.last_observed_times+1e-7)
        cov=belief.covariance;mean=belief.residual_mean
        r2=self.log_readout.exp().clamp(.01,5.).square()
        H=torch.stack([torch.ones_like(lag),-lag],-1).to(c.positions)[:,None]
        physical_lag=lag.to(c.positions)
        innovation=measurement[:,None]-(c.positions-physical_lag[:,None,:,None]*c.velocities)-(H[...,None]*mean).sum(-2)
        ph=(cov@H[...,None]).squeeze(-1)
        var=((H*ph).sum(-1)+r2).clamp_min(1e-8)
        logp=(-.5*innovation.square()/var[...,None]-.5*torch.log(2*math.pi*var)[...,None]).sum(-1)
        weights=(c.scenario_weights.clamp_min(1e-12).log()+(logp*fresh[:,None]).sum(-1)).softmax(-1)
        gain=ph/var[...,None]
        mean=mean+gain[...,None]*innovation[...,None,:]*fresh[:,None,:,None,None]
        cov=cov-gain[..., :,None]*ph[...,None,:]*fresh[:,None,:,None,None]
        corrected_position=c.positions+mean[...,0,:]
        corrected_velocity=c.velocities+mean[...,1,:]
        if self.noise=='iid':
            corrected_position=torch.where(fresh[:,None,:,None],measurement[:,None]+physical_lag[:,None,:,None]*c.velocities,c.positions)
        c=replace(c,positions=corrected_position,velocities=corrected_velocity,
            scenario_weights=weights,age=torch.where(fresh,lag.to(c.age),c.age),
            last_observed_times=torch.where(fresh,acquired,c.last_observed_times),
            last_observed_positions=torch.where(fresh[...,None],measurement,c.last_observed_positions))
        return SceneBelief(c,cov,torch.zeros_like(mean))


def trajectory_log_probability(forecast,truth,mask):
    """Exact joint mixture likelihood, marginalizing missing measurements.

    Kalman innovations are likelihood computation only, never imagined inputs.
    One mixture mode covers all supported humans and times in a root.
    """
    mu=forecast.positions;b,s,t,n,_=mu.shape
    residual=mu.new_zeros(b,s,n,2,2);cov=forecast.initial_covariance
    f=mu.new_tensor([[1.,forecast.dt],[0.,1.]])
    g=mu.new_tensor([.5*forecast.dt**2,forecast.dt]);q=forecast.acceleration_scale
    r2=forecast.readout_scale.square();logp=mu.new_zeros(b,s)
    for j in range(t):
        if forecast.noise_kind=='integrated':
            residual=f@residual;cov=f@cov@f.T+q.square()*g[:,None]*g[None,:]
        innovation=truth[:,None,j]-mu[:,:,j]-residual[...,0,:]
        variance=(cov[...,0,0]+r2).clamp_min(1e-8)
        valid=mask[:,None,j]
        logp+=((-.5*innovation.square()/variance[...,None]-.5*torch.log(2*math.pi*variance)[...,None])*valid[...,None]).sum((2,3))
        if forecast.noise_kind=='integrated':
            gain=cov[..., :,0]/variance[...,None]
            residual=residual+gain[...,None]*innovation[...,None,:]*valid[...,None,None]
            cov=cov-gain[..., :,None]*cov[...,0,None,:]*valid[...,None,None]
    count=mask.sum((1,2))*2
    joint=torch.logsumexp(forecast.weights.clamp_min(1e-12).log()+logp,1)
    return joint,count


def sample_components(forecast,draws=1,generator=None,*,initial_state=None,return_state=False):
    """Conditional coherent paths, B,S*draws,T,N,2 with stratified weights."""
    mu=forecast.positions;b,s,t,n,_=mu.shape
    rand=lambda shape:torch.randn(shape,device=mu.device,dtype=mu.dtype,generator=generator)
    if initial_state is not None:
        if initial_state.shape!=(b,s,draws,n,2,2):raise ValueError('Persistent sample identity/dimension mismatch')
        state=initial_state
    elif forecast.noise_kind=='integrated':
        eye=torch.eye(2,device=mu.device,dtype=mu.dtype)
        chol=torch.linalg.cholesky(forecast.initial_covariance+1e-8*eye)
        state=chol[:,:,None]@rand((b,s,draws,n,2,2))
    else:state=mu.new_zeros(b,s,draws,n,2,2)
    f=mu.new_tensor([[1.,forecast.dt],[0.,1.]])
    g=mu.new_tensor([.5*forecast.dt**2,forecast.dt]);values=[]
    for j in range(t):
        if forecast.noise_kind=='integrated':
            state=f@state+g[None,None,None,None,:,None]*forecast.acceleration_scale*rand((b,s,draws,n,1,2))
        values.append(mu[:,:,j,None]+state[...,0,:]+forecast.readout_scale*rand((b,s,draws,n,2)))
    positions=torch.stack(values,3).reshape(b,s*draws,t,n,2)
    weights=(forecast.weights[:,:,None]/draws).expand(b,s,draws).reshape(b,s*draws)
    return (positions,weights,state) if return_state else (positions,weights)


def energy_score(forecast,truth,mask,draws=1,generator=None):
    """Unbiased stratified Monte Carlo energy score of the SAME NLL distribution.

    Independent A/B paths retain within-component variance in the pair term.
    Normalization is fixed by the supported coordinates for each root.
    """
    a,w=sample_components(forecast,draws,generator);b,_=sample_components(forecast,draws,generator)
    m=mask[...,None].to(a.dtype);count=m.sum((1,2,3))*2
    scale=count.clamp_min(1)
    first=(((a-truth[:,None])*m[:,None]).square().sum((2,3,4))/scale[:,None]+1e-10).sqrt()
    pair=(((a[:,:,None]-b[:,None])*m[:,None,None]).square().sum((3,4,5))/scale[:,None,None]+1e-10).sqrt()
    return (w*first).sum(1)-.5*(w[:,:,None]*w[:,None]*pair).sum((1,2)),count>0
