"""A minimal legal relative space-time feature projection into existing query tokens."""
import torch
from torch import nn
from .model import EventQueryWM,ModelConfig
from .inputs import collate,ModelInput

INTERACTION_FEATURES=('nearest_dx','nearest_dz','nearest_distance','relative_vx','relative_vz',
    'closing_speed','observation_age','visible','velocity_valid','observed_track_count','physical_horizon','proximity')

def interaction_features(value):
    h=value.humans;b,n,t,_=h.shape
    last=(~value.history_padding).sum(1)-1
    humans=h[torch.arange(b,device=h.device),:,last,:]
    valid=(~value.human_padding)&(humans[:,:,6]>.5)&(humans[:,:,10]>.5)
    velocity_valid=valid&(humans[:,:,5]>.5)
    velocity=torch.where(velocity_valid[:,:,None],humans[:,:,2:4],torch.zeros_like(humans[:,:,2:4]))
    age=humans[:,:,7].clamp(min=0)
    time=value.query[:,:,2]
    # Tracker positions are last observed, not already extrapolated. This is a legal
    # constant-velocity hypothesis, never a stored future trajectory.
    predicted=humans[:,:,None,:2]+velocity[:,:,None,:]*(time[:,None,:,None]+age[:,:,None,None])
    delta=predicted-value.query[:,None,:,:2]
    distance=delta.square().sum(-1).sqrt();distance=distance.masked_fill(~valid[:,:,None],float('inf'))
    nearest=distance.argmin(1);present=valid.any(1)[:,None,None]
    def gather(v):
        return v.permute(0,2,1,3).gather(2,nearest[:,:,None,None].expand(-1,-1,1,v.shape[-1])).squeeze(2)
    dx=gather(delta);rv=gather(velocity[:,:,None,:].expand(-1,-1,time.shape[1],-1)-value.query[:,None,:,3:5])
    dist=distance.min(1).values
    closing=-(dx*rv).sum(-1)/(dist+1e-5)
    attrs=humans[:,:,[7,4,5]][:,:,None,:].expand(-1,-1,time.shape[1],-1);attrs=gather(attrs)
    counts=torch.log1p(valid.sum(1).float())[:,None].expand_as(time)
    features=torch.cat([dx/6,dist[:,:,None]/6,rv/2,closing[:,:,None]/2,
        attrs[:,:,:1]/3,attrs[:,:,1:],counts[:,:,None],time[:,:,None]/3,
        torch.exp(-dist.square()/0.5)[:,:,None]],-1)
    features=torch.where(present,features,torch.zeros_like(features))
    assert features.shape[-1]==len(INTERACTION_FEATURES)
    return features.clamp(-10,10)

class InteractionWM(EventQueryWM):
    def __init__(self,config=ModelConfig()):
        super().__init__(config)
        self.interaction_projection=nn.Linear(len(INTERACTION_FEATURES),config.width)
        nn.init.zeros_(self.interaction_projection.weight);nn.init.zeros_(self.interaction_projection.bias)

    def project_query(self,value):
        return super().project_query(value)+self.interaction_projection(interaction_features(value))

def interaction_checks(data):
    values=[v for v,y in data if v.humans.shape[0]>0][:4]
    batch=collate(values);features=interaction_features(batch);assert torch.isfinite(features).all()
    reference=EventQueryWM(ModelConfig.family('M')).eval();candidate=InteractionWM(ModelConfig.family('M')).eval()
    missing=candidate.load_state_dict(reference.state_dict(),strict=False)
    assert set(missing.missing_keys)=={'interaction_projection.weight','interaction_projection.bias'} and not missing.unexpected_keys
    with torch.no_grad():
        a=reference(batch);b=candidate(batch);torch.testing.assert_close(a,b,rtol=0,atol=0)
    changed=ModelInput(batch.humans,batch.human_padding,batch.robot,batch.history_padding,batch.query.clone())
    changed.query[:,:,2]+=.5
    assert not torch.equal(interaction_features(changed),features)
    return dict(passed=True,zero_projection_max_logit_error=float((a-b).abs().max()),
        feature_shape=list(features.shape),features=INTERACTION_FEATURES,
        added_parameters=sum(p.numel() for p in candidate.interaction_projection.parameters()),
        inputs='Arrived legal last-observed positions, estimated velocities, observation age/status and nominal physical-time query only',
        source_change='Two-line query projection extension; shared history, attention, fusion, SR head, loss and training schedule preserved')
