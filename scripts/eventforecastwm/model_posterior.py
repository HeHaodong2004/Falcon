"""Learn a legal-history-conditioned initial belief for lost/velocity-unknown tracks.

The baseline extrapolates all unobserved history with CV before learned future
recurrence starts. This small head can revise that initial physical belief.
It runs only during history encoding, never during imagined transitions.
"""
from dataclasses import replace
import torch
from torch import nn
from eventforecastwm.model import ForecastWorldModel


def attach_posterior(model):
    d=model.core.config.width
    # Register with the physical parameters so it receives the physical LR.
    model.core.posterior_head=nn.Sequential(nn.Linear(d+8+2+2+1+1,64),nn.GELU(),nn.Linear(64,4))
    nn.init.zeros_(model.core.posterior_head[-1].weight);nn.init.zeros_(model.core.posterior_head[-1].bias)


def correct_initial(model,belief):
    c=belief.core;b,s,n,d=c.hidden.shape;age=c.age[:,None,:,None].expand(b,s,n,1)
    vv=c.velocity_valid[:,None,:,None].expand(b,s,n,1).to(c.positions)
    mode=model.core.mode_embedding.weight[None,:,None].expand(b,s,n,8)
    feature=torch.cat([c.hidden,mode,c.positions/6.,c.velocities/2.,age/5.,vv],-1)
    adjustment=model.core.posterior_head(feature).tanh()*c.valid[:,None,:,None]
    # Exact observations receive no position adjustment. Legal time since the
    # observation bounds the size of an occluded-state revision.
    position=c.positions+(3.*age).clamp(max=9.)*adjustment[...,:2]
    gate=torch.maximum(1.-torch.exp(-age),1.-vv)
    velocity=c.velocities+1.5*gate*adjustment[...,2:]
    velocity=velocity*(3./velocity.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp_max(1.)
    return replace(belief,core=replace(c,positions=position,velocities=velocity))


class PosteriorForecastWorldModel(ForecastWorldModel):
    def __init__(self,variant='posterior_v1',**kwargs):
        if variant!='posterior_v1':raise ValueError(variant)
        super().__init__(**kwargs);attach_posterior(self);self.recipe['variant']=variant

    def encode_history(self,*args,**kwargs):
        return correct_initial(self,super().encode_history(*args,**kwargs))
