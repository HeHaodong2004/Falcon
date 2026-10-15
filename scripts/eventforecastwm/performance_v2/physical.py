"""Small, capacity-matched interventions in the incumbent recurrent mean.

No observation is synthesized. The arrival encoder consumes only distinct
positively observed acquisition times, holding its latent through missing frames.
Elapsed observation age remains explicit in the initial correction and dynamics.
"""
from dataclasses import replace
from types import MethodType

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from eventforecastwm.model import ForecastWorldModel
from eventforecastwm.model_independent import independent_attention
from eventforecastwm.model_posterior import attach_posterior, correct_initial


VARIANTS = {
    'posterior_arrival_v2': ('arrival',),
    'posterior_self_v2': ('self',),
    'posterior_polar_v2': ('polar',),
    'posterior_velocity_v2': ('free_velocity',),
    'posterior_arrival_self_v2': ('arrival','self'),
    'posterior_arrival_polar_v2': ('arrival','polar'),
}


def arrival_forward(gru, inputs, hx=None):
    """Same GRU parameters; absent/repeated packets cannot erase motion memory."""
    if hx is not None:
        raise ValueError('Arrival encoding initializes from an explicit legal history')
    if not isinstance(inputs,torch.Tensor) or inputs.ndim!=3 or inputs.shape[-1]!=13:
        raise ValueError('Expected batch-first prepared legal H13')
    positive=(inputs[...,6]>.5)&(inputs[...,10]>.5)
    acquired=inputs[...,8]-inputs[...,7]
    legal_time=acquired.masked_fill(~positive,-float('inf'))
    previous=torch.cat([torch.full_like(legal_time[:,:1],-float('inf')),
                        legal_time.cummax(1).values[:,:-1]],1)
    fresh=positive & (acquired>previous+1e-6)
    rank=fresh.long().cumsum(1);lengths=rank[:,-1];maximum=max(1,int(lengths.max()))
    packed_input=inputs.new_zeros(inputs.shape[0],maximum,13)
    row,column=fresh.nonzero(as_tuple=True)
    packed_input[row,rank[row,column]-1]=inputs[row,column]
    packed=pack_padded_sequence(packed_input,lengths.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
    output,hidden=nn.GRU.forward(gru,packed)
    encoded,_=pad_packed_sequence(output,batch_first=True,total_length=maximum)
    index=(rank-1).clamp_min(0)[...,None].expand(-1,-1,gru.hidden_size)
    restored=encoded.gather(1,index)*(rank>0)[...,None]
    hidden=hidden*(lengths>0)[None,:,None]
    return restored,hidden


def polar_step(core,belief):
    """Learn speed changes and heading rate, with a coherent trapezoidal step."""
    b,s,n,d=belief.hidden.shape
    interacted,_=core._interact(belief.hidden,belief.valid)
    modes=core.mode_embedding.weight[None,:,None].expand(b,s,n,-1)
    features=torch.cat([belief.positions/6,belief.velocities/2,
        belief.age[:,None,:,None].expand(b,s,n,1)/5,
        belief.velocity_valid[:,None,:,None].expand(b,s,n,1),
        core._geometry_at(belief,belief.positions),modes],-1)
    hidden=core.transition(core.state_input(features).reshape(-1,d),interacted.reshape(-1,d)).reshape(b,s,n,d)
    command=torch.tanh(core.acceleration(hidden))
    speed=belief.velocities.norm(dim=-1,keepdim=True)
    # Zero-speed modes retain the fixed mode direction as an interpretable
    # departure heading. This reads no target or future controller state.
    fallback=core.mode_directions[None,:,None].expand(b,s,n,-1)
    direction=torch.where(speed>1e-5,belief.velocities/speed.clamp_min(1e-5),fallback)
    angle=2.*core.config.dt*command[...,1:2]
    perpendicular=torch.stack([-direction[...,1],direction[...,0]],-1)
    direction=angle.cos()*direction+angle.sin()*perpendicular
    updated_speed=(speed+core.config.dt*core.config.max_acceleration*command[...,:1]).clamp(0,core.config.max_speed)
    velocity=updated_speed*direction
    position=belief.positions+.5*core.config.dt*(belief.velocities+velocity)
    return replace(belief,hidden=hidden,positions=position,velocities=velocity,
                   age=belief.age+core.config.dt,clock=belief.clock+core.config.dt)


def attach(model,variant):
    options=VARIANTS[variant]
    attach_posterior(model)
    if 'arrival' in options:
        model.core.history_gru.forward=MethodType(arrival_forward,model.core.history_gru)
    if 'self' in options:
        model.core._interact=MethodType(independent_attention,model.core)
    if 'polar' in options:
        model.core._step=MethodType(polar_step,model.core)


def correct(model,belief):
    variant=model.recipe.get('mean_variant',model.recipe.get('variant'))
    if 'free_velocity' not in VARIANTS[variant]:return correct_initial(model,belief)
    c=belief.core;b,s,n,d=c.hidden.shape
    age=c.age[:,None,:,None].expand(b,s,n,1)
    vv=c.velocity_valid[:,None,:,None].expand(b,s,n,1).to(c.positions)
    mode=model.core.mode_embedding.weight[None,:,None].expand(b,s,n,8)
    feature=torch.cat([c.hidden,mode,c.positions/6.,c.velocities/2.,age/5.,vv],-1)
    adjustment=model.core.posterior_head(feature).tanh()*c.valid[:,None,:,None]
    # A fresh measured position is retained. A finite-difference velocity is
    # an estimate, so legal history may revise it even when its support is valid.
    position=c.positions+(3.*age).clamp(max=9.)*adjustment[...,:2]
    velocity=c.velocities+1.5*adjustment[...,2:]
    velocity=velocity*(3./velocity.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp_max(1.)
    return replace(belief,core=replace(c,positions=position,velocities=velocity))


class PhysicalWorldModel(ForecastWorldModel):
    def __init__(self,variant='posterior_arrival_v2',**kwargs):
        if variant not in VARIANTS:raise ValueError(variant)
        super().__init__(**kwargs);attach(self,variant);self.recipe['variant']=variant

    def encode_history(self,*args,**kwargs):
        return correct(self,super().encode_history(*args,**kwargs))
