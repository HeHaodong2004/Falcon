"""Capacity-matched ablation of cross-agent physical attention.

Each agent interacts with its own copy of the learned scene token. Shared scene
mode probabilities still pool legal observations, but a component's physical
transition cannot read another agent's features. All parameters are retained.
"""
from types import MethodType
import torch
from eventforecastwm.model import ForecastWorldModel


def independent_attention(core,hidden,valid):
    b,s,n,d=hidden.shape;scene=core.scene_token.expand(b*s*n,1,d)
    tokens=torch.cat([scene,hidden.reshape(b*s*n,1,d)],1)
    mask=(~valid[:,None]).expand(b,s,n).reshape(b*s*n,1)
    padding=torch.cat([torch.zeros_like(mask),mask],1)
    for layer in core.interaction:tokens=layer(tokens,src_key_padding_mask=padding)
    summaries=tokens[:,0].reshape(b,s,n,d)
    pooled=(summaries*valid[:,None,:,None]).sum(2)/valid.sum(1)[:,None,None].clamp_min(1)
    pooled=torch.where(valid.any(1)[:,None,None],pooled,summaries[:,:,0])
    return tokens[:,1].reshape(b,s,n,d),pooled


class IndependentAttentionWorldModel(ForecastWorldModel):
    def __init__(self,variant='independent_attention_v1',**kwargs):
        if variant!='independent_attention_v1':raise ValueError(variant)
        super().__init__(**kwargs);self.core._interact=MethodType(independent_attention,self.core)
        self.recipe['variant']=variant
