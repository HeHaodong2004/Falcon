"""Directional local static geometry for each predicted agent state.

Five-by-five samples cover a4m local patch. Unknown/off-map support is explicit.
The encoder is a small MLP, with no spatial recurrent grid or semantic labels.
"""
from types import MethodType
import torch
from torch import nn
import torch.nn.functional as F
from eventforecastwm.model import ForecastWorldModel


def geometry_at(core,belief,positions):
    b,s,n,_=positions.shape
    axis=torch.linspace(-2.,2.,5,device=positions.device,dtype=positions.dtype)
    z,x=torch.meshgrid(axis,axis,indexing='ij');offset=torch.stack([x,z],-1).reshape(25,2)
    points=positions[...,None,:]+offset
    supported=(points.abs()<=core.config.grid_extent_m-.25).all(-1)&belief.geometry_valid[:,None,None,None]
    grid=points.reshape(b,-1,1,2)/core.config.grid_extent_m
    sampled=F.grid_sample(belief.geometry,grid,align_corners=False,padding_mode='zeros').squeeze(-1).transpose(1,2)
    sampled=sampled.reshape(b,s,n,25,2)*supported[...,None]
    feature=torch.cat([sampled,supported[...,None].to(sampled)],-1).flatten(-2)
    return core.map_patch_encoder(feature)*supported.any(-1)[...,None]


def attach_map_patch(model):
    d=model.core.config.width
    model.core.map_patch_encoder=nn.Sequential(nn.Linear(75,32),nn.GELU(),nn.Linear(32,16))
    model.core.state_input=nn.Linear(2+2+1+1+16+8,d)
    model.core._geometry_at=MethodType(geometry_at,model.core)


class MapPatchForecastWorldModel(ForecastWorldModel):
    def __init__(self,variant='map_patch_v1',**kwargs):
        if variant!='map_patch_v1':raise ValueError(variant)
        super().__init__(**kwargs);attach_map_patch(self)
        self.recipe['variant']=variant
