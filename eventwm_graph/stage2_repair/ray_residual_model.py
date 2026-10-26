"""Add trajectory-conditioned learned spatial lines to the verified model.

All original measured geometry features and trained weights are retained.
The added logit residual starts at zero. Its usefulness is an experiment,
not an assertion that missing geometry can always be inferred.
"""
import numpy as np
import torch
from torch import nn
from .range_spatial import SpatialEvidenceNet
from .range_ray_model import RayEvidenceNet


class RayResidualEvidenceNet(SpatialEvidenceNet):
    def __init__(self):
        super().__init__();self.ray_map=nn.Conv2d(24,8,1);self.ray_head=nn.Sequential(nn.Linear(152,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,2));nn.init.zeros_(self.ray_head[-1].weight);nn.init.zeros_(self.ray_head[-1].bias)

    def initialize_spatial(self,state):
        result=self.load_state_dict(state,strict=False)
        if result.unexpected_keys or any(not k.startswith(('ray_map.','ray_head.'))for k in result.missing_keys):raise ValueError('unexpected spatial warm-start state')

    def encode(self,maps):
        local,global_=super().encode(maps);return local,global_,self.ray_map(local)

    def raw(self,x,extra,indices,mapids,encoded,origins):
        r,b,h=super().raw(x,extra,indices,mapids,encoded[:2]);rays=RayEvidenceNet.pool_rays(encoded[2],indices,mapids,origins);delta=self.ray_head(torch.cat([x,extra,rays.float()],1));return r+delta[:,0],b+delta[:,1],h

    def probabilities(self,x,extra,indices,mapids,encoded,origins):
        r,b,h=self.raw(x,extra,indices,mapids,encoded,origins);r=(r-np.log(8.)).sigmoid();b=b.sigmoid();return torch.stack([1-r,r*(1-b),r*b],1),h
