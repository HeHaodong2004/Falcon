"""Original model plus a zero-initialized starting-visibility embedding."""
import numpy as np
import torch
from torch import nn
from .range_spatial import SpatialEvidenceNet


class InitialVisibilityNet(SpatialEvidenceNet):
    def __init__(self):
        super().__init__();self.initial_projection=nn.Linear(3,128,bias=False);nn.init.zeros_(self.initial_projection.weight)

    def initialize_spatial(self,state):
        result=self.load_state_dict(state,strict=False)
        if result.unexpected_keys or any(k!='initial_projection.weight'for k in result.missing_keys):raise ValueError('unexpected spatial warm-start state')

    def raw(self,x,extra,indices,mapids,encoded):
        if extra.shape[1]!=19:raise ValueError('current partial-render19feature schema required')
        local,global_=encoded;_,_,h,w=local.shape;ij=torch.div(indices,2,rounding_mode='floor');valid=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<h)&(ij[:,1]<w);context=local[mapids,:,ij[:,0].clamp(0,h-1),ij[:,1].clamp(0,w-1)]*valid[:,None]
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=False):r,b,height=self.point.raw(x.float(),extra[:,:16].float())
        hidden=self.head[0](torch.cat([x,extra[:,:16],context.float(),global_[mapids].float()],1))+self.initial_projection(extra[:,16:]);delta=self.head[4](self.head[3](self.head[2](self.head[1](hidden))));return r+delta[:,0],b+delta[:,1],height

    def probabilities(self,x,extra,indices,mapids,encoded):
        r,b,h=self.raw(x,extra,indices,mapids,encoded);r=(r-np.log(8.)).sigmoid();b=b.sigmoid();return torch.stack([1-r,r*(1-b),r*b],1),h
