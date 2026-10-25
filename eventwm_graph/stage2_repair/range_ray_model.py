"""Shared legal-map/range context with learned line-of-sight integration.

The map encoder is shared across trajectories. This candidate replaces the
expensive observed-triangle visibility covariate with learned context sampled
along three supplied camera-to-cell lines. Accuracy must be measured; this is
not an exact certificate renderer or a claim that unobserved space is free.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .range_spatial import SpatialEvidenceNet
from .range_bev import SIZE


class RayEvidenceNet(SpatialEvidenceNet):
    def __init__(self):
        super().__init__()
        self.ray_map=nn.Conv2d(24,8,1)
        self.head=nn.Sequential(nn.Linear(192,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def initialize_spatial(self,checkpoint):
        saved=torch.load(checkpoint,map_location='cpu')['model']
        state=self.state_dict()
        for key,value in saved.items():
            if not key.startswith('head.')and key in state and state[key].shape==value.shape:state[key]=value
        self.load_state_dict(state)

    def encode(self,maps):
        local,global_=super().encode(maps)
        return local,global_,self.ray_map(local)

    @staticmethod
    def pool_rays(raymap,indices,mapids,origins):
        result=raymap.new_zeros((len(indices),48))
        progress=torch.arange(1,17,device=indices.device,dtype=torch.float32)/17
        end=indices.float()+.5
        for mi in range(len(raymap)):
            at=torch.nonzero(mapids==mi,as_tuple=False)[:,0]
            if not len(at):continue
            begin=origins[at].float()
            points=begin[:,:,None,:]+(end[at,None,None,:]-begin[:,:,None,:])*progress[None,None,:,None]
            # CUDA grid_sample uses width(z),height(x); physical tensors use x,z.
            grid=(points/SIZE-1.)[:,:,:,[1,0]].reshape(1,len(at),48,2)
            sampled=F.grid_sample(raymap[mi:mi+1].float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)
            sampled=sampled[0].permute(1,2,0).reshape(len(at),3,16,8)
            result[at]=torch.cat([sampled.mean(2),sampled.amax(2)],2).reshape(len(at),48).to(result.dtype)
        return result

    def raw(self,x,extra,indices,mapids,encoded,origins):
        local,global_,raymap=encoded;_,_,h,w=local.shape
        ij=torch.div(indices,2,rounding_mode='floor')
        valid=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<h)&(ij[:,1]<w)
        context=local[mapids,:,ij[:,0].clamp(0,h-1),ij[:,1].clamp(0,w-1)]*valid[:,None]
        rays=self.pool_rays(raymap,indices,mapids,origins)
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=False):
            # Keep the original point prior/height; no missing-mesh placeholder
            # is fed to the old measured-geometry residual head.
            r,b,height=self.point.point.raw(x.float(),torch.zeros((len(x),32),device=x.device))
        legal_extra=torch.cat([torch.zeros_like(extra[:,:8]),extra[:,8:]],1)
        delta=self.head(torch.cat([x,legal_extra,context.float(),global_[mapids].float(),rays.float()],1))
        return r+delta[:,0],b+delta[:,1],height

    def probabilities(self,x,extra,indices,mapids,encoded,origins):
        r,b,h=self.raw(x,extra,indices,mapids,encoded,origins)
        r=(r-np.log(8.)).sigmoid();b=b.sigmoid()
        return torch.stack([1-r,r*(1-b),r*b],1),h
