"""One shared spatial encoder for measured-range and legal-map context."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .range_evidence import EvidenceNet
from .range_bev import CHANNELS


class Block(nn.Module):
    def __init__(self,a,b,dilation=1):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=dilation,dilation=dilation),nn.SiLU(),nn.Conv2d(b,b,3,padding=dilation,dilation=dilation),nn.SiLU())
    def forward(self,x):return self.net(x)


class SpatialEvidenceNet(nn.Module):
    def __init__(self):
        super().__init__();self.point=EvidenceNet();self.train_range_head=False
        for p in self.point.parameters():p.requires_grad_(False)
        self.e0=Block(CHANNELS,16);self.e1=Block(16,32);self.e2=Block(32,64);self.e3=Block(64,96);self.context=nn.Sequential(Block(96,96,2),Block(96,96,4));self.mix=nn.Sequential(nn.Conv2d(208,32,1),nn.SiLU(),Block(32,24));self.global_=nn.Linear(96,16);self.aux=nn.Conv2d(24,1,1)
        self.head=nn.Sequential(nn.Linear(144,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def initialize(self,checkpoint):self.point.load_state_dict(torch.load(checkpoint,map_location='cpu')['model'])

    def set_range_head_training(self,enabled):
        """Optional joint fine-tuning of the existing measured-range MLP.

        The graph/time prior and its height output stay frozen. No parameters
        or deployed input/feature dimensions are added.
        """
        self.train_range_head=bool(enabled)
        for p in self.point.head.parameters():p.requires_grad_(self.train_range_head)

    def encode(self,maps):
        f0=self.e0(maps);f1=self.e1(F.avg_pool2d(f0,2));f2=self.e2(F.avg_pool2d(f1,2));f3=self.e3(F.avg_pool2d(f2,2));f3=f3+self.context(f3)
        local=self.mix(torch.cat([f0,F.interpolate(f1,scale_factor=2,mode='nearest'),F.interpolate(f2,scale_factor=4,mode='nearest'),F.interpolate(f3,scale_factor=8,mode='nearest')],1));return local,self.global_(f3.mean((2,3)))

    def raw(self,x,extra,indices,mapids,encoded):
        local,global_=encoded;_,_,h,w=local.shape;ij=torch.div(indices,2,rounding_mode='floor');valid=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<h)&(ij[:,1]<w);context=local[mapids,:,ij[:,0].clamp(0,h-1),ij[:,1].clamp(0,w-1)]*valid[:,None]
        with torch.set_grad_enabled(torch.is_grad_enabled()and self.train_range_head),torch.cuda.amp.autocast(enabled=False):r,b,height=self.point.raw(x.float(),extra.float())
        delta=self.head(torch.cat([x,extra,context.float(),global_[mapids].float()],1));return r+delta[:,0],b+delta[:,1],height

    def probabilities(self,x,extra,indices,mapids,encoded):
        r,b,h=self.raw(x,extra,indices,mapids,encoded);r=(r-np.log(8.)).sigmoid();b=b.sigmoid();return torch.stack([1-r,r*(1-b),r*b],1),h
