"""Shared legal-map encoder and trajectory-conditioned cell reconstruction head.

The map encoding depends only on G_t and is reused across trial trajectories.
Every original candidate cell is decoded, including cells outside the finite
learned map canvas (zero local embedding plus learned global map context).
Publication-age channels mean certificate age, not repeated camera visibility.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F
from eventwm_graph.stage2_hf.time_model import TimeNet

HALO=96
MAP_CHANNELS=8


def map_canvas(keys,states,heights,last,now,geometry,ego,anchor=False,halo=96):
    resolution,height,_=geometry;keys=np.asarray(keys,np.int32);ego_key=np.floor(np.asarray(ego)[[0,2]]/resolution).astype(np.int64)
    low=np.minimum(keys.min(0),ego_key)if len(keys)else ego_key;high=np.maximum(keys.max(0)+1,ego_key+1)if len(keys)else ego_key+1
    lo=((low-halo)//32)*32;hi=((high+halo+31)//32)*32;shape=hi-lo
    if int(np.prod(shape))>6000000:raise ValueError('shared legal map canvas exceeds six million cells')
    nx,nz=map(int,shape);s=np.zeros((nx,nz),np.uint8);h=np.zeros((nx,nz),np.float32);age=np.zeros((nx,nz),np.float32);ij=keys-lo
    if len(keys):
        last=np.asarray(last,np.float64)
        if len(last)!=len(keys)or not np.isfinite(last).all()or np.any(last>now+1e-7):raise ValueError('invalid/future starting certificate timestamp')
        s[ij[:,0],ij[:,1]]=states;h[ij[:,0],ij[:,1]]=heights-height;age[ij[:,0],ij[:,1]]=np.exp(-np.maximum(now-last,0)/.25)
    def coarse(a):return a.reshape(nx//2,2,nz//2,2).mean((1,3))
    free=coarse((s==1).astype(np.float32));blocked=coarse((s==2).astype(np.float32));known=free+blocked
    hh=coarse(h)/np.maximum(known,.25)
    distances=[coarse(distance_transform_edt(s!=state).astype(np.float32)*resolution/10)if np.any(s==state)else np.full_like(free,1.1)for state in (1,2)]
    maps=np.stack([free,blocked,hh,*distances,known,coarse(age*(s==1)),coarse(age*(s==2))]).astype(np.float32)
    if anchor:
        x=((lo[0]+2*np.arange(nx//2)+1)*resolution-ego[0])/10;z=((lo[1]+2*np.arange(nz//2)+1)*resolution-ego[2])/10
        maps=np.concatenate([maps,np.broadcast_to(x[:,None],(nx//2,nz//2))[None],np.broadcast_to(z[None,:],(nx//2,nz//2))[None]]).astype(np.float32)
    fine=np.stack([(s==1).astype(np.float32),(s==2).astype(np.float32),h,age*(s==1),age*(s==2)]).astype(np.float32)
    return dict(maps=maps,fine=fine,lo=lo,shape=shape,geometry=np.asarray(geometry),now=float(now))


class MaskedBlock(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.c1=nn.Conv2d(a,b,3,padding=1);self.c2=nn.Conv2d(b,b,3,padding=1)
    def forward(self,x,mask):return F.silu(self.c2(F.silu(self.c1(x))*mask))*mask


class SharedMapNet(nn.Module):
    def __init__(self,map_channels=8,fine=False):
        super().__init__();self.point=TimeNet(train_occupancy=True);self.enc0=MaskedBlock(map_channels,12);self.enc1=MaskedBlock(12,16);self.enc2=MaskedBlock(16,24);self.enc3=MaskedBlock(24,32);self.use_fine=fine
        self.map_project=nn.Conv2d(84,24,1);self.map_mix=MaskedBlock(24,24);self.global_project=nn.Linear(32,16)
        self.head=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))
        if fine:self.fine1=nn.Conv2d(29,12,3,padding=1);self.fine2=nn.Conv2d(12,24,3,padding=1)
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        for p in self.point.parameters():p.requires_grad_(False)

    def initialize(self,point_checkpoint,spatial_checkpoint=None):
        self.point.load_state_dict(torch.load(point_checkpoint,map_location='cpu')['model'])
        if spatial_checkpoint:
            state=torch.load(spatial_checkpoint,map_location='cpu')['model']
            with torch.no_grad():
                self.enc0.c1.weight.zero_();self.enc0.c1.weight[:,:6].copy_(state['map_encoder.0.weight']);self.enc0.c1.bias.copy_(state['map_encoder.0.bias'])
                self.enc0.c2.weight.copy_(state['map_encoder.2.weight']);self.enc0.c2.bias.copy_(state['map_encoder.2.bias'])

    def encode(self,maps,sizes,fine=None):
        n,_,h,w=maps.shape;rows=torch.arange(h,device=maps.device)[None,:,None];cols=torch.arange(w,device=maps.device)[None,None,:]
        mask=((rows<sizes[:,0,None,None])&(cols<sizes[:,1,None,None]))[:,None].to(maps.dtype)
        f0=self.enc0(maps,mask);m1=F.avg_pool2d(mask,2);f1=self.enc1(F.avg_pool2d(f0,2),m1);m2=F.avg_pool2d(m1,2);f2=self.enc2(F.avg_pool2d(f1,2),m2);m3=F.avg_pool2d(m2,2);f3=self.enc3(F.avg_pool2d(f2,2),m3)
        all_features=torch.cat([f0,F.interpolate(f1,scale_factor=2,mode='nearest'),F.interpolate(f2,scale_factor=4,mode='nearest'),F.interpolate(f3,scale_factor=8,mode='nearest')],1)
        local=self.map_mix(F.silu(self.map_project(all_features))*mask,mask)
        if self.use_fine:
            if fine is None:raise ValueError('fine current-map channels are required')
            fmask=F.interpolate(mask,scale_factor=2,mode='nearest');local=F.interpolate(local,scale_factor=2,mode='nearest');local=F.silu(self.fine1(torch.cat([local,fine],1)))*fmask;local=F.silu(self.fine2(local))*fmask
        global_=self.global_project((f3*m3).sum((2,3))/m3.sum((2,3)).clamp_min(1))
        return local,global_

    def raw(self,x,indices,map_ids,encoded):
        local,global_=encoded;n,_,h,w=local.shape;ij=torch.div(indices,1 if self.use_fine else 2,rounding_mode='floor');valid=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<h)&(ij[:,1]<w)
        gathered=local[map_ids,:,ij[:,0].clamp(0,h-1),ij[:,1].clamp(0,w-1)]*valid[:,None]
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=False):r,b,height=self.point.raw(x.float(),torch.zeros((len(x),32),device=x.device))
        delta=self.head(torch.cat([x,gathered.float(),global_[map_ids].float()],1));return r+delta[:,0],b+delta[:,1],height

    def probabilities(self,x,indices,map_ids,encoded):
        r,b,h=self.raw(x,indices,map_ids,encoded);r=(r-np.log(8.)).sigmoid();b=b.sigmoid()
        return torch.stack([1-r,r*(1-b),r*b],1),h
