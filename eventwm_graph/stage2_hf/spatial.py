"""Legal-map/whole-segment dense features on an untruncated dynamic domain."""
import math
import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F
from eventwm_graph.stage2_quality.features import candidate_mask

HALO=32
MAP_CHANNELS=6
QUERY_CHANNELS=32

@njit(cache=True)
def trajectory_features(lo,shape,resolution,path,free,blocked):
 nx,nz=shape;out=np.zeros((32,nx,nz),np.float32);duration=max(path[-1,0]-path[0,0],1e-4);rotation=0.;length=0.;stationary=0.
 for j in range(1,len(path)):
  dx=path[j,1]-path[j-1,1];dz=path[j,3]-path[j-1,3];dd=math.sqrt(dx*dx+dz*dz);length+=dd
  rotation+=abs(math.atan2(math.sin(path[j,4]-path[j-1,4]),math.cos(path[j,4]-path[j-1,4])))
  if dd<1e-4:stationary+=path[j,0]-path[j-1,0]
 cosine=np.cos(path[:,4]);sine=np.sin(path[:,4]);idx=(min(1,len(path)-1),len(path)//2,len(path)-1)
 for i in range(nx):
  x=(lo[0]+2*i+1)*resolution
  for k in range(nz):
   z=(lo[1]+2*k+1)*resolution;mind=100.;best=-1.;maxw=0.;f5=0.;f8=0.;count=0.;first=1.;last=0.;soft=0.
   for j in range(len(path)):
    dx=x-path[j,1];dz=z-path[j,3];d=math.sqrt(dx*dx+dz*dz);mind=min(mind,d)
    if not j:continue
    facing=max(-1.,min(1.,(dx*cosine[j]-dz*sine[j])/max(d,1e-4)));dt=path[j,0]-path[j-1,0]
    if d<=10.:
     best=max(best,facing);w=max(0.,facing)*max(0.,1-d/11.);maxw=max(maxw,w);soft+=w*dt
     if facing>=.5:f5+=dt;count+=1;when=(path[j,0]-path[0,0])/duration;first=min(first,when);last=max(last,when)
     if facing>=.866025403784:f8+=dt
   out[0,i,k]=mind/10;out[1,i,k]=best;out[2,i,k]=maxw;out[3,i,k]=f5/duration;out[4,i,k]=f8/duration;out[5,i,k]=count/max(len(path)-1,1);out[6,i,k]=first;out[7,i,k]=last;out[8,i,k]=soft/duration
   out[9,i,k]=(x-path[0,1])/10;out[10,i,k]=(z-path[0,3])/10;out[11,i,k]=(x-path[-1,1])/10;out[12,i,k]=(z-path[-1,3])/10
   out[13,i,k]=duration/5;out[14,i,k]=length/5;out[15,i,k]=math.cos(path[-1,4]-path[0,4]);out[16,i,k]=math.sin(path[-1,4]-path[0,4]);out[17,i,k]=rotation/(2*math.pi);out[18,i,k]=stationary/duration;out[19,i,k]=(path[-1,2]-path[0,2])
   for r in range(3):
    j=idx[r];sx=path[j,1];sz=path[j,3];fr=0.;bl=0.;fu=1.;fb=1.
    for step in range(1,17):
     t=step/17.;a=int(math.floor(((sx+(x-sx)*t)/resolution-lo[0])/2));b=int(math.floor(((sz+(z-sz)*t)/resolution-lo[1])/2))
     if 0<=a<nx and 0<=b<nz:
      f=free[a,b];v=blocked[a,b];fr+=f;bl+=v
      if f+v<.999:fu=min(fu,t)
      if v>0:fb=min(fb,t)
     else:fu=min(fu,t)
    out[20+4*r,i,k]=fr/16;out[21+4*r,i,k]=bl/16;out[22+4*r,i,k]=fu;out[23+4*r,i,k]=fb
 return out

def bounds(path,resolution):
 xy=np.asarray(path[:,[1,3]],np.float32);lo=np.floor((xy.min(0)-11)/resolution).astype(np.int64)-HALO;hi=np.floor((xy.max(0)+11)/resolution).astype(np.int64)+HALO+1
 lo=(lo//16)*16;hi=((hi+15)//16)*16;shape=hi-lo
 if np.any(shape<=0)or int(np.prod(shape))>6000000:raise ValueError('spatial domain exceeds explicit sixmillioncell memory bound; no crop truncation')
 return lo,shape

def build(gkeys,gstates,gheights,path,geometry,target_keys=None,target_states=None,target_heights=None,domain_check=True,make_query=True,make_candidate=True):
 resolution,height,_=geometry;lo,shape=bounds(path,resolution);nx,nz=map(int,shape);states=np.zeros((nx,nz),np.uint8);heights=np.zeros((nx,nz),np.float32)
 ij=gkeys-lo;ok=np.all((ij>=0)&(ij<shape),axis=1);at=ij[ok];states[at[:,0],at[:,1]]=gstates[ok];heights[at[:,0],at[:,1]]=gheights[ok]-path[0,2]
 fine=np.stack([states==1,states==2,heights]).astype(np.float32)
 coarse=fine.reshape(3,nx//2,2,nz//2,2).mean((2,4));free,blocked=coarse[:2];known=free+blocked;coarse[2]/=np.maximum(known,.25)
 dist=[distance_transform_edt(channel==0).astype(np.float32)*resolution*2/10 if np.any(channel)else np.ones(channel.shape,np.float32)*1.1 for channel in (free,blocked)]
 maps=np.concatenate([coarse,np.stack(dist),known[None]],axis=0).astype(np.float32)
 query=trajectory_features(lo,(nx//2,nz//2),resolution,path,free,blocked) if make_query else None
 if make_candidate:
  unknown=np.argwhere(states==0).astype(np.int32)+lo.astype(np.int32);mask=candidate_mask(unknown,np.unique(path[:,[1,3]].astype(np.float32),axis=0),resolution);keys=unknown[mask]
 else:keys=np.empty((0,2),np.int32)
 candidate=np.zeros((nx,nz),np.float32);ci=keys-lo;candidate[ci[:,0],ci[:,1]]=1
 y=np.zeros((nx,nz),np.uint8);hgt=np.zeros((nx,nz),np.float32);miss=0
 if target_keys is not None and len(target_keys):
  ti=target_keys-lo;valid=np.all((ti>=0)&(ti<shape),axis=1);safe=np.clip(ti,0,shape-1);valid&=candidate[safe[:,0],safe[:,1]]>0;miss=int((~valid).sum())
  if domain_check and miss:raise ValueError('true revelation outside unchangedcandidate domain: '+str(miss))
  ti=ti[valid];y[ti[:,0],ti[:,1]]=target_states[valid];hgt[ti[:,0],ti[:,1]]=target_heights[valid]-path[0,2]
 return dict(maps=maps,query=query,fine=fine,candidate=candidate,labels=y,height=hgt,keys=keys,lo=lo,shape=shape,path=path,resolution=resolution,out_of_domain_positive=miss)

def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())
class SpatialNet(nn.Module):
 def __init__(self,width=12,context_dilations=None):
  super().__init__();self.width=width;self.map_encoder=block(6,width);self.mix=block(width+32,width);self.down=block(width,width*2);self.up=block(width*3,width);self.detail=nn.Sequential(nn.Conv2d(width+3,8,3,padding=1),nn.SiLU(),nn.Conv2d(8,3,3,padding=1))
  nn.init.constant_(self.detail[-1].bias[0],-3.);nn.init.constant_(self.detail[-1].bias[1],1.)
  self.context_dilations=context_dilations
  if context_dilations:
   layers=[]
   for j,dilation in enumerate(context_dilations):
    layers.append(nn.Conv2d(width*2,width*2,3,padding=dilation,dilation=dilation))
    if j<len(context_dilations)-1:layers.append(nn.SiLU())
   self.context=nn.Sequential(*layers);nn.init.zeros_(self.context[-1].weight);nn.init.zeros_(self.context[-1].bias)
 def encode_map(self,maps):return self.map_encoder(maps)
 def forward(self,maps,query,fine,encoded=None):
  e=self.encode_map(maps)if encoded is None else encoded;x=self.mix(torch.cat([e,query],1));z=self.down(F.avg_pool2d(x,2))
  if self.context_dilations:z=z+self.context(z)
  z=F.interpolate(z,scale_factor=2,mode='nearest');z=self.up(torch.cat([x,z],1));z=F.interpolate(z,scale_factor=2,mode='nearest');return self.detail(torch.cat([z,fine],1))

def spatial_probabilities(out,pos_weight=8.,calibration=None):
 rlog=out[:,0]-math.log(pos_weight);blog=out[:,1]
 if calibration:
  rlog=rlog*calibration['rev_scale']+calibration['rev_bias'];blog=blog*calibration['blocked_scale']+calibration['blocked_bias']
 r=rlog.sigmoid();b=blog.sigmoid();return torch.stack([1-r,r*(1-b),r*b],1)
