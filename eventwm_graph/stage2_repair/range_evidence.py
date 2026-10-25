"""Learned fusion of legal measured-range evidence and original cell features.

The feature extractor accepts an observed-only mesh, legal current graph
columns and the supplied base trajectory. Static labels are loss/scoring
targets only. No full scene, native support, future camera or future sensing
is accepted by the predictor.
"""
import numpy as np
import torch
from torch import nn
from .legal_camera import current_extrinsic,base_rotations
from .legal_camera_features import parameters,cell_features
from eventwm_graph.stage2_hf.time_model import TimeNet

EXTRA=16


def features(mesh,keys,x,path,gkeys,current_origin,current_rotation):
    resolution=mesh.kernel.resolution
    translation,rotation=current_extrinsic(path[:1],current_origin,current_rotation)
    base=base_rotations(path[:,4]);origin=path[:,1:4]+base@translation;rot=base@rotation
    cameras=np.concatenate([origin[1:],rot[1:].reshape(-1,9),path[1:,2:3]],1)
    low=np.minimum(keys.min(0),gkeys.min(0))-4;high=np.maximum(keys.max(0),gkeys.max(0))+5
    shape=high-low;unknown=np.ones(tuple(shape),np.uint8);ij=gkeys-low;unknown[ij[:,0],ij[:,1]]=0
    floor=x[:,24].astype(np.float64)+path[0,2]
    supported,measured_floor=mesh.floor_support(keys,floor)
    obstacle=mesh.obstacle_distance(keys,path[0,2])
    angles=np.arange(16)*2*np.pi/16
    planar=np.array([[0,0,0],[-.49*resolution,0,-.49*resolution],[-.49*resolution,0,.49*resolution],[.49*resolution,0,-.49*resolution],[.49*resolution,0,.49*resolution]])
    halo=np.column_stack([np.cos(angles)*(.25+resolution/np.sqrt(2)),np.zeros(16),np.sin(angles)*(.25+resolution/np.sqrt(2))])
    offset=np.concatenate([np.concatenate([planar,halo])+[0,y,0]for y in [.025,.7,1.4]])
    center=np.column_stack([(keys[:,0]+.5)*resolution,floor,(keys[:,1]+.5)*resolution])
    mesh.set_probes(keys,center[:,None,:]+offset)
    code=mesh.blocked(unknown,low,cameras,[keys],[np.arange(len(cameras),dtype=np.int32)])[0]
    params,_=parameters(path,translation,rotation,resolution)
    view=cell_features(keys,path,resolution,x,params)
    out=np.column_stack([code>0,code<0,code==0,supported,np.clip(measured_floor-floor,-.5,.5),np.minimum(obstacle,2.)/2,obstacle>(.25+resolution/np.sqrt(2)+.008),np.minimum(np.maximum(path[0,0]-mesh.latest_sensor_time,0),5.)/5+np.zeros(len(keys)),view]).astype(np.float32)
    assert out.shape==(len(keys),EXTRA)
    if not np.isfinite(out).all():raise ValueError('nonfinite legal range evidence')
    return out


class EvidenceNet(nn.Module):
    def __init__(self):
        super().__init__();self.point=TimeNet(train_occupancy=True)
        for p in self.point.parameters():p.requires_grad_(False)
        self.head=nn.Sequential(nn.Linear(88+EXTRA,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def initialize(self,checkpoint):self.point.load_state_dict(torch.load(checkpoint,map_location='cpu')['model'])

    def raw(self,x,extra):
        with torch.no_grad():r,b,h=self.point.raw(x,torch.zeros((len(x),32),device=x.device))
        d=self.head(torch.cat([x,extra],1));return r+d[:,0],b+d[:,1],h

    def probabilities(self,x,extra):
        r,b,h=self.raw(x,extra);r=(r-np.log(8.)).sigmoid();b=b.sigmoid()
        return torch.stack([1-r,r*(1-b),r*b],1),h
