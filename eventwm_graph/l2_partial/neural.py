"""Compact graph-native persistent intention distribution; explicit OU evolution."""
import numpy as np
import torch
from torch import nn

class GraphDynamics(nn.Module):
    def __init__(self,width=64):
        super().__init__();self.width=width
        self.cell=nn.Linear(2,16);self.cell_layers=nn.ModuleList([nn.Linear(32,16) for _ in range(2)])
        adjacency=torch.eye(9)
        for i,j in ((0,1),(0,2),(0,3),(0,4),(1,5),(2,6),(3,7),(4,8)):adjacency[i,j]=adjacency[j,i]=1
        self.register_buffer('cell_adjacency',adjacency/adjacency.sum(-1,keepdim=True))
        self.input=nn.Linear(4+16*2,width)
        self.self_layers=nn.ModuleList([nn.Linear(width*2,width) for _ in range(2)])
        self.graph_layers=nn.ModuleList([nn.Linear(width,width) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,12))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,mean,age,graph,mask):
        cells=torch.nn.functional.silu(self.cell(graph.reshape(*graph.shape[:-1],9,2)))
        for layer in self.cell_layers:
            msg=torch.einsum('ij,bnjd->bnid',self.cell_adjacency,cells)
            cells=torch.nn.functional.silu(layer(torch.cat([cells,msg],-1)))
        spatial=torch.cat([cells[:,:,0],cells.mean(-2)],-1)
        velocity=mean[...,2:];speed=velocity.square().sum(-1,keepdim=True).sqrt()
        h=torch.nn.functional.silu(self.input(torch.cat([velocity,speed,age[...,None].clamp(0,5),spatial],-1)))
        distance=(mean[:,:,None,:2]-mean[:,None,:,:2]).square().sum(-1)
        adjacency=torch.exp(-distance/4)*mask[:,None,:]*mask[:,:,None]
        adjacency=adjacency*(1-torch.eye(mean.shape[1],device=mean.device)[None]);adjacency=adjacency/(adjacency.sum(-1,keepdim=True)+1e-6)
        for sl,gl in zip(self.self_layers,self.graph_layers):
            neighbor=adjacency@h;h=h+torch.nn.functional.silu(gl(torch.nn.functional.silu(sl(torch.cat([h,neighbor],-1)))))
        raw=self.head(h);base=torch.stack([velocity,velocity*.05,torch.stack([-velocity[...,1],velocity[...,0]],-1)],-2)
        targets=base+2*torch.tanh(raw[...,:6].reshape(*raw.shape[:-1],3,2))
        norm=(targets.square().sum(-1,keepdim=True)+1e-8).sqrt();targets=targets/torch.clamp(norm/3,min=1.)
        diffusion=torch.nn.functional.softplus(raw[...,6:9]-1.2)+.015
        logits=raw[...,9:12]+torch.tensor([1.5,0.,-.5],device=raw.device)
        return targets,diffusion,logits

class LearnedDynamics:
    def __init__(self,network,scale=1.):self.network=network.eval();self.scale=scale
    def parameters_for(self,hypothesis,graph,timestamp):
        h=hypothesis
        with torch.no_grad():
            args=[torch.from_numpy(x.astype(np.float32))[None] for x in (h.mean,np.maximum(0,timestamp-h.last_seen),graph.local_features(h.mean[:,:2]),np.ones(h.n))]
            targets,q,logits=self.network(*args)
        return targets[0].numpy(),q[0].numpy()*self.scale,torch.softmax(logits[0],-1).numpy()
    @classmethod
    def load(cls,path):
        value=torch.load(path,map_location='cpu');network=GraphDynamics(value['width']);network.load_state_dict(value['state_dict']);return cls(network,value.get('diffusion_scale',1.))
