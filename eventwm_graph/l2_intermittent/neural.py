"""Small inherited graph encoder, continuous velocity modes and Stop probability."""
import numpy as np
import torch
from torch import nn
from eventwm_graph.l2_partial.neural import GraphDynamics
from eventwm_graph.l2_partial.state import ObservedGraph


def observed_features(graph,positions):
    """Reuse identical observed-cell stencils within this encoder invocation."""
    if type(graph) is not ObservedGraph:return graph.local_features(positions)
    shape=positions.shape[:-1];flat=positions.reshape(-1,2)
    keys=np.floor(flat/graph.resolution).astype(int)
    _,first,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    features=graph.local_features(flat[first])
    return features[inverse].reshape(*shape,18)


class IntermittentDynamics(GraphDynamics):
    def __init__(self,width=64):
        super().__init__(width)
        self.head[-1]=nn.Linear(width,13)
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,mean,age,graph,mask):
        cells=torch.nn.functional.silu(self.cell(graph.reshape(*graph.shape[:-1],9,2)))
        for layer in self.cell_layers:
            msg=torch.einsum('ij,bnjd->bnid',self.cell_adjacency,cells)
            cells=torch.nn.functional.silu(layer(torch.cat([cells,msg],-1)))
        spatial=torch.cat([cells[:,:,0],cells.mean(-2)],-1)
        v=mean[...,2:];speed=(v.square().sum(-1,keepdim=True)+1e-12).sqrt()
        h=torch.nn.functional.silu(self.input(torch.cat([v,speed,age[...,None].clamp(0,5),spatial],-1)))
        d=(mean[:,:,None,:2]-mean[:,None,:,:2]).square().sum(-1)
        A=torch.exp(-d/4)*mask[:,None,:]*mask[:,:,None]
        A=A*(1-torch.eye(mean.shape[1],device=mean.device)[None]);A=A/(A.sum(-1,keepdim=True)+1e-6)
        for sl,gl in zip(self.self_layers,self.graph_layers):
            h=h+torch.nn.functional.silu(gl(torch.nn.functional.silu(sl(torch.cat([h,A@h],-1)))))
        raw=self.head(h)
        base=torch.stack([v,torch.stack([-v[...,1],v[...,0]],-1),v*.5],-2)
        # Unbounded continuous targets. No clipping of sampled or mean speeds.
        target=base+raw[...,:6].reshape(*raw.shape[:-1],3,2)
        q=torch.nn.functional.softplus(raw[...,6:9]-3.)+.0001
        logits=raw[...,9:12]+raw.new_tensor([2.,-1.,-1.])
        stop_logit=raw[...,12]+3.-8.*speed[...,0]
        return target,q,logits,stop_logit


class LearnedIntermittent:
    def __init__(self,network):
        self.network=network.eval()

    def parameters(self,mean,age,graph):
        if not len(mean):
            return np.empty((0,3,2)),np.empty((0,3)),np.empty((0,3)),np.empty(0)
        args=[torch.as_tensor(x,dtype=torch.float32)[None] for x in (mean,age,graph.local_features(mean[:,:2]),np.ones(len(mean)))]
        with torch.no_grad():
            t,q,l,s=self.network(*args)
        return t[0].numpy(),q[0].numpy(),torch.softmax(l[0],-1).numpy(),torch.sigmoid(s[0]).numpy()

    def parameters_many(self,components,timestamp,graph):
        result=[None]*len(components)
        for n in sorted({c.n for c in components}):
            indices=[i for i,c in enumerate(components) if c.n==n]
            if n==0:
                for i in indices:result[i]=(np.empty((0,3,2)),np.empty((0,3)),np.empty((0,3)),np.empty(0))
                continue
            mean=np.stack([components[i].mean for i in indices]);age=np.stack([timestamp-components[i].last_seen for i in indices])
            args=[torch.as_tensor(x,dtype=torch.float32) for x in (mean,age,observed_features(graph,mean[...,:2]),np.ones(mean.shape[:2]))]
            with torch.no_grad():t,q,l,s=self.network(*args)
            arrays=(t.numpy(),q.numpy(),torch.softmax(l,-1).numpy(),torch.sigmoid(s).numpy())
            for j,i in enumerate(indices):result[i]=tuple(a[j] for a in arrays)
        return result

    @classmethod
    def load(cls,path):
        value=torch.load(path,map_location='cpu');net=IntermittentDynamics(value['width']);net.load_state_dict(value['state_dict'])
        return cls(net)
