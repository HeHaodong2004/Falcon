"""Legal track-conditioned, query-independent human-center occupancy forecast."""
import math
import torch
from torch import nn

HORIZONS=(.25,.5,1.,2.,3.)

def legal_tracks(records,device='cpu'):
    # Uncollated inputs contain only actual history steps. IDs are never features.
    count=max(1,max(len(r['input'].humans) for r in records))
    tracks=torch.zeros(len(records),count,8,device=device)
    valid=torch.zeros(len(records),count,dtype=torch.bool,device=device)
    for i,r in enumerate(records):
        value=r['input'];h=value.humans[:,-1].to(device)
        n=len(h)
        if not n:continue
        tracks[i,:n,:2]=h[:,:2]
        tracks[i,:n,2:4]=h[:,2:4]*(h[:,5:6]>=.5)
        tracks[i,:n,4]=h[:,7].clamp_min(0)
        tracks[i,:n,5]=h[:,4]
        tracks[i,:n,6]=h[:,5]
        tracks[i,:n,7]=h[:,6]
        valid[i,:n]=(h[:,6]>=.5)&(h[:,10]>=.5)&~value.human_padding.to(device)
    assert torch.isfinite(tracks).all()
    return tracks,valid

class CVResidual(nn.Module):
    def __init__(self,background):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(8,64),nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,5))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        # Three train-configured scalar CV parameters; frozen during residual run.
        self.log_scale=nn.Parameter(torch.tensor(math.log(math.expm1(.25))))
        self.log_growth=nn.Parameter(torch.tensor(math.log(math.expm1(.15))))
        self.confidence_logit=nn.Parameter(torch.tensor(math.log(.9/.1)))
        self.register_buffer('background',torch.as_tensor(background,dtype=torch.float32).clamp(1e-7,.02))
        self.register_buffer('horizons',torch.tensor(HORIZONS,dtype=torch.float32))
        self.register_buffer('edges',torch.linspace(-6,6,25))

    def parameters_for_tracks(self,tracks,valid,mode='full'):
        n,c,_=tracks.shape
        age=tracks[:,:,4,None]+self.horizons[None,None,:]
        center=tracks[:,:,:2,None].transpose(2,3)+tracks[:,:,2:4,None].transpose(2,3)*age[:,:,:,None]
        sigma=(.1+torch.nn.functional.softplus(self.log_scale)+torch.nn.functional.softplus(self.log_growth)*age)[:,:,:,None].expand(n,c,5,2)
        confidence=torch.sigmoid(self.confidence_logit).expand(n,c,5)
        if mode!='gaussian':
            f=torch.cat([tracks[:,:,:2]/6,tracks[:,:,2:4]/2,tracks[:,:,4:5]/5,tracks[:,:,5:7]],dim=-1)
            f=f[:,:,None,:].expand(n,c,5,7)
            f=torch.cat([f,self.horizons[None,None,:,None].expand(n,c,5,1)/3],-1)
            correction=self.net(f)
            if mode!='uncertainty_only':center=center+2*torch.tanh(correction[:,:,:,:2])
            if mode!='center_only':
                sigma=(sigma*torch.exp(1.5*torch.tanh(correction[:,:,:,2:4]))).clamp_min(.1)
                confidence=torch.sigmoid(self.confidence_logit+6*torch.tanh(correction[:,:,:,4]))
        return center,sigma,confidence*valid[:,:,None]

    def forward(self,tracks,valid,mode='full',return_tracks=False):
        center,sigma,confidence=self.parameters_for_tracks(tracks,valid,mode)
        scaled=(self.edges[None,None,None,None,:]-center[:,:,:,:,None])/(sigma[:,:,:,:,None]*math.sqrt(2))
        cdf=.5*(1+torch.erf(scaled))
        mass=(cdf[...,1:]-cdf[...,:-1]).clamp_min(0)
        per_track=mass[:,:,:,1,:,None]*mass[:,:,:,0,None,:]*confidence[:,:,:,None,None]
        # No crop renormalization: off-grid probability remains outside the grid.
        survival=torch.log1p(-per_track.clamp_max(1-1e-6)).sum(1)+torch.log1p(-self.background)[None]
        p=(-torch.expm1(survival)).clamp(1e-7,1-1e-6)
        return (p,center,sigma,confidence) if return_tracks else p
