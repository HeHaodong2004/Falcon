"""Uncertainty-only physical occupancy readouts with exactly fixed legal CV centers."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from .cv_residual import CVResidual,legal_tracks

HISTORY_FEATURES=('relative_x','relative_z','fd_velocity_x','fd_velocity_z',
                  'snapshot_relative_time','observation_relative_time','observation_age',
                  'visible','fd_velocity_valid','valid')

def short_history(records,device='cpu',steps=16,window_s=2.):
    """Only cached legal arrived states; derive velocity from observed positions/time.

    Stale position is not a new observation: timestamp = snapshot time - age.
    All positions already use the fixed decision-time robot frame. No query or
    target field is accessed. Valid snapshots are sampled over the last 2s.
    """
    tracks,valid=legal_tracks(records,device)
    n,c,_=tracks.shape
    sequence=torch.zeros(n,c,steps,10,device=device)
    sequence_mask=torch.zeros(n,c,steps,dtype=torch.bool,device=device)
    for i,record in enumerate(records):
        value=record['input'];humans=value.humans.cpu()
        for k in range(len(humans)):
            previous_time=None;previous_position=None;fd=torch.zeros(2);fd_valid=False
            arrived=[];features=[]
            for row in humans[k]:
                if row[6]<.5 or row[10]<.5:continue
                snapshot=float(row[8]);age=max(0.,float(row[7]));observed=snapshot-age
                assert snapshot<=1e-6 and observed<=1e-6
                position=row[:2]
                if previous_time is not None and observed-previous_time>1e-5:
                    fd=(position-previous_position)/(observed-previous_time);fd_valid=True
                if previous_time is None or observed-previous_time>1e-5:
                    previous_time=observed;previous_position=position.clone()
                if snapshot>=-window_s-1e-6:
                    f=torch.cat([position/6,fd/2,torch.tensor([snapshot/2,observed/2,age/5,float(row[4]),float(fd_valid),1.])])
                    arrived.append(snapshot);features.append(f)
            if not features:continue
            if len(features)>steps:
                # Uniform index spacing over actual legal history, including newest state.
                indices=torch.linspace(0,len(features)-1,steps).round().long().tolist()
                features=[features[z] for z in indices]
            length=len(features)
            sequence[i,k,:length]=torch.stack(features).to(device);sequence_mask[i,k,:length]=True
    assert torch.isfinite(sequence).all()
    return tracks,valid,sequence,sequence_mask

class TrackUncertainty(CVResidual):
    """A 4,931-param MLP or capacity-matched 5,043-param small GRU decoder."""
    def __init__(self,background,kind):
        super().__init__(background)
        if kind not in ('mlp','history'):raise ValueError(kind)
        self.kind=kind
        if kind=='history':
            self.history_gru=nn.GRU(10,16,batch_first=True)
            width=48;input_width=8+16
        else:width=64;input_width=8
        self.net=nn.Sequential(nn.Linear(input_width,width),nn.GELU(),nn.Linear(width,width),nn.GELU(),nn.Linear(width,3))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        for p in (self.log_scale,self.log_growth,self.confidence_logit):p.requires_grad_(False)
        self._history=None

    def encode_history(self,sequence,sequence_mask):
        b,c,s,f=sequence.shape;flat=sequence.reshape(b*c,s,f)
        lengths=sequence_mask.sum(-1).reshape(-1)
        # Padded values cannot change the representation, even if poisoned with NaN.
        flat=torch.where(sequence_mask.reshape(b*c,s,1),flat,torch.zeros_like(flat))
        packed=pack_padded_sequence(flat,lengths.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
        _,hidden=self.history_gru(packed)
        return hidden[0].reshape(b,c,16)*(lengths>0).reshape(b,c,1)

    def parameters_for_tracks(self,tracks,valid,mode='full'):
        center,sigma,confidence=super().parameters_for_tracks(tracks,valid,'gaussian')
        if mode=='gaussian':return center,sigma,confidence
        b,c,_=tracks.shape
        features=torch.cat([tracks[:,:,:2]/6,tracks[:,:,2:4]/2,tracks[:,:,4:5]/5,tracks[:,:,5:7]],-1)
        features=features[:,:,None,:].expand(b,c,5,7)
        features=torch.cat([features,self.horizons[None,None,:,None].expand(b,c,5,1)/3],-1)
        if self.kind=='history':
            assert self._history is not None
            history=self.encode_history(*self._history)
            features=torch.cat([features,history[:,:,None,:].expand(b,c,5,16)],-1)
        output=self.net(features)
        sigma=(sigma*torch.exp(1.5*torch.tanh(output[...,:2]))).clamp_min(.1)
        confidence=torch.sigmoid(self.confidence_logit+6*torch.tanh(output[...,2]))*valid[:,:,None]
        return center,sigma,confidence

    def forward(self,tracks,valid,sequence=None,sequence_mask=None,mode='full',return_tracks=False):
        self._history=(sequence,sequence_mask) if self.kind=='history' else None
        try:return super().forward(tracks,valid,mode,return_tracks)
        finally:self._history=None

def initialize_from_gaussian(model,checkpoint):
    # Reuse exactly the previous optimization-only Gaussian configuration.
    with torch.no_grad():
        for key in ('log_scale','log_growth','confidence_logit','background','horizons','edges'):
            getattr(model,key).copy_(checkpoint['model'][key])
