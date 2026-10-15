"""Joint SR and masked successor prediction on the original EventQueryWM-M."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import EventQueryWM, ModelConfig
from .successor import nominal_at_horizons

class JointEventWM(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__(); self.config=config; self.backbone=EventQueryWM(config)
        self.register_buffer('horizons',torch.tensor([.25,.5,1.,2.,3.]))
        def head(n,o):
            return nn.Sequential(nn.Linear(n,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,o))
        self.physical_head=head(config.width+1,576)
        self.observed_head=head(config.width+11,576)
        self.terminal_head=head(config.width+20,583)

    def freeze_backbone(self,frozen):
        for p in self.backbone.parameters(): p.requires_grad_(not frozen)
        if frozen:self.backbone.eval()

    def successor_from_scene(self,value,scene):
        history=scene[:,0]; query=value.query; b=len(history)
        history=history[:,None].expand(-1,5,-1)
        clock=(self.horizons/3)[None,:,None].expand(b,-1,-1)
        physical=self.physical_head(torch.cat([history,clock],-1)).reshape(b,5,24,24)
        observed=self.observed_head(torch.cat([history,clock,nominal_at_horizons(query,self.horizons)],-1)).reshape(b,5,24,24)
        terminal=self.terminal_head(torch.cat([scene[:,0],query.mean(1),query.max(1).values],-1))
        return dict(physical_logits=physical,observed_logits=observed,terminal_physical_logits=terminal[:,:576].reshape(b,24,24),
                    terminal_robot_pose=terminal[:,576:580],event_logits=terminal[:,580:583])

    def forward(self,value,with_successor=True):
        scene,padding=self.backbone.encode_scene(value)
        sr=self.backbone.sr_from_scene(value,scene,padding)
        return dict(sr_logits=sr,p_safe=sr.sigmoid(),successor_prediction=self.successor_from_scene(value,scene) if with_successor else None)

    def successor(self,value):
        return self.successor_from_scene(value,self.backbone.encode_scene(value)[0])

    @torch.no_grad()
    def predict(self,value):
        self.eval(); raw=self(value); state=raw['successor_prediction']
        return dict(p_safe=raw['p_safe'],successor_prediction=dict(horizons_s=self.horizons,
            physical_occupancy_probability=state['physical_logits'].sigmoid(),
            observed_occupancy_probability=state['observed_logits'].sigmoid(),
            terminal_physical_occupancy_probability=state['terminal_physical_logits'].sigmoid(),
            termination_probabilities=state['event_logits'].softmax(-1),
            terminal_robot_pose_mean=state['terminal_robot_pose']*state['terminal_robot_pose'].new_tensor([6,6,1,1])))

def masked_bce(logits,target,mask):
    active=mask.bool()
    while active.ndim<logits.ndim:active=active.unsqueeze(-1)
    active=active.expand_as(logits)
    return F.binary_cross_entropy_with_logits(logits[active],target.float()[active]) if active.any() else logits.sum()*0

def successor_loss(prediction,target):
    components={}; active=[]
    for name,key,tkey,mkey in [('physical','physical_logits','physical_occupancy','physical_mask'),
                             ('observed','observed_logits','observed_occupancy','observed_mask'),
                             ('terminal_occupancy','terminal_physical_logits','terminal_physical_occupancy','terminal_mask')]:
        components[name]=masked_bce(prediction[key],target[tkey],target[mkey])
        if target[mkey].any():active.append(components[name])
    components['event']=F.cross_entropy(prediction['event_logits'],target['event_class']);active.append(components['event'])
    mask=target['terminal_mask'].bool()
    components['terminal_pose']=F.smooth_l1_loss(prediction['terminal_robot_pose'][mask],
        target['terminal_robot_pose'][mask]/target['terminal_robot_pose'].new_tensor([6,6,1,1])) if mask.any() else prediction['terminal_robot_pose'].sum()*0
    if mask.any():active.append(components['terminal_pose'])
    return torch.stack(active).mean(),components
