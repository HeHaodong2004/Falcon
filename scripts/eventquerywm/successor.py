"""Minimal successor heads on the existing, frozen legal-history SR backbone."""
import torch
from torch import nn
from .model import EventQueryWM,ModelConfig

def history_embedding(backbone,value):
    humans=value.humans.masked_fill(value.human_padding[:,:,None,None],0.)
    humans=humans.masked_fill(value.history_padding[:,None,:,None],0.)
    b,n,t,f=humans.shape;lengths=(~value.history_padding).sum(1)
    encoded,_=backbone.human_gru(humans.reshape(b*n,t,f))
    last=(lengths-1).repeat_interleave(n)
    human=backbone.human_norm(encoded[torch.arange(b*n,device=humans.device),last].reshape(b,n,-1))
    robot=value.robot.masked_fill(value.history_padding[:,:,None],0.)
    steps=backbone.robot_step(robot).masked_fill(value.history_padding[:,:,None],0.)
    mean=steps.sum(1)/lengths[:,None];last=steps[torch.arange(b,device=humans.device),lengths-1]
    ego=backbone.robot_pool(torch.cat([mean,last],-1))
    scene=torch.cat([ego[:,None],human],1)
    padding=torch.cat([torch.zeros(b,1,dtype=torch.bool,device=humans.device),value.human_padding],1)
    for layer in backbone.scene:scene=layer(scene,src_key_padding_mask=padding)
    return scene[:,0]

def nominal_at_horizons(query,horizons):
    time=(horizons[None,:]/query[:,-1,2:3]).clamp(0,1)*15
    earlier=time.floor().long().clamp(max=14);fraction=(time-earlier).unsqueeze(-1)
    a=query.gather(1,earlier[:,:,None].expand(-1,-1,query.shape[-1]))
    b=query.gather(1,(earlier+1)[:,:,None].expand(-1,-1,query.shape[-1]))
    return a+fraction*(b-a)

class SuccessorPrototype(nn.Module):
    def __init__(self,config=ModelConfig()):
        super().__init__();self.config=config;self.backbone=EventQueryWM(config)
        for p in self.backbone.parameters():p.requires_grad_(False)
        self.register_buffer('horizons',torch.tensor([.25,.5,1.,2.,3.]))
        def head(features,outputs):
            return nn.Sequential(nn.Linear(features,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,outputs))
        self.physical_head=head(config.width+1,576)
        self.observed_head=head(config.width+11,576)
        self.terminal_head=head(config.width+20,584)

    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self

    def encode(self,value):
        self.backbone.eval()
        with torch.no_grad():return dict(history=history_embedding(self.backbone,value),query=value.query)

    def decode(self,features):
        history,query=features['history'],features['query'];b=history.shape[0]
        horizons=self.horizons.to(history.device);clock=(horizons/3)[None,:,None].expand(b,-1,-1)
        physical=self.physical_head(torch.cat([history[:,None,:].expand(-1,5,-1),clock],-1)).reshape(b,5,24,24)
        observed=self.observed_head(torch.cat([history[:,None,:].expand(-1,5,-1),clock,nominal_at_horizons(query,horizons)],-1)).reshape(b,5,24,24)
        terminal=self.terminal_head(torch.cat([history,query.mean(1),query.max(1).values],-1))
        return dict(physical_logits=physical,observed_logits=observed,
            terminal_physical_logits=terminal[:,:576].reshape(b,24,24),
            terminal_robot_pose=terminal[:,576:580],event_logits=terminal[:,580:583],
            collision_time_s=torch.nn.functional.softplus(terminal[:,583]))

    def forward(self,value):
        successor=self.decode(self.encode(value))
        with torch.no_grad():p_safe=self.backbone(value).sigmoid()
        return dict(p_safe=p_safe,successor_prediction=successor)

    @torch.no_grad()
    def predict(self,value):
        """Inference uses legal tensors only; no supervision masks or actual end time."""
        self.eval();raw=self.forward(value);state=raw['successor_prediction']
        return dict(p_safe=raw['p_safe'],successor_prediction=dict(
            horizons_s=self.horizons,physical_occupancy_probability=state['physical_logits'].sigmoid(),
            observed_occupancy_probability=state['observed_logits'].sigmoid(),
            terminal_physical_occupancy_probability=state['terminal_physical_logits'].sigmoid(),
            termination_probabilities=state['event_logits'].softmax(-1),
            conditional_collision_time_mean_s=state['collision_time_s'],
            terminal_robot_pose_mean=state['terminal_robot_pose']*state['terminal_robot_pose'].new_tensor([6,6,1,1])))
