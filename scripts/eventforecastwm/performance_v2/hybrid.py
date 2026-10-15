"""Recurrent trajectory belief with a fixed-frame anonymous occupancy marginal.

The anonymous component is a learned static-geometry prior, not identity paths.
Its forecast clock persists across composition. Stable-track observations update
the tracked belief; they do not pretend to reveal unobserved anonymous agents.
Re-encoding legal history may establish a new robot frame and map grid.
"""
from dataclasses import dataclass,replace
import torch
from torch import nn

from eventforecastwm.model_agent_scale import AgentScaleWorldModel


@dataclass
class OccupancyBelief:
    physical: object
    geometry_categories: torch.Tensor
    origin_clock: torch.Tensor

    @property
    def core(self):return self.physical.core

    @property
    def covariance(self):return self.physical.covariance

    def detach(self):
        return OccupancyBelief(self.physical.detach(),self.geometry_categories.detach(),self.origin_clock.detach())


class OccupancyWorldModel(nn.Module):
    def __init__(self,physical_recipe,variant='geometry_occupancy_v2',physical_uses_geometry=False):
        super().__init__()
        if variant!='geometry_occupancy_v2' or physical_recipe.get('variant') not in ('agent_scene_scale_v1','stopped_agent_scale_v2'):
            raise ValueError('This release adapter requires a calibrated per-agent scale trajectory model')
        if physical_recipe.get('variant')=='stopped_agent_scale_v2':
            from .regime import RegimeWorldModel
            self.physical=RegimeWorldModel(**physical_recipe)
        else:self.physical=AgentScaleWorldModel(**physical_recipe)
        self.register_buffer('anonymous_probabilities',torch.zeros(4,20,32,32))
        self.physical_uses_geometry=physical_uses_geometry
        self.recipe=dict(variant=variant,physical_recipe=physical_recipe,physical_uses_geometry=physical_uses_geometry)

    @property
    def core(self):return self.physical.core

    @property
    def noise(self):return self.physical.noise

    @property
    def raw_decay_rate(self):return self.physical.raw_decay_rate

    def parameter_counts(self):
        result=self.physical.parameter_counts()
        result.update(recipe=self.recipe,anonymous_probability_values=self.anonymous_probabilities.numel(),
                      stored_state_values=sum(v.numel() for v in self.state_dict().values()))
        return result

    def encode_history(self,history,static_geometry=None,sensor_config=None,robot_geometry=None,**kwargs):
        belief=self.physical.encode_history(history,static_geometry if self.physical_uses_geometry else None,
                                             sensor_config,robot_geometry,**kwargs)
        b=belief.core.positions.shape[0];device=belief.core.positions.device
        categories=torch.zeros(b,32,32,dtype=torch.long,device=device)
        if static_geometry is not None:
            if not isinstance(static_geometry,dict) or set(static_geometry)-{'features','support'}:
                raise ValueError('Anonymous occupancy accepts only known static features and their support')
            features=torch.as_tensor(static_geometry['features'],device=device)
            support=torch.as_tensor(static_geometry.get('support',False),device=device,dtype=torch.bool)
            if features.ndim==3:features=features[None]
            features=features.expand(b,-1,-1,-1);support=support.expand(b,32,32)
            if features.shape!=(b,2,32,32) or not torch.isfinite(features).all():raise ValueError('Finite 2x32x32 static map required')
            categories=torch.where(support,1+(features[:,1]>.5).long()*(1+(features[:,0]>.25).long()),0)
        return OccupancyBelief(belief,categories,belief.core.clock.clone())

    def imagine(self,belief,candidate_query=None,horizon_s=3.):
        forecast=self.physical.imagine(belief.physical,candidate_query,horizon_s)
        ticks=torch.round((forecast.times-belief.origin_clock[:,None])/.25).long()-1
        supported=(ticks>=0)&(ticks<20)
        table=self.anonymous_probabilities.permute(1,0,2,3)[ticks.clamp(0,19)]
        index=belief.geometry_categories[:,None,None].expand(-1,ticks.shape[1],1,-1,-1)
        probability=table.gather(2,index).squeeze(2)*supported[...,None,None]
        successor=OccupancyBelief(forecast.next_belief,belief.geometry_categories,belief.origin_clock)
        forecast=replace(forecast,next_belief=successor)
        forecast.anonymous_occupancy=probability
        forecast.anonymous_occupancy_supported=supported
        return forecast

    def observe(self,belief,actual_arrived_observation):
        physical=self.physical.observe(belief.physical,actual_arrived_observation)
        return OccupancyBelief(physical,belief.geometry_categories,belief.origin_clock)


def assemble(physical,tables,strength=128.):
    """Compose already trained components; no fitting or target access occurs."""
    model=OccupancyWorldModel(physical.recipe)
    model.physical.load_state_dict(physical.state_dict())
    positive=tables['positive'].sum(0);exposure=tables['exposure'].sum(0)
    probabilities=(positive+strength*tables['prior'])/(exposure+strength)
    if not torch.isfinite(probabilities).all() or not ((probabilities>=0)&(probabilities<=1)).all():
        raise ValueError('Invalid anonymous occupancy probabilities')
    model.anonymous_probabilities.copy_(probabilities)
    return model
