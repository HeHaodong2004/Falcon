"""Mask missing sensor values before probabilistic observation arithmetic.

The frozen training sources stay byte-identical for exact recovery. Exported
models receive this inference adapter; no state or prediction parameters change.
Filling a masked numerical placeholder does not assert physical free space.
"""
from types import MethodType
import torch


def protect_observe(model):
    if getattr(model,'_missing_observation_protected',False):return model
    original=model.observe
    def observe(self,belief,observation):
        c=belief.core;obs=dict(observation)
        mask=obs['observed'].to(c.clock.device).bool()
        acquired=obs['acquisition_times'].to(c.clock.device)
        if mask.shape!=c.valid.shape or acquired.shape!=mask.shape:raise ValueError('Observation masks/timestamps must match existing legal track slots')
        if not torch.isfinite(acquired[mask]).all():raise ValueError('Observed acquisition timestamps must be finite')
        positions=obs['positions'].to(c.positions)
        if positions.shape!=c.last_observed_positions.shape:raise ValueError('Observation positions must match existing legal track slots')
        obs.update(observed=mask,positions=torch.where(mask[...,None],positions,torch.zeros_like(positions)),
            acquisition_times=torch.where(mask,acquired,c.clock[:,None].expand_as(acquired)))
        return original(belief,obs)
    model.observe=MethodType(observe,model);model._missing_observation_protected=True
    return model
