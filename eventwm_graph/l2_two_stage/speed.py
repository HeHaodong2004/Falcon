"""Smooth bijection from R² to the known exogenous human speed ball.

T(u)=u/sqrt(1+||u||²/R²); its inverse Jacobian determinant is
(1-||v||²/R²)^(-2). This is a probability pushforward, not clipping samples.
The source collector config supplies the3m/s physical bound, never future truth.
"""
import numpy as np
import torch


def bound_numpy(value,maximum):
    value=np.asarray(value)
    return value/np.sqrt(1+(value**2).sum(-1,keepdims=True)/maximum**2)


def bound_torch(value,maximum):
    return value/torch.sqrt(1+value.square().sum(-1,keepdim=True)/maximum**2)


def inverse_numpy(value,maximum):
    value=np.asarray(value);inside=1-(value**2).sum(-1,keepdims=True)/maximum**2
    if np.any(inside<=0):raise ValueError('native speed reaches or exceeds declared open speed support')
    return value/np.sqrt(inside)
