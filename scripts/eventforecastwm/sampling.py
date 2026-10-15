"""IID scene draws with one categorical mode shared by every agent and time."""
import torch


def joint_samples(forecast,k=6,generator=None):
    # Each draw index has independent residual noise, including when its mode
    # repeats. Resampling from a small finite path bank would create duplicates
    # and distort fixed-K minADE/minFDE comparisons.
    conditional,_,_=forecast.sample(draws=k,generator=generator)
    b,s=forecast.weights.shape;t,n=forecast.positions.shape[2:4]
    conditional=conditional.reshape(b,s,k,t,n,2)
    modes=torch.multinomial(forecast.weights,k,replacement=True,generator=generator)
    batch=torch.arange(b,device=conditional.device)[:,None]
    draws=torch.arange(k,device=conditional.device)[None,:]
    return conditional[batch,modes,draws],modes
