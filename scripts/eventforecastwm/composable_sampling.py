"""Persistent categorical scene identities and residuals across imagined chunks.

The benchmark's historical fixed-K sampler remains unchanged. This interface
draws the categorical identities before residuals, making segmented and direct
joint samples identical under the same RNG. Reset the sample state after a real
observation; posterior paths must be drawn from the corrected belief.
"""
from dataclasses import dataclass
import torch


@dataclass
class JointSampleState:
    modes: torch.Tensor
    conditional_state: object
    end_clock: torch.Tensor
    end_base_covariance: torch.Tensor
    weights: torch.Tensor


def sample_transition(forecast,k=6,generator=None,state=None):
    b,s=forecast.weights.shape;t,n=forecast.positions.shape[2:4]
    if state is None:
        modes=torch.multinomial(forecast.weights,k,replacement=True,generator=generator)
        residual=None
    else:
        if (state.modes.shape!=(b,k) or not torch.equal(state.end_clock+forecast.dt,forecast.times[:,0]) or
            not torch.equal(state.end_base_covariance,forecast.initial_covariance) or
            not torch.equal(state.weights,forecast.weights)):
            raise ValueError('Sample identities, clock or belief changed; draw new posterior paths after observe')
        modes=state.modes;residual=state.conditional_state
    conditional,_,next_residual=forecast.sample(draws=k,generator=generator,initial_state=residual)
    conditional=conditional.reshape(b,s,k,t,n,2)
    batch=torch.arange(b,device=conditional.device)[:,None];draw=torch.arange(k,device=conditional.device)[None]
    successor=JointSampleState(modes,next_residual,forecast.times[:,-1],forecast.covariance[:,:,-1],forecast.weights)
    return conditional[batch,modes,draw],successor
