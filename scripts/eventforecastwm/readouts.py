"""Physical human-center occupancy marginals of the trajectory distribution."""
import math
import torch


def marginal_cdf(forecast,locations):
    if hasattr(forecast,'agent_scale_probabilities'):
        from eventforecastwm.model_agent_scale import marginal_cdf as agent
        return agent(forecast,locations)
    if hasattr(forecast,'scale_values'):
        from eventforecastwm.model_scale_mixture import marginal_cdf as mixture
        return mixture(forecast,locations)
    if hasattr(forecast,'concentration'):
        from eventforecastwm.model_student import marginal_cdf as student
        return student(forecast,locations)
    mean=forecast.positions;sigma=forecast.position_variance.sqrt()[...,None]
    if locations.ndim==6:mean=mean[...,None];sigma=sigma[...,None]
    return .5*(1+torch.erf((locations-mean)/(sigma*math.sqrt(2))))


def tracked_occupancy(forecast,extent=8.,size=32):
    if hasattr(forecast,'agent_scale_probabilities'):
        from eventforecastwm.model_agent_scale import occupancy as agent
        return agent(forecast,extent,size)
    if hasattr(forecast,'scale_values'):
        from eventforecastwm.model_scale_mixture import occupancy
        return occupancy(forecast,extent,size)
    if hasattr(forecast,'concentration'):
        return student_occupancy(forecast,extent,size)
    p=forecast.positions;edges=torch.linspace(-extent,extent,size+1,device=p.device,dtype=p.dtype)
    cdf=marginal_cdf(forecast,edges[None,None,None,None,None])
    mass=(cdf[...,1:]-cdf[...,:-1]).clamp_min(0.)
    per_agent=mass[...,1,:,None]*mass[...,0,None,:]*forecast.valid[:,None,None,:,None,None]
    return ((1-(1-per_agent).prod(3))*forecast.weights[:,:,None,None,None]).sum(1)


@torch.no_grad()
def student_occupancy(forecast,extent=8.,size=32,nodes=32):
    """Integrate the shared scale AFTER conditional agent/axis probabilities.

    Products of marginal Student CDF differences would incorrectly assume that
    axes and humans have independent scales. Gauss-Legendre quadrature over the
    Gamma quantile keeps their shared scene variance intact.
    """
    import numpy as np
    from scipy.special import gammaincinv
    points,weights=np.polynomial.legendre.leggauss(nodes);uniform=(points+1)/2;weights=weights/2
    alpha=forecast.concentration.detach().cpu().numpy();beta=forecast.rate.detach().cpu().numpy()
    p=forecast.positions;edges=torch.linspace(-extent,extent,size+1,device=p.device,dtype=p.dtype)
    base_variance=forecast.covariance[...,0,0]+forecast.readout_scale.square();result=0.
    for u,w in zip(uniform,weights):
        v=torch.as_tensor(beta/gammaincinv(alpha,u),device=p.device,dtype=p.dtype)
        sigma=(base_variance*v[:,:,None,None]).sqrt()
        cdf=.5*(1+torch.erf((edges-p[...,None])/(sigma[...,None,None]*math.sqrt(2))))
        mass=(cdf[...,1:]-cdf[...,:-1]).clamp_min(0.)
        per_agent=mass[...,1,:,None]*mass[...,0,None,:]*forecast.valid[:,None,None,:,None,None]
        result=result+w*((1-(1-per_agent).prod(3))*forecast.weights[:,:,None,None,None]).sum(1)
    return result
