"""Marked diagonal-Gaussian mixtures restricted to a finite XY rectangle.

All component masses remain in log space. Sampling changes neither the mixture
law nor its Gaussian tails. A stable narrow-interval integral avoids subtracting
indistinguishable CDFs; conditional sampling there uses exact uniform rejection,
not a uniform approximation. Other intervals use SciPy's log-CDF truncnorm.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.special import log_ndtr,logsumexp,ndtr
from scipy.stats import truncnorm

from .types import immutable_array

VERSION='marked_diagonal_gaussian_rectangle_logmass_1'
_LOG_SQRT_2PI=.5*math.log(2*math.pi)
_GL_X,_GL_W=np.polynomial.legendre.leggauss(16)
_GL_X=immutable_array(_GL_X);_GL_W=immutable_array(_GL_W/2)


def _finite64(value,name):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'fiu':raise ValueError(name+' must contain real numeric values')
    try:array=np.asarray(value,np.float64)
    except (ValueError,TypeError,OverflowError) as error:raise ValueError(name+' outside float64 support') from error
    if not np.isfinite(array).all():raise ValueError(name+' must be finite float64')
    return array


def _standardize(x,mean,scale):
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        answer=(x-mean)/scale
        # Prevent an intermediate subtraction overflow when the final quotient
        # is finite, e.g. (-1e308-1e308)/1e308 == -2 mathematically.
        alternate=x/scale-mean/scale
    return np.where(~np.isfinite(answer)&np.isfinite(alternate),alternate,answer)


def _log_width(lo,hi):
    with np.errstate(over='ignore'):
        width=hi-lo
    if math.isfinite(float(width)):return math.log(float(width))
    return float(np.logaddexp(math.log(abs(float(lo))),math.log(abs(float(hi)))))


def _log_difference(large,small):
    if small==-math.inf:return float(large)
    difference=float(small-large)
    if difference>=0.:raise FloatingPointError('Gaussian interval CDF difference unresolved')
    return float(large+math.log(-math.expm1(difference)))


def _axis_mass(lo,hi,mean,scale,a,b):
    log_width=_log_width(lo,hi)-math.log(scale)
    if math.isfinite(a) and math.isfinite(b):
        # In this regime the exponent variation over the interval is <0.001.
        # Sixteen positive Gauss-Legendre nodes integrate the exponential's
        # series through degree31; the remaining analytic contribution is far
        # below float64 roundoff (<1e-60). No probability support is truncated.
        if log_width+math.log1p(max(abs(a),abs(b)))<=math.log(1e-3):
            width=math.exp(log_width) # May underflow; log_width is still retained.
            middle=a+.5*width;shift=.5*width*_GL_X
            square=middle*middle
            if not math.isfinite(square):raise FloatingPointError('Gaussian log mass exceeds float64 range')
            ratio=float(np.dot(_GL_W,np.exp(-middle*shift-.5*shift*shift)))
            return log_width-.5*square-_LOG_SQRT_2PI+math.log(ratio)
    if not a<b:raise FloatingPointError('Gaussian standardized interval collapsed outside narrow support')
    if b<=0.:
        return _log_difference(log_ndtr(b),log_ndtr(a))
    if a>=0.:
        return _log_difference(log_ndtr(-a),log_ndtr(-b))
    return float(np.log1p(-ndtr(a)-ndtr(-b)))


@dataclass(frozen=True,init=False)
class GaussianRectangle:
    lower: np.ndarray
    upper: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    standardized_lower: np.ndarray
    standardized_upper: np.ndarray
    axis_log_mass: np.ndarray
    component_log_mass: np.ndarray
    conditional_log_weights: np.ndarray
    uniform_rejection: np.ndarray
    log_mass: float
    version: str=VERSION

    def sample(self,worlds,rng):
        if (not isinstance(worlds,(int,np.integer)) or isinstance(worlds,(bool,np.bool_))
                or worlds<0 or not isinstance(rng,np.random.Generator)):
            raise ValueError('nonnegative integer worlds and numpy Generator required')
        worlds=int(worlds)
        if not worlds:return np.empty(0,np.int64),np.empty((0,2),np.float64)
        # Log-domain Gumbel-max retains every finite component log mass, including
        # components whose ordinary exp(weight) would numerically underflow.
        # Exact -inf weights are legitimate inactive batch-padding slots. Only
        # those zero-mass slots are skipped; every FINITE weight remains, even
        # when exp(weight) underflows. Map marks back to their original indices.
        active=np.flatnonzero(np.isfinite(self.conditional_log_weights))
        component=active[np.argmax(self.conditional_log_weights[active][None]+
            rng.gumbel(size=(worlds,len(active))),axis=1)]
        means=self.means[component];scales=self.scales[component]
        a=self.standardized_lower[component];b=self.standardized_upper[component]
        uniform=self.uniform_rejection[component];positions=np.empty((worlds,2),np.float64)
        ordinary=~uniform
        if ordinary.any():
            z=truncnorm.rvs(a[ordinary],b[ordinary],random_state=rng)
            with np.errstate(over='ignore',invalid='ignore'):
                x=means[ordinary]+scales[ordinary]*z
            low=np.broadcast_to(self.lower,(worlds,2))[ordinary]
            high=np.broadcast_to(self.upper,(worlds,2))[ordinary]
            bad=~np.isfinite(x)|(x<low)|(x>high)
            if bad.any():
                # Same affine map, evaluated without overflowing an intermediate
                # loc+scale*z. This is not clipping, censoring or a redraw.
                t=(z[bad]-a[ordinary][bad])/(b[ordinary][bad]-a[ordinary][bad])
                with np.errstate(over='ignore',invalid='ignore'):
                    width=high[bad]-low[bad]
                    affine=low[bad]+width*t
                    convex=low[bad]*(1-t)+high[bad]*t
                x[bad]=np.where(np.isfinite(width),affine,convex)
            positions[ordinary]=x
        pending=np.flatnonzero(uniform.ravel())
        while len(pending):
            rows=pending//2;axes=pending%2;u=rng.random(len(pending))
            low=self.lower[axes];high=self.upper[axes]
            with np.errstate(over='ignore',invalid='ignore'):
                width=high-low
                x=np.where(np.isfinite(width),low+width*u,low*(1-u)+high*u)
            mu=means[rows,axes];sigma=scales[rows,axes];mode=np.minimum(np.maximum(mu,low),high)
            zx=_standardize(x,mu,sigma);zm=_standardize(mode,mu,sigma)
            step=_standardize(x,mode,sigma)
            log_ratio=-.5*step*(zx+zm)
            if np.any(~np.isfinite(log_ratio)) or np.any(log_ratio>0.):
                raise FloatingPointError('conditional Gaussian rejection density unresolved')
            accepted=np.log1p(-rng.random(len(pending)))<=log_ratio
            positions.ravel()[pending[accepted]]=x[accepted]
            pending=pending[~accepted]
        if (not np.isfinite(positions).all() or np.any(positions<self.lower)
                or np.any(positions>self.upper)):
            raise FloatingPointError('conditional Gaussian sample outside rectangle; no clipping')
        return component.astype(np.int64),positions


def prepare_rectangle(parameters,robotxy,lo,hi):
    """Prepare f(component, absolute XY | lo <= XY <= hi), preserving marks."""
    robot=_finite64(robotxy,'robotxy');lower=_finite64(lo,'lo');upper=_finite64(hi,'hi')
    relative=_finite64(parameters['spatial_mean'],'spatial_mean')
    scales=_finite64(parameters['spatial_scale'],'spatial_scale')
    raw_logits=np.asarray(parameters['spatial_logits'])
    if raw_logits.dtype.kind not in 'fiu':raise ValueError('spatial_logits must be real numeric values')
    logits=np.asarray(raw_logits,np.float64)
    if np.isnan(logits).any() or np.isposinf(logits).any() or not np.isfinite(logits).any():
        raise ValueError('spatial_logits need finite active weights; only exact -inf padding is allowed')
    if (robot.shape!=(2,) or lower.shape!=(2,) or upper.shape!=(2,) or np.any(lower>=upper)
            or relative.ndim!=2 or relative.shape[1]!=2 or len(relative)==0
            or scales.shape!=relative.shape or logits.shape!=(len(relative),) or np.any(scales<=0)):
        raise ValueError('finite positive rectangle and diagonal Gaussian mixture shapes required')
    with np.errstate(over='ignore'):
        means=relative+robot
    if not np.isfinite(means).all():raise FloatingPointError('absolute Gaussian mean overflow')
    a=_standardize(lower,means,scales);b=_standardize(upper,means,scales)
    axis=np.empty_like(means)
    for component in range(len(means)):
        for dimension in range(2):
            axis[component,dimension]=_axis_mass(float(lower[dimension]),float(upper[dimension]),
                float(means[component,dimension]),float(scales[component,dimension]),
                float(a[component,dimension]),float(b[component,dimension]))
    if not np.isfinite(axis).all():raise FloatingPointError('Gaussian rectangle log mass unresolved; no component dropped')
    joint=logits-logsumexp(logits)+axis.sum(-1);mass=float(logsumexp(joint))
    if not np.isfinite(mass):raise FloatingPointError('Gaussian rectangle normalizer exceeds float64 support')
    mode=np.minimum(np.maximum(0.,a),b)
    with np.errstate(over='ignore',invalid='ignore'):
        variation=.5*np.maximum((a-mode)*(a+mode),(b-mode)*(b+mode))
    uniform=np.isfinite(variation)&(variation<=2.)
    if np.any((a>=b)&~uniform):raise FloatingPointError('Gaussian interval has no supported inverse sampler')
    result=object.__new__(GaussianRectangle)
    values=dict(lower=lower,upper=upper,means=means,scales=scales,standardized_lower=a,
        standardized_upper=b,axis_log_mass=axis,component_log_mass=joint,
        conditional_log_weights=joint-mass,uniform_rejection=uniform)
    for name,value in values.items():object.__setattr__(result,name,immutable_array(value))
    object.__setattr__(result,'log_mass',mass);object.__setattr__(result,'version',VERSION)
    return result
