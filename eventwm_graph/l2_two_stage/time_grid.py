"""Time-indexed worlds that preserve every piecewise velocity change exactly."""
import numpy as np
from .neural import KNOTS
from .types import immutable_array


def prediction_grid(horizon,count=64):
    if not np.isfinite(horizon) or not 0<float(horizon)<=3.028001:
        raise ValueError('unsupported finite prediction horizon')
    knots=np.array([t for t in KNOTS[1:-1] if t<horizon],np.float64)
    if not isinstance(count,(int,np.integer)) or isinstance(count,(bool,np.bool_)) or count<len(knots)+2:
        raise ValueError('time grid must contain endpoints and every internal motion knot')
    times=np.linspace(0.,float(horizon),count)
    available=np.ones(count-2,bool)
    for knot in knots:
        distance=np.where(available,abs(times[1:-1]-knot),np.inf)
        index=int(distance.argmin());times[index+1]=knot;available[index]=False
    times.sort()
    if np.any(np.diff(times)<=0):raise RuntimeError('motion-preserving time grid is not strictly increasing')
    return times


def interpolate_positions(bank,requested_times):
    """Return [world,requested_time,person,XZ], without inference or resampling.

    Exact for the declared piecewise-linear continuous trajectory when every
    internal velocity knot is present. Native-only diagnostic banks may omit a
    knot; off-grid interpolation then refuses instead of silently approximating.
    """
    raw=np.asarray(requested_times);raw_times=np.asarray(bank.times)
    if raw.dtype.kind not in 'fiu' or raw_times.dtype.kind not in 'fiu':
        raise ValueError('relative times must be real numeric values')
    requested=np.asarray(raw,np.float64);times=np.asarray(raw_times,np.float64)
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0)):
        raise ValueError('stored bank times must be finite and strictly increasing')
    raw_positions=np.asarray(bank.positions)
    if raw_positions.dtype.kind not in 'fiu':
        raise ValueError('stored world positions must be real numeric arrays')
    with np.errstate(over='ignore',invalid='ignore'):
        positions=np.asarray(raw_positions,np.float64)
    if (positions.ndim!=4 or positions.shape[1]!=len(times)
            or positions.shape[-1]!=2 or not np.isfinite(positions).all()):
        raise ValueError('stored world positions must be finite real [K,T,N,2] arrays')
    if requested.ndim!=1 or not np.isfinite(requested).all():
        raise ValueError('finite one-dimensional relative times required')
    if np.any(requested<times[0]) or np.any(requested>times[-1]):
        raise ValueError('requested time outside sampled support; extrapolation forbidden')
    indices=np.searchsorted(times,requested)
    exact=(indices<len(times))
    exact[exact]&=times[indices[exact]]==requested[exact]
    if exact.all():return immutable_array(positions[:,indices])
    if any(knot<times[-1] and not np.any(times==knot) for knot in KNOTS[1:-1]):
        raise ValueError('bank omits an internal velocity knot; exact off-grid reuse unavailable')
    upper=np.clip(indices,1,len(times)-1);lower=upper-1
    fraction=(requested-times[lower])/(times[upper]-times[lower])
    result=positions[:,lower]+fraction[None,:,None,None]*(positions[:,upper]-positions[:,lower])
    # Exact stored columns are copied, including when mixed with off-grid
    # requests. Re-evaluating a+(b-a) may differ from b by one floating ULP.
    result[:,exact]=positions[:,indices[exact]]
    return immutable_array(result)
