"""Legal range-history selection, independent of simulator reconstruction."""
import numpy as np


def ancestor_frames(parents,history_count,start):
    ancestors=[];seen=set()
    while start is not None and start>=0:
        start=int(start)
        if start in seen or start>=len(parents):raise ValueError('invalid sensor ancestry')
        seen.add(start);ancestors.append(start+int(history_count));parent=int(parents[start])
        if parent>=start:raise ValueError('sensor parent must precede child')
        start=parent
    return np.array(list(range(int(history_count)))+list(reversed(ancestors)),np.int64)


def validate(ranges,origins,rotations,timestamps,now):
    if np.asarray(ranges).ndim!=3:raise ValueError('range images must have frame/row/column axes')
    n=len(ranges);times=np.asarray(timestamps)
    if np.shape(origins)!=(n,3)or np.shape(rotations)!=(n,3,3)or times.shape!=(n,):raise ValueError('range history alignment')
    if not n or not np.isfinite(now)or not np.isfinite(times).all():raise ValueError('invalid range clock')
    if np.any(times>now+1.e-7):raise ValueError('future range observation forbidden')
    if np.any(np.diff(times)<0):raise ValueError('noncausal range history order')
    if not np.isfinite(origins).all()or not np.isfinite(rotations).all():raise ValueError('nonfinite sensor pose')
