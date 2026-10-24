"""Immutable numeric facts for exact historical observation-view features.

All actual frames remain represented, including empty, invalid and unavailable
frames. This caches sensor facts, never inferred visibility or future evidence.
Camera predicates use guarded float64 signs; ambiguous frames retain the original
full-point-shape NumPy camera computation in features.py.
"""
from dataclasses import dataclass
import math

import numpy as np
from numba import njit

from .types import SealedObservation,immutable_array

_CERTIFICATE=object()
VERSION='immutable_arrived_view_facts_1'


def _positive_xy(frame):
    if not frame.get('observation_valid',True) or not frame.get('human_sensor_available',True):
        return np.empty((0,2),np.float64)
    return np.asarray([np.asarray(track['position'],np.float64)[[0,2]] for track in frame['tracks']
        if track.get('visible_now') and track.get('valid',True)
        and abs(float(track['last_seen_time'])-float(frame['timestamp']))<1e-7],np.float64).reshape(-1,2)


@dataclass(frozen=True,init=False)
class ViewSummary:
    timestamps: np.ndarray
    origins: np.ndarray
    rotations: np.ndarray
    valid: np.ndarray
    packed_supported: np.ndarray
    code_offsets: np.ndarray
    codes: np.ndarray
    positive_offsets: np.ndarray
    positive_xy: np.ndarray
    version: str=VERSION

    @property
    def frame_count(self):return len(self.timestamps)

    @property
    def timestamp(self):return float(self.timestamps[-1])

    def validate_certificate(self):
        if getattr(self,'_certificate',None) is not _CERTIFICATE:
            raise ValueError('uncertified arrived view facts')
        return self

    @classmethod
    def _verified(cls,timestamps,origins,rotations,valid,supported,offsets,codes_payload,positive_offsets,xy_payload):
        result=object.__new__(cls)
        for key,value in (('timestamps',timestamps),('origins',origins),('rotations',rotations),('valid',valid),
                          ('packed_supported',supported),('code_offsets',offsets),('positive_offsets',positive_offsets)):
            object.__setattr__(result,key,immutable_array(value))
        object.__setattr__(result,'codes',np.frombuffer(codes_payload,np.uint64))
        object.__setattr__(result,'positive_xy',np.frombuffer(xy_payload,np.float64).reshape(-1,2))
        for key,value in dict(_codes_payload=codes_payload,_xy_payload=xy_payload,
                              version=VERSION,_certificate=_CERTIFICATE).items():object.__setattr__(result,key,value)
        return result

    @classmethod
    def from_frames(cls,frames):
        sealed=tuple(SealedObservation.from_raw(frame) for frame in frames)
        if not sealed:raise ValueError('arrived view facts need an actual frame')
        times=np.asarray([frame['timestamp'] for frame in sealed],np.float64)
        if np.any(np.diff(times)<=0):raise ValueError('noncausal arrived view facts')
        codes=[frame.free_codes for frame in sealed]
        points=[_positive_xy(frame) for frame in sealed]
        return cls._verified(times,np.asarray([frame['camera_origin'] for frame in sealed],np.float64),
            np.asarray([frame['camera_rotation'] for frame in sealed],np.float64),
            np.asarray([frame.get('observation_valid',True) and frame.get('human_sensor_available',True) for frame in sealed],bool),
            np.asarray([code is not None for code in codes],bool),
            np.r_[0,np.cumsum([0 if code is None else len(code) for code in codes])].astype(np.int64),
            b''.join(code.tobytes() for code in codes if code is not None),
            np.r_[0,np.cumsum([len(point) for point in points])].astype(np.int64),
            b''.join(point.tobytes() for point in points))

    def append(self,frame):
        self.validate_certificate();frame=SealedObservation.from_raw(frame)
        if float(frame['timestamp'])<=self.timestamp:raise ValueError('noncausal arrived view append')
        code=frame.free_codes;points=_positive_xy(frame)
        return type(self)._verified(np.r_[self.timestamps,float(frame['timestamp'])],
            np.concatenate((self.origins,np.asarray(frame['camera_origin'],np.float64)[None])),
            np.concatenate((self.rotations,np.asarray(frame['camera_rotation'],np.float64)[None])),
            np.r_[self.valid,frame.get('observation_valid',True) and frame.get('human_sensor_available',True)],
            np.r_[self.packed_supported,code is not None],
            np.r_[self.code_offsets,self.code_offsets[-1]+(len(code) if code is not None else 0)],
            self._codes_payload+(code.tobytes() if code is not None else b''),
            np.r_[self.positive_offsets,self.positive_offsets[-1]+len(points)],self._xy_payload+points.tobytes())


@njit(cache=True)
def reduce_view_evidence(frustum,flat,keys,decay,indices,code_offsets,codes,positive_offsets,positive_xy):
    """Exact integer support and original sqrt-distance threshold, all frames."""
    points=len(flat);out=np.zeros((points,3),np.float64)
    for i in range(len(indices)):
        frame=indices[i];weight=decay[i]
        for j in range(points):
            near=False
            for h in range(positive_offsets[frame],positive_offsets[frame+1]):
                dx=flat[j,0]-positive_xy[h,0];dz=flat[j,1]-positive_xy[h,1]
                if math.sqrt(dx*dx+dz*dz)<.5:
                    near=True;break
            if near and weight>out[j,1]:out[j,1]=weight
            if not frustum[i,j]:continue
            if weight>out[j,0]:out[j,0]=weight
            if near:continue
            low=code_offsets[frame];high=code_offsets[frame+1];end=high
            while low<high:
                middle=(low+high)//2
                if codes[middle]<keys[j]:low=middle+1
                else:high=middle
            if low<end and codes[low]==keys[j] and weight>out[j,2]:out[j,2]=weight
    return out


def certified_camera_frustum(xyz,origins,rotations):
    """Internal float64-only certificate; other arithmetic is outside its proof."""
    if any(type(value) is not np.ndarray or value.dtype!=np.float64
           for value in (xyz,origins,rotations)):
        raise ValueError('camera certificate requires plain float64 NumPy arrays')
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or origins.ndim!=2 or origins.shape[1]!=3
            or rotations.shape!=(len(origins),3,3)):
        raise ValueError('camera certificate requires XYZ[P,3], origins[F,3], rotations[F,3,3]')
    return _certified_camera_frustum64(xyz,origins,rotations)


@njit(cache=True)
def _certified_camera_frustum64(xyz,origins,rotations):
    """Certify original float64 camera predicates, or request full-frame NumPy.

    Each three-term dot carries delta=1e-12+32eps*sum(abs(terms)), exceeding
    roundoff differences between ordinary float64 dot evaluation orders. Norm
    uncertainty uses the component-error L1 bound plus conservative operation
    roundoff. No predicate threshold is expanded: an ambiguous frame is entirely
    recomputed with the original NumPy matmul/norm and unchanged point shape.
    """
    frames=len(origins);points=len(xyz);answer=np.zeros((frames,points),np.bool_)
    fallback=np.zeros(frames,np.bool_);eps=np.finfo(np.float64).eps
    for i in range(frames):
        for j in range(points):
            a0=xyz[j,0]-origins[i,0];a1=xyz[j,1]-origins[i,1];a2=xyz[j,2]-origins[i,2]
            x0=a0*rotations[i,0,0];x1=a1*rotations[i,1,0];x2=a2*rotations[i,2,0]
            y0=a0*rotations[i,0,1];y1=a1*rotations[i,1,1];y2=a2*rotations[i,2,1]
            z0=a0*rotations[i,0,2];z1=a1*rotations[i,1,2];z2=a2*rotations[i,2,2]
            x=(x0+x1)+x2;y=(y0+y1)+y2;z=(z0+z1)+z2;forward=-z
            dx=1e-12+32*eps*(abs(x0)+abs(x1)+abs(x2))
            dy=1e-12+32*eps*(abs(y0)+abs(y1)+abs(y2))
            dz=1e-12+32*eps*(abs(z0)+abs(z1)+abs(z2))
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and
                    math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dz)):
                fallback[i]=True;break
            near=forward-1e-6;horizontal=forward-abs(x);vertical=forward-abs(y)
            dh=dx+dz+eps*(abs(forward)+abs(x));dv=dy+dz+eps*(abs(forward)+abs(y))
            # Any independently certified false conjunct makes the whole result
            # false, even if an irrelevant other predicate lies on its boundary.
            if near < -dz or horizontal < -dh or vertical < -dv:
                continue
            if abs(near)<=dz or abs(horizontal)<=dh or abs(vertical)<=dv:
                fallback[i]=True;break
            norm=math.sqrt((x*x+y*y)+z*z)
            dn=dx+dy+dz+32*eps*(abs(x)+abs(y)+abs(z)+1.)
            if not math.isfinite(norm) or not math.isfinite(dn) or abs(norm-10.)<=dn:
                fallback[i]=True;break
            answer[i,j]=norm<10.
    return answer,fallback
