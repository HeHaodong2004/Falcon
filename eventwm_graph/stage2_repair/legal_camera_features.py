"""Continuous certificate-view covariates from LEGAL sensor calibration.

Only the current observed camera extrinsic, supplied base trajectory and
nearest-known map height enter. These are features, not visibility labels or
a hard candidate mask. The range term is a conservative cylinder bound.
"""
import math
import numpy as np
from numba import njit
from .legal_camera import base_rotations


def parameters(path,translation,rotation,resolution):
    base=base_rotations(path[:,4]);origins=path[:,1:4]+base@translation;rotations=base@rotation;out=np.empty((len(path),23),np.float64);out[:,:3]=origins;radius=.25+resolution/math.sqrt(2);angles=np.arange(16)*(2*math.pi/16);halo=np.column_stack([np.cos(angles)*radius,np.zeros(16),np.sin(angles)*radius])
    for i,r in enumerate(rotations):
        forward=-r[:,2];planes=np.array([forward,forward+r[:,0],forward-r[:,0],forward+r[:,1],forward-r[:,1]])
        for j,n in enumerate(planes):
            norm=np.linalg.norm(n);out[i,3+4*j:6+4*j]=n/norm;out[i,6+4*j]=(np.min(halo@n)+min(n[1]*.025,n[1]*1.4))/norm
    signature=np.r_[np.asarray(translation)/2,np.asarray(rotation).ravel()].astype(np.float32)
    return out,signature


@njit(cache=True)
def cell_features(keys,path,resolution,x88,params):
    out=np.zeros((len(keys),8),np.float32);duration=max(path[-1,0]-path[0,0],1.e-4);radius=.25+resolution/math.sqrt(2)
    for i in range(len(keys)):
        x=(keys[i,0]+.5)*resolution;z=(keys[i,1]+.5)*resolution;y=float(x88[i,24])+path[0,2];first=1.;last=0.;best=-10.;count=0.;weighted=0.;positive=0.;initial=-10.
        for j in range(len(path)):
            p=params[j];dx=x-p[0];dy=y-p[1];dz=z-p[2];margin=100.
            for k in range(5):
                at=3+4*k;margin=min(margin,dx*p[at]+dy*p[at+1]+dz*p[at+2]+p[at+3])
            horizontal=math.sqrt(dx*dx+dz*dz)+radius;vertical=max(abs(dy+.025),abs(dy+1.4));margin=min(margin,10.-math.sqrt(horizontal*horizontal+vertical*vertical))
            if j==0:initial=margin;continue
            best=max(best,margin)
            if margin>=0:
                count+=1.;weighted+=path[j,0]-path[j-1,0];positive+=margin;when=(path[j,0]-path[0,0])/duration;first=min(first,when);last=max(last,when)
        den=max(1,len(path)-1);out[i,0]=max(-2.,min(2.,initial/2));out[i,1]=max(-2.,min(2.,best/2));out[i,2]=count/den;out[i,3]=weighted/duration;out[i,4]=positive/(2*den);out[i,5]=first;out[i,6]=last;out[i,7]=max(-2.,min(2.,(best-initial)/2))
    return out
