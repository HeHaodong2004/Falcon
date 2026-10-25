"""Signed bearings of supplied poses, preserving the starting view direction.

Yaw was always present in the trajectory. These features retain information
that order-aggregated future-facing summaries can discard. They use no camera
truth, observation labels or unobserved state.
"""
import math
import numpy as np
from numba import njit

DIMENSIONS=23


@njit(cache=True)
def signed_pose_features(keys,path,resolution,x88):
    n=len(keys);out=np.empty((n,23),np.float32);indices=(0,min(1,len(path)-1),len(path)//2,len(path)-1)
    cosine=np.empty(4,np.float64);sine=np.empty(4,np.float64)
    for k in range(4):cosine[k]=math.cos(path[indices[k],4]);sine[k]=math.sin(path[indices[k],4])
    for i in range(n):
        x=(keys[i,0]+.5)*resolution;z=(keys[i,1]+.5)*resolution
        for k in range(4):
            j=indices[k];dx=x-path[j,1];dz=z-path[j,3];dist=math.sqrt(dx*dx+dz*dz);den=max(dist,1e-6)
            out[i,4*k]=(dx*cosine[k]-dz*sine[k])/den;out[i,4*k+1]=(dx*sine[k]+dz*cosine[k])/den;out[i,4*k+2]=dist/10.;out[i,4*k+3]=x88[i,24]+path[0,2]-path[j,2]
        out[i,16]=cosine[0];out[i,17]=sine[0];out[i,18]=cosine[3];out[i,19]=sine[3]
        out[i,20]=out[i,0]-math.sqrt(.5);out[i,21]=x88[i,32]-math.sqrt(.5);out[i,22]=x88[i,32]-out[i,0]
    return out
