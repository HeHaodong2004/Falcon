"""Camera geometry derived solely from the current observed camera and trial.

The rigid-current-extrinsic proxy is deliberately simple. Recorded FUTURE
camera poses and hidden scene geometry are not accepted by these functions.
"""
import numpy as np


def base_rotations(yaw):
    yaw=np.asarray(yaw,np.float64);c,s=np.cos(yaw),np.sin(yaw);out=np.zeros((len(yaw),3,3),np.float64);out[:,0,0]=c;out[:,0,2]=s;out[:,1,1]=1.;out[:,2,0]=-s;out[:,2,2]=c;return out


def current_extrinsic(path,current_origin,current_rotation):
    p=np.asarray(path,np.float64);origin=np.asarray(current_origin,np.float64);rotation=np.asarray(current_rotation,np.float64)
    if origin.shape!=(3,)or rotation.shape!=(3,3)or not np.isfinite(origin).all()or not np.isfinite(rotation).all():raise ValueError('current observed camera geometry required')
    frame=base_rotations(p[:1,4])[0]
    return frame.T@(origin-p[0,1:4]),frame.T@rotation


def proxy_cameras(path,current_origin,current_rotation):
    p=np.asarray(path,np.float64);translation,rotation=current_extrinsic(p,current_origin,current_rotation);base=base_rotations(p[:,4]);return p[:,1:4]+base@translation,base@rotation
