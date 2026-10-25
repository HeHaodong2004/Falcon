"""Vectorized current-calibration camera planes, same mathematical features."""
import numpy as np
from .legal_camera import base_rotations


def parameters(path,translation,rotation,resolution):
    base=base_rotations(path[:,4]);origins=path[:,1:4]+base@translation;rotations=base@rotation;forward=-rotations[:,:,2];planes=np.stack([forward,forward+rotations[:,:,0],forward-rotations[:,:,0],forward+rotations[:,:,1],forward-rotations[:,:,1]],1);norm=np.sqrt((planes[...,None,:]@planes[..., :,None])[...,0,0]);angles=np.arange(16)*2*np.pi/16;radius=.25+resolution/np.sqrt(2);halo=np.column_stack([np.cos(angles)*radius,np.zeros(16),np.sin(angles)*radius]);offset=(planes@halo.T).min(-1)+np.minimum(planes[:,:,1]*.025,planes[:,:,1]*1.4);out=np.empty((len(path),23),np.float64);out[:,:3]=origins;block=out[:,3:].reshape(-1,5,4);block[:,:,:3]=planes/norm[:,:,None];block[:,:,3]=offset/norm;signature=np.r_[np.asarray(translation)/2,np.asarray(rotation).ravel()].astype(np.float32);return out,signature
