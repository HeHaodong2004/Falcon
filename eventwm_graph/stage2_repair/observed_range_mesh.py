"""Mesh estimated ONLY from supplied past/current measured range images.

This module accepts arrays, not a simulator, scene path, navigation mesh or
future observations. The CPU BVH is a bounded quality prototype; its complete
preparation cost is reported and is not yet a latency acceptance result.
"""
import time
import ctypes as ct
import numpy as np
import torch
from eventwm_graph.l4.mesh_visibility import _build_bvh
from .geometry_cuda import GeometryKernel
from .range_history import validate


def reconstruct(ranges,origins,rotations,timestamps,now,stride=1,tracks=None,legacy_dedup=False):
    validate(ranges,origins,rotations,timestamps,now)
    n,h,w=np.asarray(ranges).shape
    if len(origins)!=n or len(rotations)!=n or len(timestamps)!=n:raise ValueError('range history alignment')
    u,v=np.meshgrid((np.arange(w)+.5)*2/w-1,1-(np.arange(h)+.5)*2/h,indexing='xy');local=np.stack([u,v,-np.ones_like(u)],axis=-1);local/=np.linalg.norm(local,axis=-1,keepdims=True);triangles=[];selected=[];last=None
    for i in range(n):
        if last is not None and i!=n-1:
            if np.linalg.norm(origins[i]-origins[last])<.07 and np.max(abs(rotations[i]-rotations[last]))<.07:continue
        selected.append(i);last=i
        q=origins[i]+(local@rotations[i].T)*ranges[i,:,:,None];q=q[::stride,::stride];valid=np.isfinite(q).all(-1)&(ranges[i,::stride,::stride]>.03)&(ranges[i,::stride,::stride]<=10.00001)
        if tracks is not None:
            for track in tracks[i]:
                if not track.get('valid',True):continue
                pos=np.asarray(track['position'],np.float64)
                if track.get('velocity_valid')and track.get('estimated_velocity')is not None:pos=pos+np.asarray(track['estimated_velocity'])*min(float(track.get('observation_age',0)),.5)
                person=(np.square(q[...,[0,2]]-pos[[0,2]]).sum(-1)<.5**2)&(q[...,1]>pos[1]-.05)&(q[...,1]<pos[1]+2.)
                valid&=~person
        q[~valid]=np.nan
        for a,b,c in [(q[:-1,:-1],q[1:,:-1],q[:-1,1:]),(q[1:,1:],q[:-1,1:],q[1:,:-1])]:
            good=np.isfinite(a).all(-1)&np.isfinite(b).all(-1)&np.isfinite(c).all(-1)
            good&=(np.linalg.norm(a-b,axis=-1)<.3)&(np.linalg.norm(a-c,axis=-1)<.3)&(np.linalg.norm(b-c,axis=-1)<.3)
            t=np.stack([a,b,c],axis=-2)[good];area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1);triangles.append(t[area>1.e-8])
    t=np.concatenate(triangles).astype(np.float32)
    if not len(t):raise ValueError('no observed range surface')
    # A centroid/normal voxel does NOT identify duplicate triangle footprints.
    # Retaining one triangle per voxel cuts holes in a measured plane. Keep
    # all measured surfaces; the legacy option is a diagnostic ablation only.
    if not legacy_dedup:return t,selected
    centroid=t.mean(1);normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    key=np.column_stack([np.floor(centroid/.035),np.rint(normal/.2)]).astype(np.int32);_,take=np.unique(key,axis=0,return_index=True);return t[np.sort(take)],selected


class ObservedRangeMesh:
    def __init__(self,ranges,origins,rotations,timestamps,now,resolution,device='cuda:0',stride=1,tracks=None,plane_extension=0.,free_keys=None,legacy_dedup=False):
        tick=time.perf_counter();triangles,self.keyframes=reconstruct(ranges,origins,rotations,timestamps,now,stride,tracks,legacy_dedup)
        self.planes=[]
        if plane_extension>0:
            from .planar_completion import complete
            triangles,self.planes=complete(triangles,ranges,origins,rotations,np.empty((0,2))if free_keys is None else free_keys,resolution,plane_extension);triangles=triangles.astype(np.float32)
        bvh=_build_bvh(triangles.astype(np.float64),12);packed=np.concatenate([triangles[:,0],triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]],1);arrays=dict(keys=np.empty((0,2),np.int32),targets=np.empty((0,63,3),np.float64),triangles=packed)
        for name,a in zip(['bounds','left','right','first','count','indices'],bvh):arrays[name]=np.asarray(a,np.float32 if name=='bounds'else np.int32)
        self.triangles=len(triangles);self.kernel=GeometryKernel(arrays,resolution,device);self.prepare_seconds=time.perf_counter()-tick

    def blocked(self,unknown,lo,cameras,candidates,steps):
        return self.kernel.compute(unknown,lo,cameras,candidates,steps)

    def set_probes(self,keys,targets):
        self.kernel.arrays['keys']=keys;self.kernel.arrays['targets']=targets
        self.kernel.static['keys']=self.kernel.tensor(keys,np.int32);self.kernel.static['targets']=self.kernel.tensor(targets,np.float64)

    def floor_support(self,keys,heights):
        """Estimate local square support from OBSERVED triangles, never navmesh.

        Nine downward ray hits and the recorded observer's height tolerances
        are a geometric prediction, not a certified traversability decision.
        """
        k=self.kernel;r=k.resolution
        offsets=np.array([(0,0),(-.49,-.49),(-.49,.49),(.49,-.49),(.49,.49),(-.49,0),(.49,0),(0,-.49),(0,.49)])*r
        xz=(np.asarray(keys)[:,None,:]+.5)*r+offsets[None]
        origins=np.empty((len(keys),9,3),np.float32);origins[:,:,[0,2]]=xz;origins[:,:,1]=np.asarray(heights)[:,None]+.18
        directions=np.zeros_like(origins);directions[:,:,1]=-1
        limits=np.full((len(keys),9),.36,np.float32)
        out=torch.empty(len(keys)*9,device=k.device)
        k.module.launch('test_rays',len(keys)*9,[k.tensor(origins),k.tensor(directions),k.tensor(limits),ct.c_int(len(keys)*9),out,*k.mesh])
        hit=out.cpu().numpy().reshape(-1,9)
        if np.any(hit<0):raise RuntimeError('floor support BVH stack exhausted')
        support=np.all(hit<.3599,axis=1)&(np.max(abs(hit-hit[:,:1]),axis=1)<=.08)
        return support,np.asarray(heights)+.18-hit[:,0]
