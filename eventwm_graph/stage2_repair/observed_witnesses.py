"""Estimate wall witnesses from legal published inflation footprints.

The measured ray-hit locations were not retained in the recorded input.
This inverse raster lookup estimates them using only the legal update rows.
It never reads a scene mesh, native navigation data or future observations.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=4)
def templates(resolution):
    span=int(np.ceil(.25/resolution))+2;grid=np.array([(x,z)for x in range(-span,span+1)for z in range(-span,span+1)],np.int32);centers=(grid+.5)*resolution;acc={}
    for x in np.linspace(0,resolution,128,endpoint=False):
        for z in np.linspace(0,resolution,128,endpoint=False):
            q=np.array([x,z]);keys=grid[np.square(centers-q).sum(1)<=.0625];lo=keys.min(0);pattern=(keys-lo).astype(np.int8).tobytes();offset=q-lo*resolution
            if pattern not in acc:acc[pattern]=[offset.copy(),1]
            else:acc[pattern][0]+=offset;acc[pattern][1]+=1
    return {k:v[0]/v[1]for k,v in acc.items()}


def decode(observation,resolution):
    keys=np.array([u['cell']for u in observation['static_updates']if u['state']==2],np.int32).reshape(-1,2)
    if not len(keys):return np.empty((0,3)),dict(rows=0,patches=0,recovered_rows=0)
    table=templates(float(resolution));cuts=[0];low=keys[0].copy();high=keys[0].copy()
    for i in range(1,len(keys)):
        a,b=keys[i-1],keys[i];reset=b[0]<a[0]or(b[0]==a[0]and b[1]<=a[1]);newlo=np.minimum(low,b);newhi=np.maximum(high,b)
        if reset or np.any((newhi-newlo)*resolution>.50000001):cuts.append(i);low=b.copy();high=b.copy()
        else:low=newlo;high=newhi
    cuts.append(len(keys));points=[];recovered=0
    for start,end in zip(cuts[:-1],cuts[1:]):
        patch=keys[start:end]
        if len(patch)<8 or len(patch)>32:continue
        lo=patch.min(0);key=(patch-lo).astype(np.int8).tobytes();offset=table.get(key)
        if offset is None:continue
        point=lo*resolution+offset;points.append([point[0],point[1],observation['robot_pose'][1]]);recovered+=len(patch)
    return np.array(points,np.float64).reshape(-1,3),dict(rows=len(keys),patches=len(points),recovered_rows=recovered)


def segments(points,free_keys,resolution,extension=2.,seed=9128):
    if len(points)<4:return np.empty((0,5),np.float64)
    # Duplicate time samples do not increase confidence in a single witness.
    _,take=np.unique(np.rint(points/.02).astype(np.int64),axis=0,return_index=True);cloud=points[take];p=cloud[:,:2];free=(free_keys+.5)*resolution;active=np.ones(len(p),bool);rng=np.random.default_rng(seed);out=[]
    for _ in range(32):
        ids=np.flatnonzero(active)
        if len(ids)<4:break
        pairs=rng.choice(ids,(128,2),replace=True);v=p[pairs[:,1]]-p[pairs[:,0]];length=np.linalg.norm(v,axis=1);valid=(length>.25)&(abs(cloud[pairs[:,1],2]-cloud[pairs[:,0],2])<.15)
        if not valid.any():break
        pairs=pairs[valid];v=v[valid]/length[valid,None];n=np.column_stack([-v[:,1],v[:,0]]);offset=(n*p[pairs[:,0]]).sum(1);error=abs(p[ids]@n.T-offset);height=abs(cloud[ids,2,None]-cloud[pairs[:,0],2]);counts=((error<.035)&(height<.18)).sum(0);win=int(counts.argmax())
        if counts[win]<4:break
        used=ids[(error[:,win]<.035)&(height[:,win]<.18)];c=p[used].mean(0);_,axes=np.linalg.eigh((p[used]-c).T@(p[used]-c));t=axes[:,-1];normal=np.array([-t[1],t[0]]);d=float(np.median(p[used]@normal));h=float(np.median(cloud[used,2]));used=np.flatnonzero(active&(abs(p@normal-d)<.035)&(abs(cloud[:,2]-h)<.18));active[used]=False;along=p[used]@t;order=np.argsort(along);chunks=np.split(order,np.flatnonzero(np.diff(along[order])>.8)+1)
        for chunk in chunks:
            if len(chunk)<3:continue
            low,high=float(along[chunk].min()),float(along[chunk].max())
            if high-low<.25:continue
            lower,upper=low-extension,high+extension
            if len(free):
                valid=abs(free@normal-d)<.28;ft=free[valid]@t;left=ft[(ft<low)&(ft>lower)];right=ft[(ft>high)&(ft<upper)]
                if len(left):lower=max(lower,float(left.max())+.28)
                if len(right):upper=min(upper,float(right.min())-.28)
            if upper<=lower:continue
            a=t*lower+normal*d;b=t*upper+normal*d;out.append([a[0],a[1],b[0],b[1],h])
    return np.array(out,np.float64).reshape(-1,5)
