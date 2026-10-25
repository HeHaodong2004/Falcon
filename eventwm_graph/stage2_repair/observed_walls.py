"""Estimate optical wall segments from LEGALLY OBSERVED inflated BLOCKED.

This is a geometric prediction prior, not certified occupancy. No scene ID,
navmesh, hidden mesh, future observation or future camera pose is accepted.
"""
import math
import numpy as np
from scipy.ndimage import distance_transform_edt,maximum_filter
from numba import njit


def reconstruct(keys,states,heights,resolution,extension=2.,seed=8127):
    keys=np.asarray(keys,np.int32);states=np.asarray(states);heights=np.asarray(heights);blocked=keys[states==2];floor=heights[states==2]
    if len(blocked)<6:return np.empty((0,5),np.float64)
    lo=blocked.min(0)-2;shape=blocked.max(0)-lo+3
    if np.prod(shape)>4000000:raise ValueError('observed wall raster too large')
    image=np.zeros(tuple(shape),bool);ij=blocked-lo;image[ij[:,0],ij[:,1]]=True;d=distance_transform_edt(image);ridge=(d>=1.5)&(d>=maximum_filter(d,size=3)-1.e-8);p=(np.argwhere(ridge)+lo+.5)*resolution
    if len(p)<4:return np.empty((0,5),np.float64)
    allb=(blocked+.5)*resolution;free=(keys[states==1]+.5)*resolution;active=np.ones(len(p),bool);rng=np.random.default_rng(seed);segments=[]
    for _ in range(24):
        ids=np.flatnonzero(active)
        if len(ids)<4:break
        pairs=rng.choice(ids,(128,2),replace=True);vectors=p[pairs[:,1]]-p[pairs[:,0]];length=np.linalg.norm(vectors,axis=1);valid=length>.3
        if not valid.any():break
        pairs=pairs[valid];vectors=vectors[valid]/length[valid,None];normal=np.column_stack([-vectors[:,1],vectors[:,0]]);offset=(normal*p[pairs[:,0]]).sum(1);distance=abs(p[ids]@normal.T-offset);counts=(distance<=.08).sum(0);winner=int(counts.argmax())
        if counts[winner]<4:break
        inliers=ids[distance[:,winner]<=.08];points=p[inliers];center=points.mean(0);values,axes=np.linalg.eigh((points-center).T@(points-center));tangent=axes[:,-1];n=np.array([-tangent[1],tangent[0]]);offset=float(np.median(points@n));inside=active&(abs(p@n-offset)<=.08);inliers=np.flatnonzero(inside);active[inliers]=False;along=p[inliers]@tangent;order=np.argsort(along);chunks=np.split(order,np.flatnonzero(np.diff(along[order])>.8)+1)
        for chunk in chunks:
            if len(chunk)<3:continue
            low,high=float(along[chunk].min()),float(along[chunk].max())
            if high-low<.25:continue
            band=(abs(allb@n-offset)<=.30)&(allb@tangent>=low-.15)&(allb@tangent<=high+.15)
            shift=offset
            if np.count_nonzero(band)>=5:
                q=allb[band]@n;shift=float((np.quantile(q,.02)+np.quantile(q,.98))/2)
            lower,upper=low-extension,high+extension
            if len(free):
                validfree=abs(free@n-shift)<.28;ft=free[validfree]@tangent;left=ft[(ft<low)&(ft>lower)];right=ft[(ft>high)&(ft<upper)]
                if len(left):lower=max(lower,float(left.max())+.28)
                if len(right):upper=min(upper,float(right.min())-.28)
            if upper<=lower:continue
            a=tangent*lower+n*shift;b=tangent*upper+n*shift;h=float(np.median(floor[band]))if np.any(band)else float(np.median(floor));segments.append([a[0],a[1],b[0],b[1],h])
    return np.asarray(segments,np.float64).reshape(-1,5)


@njit(cache=True)
def blocked_forecast(segments,path,origins,rotations,lo,shape,unknown,resolution):
    state=np.zeros((shape[0],shape[1]),np.uint8)
    span=int(math.ceil(.25/resolution))+1
    for frame in range(1,len(path)):
        origin=origins[frame];r=rotations[frame]
        for j in range(105):
            x=-.95+1.9*(j//3)/34.;y=(j%3-1)*.5;vx=r[0,0]*x+r[0,1]*y-r[0,2];vy=r[1,0]*x+r[1,1]*y-r[1,2];vz=r[2,0]*x+r[2,1]*y-r[2,2];norm=math.sqrt(vx*vx+vy*vy+vz*vz);vx/=norm;vy/=norm;vz/=norm;best=10.
            for s in segments:
                sx=s[2]-s[0];sz=s[3]-s[1];det=vx*sz-vz*sx
                if abs(det)<1.e-10:continue
                dx=s[0]-origin[0];dz=s[1]-origin[2];t=(dx*sz-dz*sx)/det;u=(dx*vz-dz*vx)/det
                if t>=0 and t<best and 0<=u<=1:
                    yy=origin[1]+vy*t
                    if s[4]-.05<=yy<=s[4]+2.5:best=t
            if best>=10.:continue
            px=origin[0]+vx*best;py=origin[1]+vy*best;pz=origin[2]+vz*best
            if not path[frame,2]+.2<py<path[frame,2]+1.4:continue
            ix=int(math.floor(px/resolution));iz=int(math.floor(pz/resolution))
            for a in range(ix-span,ix+span+1):
                for b in range(iz-span,iz+span+1):
                    aa=a-lo[0];bb=b-lo[1]
                    if aa<0 or bb<0 or aa>=shape[0]or bb>=shape[1]or not unknown[aa,bb]:continue
                    if ((a+.5)*resolution-px)**2+((b+.5)*resolution-pz)**2<=.0625:state[aa,bb]=2
    return state
