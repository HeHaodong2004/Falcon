"""Predict bounded continuations of observed wall planes.

Unseen geometry is explicitly a hypothesis. Extensions contradicted by legal
past range free space or known FREE certificates are rejected. No real scene
geometry, navigation data, future camera or future observation is accepted.
"""
import numpy as np
from scipy.spatial import cKDTree


def complete(triangles,ranges,origins,rotations,free_keys,resolution,extension=2.):
    t=np.asarray(triangles,np.float64);cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);area=np.linalg.norm(cross,axis=1)/2;normal=cross/np.maximum(2*area[:,None],1.e-12);centroid=t.mean(1);eligible=(abs(normal[:,1])<.15)&(area>1.e-7);ids=np.flatnonzero(eligible)
    if len(ids)<30:return t,np.empty((0,8))
    rng=np.random.default_rng(9187);pool=ids if len(ids)<=6000 else rng.choice(ids,6000,replace=False,p=area[ids]/area[ids].sum());p=centroid[pool];n=normal[pool];weight=area[pool];active=np.ones(len(pool),bool);planes=[];extra=[];free_tree=cKDTree((free_keys+.5)*resolution)if len(free_keys)else None;h,w=ranges.shape[1:]
    def contradicted(points):
        bad=np.zeros(len(points),bool);seen=np.zeros(len(points),bool)
        for image,origin,rotation in zip(ranges,origins,rotations):
            local=(points-origin)@rotation;forward=-local[:,2];valid=(forward>1.e-6)&(abs(local[:,0])<=forward)&(abs(local[:,1])<=forward);indices=np.flatnonzero(valid)
            if not len(indices):continue
            q=local[indices];u=np.clip(((q[:,0]/-q[:,2]+1)*w/2).astype(int),0,w-1);v=np.clip(((1-q[:,1]/-q[:,2])*h/2).astype(int),0,h-1);depth=np.nan_to_num(image[v,u],nan=10.);distance=np.linalg.norm(q,axis=1);bad[indices]|=distance+.12<depth;seen[indices]|=abs(distance-depth)<.10
        return bad,seen
    for _ in range(16):
        active_ids=np.flatnonzero(active)
        if len(active_ids)<15:break
        candidate=rng.choice(active_ids,min(128,len(active_ids)),replace=False);nn=n[candidate];dd=(nn*p[candidate]).sum(1);distance=abs(p@nn.T-dd);orientation=abs(n@nn.T);inlier=(distance<.025)&(orientation>.97)&active[:,None];score=(inlier*weight[:,None]).sum(0);best=int(score.argmax());select=inlier[:,best]
        if np.count_nonzero(select)<15:break
        q=p[select];active[select]=False;nxz=nn[best,[0,2]];nxz/=np.linalg.norm(nxz);d=float(np.median(q[:,[0,2]]@nxz));tangent=np.array([-nxz[1],nxz[0]]);along=q[:,[0,2]]@tangent;low,high=np.quantile(along,[.005,.995]);ylow,yhigh=np.quantile(q[:,1],[.005,.995])
        if yhigh-ylow<.7 or high-low<.4:continue
        planes.append([nxz[0],nxz[1],d,low,high,ylow,yhigh,float(score[best])]);segments=[(low-extension,low),(high,high+extension)]
        for begin,end in segments:
            step=.15;edges=np.linspace(begin,end,max(2,int(np.ceil((end-begin)/step))+1));mids=(edges[:-1]+edges[1:])/2;xy=mids[:,None]*tangent+nxz*d
            free_ok=np.ones(len(mids),bool)
            if free_tree is not None:free_ok=free_tree.query(xy,k=1)[0]>.30
            heights=np.array([ylow+.05,(ylow+yhigh)/2,yhigh-.05]);samples=np.column_stack([np.repeat(xy[:,0],3),np.tile(heights,len(xy)),np.repeat(xy[:,1],3)]);bad,seen=contradicted(samples);keep=free_ok&~bad.reshape(-1,3).any(1)&~seen.reshape(-1,3).all(1)
            for j in np.flatnonzero(keep):
                a=edges[j]*tangent+nxz*d;b=edges[j+1]*tangent+nxz*d;v=np.array([[a[0],ylow,a[1]],[b[0],ylow,b[1]],[a[0],yhigh,a[1]],[b[0],yhigh,b[1]]]);extra.extend([v[[0,1,2]],v[[1,3,2]]])
    if extra:t=np.concatenate([t,np.asarray(extra)])
    return t,np.asarray(planes).reshape(-1,8)
