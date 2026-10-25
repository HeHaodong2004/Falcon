"""Legal full-trajectory opportunity proxies and explicit-UNKNOWN map summaries.

No camera states, ray casts, future observations or hidden geometry are inputs.
The baseline crop semantics survive in the common31 map/segment columns.
"""
from collections import OrderedDict
import math
import time
import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from eventwm_graph.stage2.features import SceneFeatures,validate,RADIUS
from .geometry import graph_identity
from .common import FEATURES,MODES


@njit(cache=True)
def opportunity(keys,path,resolution):
    n=len(keys);out=np.zeros((n,20),np.float32);keep=np.zeros(n,np.bool_)
    cosine=np.cos(path[:,4]);sine=np.sin(path[:,4])
    duration=path[-1,0]-path[0,0];rotation=0.;stationary=0.;unwrapped=0.;ylo=0.;yhi=0.
    for j in range(1,len(path)):
        dy=math.atan2(math.sin(path[j,4]-path[j-1,4]),math.cos(path[j,4]-path[j-1,4]))
        rotation+=abs(dy);unwrapped+=dy;ylo=min(ylo,unwrapped);yhi=max(yhi,unwrapped)
        if (path[j,1]-path[j-1,1])**2+(path[j,3]-path[j-1,3])**2<1e-8:
            stationary+=path[j,0]-path[j-1,0]
    for i in range(n):
        # Preserve original float32 cell-center / planar-pose coordinates.
        x=np.float32(np.float32(keys[i,0])+.5)*np.float32(resolution)
        z=np.float32(np.float32(keys[i,1])+.5)*np.float32(resolution)
        mind=1e30;domain_min=1e30
        for j in range(len(path)):
            dx=float(x)-float(np.float32(path[j,1]));dz=float(z)-float(np.float32(path[j,3]))
            mind=min(mind,dx*dx+dz*dz)
            domain_x=np.float64(x)-np.float64(np.float32(path[j,1]));domain_z=np.float64(z)-np.float64(np.float32(path[j,3]))
            domain_min=min(domain_min,domain_x*domain_x+domain_z*domain_z)
        if math.sqrt(domain_min)>11.:continue
        keep[i]=True;out[i,0]=math.sqrt(mind)/10.
        best=-1.;weighted=0.;min0=11.;min5=11.;min8=11.;first=1.;last=0.;s=0.;maxall=-1.
        c0=0.;c5=0.;c8=0.;t0=0.;t5=0.;t8=0.
        for j in range(1,len(path)):
            dx=float(x)-path[j,1];dz=float(z)-path[j,3];dist=math.sqrt(dx*dx+dz*dz)
            facing=max(-1.,min(1.,(dx*cosine[j]-dz*sine[j])/max(dist,1e-4)))
            maxall=max(maxall,facing);w=max(0.,1.-dist/11.);s+=max(0.,facing)*w
            weighted=max(weighted,max(0.,facing)*w)
            if dist<=10.:
                best=max(best,facing);dt=path[j,0]-path[j-1,0]
                if facing>=0.:c0+=1.;t0+=dt;min0=min(min0,dist)
                if facing>=.5:
                    c5+=1.;t5+=dt;min5=min(min5,dist)
                    when=(path[j,0]-path[0,0])/max(duration,1e-4);first=min(first,when);last=max(last,when)
                if facing>=.866025403784:c8+=1.;t8+=dt;min8=min(min8,dist)
        count=max(1,len(path)-1);den=max(duration,1e-4)
        out[i,1]=best;out[i,2]=weighted;out[i,3]=min0/10.;out[i,4]=min5/10.;out[i,5]=min8/10.
        out[i,6]=c0/count;out[i,7]=c5/count;out[i,8]=c8/count
        out[i,9]=t0/den;out[i,10]=t5/den;out[i,11]=t8/den
        out[i,12]=first;out[i,13]=last;out[i,14]=maxall;out[i,15]=s/count
        out[i,16]=rotation/(2*math.pi);out[i,17]=(yhi-ylo)/(2*math.pi)
        out[i,18]=stationary/den;out[i,19]=(len(path)-1)/128.
    return keep,out


@njit(cache=True)
def inside_descriptors(keys,path,resolution):
    n=len(keys);out=np.zeros((n,20),np.float32);keep=np.zeros(n,np.bool_)
    cosine=np.cos(path[:,4]);sine=np.sin(path[:,4])
    duration=path[-1,0]-path[0,0];rotation=0.;stationary=0.;unwrapped=0.;ylo=0.;yhi=0.
    for j in range(1,len(path)):
        dy=math.atan2(math.sin(path[j,4]-path[j-1,4]),math.cos(path[j,4]-path[j-1,4]))
        rotation+=abs(dy);unwrapped+=dy;ylo=min(ylo,unwrapped);yhi=max(yhi,unwrapped)
        if (path[j,1]-path[j-1,1])**2+(path[j,3]-path[j-1,3])**2<1e-8:
            stationary+=path[j,0]-path[j-1,0]
    for i in range(n):
        # Preserve original float32 cell-center / planar-pose coordinates.
        x=np.float32(np.float32(keys[i,0])+.5)*np.float32(resolution)
        z=np.float32(np.float32(keys[i,1])+.5)*np.float32(resolution)
        mind=1e30
        for j in range(len(path)):
            if j>0 and np.float32(path[j,1])==np.float32(path[j-1,1]) and np.float32(path[j,3])==np.float32(path[j-1,3]):continue
            dx=float(x)-float(np.float32(path[j,1]));dz=float(z)-float(np.float32(path[j,3]))
            mind=min(mind,dx*dx+dz*dz)
        keep[i]=True;out[i,0]=math.sqrt(mind)/10.
        best=-1.;weighted=0.;min0=11.;min5=11.;min8=11.;first=1.;last=0.;s=0.;maxall=-1.
        c0=0.;c5=0.;c8=0.;t0=0.;t5=0.;t8=0.
        rel_x=0.;rel_z=0.;dist=0.
        for j in range(1,len(path)):
            if j==1 or path[j,1]!=path[j-1,1] or path[j,3]!=path[j-1,3]:
                rel_x=float(x)-path[j,1];rel_z=float(z)-path[j,3];dist=math.sqrt(rel_x*rel_x+rel_z*rel_z)
            facing=max(-1.,min(1.,(rel_x*cosine[j]-rel_z*sine[j])/max(dist,1e-4)))
            maxall=max(maxall,facing);w=max(0.,1.-dist/11.);s+=max(0.,facing)*w
            weighted=max(weighted,max(0.,facing)*w)
            if dist<=10.:
                best=max(best,facing);dt=path[j,0]-path[j-1,0]
                if facing>=0.:c0+=1.;t0+=dt;min0=min(min0,dist)
                if facing>=.5:
                    c5+=1.;t5+=dt;min5=min(min5,dist)
                    when=(path[j,0]-path[0,0])/max(duration,1e-4);first=min(first,when);last=max(last,when)
                if facing>=.866025403784:c8+=1.;t8+=dt;min8=min(min8,dist)
        count=max(1,len(path)-1);den=max(duration,1e-4)
        out[i,1]=best;out[i,2]=weighted;out[i,3]=min0/10.;out[i,4]=min5/10.;out[i,5]=min8/10.
        out[i,6]=c0/count;out[i,7]=c5/count;out[i,8]=c8/count
        out[i,9]=t0/den;out[i,10]=t5/den;out[i,11]=t8/den
        out[i,12]=first;out[i,13]=last;out[i,14]=maxall;out[i,15]=s/count
        out[i,16]=rotation/(2*math.pi);out[i,17]=(yhi-ylo)/(2*math.pi)
        out[i,18]=stationary/den;out[i,19]=(len(path)-1)/128.
    return keep,out


@njit(cache=True)
def candidate_mask(keys,xy,resolution):
    keep=np.zeros(len(keys),np.bool_)
    for i in range(len(keys)):
        x=np.float32(np.float32(keys[i,0])+.5)*np.float32(resolution)
        z=np.float32(np.float32(keys[i,1])+.5)*np.float32(resolution)
        for j in range(len(xy)):
            dx=np.float64(x)-np.float64(xy[j,0]);dz=np.float64(z)-np.float64(xy[j,1])
            if math.sqrt(dx*dx+dz*dz)<=11.:keep[i]=True;break
    return keep


@njit(cache=True)
def boxes(keys,lo,free,blocked):
    out=np.zeros((len(keys),25),np.float32);nx=free.shape[0]-1;nz=free.shape[1]-1
    for i in range(len(keys)):
        x=keys[i,0]-lo[0];z=keys[i,1]-lo[1];col=0
        for radius in (1,4,12):
            a=max(0,min(nx,x-radius));b=max(0,min(nx,x+radius+1))
            c=max(0,min(nz,z-radius));d=max(0,min(nz,z+radius+1));area=float((2*radius+1)**2)
            f=(free[b,d]-free[a,d]-free[b,c]+free[a,c])/area
            h=(blocked[b,d]-blocked[a,d]-blocked[b,c]+blocked[a,c])/area
            out[i,col]=f;out[i,col+1]=h;out[i,col+2]=max(0.,1.-f-h);col+=3
        for radius in (4,12):
            for sx,sz in ((-1,-1),(-1,1),(1,-1),(1,1)):
                xa=x-radius if sx<0 else x;xb=x+1 if sx<0 else x+radius+1
                za=z-radius if sz<0 else z;zb=z+1 if sz<0 else z+radius+1
                a=max(0,min(nx,xa));b=max(0,min(nx,xb));c=max(0,min(nz,za));d=max(0,min(nz,zb))
                area=float((radius+1)**2)
                out[i,col]=(free[b,d]-free[a,d]-free[b,c]+free[a,c])/area
                out[i,col+1]=(blocked[b,d]-blocked[a,d]-blocked[b,c]+blocked[a,c])/area;col+=2
    return out


class PreparedGraph:
    """Owned feature arrays only; never cache mutable graph output metadata."""
    def __init__(self,graph,graph_key=None):
        validate(graph,[[graph['timestamp'],0.,0.,0.,0.]])
        self.identity=graph_identity(graph) if graph_key is None else graph_key;self.resolution=float(graph['resolution']);self.height=float(graph['height'])
        self.keys=np.asarray([c['cell'] for c in graph['cells']],np.int32).reshape(-1,2)
        self.values=np.asarray([c['state'] for c in graph['cells']],np.uint8)
        self.heights=np.asarray([c['height'] for c in graph['cells']],np.float32)
        self.crops=OrderedDict();self.crop_hits=0;self.crop_misses=0
        self.integrals=None

    def crop(self,lo,shape):
        key=tuple(lo)+tuple(shape)
        if key in self.crops:
            self.crop_hits+=1;self.crops.move_to_end(key);return self.crops[key]
        self.crop_misses+=1
        states=np.zeros(tuple(shape),np.uint8);heights=np.full(tuple(shape),self.height,np.float32)
        indices=self.keys-lo;ok=np.all((indices>=0)&(indices<shape),axis=1);ij=indices[ok]
        states[ij[:,0],ij[:,1]]=self.values[ok];heights[ij[:,0],ij[:,1]]=self.heights[ok]
        distance=[distance_transform_edt(states!=state).astype(np.float32)*self.resolution
            if np.any(states==state) else np.full(tuple(shape),RADIUS,np.float32) for state in (1,2)]
        if np.any(states):
            inds=distance_transform_edt(states==0,return_distances=False,return_indices=True);baseline=heights[inds[0],inds[1]]
        else:baseline=heights
        value=(states,baseline,distance)
        self.crops[key]=value
        while len(self.crops)>4:self.crops.popitem(last=False)
        return value

    def map_features(self,keys):
        if self.integrals is None:
            lo=self.keys.min(0) if len(self.keys) else np.zeros(2,np.int32)
            hi=self.keys.max(0)+1 if len(self.keys) else np.ones(2,np.int32)
            shape=hi-lo
            if int(np.prod(shape))>4000000:raise ValueError('legal-map integral domain exceeds four million cells')
            ij=self.keys-lo;integrals=[]
            for state in (1,2):
                image=np.zeros(tuple(shape),np.int32);at=ij[self.values==state];image[at[:,0],at[:,1]]=1
                integrals.append(np.pad(image.cumsum(0).cumsum(1),((1,0),(1,0))))
            self.integrals=(lo,*integrals)
        return boxes(keys,*self.integrals)


class Features(SceneFeatures):
    def __init__(self,graph,trajectory,prepared=None):
        self.graph=graph;self.prepared=prepared or PreparedGraph(graph)
        p=np.asarray(trajectory,np.float64)
        if p.ndim!=2 or p.shape[1]!=5 or len(p)==0 or not np.isfinite(p).all():raise ValueError('trajectory must be finite [time,x,y,z,yaw] rows')
        if np.any(np.diff(p[:,0])<=0) or abs(p[0,0]-float(graph['timestamp']))>1e-7:raise ValueError('invalid trajectory clock')
        self.path=p;self.resolution=self.prepared.resolution;self.xy=p[:,[1,3]].astype(np.float32)
        self.pose_cos=np.cos(p[:,4]);self.pose_sin=np.sin(p[:,4]);self.duration=float(p[-1,0]-p[0,0])
        self.pose_timefrac=(p[:,0]-p[0,0])/max(self.duration,1e-4)
        self.lo=np.floor((self.xy.min(0)-RADIUS)/self.resolution).astype(np.int32)-4
        hi=np.floor((self.xy.max(0)+RADIUS)/self.resolution).astype(np.int32)+5;self.shape=hi-self.lo
        if int(np.prod(self.shape))>4000000:raise ValueError('trajectory domain exceeds four million cells')
        self.states,self.base_height,self.dist=self.prepared.crop(self.lo,self.shape)
        self._candidate_keys=None;self._candidate_neighbors=None;self.tree=None;self._view=None

    def candidates(self,mode='both',view=True):
        ij=np.argwhere(self.states==0).astype(np.int32);keys=ij+self.lo
        if mode in ('baseline','map'):
            self.tree=cKDTree(self.xy);xy=(keys.astype(np.float32)+.5)*self.resolution;count=min(4,len(self.path))
            distance,index=self.tree.query(xy,k=count,workers=1)
            distance=np.asarray(distance).reshape(len(keys),count);index=np.asarray(index).reshape(len(keys),count)
            keep=distance[:,0]<=RADIUS;self._candidate_neighbors=(distance[keep].astype(np.float32),index[keep])
        else:
            # Deduplicate POSITIONS for geometric support only. Every heading,
            # time and repeated viewing opportunity remains in descriptors.
            domain_xy=np.unique(self.xy,axis=0)
            keep=candidate_mask(keys,domain_xy,self.resolution)
        self._candidate_keys=keys[keep];return self._candidate_keys

    def common(self,keys):
        keys=np.asarray(keys,np.int32).reshape(-1,2);ij=keys-self.lo;n=len(keys)
        if np.any(ij<0) or np.any(ij>=self.shape):raise ValueError('feature key outside crop')
        out=np.zeros((n,FEATURES),np.float32)
        from eventwm_graph.stage2.features import OFFSETS
        if np.all(ij>=3) and np.all(ij<self.shape-3):
            center=ij[:,0]*int(self.shape[1])+ij[:,1];offsets=OFFSETS[:,0]*int(self.shape[1])+OFFSETS[:,1]
            states=self.states.ravel()[center[:,None]+offsets[None,:]]
        else:
            neighbor=np.clip(ij[:,None,:]+OFFSETS[None,:,:],0,self.shape-1)
            states=self.states[neighbor[:,:,0],neighbor[:,:,1]]
        out[:,16:25]=states==1;out[:,25:34]=states==2
        baseline=self.base_height[ij[:,0],ij[:,1]]
        out[:,34]=self.dist[0][ij[:,0],ij[:,1]]/10;out[:,35]=self.dist[1][ij[:,0],ij[:,1]]/10
        xy=(keys.astype(np.float32)+.5)*self.resolution
        out[:,36:38]=(xy-self.xy[0])/10;out[:,38:40]=(xy-self.xy[-1])/10
        out[:,40]=baseline-self.path[0,2]
        length=float(np.linalg.norm(np.diff(self.xy,axis=0),axis=1).sum())
        out[:,41:47]=np.array([self.duration/5,length/5,self.resolution,self.path[-1,2]-self.path[0,2],
            np.cos(self.path[-1,4]-self.path[0,4]),np.sin(self.path[-1,4]-self.path[0,4])],np.float32)
        return out,baseline

    def transform(self,keys,mode='both'):
        begun=time.perf_counter()
        if mode not in MODES:raise ValueError('unrecognized feature mode')
        if mode in ('baseline','map'):
            if self.tree is None:self.tree=cKDTree(self.xy)
            legacy,baseline=super().transform(keys);out=np.zeros((len(keys),FEATURES),np.float32);out[:,:47]=legacy
        else:
            out,baseline=self.common(keys)
            common_done=time.perf_counter()
            if keys is self._candidate_keys:_,view=inside_descriptors(np.asarray(keys,np.int32),self.path,self.resolution)
            else:_,view=opportunity(np.asarray(keys,np.int32),self.path,self.resolution)
            out[:,47:67]=view
        trajectory_done=time.perf_counter()
        if mode in ('map','both'):out[:,67:92]=self.prepared.map_features(keys)
        finished=time.perf_counter()
        self.last_transform_profile=dict(trajectory_seconds=trajectory_done-common_done if mode in ('trajectory','both') else None,
            map_seconds=(common_done-begun)+(finished-trajectory_done) if mode in ('trajectory','both') else None,
            legacy_combined_seconds=finished-begun if mode in ('baseline','map') else None)
        if not np.isfinite(out).all():raise ValueError('nonfinite legal features')
        return out,baseline
