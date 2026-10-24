"""Raster of supplied certified configuration-space support; UNKNOWN is blocked.

Native construction reads only static PathFinder geometry. No graph edges are
added. The disk-cover footprint follows partial.query's audited gait envelope.
"""
import math
import numpy as np
from numba import njit
from eventwm_graph.config import digest
from eventwm_graph.partial.query import ROBOT_BOUNDS


@njit(cache=True)
def body_free(pose, grid, heights, origin, resolution, extension=0.):
    c=math.cos(pose[3]); s=math.sin(pose[3])
    # .08 m spacing in .25 m support disks covers .2467 m lateral half-width.
    # The audited max half-width is .242846 m, leaving sweep sampling allowance.
    lo=-.646495; hi=.584202+extension
    n=int(math.ceil((hi-lo)/.08))+1
    for i in range(n):
        x=lo+(hi-lo)*i/(n-1)
        ix=int(math.floor((pose[0]+x*c-origin[0])/resolution))
        iz=int(math.floor((pose[2]-x*s-origin[1])/resolution))
        if ix<0 or iz<0 or ix>=grid.shape[0] or iz>=grid.shape[1] or not grid[ix,iz]: return False
        if abs(pose[1]-heights[ix,iz])>.12: return False
    return True


@njit(cache=True)
def swept_free(a,b,grid,heights,origin,resolution):
    # Exact sufficient shortcut: every support square intersecting an AABB
    # containing ALL longitudinal disk centers is certified. This covers the
    # complete continuous translation/rotation, without skipping a collision
    # check or rounding a pose-cache key. Near obstacles use the refined cover.
    xmin=int(math.floor((min(a[0],b[0])-.646496-origin[0])/resolution))
    xmax=int(math.floor((max(a[0],b[0])+.646496-origin[0])/resolution))
    zmin=int(math.floor((min(a[2],b[2])-.646496-origin[1])/resolution))
    zmax=int(math.floor((max(a[2],b[2])+.646496-origin[1])/resolution))
    rectangle=(xmin>=0 and zmin>=0 and xmax<grid.shape[0] and zmax<grid.shape[1])
    if rectangle:
        for ix in range(xmin,xmax+1):
            for iz in range(zmin,zmax+1):
                if not grid[ix,iz] or abs(a[1]-heights[ix,iz])>.12 or abs(b[1]-heights[ix,iz])>.12:
                    rectangle=False;break
            if not rectangle:break
        if rectangle:return True
    angle=math.atan2(math.sin(b[3]-a[3]), math.cos(b[3]-a[3]))
    distance=math.sqrt((b[0]-a[0])**2+(b[2]-a[2])**2)
    if abs(angle)<1e-8 and abs(b[1]-a[1])<1e-8:
        return body_free(a,grid,heights,origin,resolution,distance)
    n=max(1,int(math.ceil(abs(angle)/.005)),int(math.ceil(distance/.02)))
    for i in range(n+1):
        p=a+(b-a)*(i/n);p[3]=a[3]+angle*i/n
        if not body_free(p,grid,heights,origin,resolution): return False
    return True


class StaticGrid:
    def __init__(self, free, heights, origin, resolution, version, navigation=None):
        self.free=np.ascontiguousarray(free,dtype=np.bool_)
        self.heights=np.ascontiguousarray(heights,dtype=np.float64)
        self.origin=np.asarray(origin,dtype=float);self.resolution=float(resolution)
        self.version=str(version);self.navigation=navigation
        self.precise=None;self.predictor=None
        if self.free.shape!=self.heights.shape: raise ValueError('Height/support mismatch')
        self.free.flags.writeable=False; self.heights.flags.writeable=False

    @classmethod
    def from_partial(cls,graph):
        keys=list(graph.cells)
        if not keys: raise ValueError('Empty legal graph')
        lo=np.min(keys,axis=0)-1;hi=np.max(keys,axis=0)+2
        free=np.zeros(tuple(hi-lo),dtype=bool); heights=np.full(free.shape,graph.height)
        for (x,z),v in graph.cells.items():
            free[x-lo[0],z-lo[1]]=v['state']==1
            heights[x-lo[0],z-lo[1]]=v['height']
        return cls(free,heights,lo*graph.resolution,graph.resolution,
                   digest(graph.snapshot()))

    @classmethod
    def from_navigation(cls,navigation,route,margin=3.,resolution=.10):
        """Explicit FULL STATIC MAP profile, never a partial-observation claim.

        A whole grid square must lie inside the .25 m inflated certified mesh.
        No interpolation or ego singleton adds support. Local floor only.
        """
        p=np.asarray(route);lo=np.floor((p[:,[0,2]].min(0)-margin)/resolution).astype(int)
        hi=np.ceil((p[:,[0,2]].max(0)+margin)/resolution).astype(int)
        shape=tuple(map(int,hi-lo+1))
        if np.prod(shape)>1000000: raise ValueError('Static patch exceeds memory budget')
        free=np.zeros(shape,dtype=bool);heights=np.zeros(shape);pf=navigation.pf
        height=float(p[0,1]); radius=resolution/math.sqrt(2)+.008
        for ix in range(shape[0]):
            for iz in range(shape[1]):
                q=np.array([(lo[0]+ix+.5)*resolution,height,(lo[1]+iz+.5)*resolution])
                # Floor selection follows the immutable route. Overlapping
                # floors remain a limitation; exact native fallback checks every
                # executed transition and never infers new graph connectivity.
                if np.ptp(p[:,1])>.18:
                    nearest=int(np.argmin(np.linalg.norm(p[:,[0,2]]-q[[0,2]],axis=1)))
                    q[1]=p[nearest,1]
                s=np.asarray(pf.snap_point(q),dtype=float)
                if not np.isfinite(s).all() or abs(s[1]-q[1])>.22 or np.linalg.norm((s-q)[[0,2]])>.002: continue
                if float(pf.distance_to_closest_obstacle(s,radius+.01))<radius: continue
                free[ix,iz]=True;heights[ix,iz]=s[1]
        return cls(free,heights,lo*resolution,resolution,
                   digest(dict(navmesh=navigation.identity,origin=lo.tolist(),shape=shape,resolution=resolution,height=height)),navigation)

    def kernel_args(self): return self.free,self.heights,self.origin,self.resolution

    def motion_free(self,a,b):
        supported=bool(swept_free(np.asarray(a,dtype=float),np.asarray(b,dtype=float),*self.kernel_args()))
        if supported:return True
        return self.precise is not None and self.precise.swept(a,b,margin=.025)

    def predict(self,pose,action,spec):
        if self.predictor is not None:return self.predictor.step(pose,action)
        p=spec.step(pose,action);p[1]=self.height_at(p[:3]);return p

    def height_at(self,p):
        k=np.floor((np.asarray(p)[[0,2]]-self.origin)/self.resolution).astype(int)
        if np.any(k<0) or np.any(k>=self.free.shape) or not self.free[tuple(k)]: return float(p[1])
        return float(self.heights[tuple(k)])

    def segment_free(self,a,b):
        if self.navigation is not None: return self.navigation.segment_free(a,b)
        from eventwm_graph.partial.map import PartialGraph
        g=PartialGraph(self.resolution,float(a[1]))
        for i in range(max(1,int(np.linalg.norm(np.asarray(b)-a)/(.25*self.resolution)))+1):
            t=i/max(1,int(np.linalg.norm(np.asarray(b)-a)/(.25*self.resolution)))
            p=np.asarray(a)+(np.asarray(b)-a)*t
            k=np.floor((p[[0,2]]-self.origin)/self.resolution).astype(int)
            if np.any(k<0) or np.any(k>=self.free.shape) or not self.free[tuple(k)]: return False
        return True
