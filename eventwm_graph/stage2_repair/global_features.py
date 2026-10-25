"""Shared full-current-map feature grids, with the original candidate universe.

Distances/heights use all supplied known cells. This is a feature change from
legacy per-query clipped distance grids, never hidden geometry or pruning.
Queries outside the bounded reusable canvas take the original local path.
"""
from collections import OrderedDict
import time
import numpy as np
from eventwm_graph.stage2_segment.batch import SharedCompiledGraph
from eventwm_graph.stage2_segment.features import SegmentFeatures
from eventwm_graph.stage2_segment.prepared import content_key
from eventwm_graph.stage2_segment.native import Base
from .api import TimePredictor


class GlobalPrepared(SharedCompiledGraph):
    def __init__(self,graph):
        super().__init__(graph);self.crop_cache_size=8;self.device_cache_size=8;self.global_fields=None
        ego=graph.get('ego_support');center=np.floor(np.asarray(ego)[[0,2]]/self.resolution).astype(np.int64)if ego is not None else np.zeros(2,np.int64)
        low=np.minimum(self.known_lo,center)if self.known_lo is not None else center;high=np.maximum(self.known_hi,center+1)if self.known_hi is not None else center+1
        lo=((low-128)//32)*32;hi=((high+128+31)//32)*32;shape=hi-lo
        if int(np.prod(shape))<=6000000:
            self.global_lo=lo;self.global_shape=shape;self.global_fields=super().crop(lo,shape)

    def contains_domain(self,path):
        if self.global_fields is None:return False
        p=np.asarray(path,np.float64)
        if p.ndim!=2 or p.shape[1]!=5 or not len(p)or not np.isfinite(p).all():return False
        xy=p[:,[1,3]].astype(np.float32);lo=np.floor((xy.min(0)-11)/self.resolution).astype(np.int64)-4;hi=np.floor((xy.max(0)+11)/self.resolution).astype(np.int64)+5
        return np.all(lo>=self.global_lo)and np.all(hi<=self.global_lo+self.global_shape)

    def prewarm(self,paths):
        if all(self.contains_domain(p)for p in paths):return
        return super().prewarm(paths)


class GlobalFeatures(SegmentFeatures):
    def __init__(self,graph,trajectory,prepared):
        if not prepared.contains_domain(trajectory):
            super().__init__(graph,trajectory,prepared);return
        self.graph=graph;self.prepared=prepared;p=np.asarray(trajectory,np.float64)
        if np.any(np.diff(p[:,0])<=0)or abs(p[0,0]-float(graph['timestamp']))>1e-7:raise ValueError('invalid trajectory clock')
        self.path=p;self.resolution=prepared.resolution;self.xy=p[:,[1,3]].astype(np.float32);self.pose_cos=np.cos(p[:,4]);self.pose_sin=np.sin(p[:,4]);self.duration=float(p[-1,0]-p[0,0]);self.pose_timefrac=(p[:,0]-p[0,0])/max(self.duration,1e-4)
        dlo=np.floor((self.xy.min(0)-11)/self.resolution).astype(np.int32)-4;dhi=np.floor((self.xy.max(0)+11)/self.resolution).astype(np.int32)+5;self.candidate_lo=dlo;self.candidate_shape=dhi-dlo
        self.lo=prepared.global_lo;self.shape=prepared.global_shape;self.states,self.base_height,self.dist=prepared.global_fields
        a,b=map(int,dlo-self.lo);h,w=map(int,self.candidate_shape);self.candidate_states=self.states[a:a+h,b:b+w]
        self._candidate_keys=None;self._candidate_neighbors=None;self.tree=None;self._view=None

    def candidates(self,mode='both',view=True):
        if not hasattr(self,'candidate_lo'):return super().candidates(mode,view)
        from eventwm_graph.stage2_quality.features import candidate_mask
        keys=np.argwhere(self.candidate_states==0).astype(np.int32)+self.candidate_lo;self._candidate_keys=keys[candidate_mask(keys,np.unique(self.xy,axis=0),self.resolution)];return self._candidate_keys


class GlobalTimePredictor(TimePredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.feature_class=GlobalFeatures

    def prepare_graph(self,graph):
        key=content_key(graph)
        if key in self.cache:self.cache_hits+=1;self.cache.move_to_end(key);return self.cache[key]
        self.cache_misses+=1;p=GlobalPrepared(graph);p.future_base=None
        if p.native_eligible:
            try:p.future_base=Base(graph,p)
            except(TypeError,ValueError,OverflowError):pass
        if self.cache_size:
            self.cache[key]=p
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return p
