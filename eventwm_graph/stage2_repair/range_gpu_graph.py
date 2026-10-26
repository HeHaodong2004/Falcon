"""Exact shared graph transforms retained on device for feature construction."""
import numpy as np
import torch
from eventwm_graph.stage2_segment.batch import SharedCompiledGraph
from eventwm_graph.stage2_segment.native import Base
from eventwm_graph.stage2_segment.prepared import content_key
from eventwm_graph.stage2_segment.prepared import CompiledGraph
from .global_features import GlobalPrepared
from .gpu_edt import GridEDT
from .range_exact_api import ExactFastSpatialPredictor


class DeviceGlobalPrepared(GlobalPrepared):
    def __init__(self,graph,edt):
        SharedCompiledGraph.__init__(self,graph);self.edt=edt;self.crop_cache_size=8;self.device_cache_size=8;self.global_fields=None
        ego=graph.get('ego_support');center=np.floor(np.asarray(ego)[[0,2]]/self.resolution).astype(np.int64)if ego is not None else np.zeros(2,np.int64)
        low=np.minimum(self.known_lo,center)if self.known_lo is not None else center;high=np.maximum(self.known_hi,center+1)if self.known_hi is not None else center+1
        lo=((low-128)//32)*32;hi=((high+159)//32)*32;shape=hi-lo
        if int(np.prod(shape))<=6000000:self.global_lo=lo;self.global_shape=shape;self.global_fields=self.crop(lo,shape)

    def crop(self,lo,shape):
        if max(shape)>8192 or int(np.prod(shape))>2000000:return CompiledGraph.crop(self,lo,shape)
        key=tuple(lo)+tuple(shape)
        if key in self.crops:self.crop_hits+=1;self.crops.move_to_end(key);return self.crops[key]
        self.crop_misses+=1;s=np.zeros(tuple(shape),np.uint8);height=np.full(tuple(shape),self.height,np.float32);ij=self.keys-lo;valid=np.all((ij>=0)&(ij<shape),1);at=ij[valid];known=self.heights[valid]
        s[at[:,0],at[:,1]]=self.values[valid];height[at[:,0],at[:,1]]=known
        uniform=bool(len(known)and np.all(known.view(np.uint32)==known[:1].view(np.uint32)))
        distance,baseline,_=self.edt.transform(s,height,self.resolution,self.height,nearest_height=not uniform)
        if uniform:baseline=torch.full(tuple(shape),float(known[0]),device=self.edt.device)
        value=(s,baseline,[distance[0],distance[1]]);self.crops[key]=value
        while len(self.crops)>self.crop_cache_size:self.crops.popitem(last=False)
        return value

    def prewarm(self,paths):
        # The global device grid already shares normal queries. Outlier grids
        # retain individual complete crops; never mix a CPU shared-slice path
        # with device arrays or restrict an outlying trajectory's universe.
        return


class DeviceGraphSpatialPredictor(ExactFastSpatialPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.edt=GridEDT(self.device)

    def predict_segments(self,*,partial_graph,trajectories,range_observations,track_history,materialize=True):
        # Inputs are read-only throughout one call. The range and point layers
        # both request the same prepared map; hash the complete content once.
        # This transient reuse ends at return, so later input edits are checked.
        self._inflight_prepared={}
        try:return super().predict_segments(partial_graph=partial_graph,trajectories=trajectories,range_observations=range_observations,track_history=track_history,materialize=materialize)
        finally:self._inflight_prepared=None

    def prepare_graph(self,graph):
        inflight=getattr(self,'_inflight_prepared',None)
        if inflight is not None and id(graph)in inflight:return inflight[id(graph)][1]
        key=content_key(graph)
        if key in self.cache:
            self.cache_hits+=1;self.cache.move_to_end(key);p=self.cache[key]
            if inflight is not None:inflight[id(graph)]=(graph,p)
            return p
        self.cache_misses+=1;p=DeviceGlobalPrepared(graph,self.edt);p.future_base=None
        if p.native_eligible:
            try:p.future_base=Base(graph,p)
            except(TypeError,ValueError,OverflowError):pass
        if self.cache_size:
            self.cache[key]=p
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        if inflight is not None:inflight[id(graph)]=(graph,p)
        return p
