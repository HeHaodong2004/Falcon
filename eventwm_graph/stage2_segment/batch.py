"""Certified exact sharing of static distance transforms across segments.

Each reusable child contains the complete known-cell set. Height tie handling
remains local to the child. Every requested prediction and future map is built
inside complete_batch; only static graph preprocessing is shared.
"""
from collections import defaultdict
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from eventwm_graph.stage2.features import RADIUS
from .prepared import CompiledGraph


class SharedCompiledGraph(CompiledGraph):
    def __init__(self,graph):
        super().__init__(graph);self.shared=None;self.shared_builds=0;self.shared_slices=0;self.shared_height_builds=0
        self.known_lo=self.keys.astype(np.int64).min(0) if len(self.keys) else None
        self.known_hi=self.keys.astype(np.int64).max(0)+1 if len(self.keys) else None
        self.uniform=bool(len(self.heights) and np.all(self.heights.view(np.uint32)==self.heights[:1].view(np.uint32)))

    def contains_known(self,lo,shape):
        return self.known_lo is not None and np.all(self.known_lo>=lo) and np.all(self.known_hi<=lo+shape)

    def prewarm(self,trajectories):
        bounds={}
        for trajectory in trajectories:
            path=np.asarray(trajectory,np.float64)
            if path.ndim!=2 or path.shape[1]!=5 or not len(path) or not np.isfinite(path).all():raise ValueError('invalid supplied trajectory')
            xy=path[:,[1,3]].astype(np.float32)
            lo=np.floor((xy.min(0)-RADIUS)/self.resolution).astype(np.int32)-4
            hi=np.floor((xy.max(0)+RADIUS)/self.resolution).astype(np.int32)+5
            lo=lo.astype(np.int64);shape=hi.astype(np.int64)-lo
            if np.all(shape>0) and self.contains_known(lo,shape):bounds[tuple(lo)+tuple(shape)]=(lo,shape)
        if len(bounds)<2 or all(key in self.crops for key in bounds):return
        lo=np.min([b[0] for b in bounds.values()],axis=0);hi=np.max([b[0]+b[1] for b in bounds.values()],axis=0);shape=hi-lo
        if int(shape[0])*int(shape[1])>4000000:return
        if int(shape[0])*int(shape[1])>=sum(int(b[1][0])*int(b[1][1]) for b in bounds.values()):return
        key=tuple(lo)+tuple(shape)
        if self.shared is not None and self.shared[0]==key:return
        crop=super().crop(lo,shape)
        self.shared=(key,lo.copy(),shape.copy(),crop);self.shared_builds+=1

    def crop(self,lo,shape):
        key=tuple(lo)+tuple(shape)
        if key in self.crops or self.shared is None:return super().crop(lo,shape)
        _,outer_lo,outer_shape,outer=self.shared
        if not self.contains_known(lo,shape) or np.any(lo<outer_lo) or np.any(lo+shape>outer_lo+outer_shape):return super().crop(lo,shape)
        offset=lo-outer_lo;a,b=map(int,offset);nx,nz=map(int,shape);region=np.s_[a:a+nx,b:b+nz]
        states=np.array(outer[0][region],copy=True,order='C')
        distance=[np.array(d[region],copy=True,order='C') for d in outer[2]]
        if self.uniform:baseline=np.full(tuple(shape),self.heights[0],np.float32)
        else:
            heights=np.full(tuple(shape),self.height,np.float32);ij=self.keys-lo
            heights[ij[:,0],ij[:,1]]=self.heights
            nearest=distance_transform_edt(states==0,return_distances=False,return_indices=True)
            baseline=heights[nearest[0],nearest[1]];self.shared_height_builds+=1
        value=(states,baseline,distance);self.crop_misses+=1;self.shared_slices+=1;self.crops[key]=value
        while len(self.crops)>getattr(self,'crop_cache_size',4):self.crops.popitem(last=False)
        return value



def complete_batch(self,inputs):
    items=list(inputs);tick=time.perf_counter();compiled={};before={};paths=defaultdict(list);objects={}
    hits=self.cache_hits;misses=self.cache_misses
    for item in items:
        graph=item['partial_graph'];identity=id(graph)
        if identity not in compiled:
            compiled[identity]=self.prepare_graph(graph);p=compiled[identity]
            if id(p) not in before:
                before[id(p)]=(p,p.crop_hits,p.crop_misses,p.integrals is not None,getattr(p,'shared_builds',0),getattr(p,'shared_slices',0),getattr(p,'shared_height_builds',0))
        p=compiled[identity];paths[id(p)].append(item['trajectory']);objects[id(p)]=p
    prewarm=time.perf_counter()
    for identity,trajectories in paths.items():
        if len(trajectories)>1 and isinstance(objects[identity],SharedCompiledGraph):objects[identity].prewarm(trajectories)
    warmed=time.perf_counter();out=[self._predict(item['partial_graph'],item['trajectory'],compiled[id(item['partial_graph'])]) for item in items]
    update=sum(self._complete(item['partial_graph'],delta,item['trajectory']) for item,delta in zip(items,out))
    self.last_batch_profile=dict(queries=len(out),distinct_input_graph_objects=len(compiled),graph_cache_hits=self.cache_hits-hits,graph_cache_misses=self.cache_misses-misses,
        within_batch_geometry_reuses=len(out)-(self.cache_hits-hits+self.cache_misses-misses),
        crop_cache_hits=sum(p.crop_hits-h for p,h,m,i,b,s,t in before.values()),crop_cache_misses=sum(p.crop_misses-m for p,h,m,i,b,s,t in before.values()),
        map_integral_builds=sum(int(p.integrals is not None and not i) for p,h,m,i,b,s,t in before.values()),
        shared_crop_builds=sum(getattr(p,'shared_builds',0)-b for p,h,m,i,b,s,t in before.values()),
        shared_crop_slices=sum(getattr(p,'shared_slices',0)-s for p,h,m,i,b,s,t in before.values()),
        child_height_transforms=sum(getattr(p,'shared_height_builds',0)-t for p,h,m,i,b,s,t in before.values()),
        shared_prewarm_seconds=warmed-prewarm,update_seconds=update,total_seconds=time.perf_counter()-tick,materialized_future_graphs=len(out))
    return out
