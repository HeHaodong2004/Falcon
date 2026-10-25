"""Execution-only compiled supercover checks for the existing graph updater.

PartialGraph.update/restore semantics are retained. The private subclass only
accelerates segment_free, which is exercised when pruning abstract edges.
"""
import math
import copy
from collections.abc import Mapping
import numpy as np
from numba import njit
from eventwm_graph.partial.map import PartialGraph
from .geometry import graph_identity
from eventwm_graph.stage2.common import SCHEMA


@njit(cache=True)
def point_free(p,ego,lo,states,heights,resolution):
    if np.sqrt(((p-ego)**2).sum())<1e-7:return True
    x=int(math.floor(p[0]/resolution))-lo[0];z=int(math.floor(p[2]/resolution))-lo[1]
    return 0<=x<states.shape[0] and 0<=z<states.shape[1] and states[x,z]==1 and abs(p[1]-heights[x,z])<=.22


@njit(cache=True)
def segment_free(a,b,ego,lo,states,heights,resolution):
    if np.sqrt(((a-b)**2).sum())<1e-8:return point_free(a,ego,lo,states,heights,resolution)
    if not point_free(a,ego,lo,states,heights,resolution) or not point_free(b,ego,lo,states,heights,resolution):return False
    if abs(a[1]-b[1])>.22:return False
    start=np.array([a[0]/resolution,a[2]/resolution]);end=np.array([b[0]/resolution,b[2]/resolution]);delta=end-start
    ts=[0.,1.]
    for axis in range(2):
        if abs(delta[axis])<1e-14:continue
        low=min(start[axis],end[axis]);high=max(start[axis],end[axis])
        for line in range(int(math.ceil(low)),int(math.floor(high))+1):
            t=(line-start[axis])/delta[axis]
            if 0.<=t<=1.:ts.append(t)
    ts.sort();count=len(ts)
    for j in range(count-1):ts.append((ts[j]+ts[j+1])/2.)
    for t in ts:
        p=start+delta*t;nx=int(round(p[0]));nz=int(round(p[1]))
        xa=nx-1 if abs(p[0]-nx)<1e-10 else int(math.floor(p[0]));xb=nx if abs(p[0]-nx)<1e-10 else xa
        za=nz-1 if abs(p[1]-nz)<1e-10 else int(math.floor(p[1]));zb=nz if abs(p[1]-nz)<1e-10 else za
        for x in range(xa,xb+1):
            for z in range(za,zb+1):
                ix=x-lo[0];iz=z-lo[1]
                if ix<0 or iz<0 or ix>=states.shape[0] or iz>=states.shape[1] or states[ix,iz]!=1:return False
    return True


class FastPartialGraph(PartialGraph):
    @classmethod
    def restore(cls,snapshot):
        immutable_input=not isinstance(snapshot,dict)
        if immutable_input:
            # Stage1 exposes an immutable legal graph. Convert only metadata
            # sequences that the inherited restore deep-copies; cell scalars
            # are already copied by that original implementation.
            def thaw(value):
                if isinstance(value,Mapping):return {k:thaw(v) for k,v in value.items()}
                if isinstance(value,(tuple,list)):return [thaw(v) for v in value]
                if isinstance(value,np.ndarray):return value.copy()
                return value
            snapshot=dict(snapshot)
            for name in ('visited','global_nodes','global_edges'):
                if name in snapshot:snapshot[name]=thaw(snapshot[name])
        graph=super().restore(snapshot)
        graph._thaw_cell_extensions=immutable_input
        return graph
    def update(self,observation):
        self._fast_grid=None
        return super().update(observation)

    def segment_free(self,a,b):
        if getattr(self,'_fast_grid',None) is None:
            keys=np.asarray(list(self.cells),np.int64).reshape(-1,2)
            if len(keys):lo=keys.min(0);shape=keys.max(0)-lo+1
            else:lo=np.zeros(2,np.int64);shape=np.ones(2,np.int64)
            # Preserve general graph support without unbounded dense allocation.
            if int(np.prod(shape))>4000000:return super().segment_free(a,b)
            states=np.zeros(tuple(shape),np.uint8);heights=np.zeros(tuple(shape),np.float64)
            for (x,z),cell in self.cells.items():states[x-lo[0],z-lo[1]]=cell['state'];heights[x-lo[0],z-lo[1]]=cell['height']
            self._fast_grid=(lo,states,heights)
        ego=np.asarray(self.ego_support,np.float64) if self.ego_support is not None else np.full(3,np.inf)
        return bool(segment_free(np.asarray(a,np.float64),np.asarray(b,np.float64),ego,*self._fast_grid,self.resolution))


def owned_snapshot_transfer(graph):
    """Consume the private updater; move its owned rows instead of recopying.

    Mutable extension values are copied just as in the frozen serializer.
    JSON object key order may differ; coordinates/values/topology/provenance
    and canonical graph identities are unchanged. No caller row is shared.
    """
    def owned_copy(value):
        # Immutable Stage1/Sandbox inputs can freeze extension metadata too.
        # Scalars keep the allocation-free path used by recorded map cells.
        if isinstance(value,Mapping):return {k:owned_copy(v) for k,v in value.items()}
        if isinstance(value,list):return [owned_copy(v) for v in value]
        if isinstance(value,tuple):
            items=(owned_copy(v) for v in value)
            return list(items) if getattr(graph,'_thaw_cell_extensions',False) else tuple(items)
        return copy.deepcopy(value)
    cells=[];owned=graph.cells;graph.cells={}
    for key,row in sorted(owned.items()):
        for name,value in list(row.items()):
            if not isinstance(value,(str,int,float,bool,type(None))):row[name]=owned_copy(value)
        row['cell']=list(key);cells.append(row)
    return dict(schema='causal_partial_cspace_graph_1',resolution=graph.resolution,height=graph.height,clearance=graph.clearance,
        timestamp=graph.timestamp,revision=graph.revision,ego_support=graph.ego_support.tolist() if graph.ego_support is not None else None,
        cells=cells,visited=graph.visited,global_nodes=graph.global_nodes,global_edges=graph.global_edges)


def apply_graph_delta(current_graph,predicted_delta):
    if predicted_delta.get('schema')!=SCHEMA or predicted_delta.get('predicted') is not True:raise ValueError('invalid predicted GraphDelta')
    if predicted_delta['base_graph_identity']!=graph_identity(current_graph):raise ValueError('GraphDelta base graph mismatch')
    graph=FastPartialGraph.restore(current_graph);now=float(predicted_delta['timestamp'])
    if now<graph.timestamp:raise ValueError('GraphDelta clock regressed')
    updates=predicted_delta['static_updates'];seen=set()
    for row in updates:
        key=tuple(row['cell'])
        if key in seen or graph.state(key)!=0 or row['state'] not in (1,2) or row['timestamp']!=now or not np.isfinite(row['height']):raise ValueError('delta must update unique initially UNKNOWN cells only')
        seen.add(key)
    if now==graph.timestamp:
        if updates:raise ValueError('zero-duration trajectory cannot reveal cells')
    else:graph.update(dict(timestamp=now,robot_pose=predicted_delta['robot_pose'],static_updates=updates,tracks=[]))
    out=owned_snapshot_transfer(graph);out['hypothetical']=True;out['prediction_checkpoint_sha256']=predicted_delta['checkpoint_sha256'];return out
