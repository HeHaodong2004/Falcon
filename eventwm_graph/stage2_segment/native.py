"""Complete immutable PartialGraph stored as concrete typed cell columns.

All columns, canonical ordering, update times and final graph metadata exist
before prediction returns. Views only format scalar access. to_snapshot() is
an explicit representation conversion, not a delayed graph update.
"""
from collections.abc import Mapping,Sequence
from types import MappingProxyType
import math
import numpy as np
from numba import njit
from eventwm_graph.stage2_quality.update import FastPartialGraph,segment_free

FIELDS=('cell','state','height','first_seen','last_seen','source')
FIELD_SET=frozenset(FIELDS)


class Immutable:
    __slots__=()
    def __setattr__(self,name,value):
        if hasattr(self,name):raise AttributeError('immutable graph storage')
        object.__setattr__(self,name,value)
    def __delattr__(self,name):raise AttributeError('immutable graph storage')


def immutable_array(value,dtype=None):
    array=np.ascontiguousarray(value,dtype=dtype)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


def freeze(value,canonical=False):
    kind=type(value)
    if kind in (str,int,float,bool,type(None)):return value
    if kind is dict:return MappingProxyType({k:freeze(v,canonical) for k,v in value.items()})
    if kind is list:return tuple(freeze(v,canonical) for v in value)
    if canonical:raise TypeError('noncanonical metadata requires legacy snapshot preservation')
    if isinstance(value,Mapping):return MappingProxyType({k:freeze(v) for k,v in value.items()})
    if isinstance(value,tuple):return tuple(freeze(v) for v in value)
    if isinstance(value,np.ndarray):return immutable_array(value)
    return value


def thaw(value):
    if isinstance(value,Mapping):return {k:thaw(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [thaw(v) for v in value]
    if isinstance(value,np.ndarray):return value.copy()
    return value


class Cell(Immutable,Mapping):
    __slots__=('table','index')
    def __init__(self,table,index):self.table=table;self.index=index
    def __iter__(self):return iter(FIELDS)
    def __len__(self):return 6
    def __getitem__(self,key):
        t=self.table;i=self.index
        if key=='cell':return t.keys[i].tolist()
        if key=='state':return int(t.states[i])
        if key=='height':return float(t.heights[i])
        if key=='first_seen':return float(t.first[i])
        if key=='last_seen':return float(t.last[i])
        if key=='source':return t.sources[int(t.source_ids[i])]
        raise KeyError(key)


class Cells(Immutable,Sequence):
    __slots__=('table',)
    def __init__(self,table):self.table=table
    def __len__(self):return len(self.table.states)
    def __getitem__(self,index):
        if isinstance(index,slice):return tuple(Cell(self.table,i) for i in range(*index.indices(len(self))))
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return Cell(self.table,index)


class Table(Immutable):
    __slots__=('keys','states','heights','first','last','source_ids','sources')
    def __init__(self,keys,states,heights,first,last,source_ids,sources):
        self.keys=immutable_array(keys,np.int64);self.states=immutable_array(states,np.uint8)
        self.heights=immutable_array(heights,np.float64);self.first=immutable_array(first,np.float64);self.last=immutable_array(last,np.float64)
        self.source_ids=immutable_array(source_ids,np.int32);self.sources=tuple(sources)
    def columns(self):
        return MappingProxyType(dict(cell=self.keys,state=self.states,height=self.heights,first_seen=self.first,last_seen=self.last,
                                     source_id=self.source_ids,source_names=self.sources))


class NativePartialGraph(Immutable,Mapping):
    __slots__=('_headers','_table','_cells')
    def __init__(self,headers,table):self._headers=MappingProxyType(dict(headers));self._table=table;self._cells=Cells(table)
    def __iter__(self):return iter((*self._headers.keys(),'cells'))
    def __len__(self):return len(self._headers)+1
    def __getitem__(self,key):return self._cells if key=='cells' else self._headers[key]
    @property
    def cell_columns(self):return self._table.columns()
    def to_snapshot(self):
        out=thaw(self._headers)
        out['cells']=[dict(cell=self._table.keys[i].tolist(),state=int(self._table.states[i]),height=float(self._table.heights[i]),
            first_seen=float(self._table.first[i]),last_seen=float(self._table.last[i]),source=self._table.sources[int(self._table.source_ids[i])]) for i in range(len(self._cells))]
        return out
    def __eq__(self,other):
        if isinstance(other,NativePartialGraph):other=other.to_snapshot()
        return self.to_snapshot()==other


@njit(cache=True)
def certify_segments(starts,ends,lo,states,heights,resolution):
    ego=np.full(3,np.inf)
    for i in range(len(starts)):
        if not segment_free(starts[i],ends[i],ego,lo,states,heights,resolution):return False
    return True


def certificate(graph,prepared):
    """Known-certified paths stay valid under initially-UNKNOWN additions."""
    try:
        nodes=graph.get('global_nodes',{});edges=graph.get('global_edges',[])
        if not nodes and not edges:return True
        keys=prepared.keys.astype(np.int64)
        if not len(keys):return False
        lo=keys.min(0);shape=keys.max(0)-lo+1
        if int(np.prod(shape))>4000000:return False
        states=np.zeros(tuple(shape),np.uint8);heights=np.zeros(tuple(shape),np.float64);ij=keys-lo
        states[ij[:,0],ij[:,1]]=prepared.values;heights[ij[:,0],ij[:,1]]=prepared.height64
        for node in nodes.values():
            raw=np.asarray(node['cell']);integer=raw.astype(np.int64)
            if raw.shape!=(2,) or not np.array_equal(raw,integer):return False
            k=integer-lo
            if np.any(k<0) or np.any(k>=shape) or states[k[0],k[1]]!=1:return False
        starts=[];ends=[]
        for edge in edges:
            if edge['source'] not in nodes or edge['target'] not in nodes or not edge['path']:return False
            for a,b in zip(edge['path'],edge['path'][1:]):starts.append(a);ends.append(b)
        if not starts:return True
        return bool(certify_segments(np.asarray(starts,np.float64),np.asarray(ends,np.float64),lo,states,heights,float(graph['resolution'])))
    except (KeyError,TypeError,ValueError,OverflowError):return False


class Base:
    __slots__=('table','headers','identity','valid_topology','known_codes')
    def __init__(self,graph,prepared):
        cells=graph['cells'];order=prepared.input_order
        if float(graph['clearance'])<=0:raise ValueError('positive clearance required')
        # Extensions and unusual cell metadata use the exact legacy fallback.
        names=[];lookup={};ids=[];first=[];last=[]
        for index in order:
            c=cells[int(index)]
            if len(c)!=6 or (tuple(c)!=FIELDS and c.keys()!=FIELD_SET) or not isinstance(c['source'],str):
                raise TypeError('native table requires canonical static-cell metadata')
            source=c['source']
            if source not in lookup:lookup[source]=len(names);names.append(source)
            ids.append(lookup[source]);first.append(float(c['first_seen']));last.append(float(c['last_seen']))
        if not np.isfinite(first).all() or not np.isfinite(last).all():raise TypeError('nonfinite cell metadata uses legacy fallback')
        self.table=Table(prepared.keys,prepared.values,prepared.height64,first,last,ids,names)
        canonical=isinstance(graph,dict)
        self.headers=freeze({k:graph[k] for k in ('schema','resolution','height','clearance','timestamp','revision','ego_support','visited')},canonical)
        self.headers=MappingProxyType(dict(self.headers,global_nodes=freeze(graph.get('global_nodes',{}),canonical),global_edges=freeze(graph.get('global_edges',[]),canonical)))
        self.identity=prepared.identity;self.valid_topology=certificate(graph,prepared)
        # Signed-coordinate bias preserves the canonical lexicographic order.
        # Vector encoding avoids a Python object/set per already-known cell.
        biased=(self.table.keys+np.int64(2147483648)).astype(np.uint64)
        self.known_codes=immutable_array((biased[:,0]<<np.uint64(32))|biased[:,1])

    def update(self,delta):
        now=float(delta['timestamp']);old=float(self.headers['timestamp']);updates=delta['static_updates']
        if not math.isfinite(now) or now<old:raise ValueError('GraphDelta clock regressed or nonfinite')
        if delta['base_graph_identity']!=self.identity:raise ValueError('GraphDelta base graph mismatch')
        if updates:
            rawkeys=np.asarray([u['cell'] for u in updates])
            if rawkeys.shape!=(len(updates),2) or rawkeys.dtype.kind not in ('i','u') or np.any(rawkeys<np.iinfo(np.int32).min) or np.any(rawkeys>np.iinfo(np.int32).max):
                raise ValueError('new cell coordinates exceed supported int32 geometry')
            biased=(rawkeys.astype(np.int64)+np.int64(2147483648)).astype(np.uint64)
            codes=(biased[:,0]<<np.uint64(32))|biased[:,1]
            indices=np.searchsorted(self.known_codes,codes);inside=indices<len(self.known_codes)
            if np.any(self.known_codes[indices[inside]]==codes[inside]):raise ValueError('known-cell correction outside new-observation scope')
        seen=set()
        for u in updates:
            key=tuple(u['cell'])
            if key in seen or u['state'] not in (1,2) or u['timestamp']!=now or not np.isfinite(u['height']):raise ValueError('invalid new-observation update')
            seen.add(key)
            if u['source']=='occupied_ego_base_support':return None
        if now==old and updates:raise ValueError('zero-duration revelations forbidden')
        if updates and not self.valid_topology:return None
        base=self.table;names=list(base.sources);lookup={v:i for i,v in enumerate(names)}
        for u in updates:
            if not isinstance(u['source'],str):return None
            if u['source'] not in lookup:lookup[u['source']]=len(names);names.append(u['source'])
        if updates:
            keys=np.concatenate((base.keys,np.asarray([u['cell'] for u in updates],np.int64)))
            order=np.lexsort((keys[:,1],keys[:,0]));keys=keys[order]
            states=np.concatenate((base.states,np.asarray([u['state'] for u in updates],np.uint8)))[order]
            heights=np.concatenate((base.heights,np.asarray([u['height'] for u in updates],np.float64)))[order]
            first=np.concatenate((base.first,np.full(len(updates),now)))[order];last=np.concatenate((base.last,np.full(len(updates),now)))[order]
            ids=np.concatenate((base.source_ids,np.asarray([lookup[u['source']] for u in updates],np.int32)))[order]
            table=Table(keys,states,heights,first,last,ids,names)
        else:table=base
        headers=dict(self.headers)
        if now>old:
            pose=delta['robot_pose'];headers['ego_support']=tuple(pose[:3]);headers['timestamp']=now
            visited=headers['visited']
            if not visited or np.linalg.norm((np.asarray(pose[:3])-np.asarray(visited[-1]['position']))[[0,2]])>=.55:
                headers['visited']=(*visited,MappingProxyType(dict(position=tuple(pose[:3]),timestamp=now)))
            if updates:headers['revision']+=1
        headers['hypothetical']=True;headers['prediction_checkpoint_sha256']=delta['checkpoint_sha256']
        return NativePartialGraph(headers,table)
