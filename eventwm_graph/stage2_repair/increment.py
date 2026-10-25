"""Concrete immutable sparse increment columns, compatible with update loops.

All numbers and provenance are materialized at construction. Row objects only
format scalar access; no prediction, decode or map update is deferred.
"""
from collections.abc import Mapping,Sequence
import numpy as np
from eventwm_graph.stage2_segment.native import immutable_array,Immutable


class UpdateRow(Immutable,Mapping):
    __slots__=('table','index')
    def __init__(self,table,index):self.table=table;self.index=index
    def __iter__(self):return iter(('cell','state','height','timestamp','source'))
    def __len__(self):return 5
    def __getitem__(self,key):
        t=self.table;i=self.index
        if key=='cell':return t.keys[i].tolist()
        if key=='state':return int(t.states[i])
        if key=='height':return float(t.heights[i])
        if key=='timestamp':return t.timestamp
        if key=='source':return t.source
        raise KeyError(key)


class SparseIncrement(Immutable,Sequence):
    __slots__=('keys','states','heights','timestamp','source')
    def __init__(self,keys,states,heights,timestamp,source='predicted_stage2_research'):
        self.keys=immutable_array(keys,np.int32).reshape(-1,2);self.states=immutable_array(states,np.uint8);self.heights=immutable_array(heights,np.float32)
        self.timestamp=float(timestamp);self.source=str(source)
        if not(len(self.keys)==len(self.states)==len(self.heights)):raise ValueError('increment column lengths disagree')
    def __len__(self):return len(self.states)
    def __getitem__(self,index):
        if isinstance(index,slice):return [UpdateRow(self,i)for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return UpdateRow(self,index)
    def to_list(self):
        return [dict(cell=k,state=s,height=h,timestamp=self.timestamp,source=self.source)for k,s,h in zip(self.keys.tolist(),self.states.tolist(),self.heights.tolist())]
