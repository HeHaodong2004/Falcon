"""Immutable ordered numeric facts about the currently accumulated legal cells.

Only the owning OnlineContext constructs this private representation after full
graph/history certification. It stores measured state and last-seen time, never
predicted geometry. Ingest copies arrays and updates touched indices from final
certified cells, preserving occupied/ego/duplicate-update semantics and order.
"""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from .types import immutable_array

_CERTIFICATE=object()
VERSION='immutable_ordered_arrived_graph_cell_facts_1'


@dataclass(frozen=True,init=False)
class OrderedCellFacts:
    order: tuple
    index: object
    states: np.ndarray
    last_seen: np.ndarray
    version: str=VERSION

    def validate_certificate(self):
        if getattr(self,'_certificate',None) is not _CERTIFICATE:
            raise ValueError('uncertified ordered graph-cell facts')
        return self

    @classmethod
    def _verified(cls,order,index,states,last_seen):
        result=object.__new__(cls)
        for key,value in dict(order=order,index=index,states=immutable_array(states),
                last_seen=immutable_array(last_seen),version=VERSION,_certificate=_CERTIFICATE).items():
            object.__setattr__(result,key,value)
        return result

    @classmethod
    def from_cells(cls,order,cells):
        """Internal factory: input cells/order must already belong to a legal context."""
        order=tuple(order)
        if len(set(order))!=len(order) or len(order)!=len(cells):
            raise ValueError('ordered graph facts require every certified cell once')
        rows=[cells[key] for key in order]
        # Exact dtypes/order of the former np.array(list) feature reconstruction.
        states=np.asarray([row['state'] for row in rows],np.int64)
        last=np.asarray([row['last_seen'] for row in rows],np.float64)
        return cls._verified(order,MappingProxyType({key:i for i,key in enumerate(order)}),states,last)

    def update(self,order,cells,touched):
        """Private causal successor; read FINAL cells after actual observation ingest."""
        self.validate_certificate();old_n=len(self.order);new_n=len(order)
        if new_n<old_n or order[:old_n]!=self.order or new_n!=len(cells):
            raise ValueError('ordered graph facts changed or removed past cell indices')
        states=np.empty(new_n,np.int64);last=np.empty(new_n,np.float64)
        states[:old_n]=self.states;last[:old_n]=self.last_seen
        if new_n==old_n:
            index=self.index
        else:
            changed=dict(self.index)
            for i in range(old_n,new_n):changed[order[i]]=i
            index=MappingProxyType(changed)
            for i in range(old_n,new_n):
                row=cells[order[i]];states[i]=row['state'];last[i]=row['last_seen']
        for key in touched:
            i=index.get(key)
            if i is not None:
                row=cells[key];states[i]=row['state'];last[i]=row['last_seen']
        return type(self)._verified(order,index,states,last)

    def update_with_rows(self,order,cells,touched,old_rows):
        """Update numeric facts and the owning context's immutable cell tuple.

        ``old_rows`` belongs to the same certified context as this cache. Copy
        its references in C, then replace only touched/fresh rows with FINAL
        arrived cells. This preserves the exact public tuple and caller order
        without looking up every accumulated cell again in Python.
        """
        self.validate_certificate();old_n=len(self.order);new_n=len(order)
        if (new_n<old_n or order[:old_n]!=self.order or new_n!=len(cells)
                or not isinstance(old_rows,tuple) or len(old_rows)!=old_n):
            raise ValueError('ordered graph facts changed or removed past cell indices')
        rows=list(old_rows)
        states=np.empty(new_n,np.int64);last=np.empty(new_n,np.float64)
        states[:old_n]=self.states;last[:old_n]=self.last_seen
        if new_n==old_n:
            index=self.index
        else:
            changed=dict(self.index)
            for i in range(old_n,new_n):changed[order[i]]=i
            index=MappingProxyType(changed)
            for i in range(old_n,new_n):
                row=cells[order[i]];states[i]=row['state'];last[i]=row['last_seen']
                rows.append(row)
        for key in touched:
            i=index.get(key)
            if i is not None:
                row=cells[key];states[i]=row['state'];last[i]=row['last_seen']
                rows[i]=row
        return type(self)._verified(order,index,states,last),tuple(rows)
