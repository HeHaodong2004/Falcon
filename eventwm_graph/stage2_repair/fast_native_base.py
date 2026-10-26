"""Fast canonical metadata extraction; exact legacy behavior remains fallback."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sysconfig
from types import MappingProxyType
import numpy as np
from .diagnose import OUT
from eventwm_graph.stage2_segment.native import Base,Table,freeze,certificate,immutable_array

_module=None


def extension():
    global _module
    if _module is None:
        source=Path(__file__).with_name('native_metadata.cpp');identity=hashlib.sha256(source.read_bytes()+str(sysconfig.get_config_var('SOABI')).encode()).hexdigest();path=OUT/'deps'/(identity+sysconfig.get_config_var('EXT_SUFFIX'))
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():subprocess.run(['g++','-O3','-std=c++14','-shared','-fPIC',str(source),'-I',sysconfig.get_path('include'),'-o',str(path)],check=True)
        spec=importlib.util.spec_from_file_location('_falcon_native_metadata',path);_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(_module)
    return _module


class FastNativeBase(Base):
    __slots__=()
    def __init__(self,graph,prepared):
        if float(graph['clearance'])<=0:raise ValueError('positive clearance required')
        cells=graph['cells'];n=len(cells);first=np.empty(n,np.float64);last=np.empty(n,np.float64);ids=np.empty(n,np.int32);names=extension().extract(cells,prepared.input_order,first,last,ids)
        if names is None:return super().__init__(graph,prepared)
        self.table=Table(prepared.keys,prepared.values,prepared.height64,first,last,ids,names);canonical=isinstance(graph,dict)
        self.headers=freeze({k:graph[k]for k in ('schema','resolution','height','clearance','timestamp','revision','ego_support','visited')},canonical)
        self.headers=MappingProxyType(dict(self.headers,global_nodes=freeze(graph.get('global_nodes',{}),canonical),global_edges=freeze(graph.get('global_edges',[]),canonical)))
        self.identity=prepared.identity;self.valid_topology=certificate(graph,prepared);biased=(self.table.keys+np.int64(2147483648)).astype(np.uint64);self.known_codes=immutable_array((biased[:,0]<<np.uint64(32))|biased[:,1])


class FastColumnsMixin:
    def _canonical_columns(self,cells):
        n=len(cells);keys=np.empty((n,2),np.int64);states=np.empty(n,np.uint8);heights=np.empty(n,np.float64)
        if extension().columns(cells,keys,states,heights)is not True:return None
        return keys,states,heights


from .range_gpu_graph import DeviceGlobalPrepared


class FastDeviceGlobalPrepared(FastColumnsMixin,DeviceGlobalPrepared):pass
