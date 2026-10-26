"""One measured-history upload shared by mesh and map encoder.

The existing full-content range identity is checked BEFORE selecting these
buffers. Pointer descriptors only associate already validated inputs with
their device copies; they never replace mutation/content validation.
"""
import numpy as np
import torch
from .native_triangle_api import NativeTrianglePredictor
from .range_planar import PlanarRangeMesh
from .range_device_canvas import DeviceRangeCanvas


def descriptor(a):return(a.__array_interface__['data'][0],a.shape,a.strides,a.dtype.str)


class SensorBuffers:
    def __init__(self,ranges,origins,rotations,device):
        self.device=torch.device(device);self.original=[];self.pinned=[];self.copies={};self.values=[]
        for a,dtype in [(ranges,np.float32),(origins,np.float64),(rotations,np.float64)]:
            a=np.asarray(a,dtype=dtype);host=torch.from_numpy(np.ascontiguousarray(a)).pin_memory();value=host.to(self.device,non_blocking=True)
            self.original.append(a);self.pinned.append(host);self.values.append(value);self.copies[descriptor(a)]=value

    def tensor(self,a,dtype):
        a=np.asarray(a,dtype=dtype);value=self.copies.get(descriptor(a))
        if value is not None:return value
        return torch.from_numpy(np.ascontiguousarray(a)).to(self.device,non_blocking=True)

    def selected(self,indices):
        take=torch.as_tensor(np.asarray(indices,np.int64),device=self.device)
        return[v.index_select(0,take)for v in self.values]


class SharedSensorCanvas(DeviceRangeCanvas):
    input_buffers=None
    def tensor(self,a,dtype):
        if self.input_buffers is not None:return self.input_buffers.tensor(a,dtype)
        return super().tensor(a,dtype)


class SharedSensorPredictor(NativeTrianglePredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.canvas=SharedSensorCanvas(self.edt)

    def _make_mesh(self,ranges,origins,rotations,timestamps,now,resolution,tracks):
        buffers=SensorBuffers(ranges,origins,rotations,self.device);mesh=PlanarRangeMesh(ranges,origins,rotations,timestamps,now,resolution,self.device,tracks=tracks,bvh='native',input_buffers=buffers,epsilon=getattr(self,'plane_epsilon',.00005));mesh.input_buffers=buffers;self.range_kernel.module.backend.prepare(mesh);return mesh

    def _encoded_map(self,details):
        self.canvas.input_buffers=self.active_mesh.input_buffers
        return super()._encoded_map(details)


class SharedFastBuildPredictor(SharedSensorPredictor):primitive_mode='native-fast'


class Plane3mmPredictor(SharedFastBuildPredictor):
    plane_epsilon=.003
    def predict_segments(self,**kwargs):
        results=super().predict_segments(**kwargs)
        for d in results:d['provenance']['range_mesh_approximation']='All original input pixels tested. Merge4x4measured patches only when allpoints lie within3mm of the cornerreferenceplane; depth holes and0.3m edge guard retained. Numerical approximation, not certificate target change.'
        return results
