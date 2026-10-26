"""Optional hardware traversal with the same63LOS certificate features."""
import ctypes as ct
import torch
from .cuda_module import CudaModule
from .range_exact_fast import source
from .range_map_reuse import SupportReuse
from .range_fast import RangeFeatureKernel
from .camera_vectorized import parameters
from .optix_backend import OptixBackend
from .range_batched_spatial import BatchedSpatialPredictor


class OptixModule(CudaModule):
    def __init__(self,device,normalization=False,primitive_mode='leaf'):
        if normalization:
            from .range_normalization import source as normalized_source
            code=normalized_source()
        else:code=source()
        super().__init__(code,device);self.backend=OptixBackend(device,normalization=normalization,primitive_mode=primitive_mode);self.active_mesh=None

    def launch(self,name,total,values,threads=128):
        if name not in ['generated_views','wall_views','square_support']:return super().launch(name,total,values,threads)
        if not total:return
        p=self.backend.parameters(self.active_mesh)
        if name=='generated_views':
            nv=values[4].value;planes=torch.empty((nv,20),device=self.device,dtype=torch.float64);offsets=torch.empty((63,3),device=self.device,dtype=torch.float64)
            super().launch('prepare_certificate',nv,[values[3],values[4],values[10],planes,offsets]);p.mode=0;p.keys=values[0].data_ptr();p.floors=values[1].data_ptr();p.nc=values[2].value;p.cameras=values[3].data_ptr();p.nv=nv;p.planes=planes.data_ptr();p.probe_offsets=offsets.data_ptr();p.unknown=values[5].data_ptr();p.lx=values[6].value;p.lz=values[7].value;p.nx=values[8].value;p.nz=values[9].value;p.resolution=values[10].value;p.frames=values[11].data_ptr();self.constants=(planes,offsets)
        elif name=='wall_views':
            p.mode=1;p.cameras=values[0].data_ptr();p.nv=values[1].value;p.unknown=values[2].data_ptr();p.lx=values[3].value;p.lz=values[4].value;p.nx=values[5].value;p.nz=values[6].value;p.resolution=values[7].value;p.frames=values[8].data_ptr()
        else:
            p.mode=2;p.keys=values[0].data_ptr();p.floors=values[1].data_ptr();p.nc=values[2].value;p.resolution=values[3].value;p.support=values[4].data_ptr();p.difference=values[5].data_ptr()
        self.backend.launch(p,total)


class OptixRangeKernel(SupportReuse,RangeFeatureKernel):
    camera_parameters=staticmethod(parameters)
    def __init__(self,device,normalization=False,primitive_mode='leaf'):self.device=torch.device(device);self.module=OptixModule(device,normalization=normalization,primitive_mode=primitive_mode)
    def compute(self,mesh,*args):
        self.module.active_mesh=mesh
        return super().compute(mesh,*args)


class OptixSpatialPredictor(BatchedSpatialPredictor):
    fast_normalization=False
    primitive_mode='leaf'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.range_kernel=OptixRangeKernel(self.device,normalization=self.fast_normalization,primitive_mode=self.primitive_mode)


class OptixNormalizedSpatialPredictor(OptixSpatialPredictor):fast_normalization=True
class OptixTriangleSpatialPredictor(OptixSpatialPredictor):primitive_mode='triangle'
