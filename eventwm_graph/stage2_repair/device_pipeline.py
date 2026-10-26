"""Same predictor/targets with original query cells retained on device longer."""
import time
import numpy as np
import torch
from .range_optix import OptixTriangleSpatialPredictor
from .device_query import DeviceQuery
from .device_point_features import extract as point_features
from .device_range_features import extract as range_features


class DevicePipelinePredictor(OptixTriangleSpatialPredictor):
    parallel_features=False
    def _probability_tiles(self,x,extra,indices,encoded,baseline,**kwargs):
        values=[];n=len(x)
        with torch.cuda.amp.autocast():
            for begin in range(0,n,self.tile_size):
                end=min(begin+self.tile_size,n);ids=torch.zeros(end-begin,dtype=torch.long,device=self.device);p,h=self.model.probabilities(x[begin:end],extra[begin:end],indices[begin:end],ids,encoded);values.append(torch.cat([p.float(),h.float()[:,None],baseline[begin:end,None]],1))
        return torch.cat(values)

    def _eligible(self,prepared,contexts):
        if not contexts or self.config.get('observed_obstacle_schema')!='capped_2m_complete':return False
        if prepared.global_fields is None or not isinstance(prepared.global_fields[1],torch.Tensor):return False
        if not all(hasattr(c,'candidate_lo')for c in contexts):return False
        if not all(np.array_equal(c.path[0],contexts[0].path[0])for c in contexts):return False
        low=np.min([c.candidate_lo for c in contexts],0)-4;high=np.max([c.candidate_lo+c.candidate_shape for c in contexts],0)+4;shape=high-low;padding=int(np.ceil(2./prepared.resolution))+2
        if np.any(low<prepared.global_lo)or np.any(high>prepared.global_lo+prepared.global_shape):return False
        if sum(int(np.prod(c.candidate_shape))for c in contexts)>2000000:return False
        if sum(len(c.path)for c in contexts)*int(np.prod(shape))*4>512*1024*1024:return False
        return bool(np.prod(shape+2*padding)<=2000000 and np.all(shape>0)and max(shape+2*padding)<=8192)

    def _forward_batch(self,prepared,contexts,partial_graph):
        if not self._eligible(prepared,contexts):return super()._forward_batch(prepared,contexts,partial_graph)
        with torch.inference_mode():
            if not hasattr(self,'device_query'):self.device_query=DeviceQuery(self.device)
            batch=self.device_query.build(prepared,contexts);n=len(batch['keys'])
            if not n:
                details=[(np.empty((0,2),np.int32),np.empty(0,np.float32),c.path)for c in contexts];return details,[np.empty((0,4),np.float32)for c in contexts],time.perf_counter(),[dict(device_pipeline=True,rows=0)]
            if self.parallel_features:
                if not hasattr(self,'geometry_stream'):self.geometry_stream=torch.cuda.Stream(device=self.device)
                current=torch.cuda.current_stream(self.device);self.geometry_stream.wait_stream(current)
                with torch.cuda.stream(self.geometry_stream):
                    extra=range_features(self,prepared,contexts,batch);ready=torch.cuda.Event();ready.record()
                # Static inputs stay owned by batch/mesh until the final copy.
                # Temporary geometry tensors were allocated on their own stream.
                encoded,low=self._encoded_map([(None,None,c.path)for c in contexts]);x,baseline,fallbacks=point_features(self.cuda,prepared,contexts,batch);current.wait_event(ready)
            else:
                x,baseline,fallbacks=point_features(self.cuda,prepared,contexts,batch);encoded,low=self._encoded_map([(None,None,c.path)for c in contexts]);extra=range_features(self,prepared,contexts,batch)
            features_at=time.perf_counter();indices=batch['keys'].long()-torch.as_tensor(low,device=self.device)
            flat=self._probability_tiles(x,extra,indices,encoded,baseline,contexts=contexts,low=low,offsets=batch['offsets']).cpu().numpy();keys=batch['keys'].cpu().numpy();details=[];returned=[]
            for i,c in enumerate(contexts):
                begin,end=map(int,batch['offsets'][i:i+2]);details.append((keys[begin:end],flat[begin:end,4],c.path));returned.append(flat[begin:end,:4])
        return details,returned,features_at,[dict(device_pipeline=True,rows=n,union_rows=len(batch['union']),boundary_fallbacks=fallbacks)]


class ConcurrentDevicePipelinePredictor(DevicePipelinePredictor):parallel_features=True


class PreparedConcurrentPipelinePredictor(ConcurrentDevicePipelinePredictor):
    def _make_mesh(self,*args,**kwargs):
        mesh=super()._make_mesh(*args,**kwargs)
        # Hardware structure depends only on this already measured mesh. Queue
        # its build before independent CPU graph parsing/metadata preparation.
        self.range_kernel.module.backend.prepare(mesh)
        return mesh


from .fast_native_base import FastNativeBase,FastDeviceGlobalPrepared,extension


class FastPreparedPipelinePredictor(PreparedConcurrentPipelinePredictor):
    native_base_class=FastNativeBase
    def __init__(self,*args,**kwargs):
        extension();super().__init__(*args,**kwargs)


class ColumnPreparedPipelinePredictor(FastPreparedPipelinePredictor):prepared_class=FastDeviceGlobalPrepared
