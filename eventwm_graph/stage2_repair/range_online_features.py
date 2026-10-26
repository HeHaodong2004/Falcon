"""Legal array-only features; future targets are excluded from this builder."""
import numpy as np
import torch
from .shared_online_train import ArrayPrepared
from .global_features import GlobalPrepared,GlobalFeatures
from .features import transform_batch
from .range_planar import PlanarRangeMesh
from .range_map_reuse import ReusedExactKernel
from .range_bev import RangeCanvas
from .range_history import validate
from eventwm_graph.stage2_segment.cuda import CudaExecution


class ArrayGlobal(ArrayPrepared):
    contains_domain=GlobalPrepared.contains_domain
    def __init__(self,keys,states,heights,geometry,ego):
        super().__init__(dict(keys=keys,states=states,heights=heights,geometry=geometry))
        center=np.floor(np.asarray(ego)[[0,2]]/self.resolution).astype(np.int64)
        low=np.minimum(self.known_lo,center)if self.known_lo is not None else center
        high=np.maximum(self.known_hi,center+1)if self.known_hi is not None else center+1
        lo=((low-128)//32)*32;hi=((high+159)//32)*32;shape=hi-lo
        self.global_fields=None
        if int(np.prod(shape))<=6000000:
            self.global_lo=lo;self.global_shape=shape;self.global_fields=self.crop(lo,shape)

    def prewarm(self,paths):
        if all(self.contains_domain(p)for p in paths):return
        return super().prewarm(paths)


class LegalFeatureBuilder:
    def __init__(self,model,device='cuda:0',device_canvas=False,device_edt=False,initial_visibility=False):
        self.device=torch.device(device)
        self.engine=CudaExecution(model.point.point.prior,device)
        self.range_kernel=ReusedExactKernel(device);self.canvas=RangeCanvas(device);self.edt=None
        if initial_visibility:
            from .initial_visibility import TrainingInitialKernel
            self.range_kernel=TrainingInitialKernel(device)
        if device_canvas or device_edt:
            from .gpu_edt import GridEDT
            self.edt=GridEDT(device)
        if device_canvas:
            from .range_device_canvas import DeviceRangeCanvas
            self.canvas=DeviceRangeCanvas(self.edt)
        self.device_edt=device_edt

    def prepare(self,*,keys,states,heights,geometry,ego,now,paths):
        prepared=GpuArrayGlobal(keys,states,heights,geometry,ego,self.edt)if self.device_edt else ArrayGlobal(keys,states,heights,geometry,ego);prepared.prewarm(paths)
        graph=dict(schema='causal_partial_cspace_graph_1',timestamp=now,resolution=geometry[0],height=geometry[1],clearance=geometry[2])
        contexts=[GlobalFeatures(graph,p,prepared)for p in paths]
        return prepared,contexts

    def build(self,*,prepared,contexts,keysets,known_keys,known_states,known_heights,ego,packet,tracks):
        """No target, class, loss mask, private geometry or scene ID argument."""
        paths=[c.path for c in contexts];now=float(paths[0][0,0]);r=prepared.resolution
        validate(packet['ranges'],packet['origins'],packet['rotations'],packet['timestamps'],now)
        if abs(packet['timestamps'][-1]-now)>1.e-7:raise ValueError('sensor packet is not current')
        with torch.no_grad():
            xs,baselines,profile=transform_batch(self.engine,contexts,keysets)
            mesh=PlanarRangeMesh(packet['ranges'],packet['origins'],packet['rotations'],packet['timestamps'],now,r,self.device,tracks=tracks,bvh='radix')
            details=[(k,b,p)for k,b,p in zip(keysets,baselines,paths)]
            extras=self.range_kernel.compute(mesh,details,known_keys,packet['origins'][-1],packet['rotations'][-1])
            canvas,low=self.canvas.build(known_keys,known_states,known_heights,ego,r,packet['ranges'],packet['origins'],packet['rotations'],packet['timestamps'],tracks)
        return xs,extras,canvas if torch.is_tensor(canvas)else canvas.astype(np.float16),low,profile


class GpuArrayGlobal(ArrayGlobal):
    def __init__(self,keys,states,heights,geometry,ego,edt):
        self.edt=edt;super().__init__(keys,states,heights,geometry,ego)

    def crop(self,lo,shape):
        if max(shape)>8192 or int(np.prod(shape))>2000000:return super().crop(lo,shape)
        key=tuple(lo)+tuple(shape)
        if key in self.crops:self.crop_hits+=1;self.crops.move_to_end(key);return self.crops[key]
        self.crop_misses+=1;state=np.zeros(tuple(shape),np.uint8);height=np.full(tuple(shape),self.height,np.float32);ij=self.keys-lo;valid=np.all((ij>=0)&(ij<shape),1);at=ij[valid];known=self.heights[valid];state[at[:,0],at[:,1]]=self.values[valid];height[at[:,0],at[:,1]]=known
        uniform=bool(len(known)and np.all(known.view(np.uint32)==known[:1].view(np.uint32)))
        distance,baseline,_=self.edt.transform(state,height,self.resolution,self.height,nearest_height=not uniform)
        if uniform:base=np.full(tuple(shape),known[0],np.float32)
        else:base=baseline.cpu().numpy()
        distances=distance.cpu().numpy();value=(state,base,[distances[0],distances[1]]);self.crops[key]=value
        while len(self.crops)>self.crop_cache_size:self.crops.popitem(last=False)
        return value
