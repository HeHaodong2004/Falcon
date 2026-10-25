"""Complete learned-ray API: legal packets in, hypothetical future graphs out."""
from types import SimpleNamespace
import numpy as np
import torch
from .range_spatial_api import SpatialRangePredictor
from .range_ray_model import RayEvidenceNet
from .range_ray_data import camera_lines
from .range_camera_kernel import AnalyticCameraKernel
from .legal_camera import current_extrinsic


class RayRangePredictor(SpatialRangePredictor):
    model_class=RayEvidenceNet
    checkpoint_schema='legal_range_ray_1'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.camera_kernel=AnalyticCameraKernel(self.device)

    def _make_mesh(self,ranges,origins,rotations,timestamps,now,resolution,tracks):
        # Parent packet validation/content identity is retained. No observed
        # triangle surface is built, and no hidden scene is read.
        return SimpleNamespace(latest_sensor_time=float(timestamps[-1]),resolution=resolution)

    def _network(self,xs,details):
        if not any(len(x)for x in xs):return [np.empty((0,4),np.float32)for x in xs]
        encoded,low=self._encoded_map(details);result=[]
        for x,(keys,baseline,path)in zip(xs,details):
            extra=self.camera_kernel.compute(keys,x,path,self.current_origin,self.current_rotation,self.active_graph.resolution)
            translation,_=current_extrinsic(path[:1],self.current_origin,self.current_rotation)
            origins=camera_lines(path,translation,self.active_graph.resolution,low)
            ij=torch.from_numpy(np.ascontiguousarray(keys-low)).to(self.device,non_blocking=True)
            camera=torch.from_numpy(origins).to(self.device,non_blocking=True);values=[]
            with torch.cuda.amp.autocast():
                for begin in range(0,len(x),32768):
                    end=min(begin+32768,len(x));ids=torch.zeros(end-begin,dtype=torch.long,device=self.device)
                    p,h=self.model.probabilities(x[begin:end],extra[begin:end],ij[begin:end],ids,encoded,camera[None].expand(end-begin,-1,-1))
                    values.append(torch.cat([p.float(),h.float()[:,None]],1).cpu().numpy())
            result.append(np.concatenate(values)if values else np.empty((0,4),np.float32))
        return result

    def predict_segments(self,**kwargs):
        out=super().predict_segments(**kwargs)
        for d in out:
            d['provenance'].pop('range_mesh_pixel_stride',None)
            d['provenance']['range_representation']='Learned shared context sampled along three supplied camera-to-cell lines; no mesh or private geometry. First8measured-mesh covariates are absent.'
        return out
