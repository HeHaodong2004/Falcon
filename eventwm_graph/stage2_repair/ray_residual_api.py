"""Preserved measured geometry plus learned legal camera-line residual."""
import numpy as np
import torch
from .device_pipeline import ColumnPreparedPipelinePredictor
from .ray_residual_model import RayResidualEvidenceNet
from .legal_camera import current_extrinsic
from .range_ray_data import camera_lines


class RayResidualPredictor(ColumnPreparedPipelinePredictor):
    model_class=RayResidualEvidenceNet
    checkpoint_schema='legal_range_ray_residual_1'

    def _origins(self,paths,counts,low):
        translation,_=current_extrinsic(paths[0][:1],self.current_origin,self.current_rotation);parts=[]
        for path,count in zip(paths,counts):
            rays=camera_lines(path,translation,self.active_graph.resolution,low);parts.append(torch.from_numpy(rays.copy()).to(self.device)[None].expand(int(count),-1,-1))
        return torch.cat(parts)

    def _probability_tiles(self,x,extra,indices,encoded,baseline,contexts,low,offsets):
        origins=self._origins([c.path for c in contexts],np.diff(offsets),low);values=[]
        with torch.cuda.amp.autocast():
            for begin in range(0,len(x),self.tile_size):
                end=min(begin+self.tile_size,len(x));ids=torch.zeros(end-begin,dtype=torch.long,device=self.device);p,h=self.model.probabilities(x[begin:end],extra[begin:end],indices[begin:end],ids,encoded,origins[begin:end]);values.append(torch.cat([p.float(),h.float()[:,None],baseline[begin:end,None]],1))
        return torch.cat(values)

    def _network(self,xs,details):
        if not any(len(x)for x in xs):return[np.empty((0,4),np.float32)for x in xs]
        encoded,low=self._encoded_map(details);extras=self.range_kernel.compute(self.active_mesh,details,self.active_graph.keys,self.current_origin,self.current_rotation);origins=self._origins([d[2]for d in details],[len(x)for x in xs],low);result=[];offset=0
        for x,extra,(keys,_,path)in zip(xs,extras,details):
            indices=torch.from_numpy(np.ascontiguousarray(keys-low)).to(self.device);values=[]
            with torch.cuda.amp.autocast():
                for begin in range(0,len(x),self.tile_size):
                    end=min(begin+self.tile_size,len(x));ids=torch.zeros(end-begin,dtype=torch.long,device=self.device);p,h=self.model.probabilities(x[begin:end],extra[begin:end],indices[begin:end],ids,encoded,origins[offset+begin:offset+end]);values.append(torch.cat([p.float(),h.float()[:,None]],1).cpu().numpy())
            offset+=len(x);result.append(np.concatenate(values)if values else np.empty((0,4),np.float32))
        return result
