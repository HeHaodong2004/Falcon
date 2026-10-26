"""Bounded neural tiles spanning all queries sharing one encoded current map."""
import numpy as np
import torch
from .range_device_canvas import DeviceCanvasSpatialPredictor


class BatchedSpatialPredictor(DeviceCanvasSpatialPredictor):
    def _network(self,xs,details):
        lengths=[len(x)for x in xs]
        if not any(lengths):return [np.empty((0,4),np.float32)for _ in xs]
        encoded,low=self._encoded_map(details)
        # Coordinate staging precedes long geometry kernels. All coordinates
        # remain original integer keys, including out-of-canvas candidates.
        coordinates=torch.from_numpy(np.ascontiguousarray(np.concatenate([d[0]-low for d in details]))).to(self.device,non_blocking=True)
        extras=self.range_kernel.compute(self.active_mesh,details,self.active_graph.keys,self.current_origin,self.current_rotation)
        pieces=[];used=0;cursor=0;outputs=[]
        def flush():
            nonlocal pieces,used,cursor
            x=torch.cat([a for a,b in pieces]);extra=torch.cat([b for a,b in pieces]);ij=coordinates[cursor:cursor+used];ids=torch.zeros(used,dtype=torch.long,device=self.device)
            with torch.cuda.amp.autocast():p,h=self.model.probabilities(x,extra,ij,ids,encoded)
            outputs.append(torch.cat([p.float(),h.float()[:,None]],1));cursor+=used;pieces=[];used=0
        for x,extra in zip(xs,extras):
            begin=0
            while begin<len(x):
                end=min(len(x),begin+self.tile_size-used);pieces.append((x[begin:end],extra[begin:end]));used+=end-begin;begin=end
                if used==self.tile_size:flush()
        if used:flush()
        flat=torch.cat(outputs).cpu().numpy();offset=np.r_[0,np.cumsum(lengths)]
        return [flat[offset[i]:offset[i+1]]for i in range(len(xs))]
