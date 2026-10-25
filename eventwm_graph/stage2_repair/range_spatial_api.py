"""Shared spatial model; legal sensor arrays in, hypothetical increments out."""
from collections import OrderedDict
import numpy as np
import torch
from .range_api import RangeEvidencePredictor
from .range_spatial import SpatialEvidenceNet
from .range_bev import RangeCanvas
from .diagnose import sha


class SpatialRangePredictor(RangeEvidencePredictor):
    model_class=SpatialEvidenceNet
    checkpoint_schema='legal_range_spatial_1'
    def __init__(self,checkpoint,*,device='cuda:0',threshold=None,**kwargs):
        saved=torch.load(checkpoint,map_location='cpu')
        if saved['schema']!=self.checkpoint_schema:raise ValueError('unsupported spatial range model')
        super().__init__(saved['config']['base_checkpoint'],device=device,threshold=saved.get('threshold',.2)if threshold is None else threshold,**kwargs);source=self.config['source'];self.config=dict(saved['config'],source=source);self.model=self.model_class().to(device).eval();self.model.load_state_dict(saved['model']);self.checkpoint_sha256=sha(checkpoint);self.predictor_schema=saved['schema'];self.decision_rule=saved.get('decision_rule','threshold');self.canvas=RangeCanvas(device);self.spatial_cache=OrderedDict()

    def clear_cache(self):
        super().clear_cache()
        if hasattr(self,'spatial_cache'):self.spatial_cache.clear()

    def predict_segments(self,*,partial_graph,trajectories,range_observations,track_history,materialize=True):
        # The parent validates protocol, timestamps and poses before it calls
        # _network. No training cache, future target or auxiliary mask is read.
        self.current_range_observations=range_observations;self.current_track_history=track_history
        return super().predict_segments(partial_graph=partial_graph,trajectories=trajectories,range_observations=range_observations,track_history=track_history,materialize=materialize)

    def _encoded_map(self,details):
        history=next(reversed(self.mesh_cache));assert self.mesh_cache[history]is self.active_mesh;key=(self.active_graph.identity,history)
        if key not in self.spatial_cache:
            p=self.current_range_observations;g=self.active_graph;ego=details[0][2][0,1:4]
            canvas,low=self.canvas.build(g.keys,g.values,g.heights,ego,g.resolution,np.asarray(p['ranges'],np.float32),np.asarray(p['origins'],np.float64),np.asarray(p['rotations'],np.float64),np.asarray(p['timestamps'],np.float64),self.current_track_history)
            with torch.cuda.amp.autocast():encoded=self.model.encode(torch.from_numpy(canvas.astype(np.float16))[None].to(self.device))
            self.spatial_cache[key]=(encoded,low)
            while len(self.spatial_cache)>2:self.spatial_cache.popitem(last=False)
        self.spatial_cache.move_to_end(key)
        return self.spatial_cache[key]

    def _network(self,xs,details):
        if not any(len(x)for x in xs):return [np.empty((0,4),np.float32)for x in xs]
        encoded,low=self._encoded_map(details);extras=self.range_kernel.compute(self.active_mesh,details,self.active_graph.keys,self.current_origin,self.current_rotation);result=[]
        for x,extra,(keys,_,path)in zip(xs,extras,details):
            ij=torch.from_numpy(np.ascontiguousarray(keys-low)).to(self.device);values=[]
            with torch.cuda.amp.autocast():
                for begin in range(0,len(x),self.tile_size):
                    end=min(begin+self.tile_size,len(x));ids=torch.zeros(end-begin,dtype=torch.long,device=self.device);p,h=self.model.probabilities(x[begin:end],extra[begin:end],ij[begin:end],ids,encoded);values.append(torch.cat([p.float(),h.float()[:,None]],1).cpu().numpy())
            result.append(np.concatenate(values)if values else np.empty((0,4),np.float32))
        return result
