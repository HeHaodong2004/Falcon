"""Complete API for a current-map encoding reused across all trial paths."""
import numpy as np
import torch
from .api import TimePredictor
from .shared_map import SharedMapNet,map_canvas
from .diagnose import sha


class SharedPredictor(TimePredictor):
    def __init__(self,checkpoint,*,device='cuda:0',threshold=None,increment_format='columns',**kwargs):
        saved=torch.load(checkpoint,map_location='cpu')
        if saved['schema']!='stage2_shared_map_1':raise ValueError('unsupported shared-map checkpoint')
        super().__init__(saved['config']['point_checkpoint'],device=device,threshold=threshold if threshold is not None else saved.get('threshold',.2),increment_format=increment_format,**kwargs)
        self.shared_config=saved.get('model_config',{});self.model=SharedMapNet(**self.shared_config).to(device).eval();self.model.load_state_dict(saved['model']);self.config=saved['config'];self.checkpoint_sha256=sha(checkpoint)

    def prepare_graph(self,graph):
        p=super().prepare_graph(graph)
        if not hasattr(p,'shared_map_encoded'):
            last=np.array([c['last_seen']for c in graph['cells']],np.float64)[p.input_order];ego=graph.get('ego_support')
            if ego is None:raise ValueError('shared map requires current legal ego support position')
            fine=self.shared_config.get('fine',False);d=map_canvas(p.keys,p.values,p.heights,last,float(graph['timestamp']),[p.resolution,p.height,float(graph['clearance'])],ego,anchor=fine,halo=128 if fine else 96)
            with torch.inference_mode():p.shared_map_encoded=self.model.encode(torch.from_numpy(d['maps'][None]).to(self.device),torch.tensor([d['maps'].shape[1:]],device=self.device),torch.from_numpy(d['fine'][None]).to(self.device)if fine else None)
            p.shared_map_lo=d['lo']
        return p

    def _projection(self,prepared,graph,path):return prepared

    def _network(self,xs,details):
        result=[np.empty((len(x),4),np.float32)for x in xs];pieces=[];used=0
        def flush():
            nonlocal pieces,used
            x=torch.cat([v[3]for v in pieces]);ij=torch.cat([v[4]for v in pieces]);p=details[pieces[0][0]][2]
            probability,height=self.model.probabilities(x,ij,torch.zeros(len(x),device=self.device,dtype=torch.long),p.shared_map_encoded);values=torch.cat([probability,height[:,None]],1).cpu().numpy();at=0
            for qi,start,end,xt,indices in pieces:result[qi][start:end]=values[at:at+len(xt)];at+=len(xt)
            pieces=[];used=0
        for qi,x in enumerate(xs):
            p=details[qi][2];ij=torch.from_numpy((details[qi][0]-p.shared_map_lo).astype(np.int64)).to(self.device);start=0
            while start<len(x):
                end=min(len(x),start+self.tile_size-used);pieces.append((qi,start,end,x[start:end],ij[start:end]));used+=end-start;start=end
                if used==self.tile_size:flush()
        if used:flush()
        return result
