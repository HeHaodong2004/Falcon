"""Complete point/recency predictor with one shared-map batched neural pass.

The deployed inputs are exactly the current legal graph and supplied Nx5
trajectory. Certificate-publication times are legal graph metadata, not a
claim about repeated camera visibility. Outputs remain hypothetical.
"""
from collections import OrderedDict
import hashlib
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from eventwm_graph.stage2_hf.common import CKPT, sha
from eventwm_graph.stage2_hf.time_model import TimeNet
from eventwm_graph.stage2_hf.time_features import moment_matrix,basis_torch
from eventwm_graph.stage2_hf.decisions import observation_probabilities
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.fused import transform
from eventwm_graph.stage2.common import SCHEMA


class TimePredictor(SegmentPredictor):
    def __init__(self, checkpoint, *, device='cuda:0', threshold=None, cache_size=4,
                 future_format='native', feature_threads=1, tile_size=65536):
        super().__init__(CKPT,cache_size=cache_size,execution='optimized',future_format=future_format,device=device,feature_threads=feature_threads)
        saved=torch.load(checkpoint,map_location='cpu')
        if saved['schema']!='stage2_hf_time_1':raise ValueError('unsupported point/recency checkpoint')
        self.model=TimeNet(**saved['model_config']).to(device).eval();self.model.load_state_dict(saved['model'])
        self.config=saved['config'];self.checkpoint_sha256=sha(checkpoint);self.threshold=float(saved.get('threshold',.2) if threshold is None else threshold)
        if not 0<=self.threshold<=1:raise ValueError('invalid revelation threshold')
        self.ablate_time=bool(self.config.get('ablate_time'));self.tile_size=int(tile_size)
        self.device=torch.device(device)

    def _projection(self,prepared,graph,path):
        if not hasattr(prepared,'time_projections'):prepared.time_projections=OrderedDict()
        key=(float(path[0,1]),float(path[0,3]))
        if key not in prepared.time_projections:
            if self.ablate_time:matrix=np.zeros((7,32),np.float32)
            else:
                last=np.array([c['last_seen'] for c in graph['cells']],np.float64)[prepared.input_order]
                matrix=moment_matrix(prepared.keys,prepared.values,last,float(path[0,0]),path[0,[1,3]],prepared.resolution)
            with torch.inference_mode():projection=torch.from_numpy(matrix).to(self.device)@self.model.correction[0].weight[:,88:].T
            prepared.time_projections[key]=projection
            while len(prepared.time_projections)>8:prepared.time_projections.popitem(last=False)
        return prepared.time_projections[key]

    def predict(self,*,partial_graph,trajectory,materialize=True):
        result=self.predict_segments(partial_graph=partial_graph,trajectories=[trajectory],materialize=materialize)[0]
        self.last_profile=dict(self.last_batch_profile)
        return result

    def _network(self,xs,details):
        """Tile across queries, sharing network launches while bounding memory."""
        returned=[np.empty((len(x),4),np.float32)for x in xs]
        pieces=[];used=0
        def flush():
            nonlocal pieces,used
            x=torch.cat([part[3]for part in pieces],dim=0)
            base=self.model.prior(x)
            h=F.linear(x,self.model.correction[0].weight[:,:88],self.model.correction[0].bias)
            at=0
            for qi,start,end,xt in pieces:
                h[at:at+len(xt)]+=basis_torch(xt)@details[qi][2];at+=len(xt)
            delta=self.model.correction[4](self.model.correction[3](self.model.correction[2](self.model.correction[1](h))))
            r=(base[:,1:3].logsumexp(-1)-base[:,0]+delta[:,0]-math.log(8.)).sigmoid()
            b=(base[:,2]-base[:,1]+(delta[:,1]if self.model.train_occupancy else 0)).sigmoid()
            values=torch.stack([1-r,r*(1-b),r*b,base[:,3]],dim=1).cpu().numpy();at=0
            for qi,start,end,xt in pieces:
                returned[qi][start:end]=values[at:at+len(xt)];at+=len(xt)
            pieces=[];used=0
        for qi,x in enumerate(xs):
            start=0
            while start<len(x):
                end=min(len(x),start+self.tile_size-used);pieces.append((qi,start,end,x[start:end]));used+=end-start;start=end
                if used==self.tile_size:flush()
        if used:flush()
        return returned

    def predict_segments(self,*,partial_graph,trajectories,materialize=True):
        start=time.perf_counter();prepared=self.prepare_graph(partial_graph);prepared_at=time.perf_counter()
        # Validation lives in the feature context, including strictly increasing
        # times and starting graph timestamp. Validate before any network call.
        contexts=[self.feature_class(partial_graph,p,prepared)for p in trajectories]
        if len(contexts)>1 and hasattr(prepared,'prewarm'):prepared.prewarm([c.path for c in contexts])
        contexts_at=time.perf_counter();xs=[];details=[];feature_profiles=[]
        with torch.inference_mode():
            for ctx in contexts:
                keys=ctx.candidates('both');x,baseline=transform(ctx,keys,cuda=self.cuda,device_output=self.cuda is not None)
                if self.cuda is None:x=torch.from_numpy(x).to(self.device)
                xs.append(x);details.append((keys,baseline,self._projection(prepared,partial_graph,ctx.path)))
                feature_profiles.append(dict(ctx.last_transform_profile))
            features_at=time.perf_counter()
            # Concatenation batches different trajectory lengths/extents while
            # preserving each exact candidate order. A bounded tile limits RAM.
            returned=self._network(xs,details)
        network_at=time.perf_counter();out=[];update_seconds=0.
        for ctx,(keys,baseline,_),values in zip(contexts,details,returned):
            prob=observation_probabilities(values[:,:3],no_future_observations=len(ctx.path)==1)
            heights=np.asarray(values[:,3]+baseline,np.float32)
            revelation=(prob[:,1]+prob[:,2]).astype(np.float64)
            states=np.where(revelation>=self.threshold,np.where(prob[:,1]>=prob[:,2],1,2),0).astype(np.uint8)
            now=float(ctx.path[-1,0]);updates=[dict(cell=keys[j].tolist(),state=int(states[j]),height=float(heights[j]),timestamp=now,source='predicted_stage2_research')for j in np.flatnonzero(states)]
            delta=dict(schema=SCHEMA,predictor_schema='stage2_hf_time_1',predicted=True,base_graph_identity=prepared.identity,
                base_timestamp=float(partial_graph['timestamp']),timestamp=now,robot_pose=ctx.path[-1,1:].tolist(),static_updates=updates,
                candidate_cells=keys,probabilities=prob,states=states,heights=heights,unknown_cells=keys[states==0],outside_domain='UNKNOWN_NO_UPDATE',checkpoint_sha256=self.checkpoint_sha256)
            if materialize:
                delta=self._with_base(delta,prepared);update_seconds+=self._complete(partial_graph,delta,ctx.path)
            else:
                delta.update(observation_scope='initially_UNKNOWN_static_cells_only',probability_columns=['UNKNOWN','FREE','BLOCKED'],provenance=dict(hypothetical=True))
            delta['provenance'].update(decision='revelation_threshold_then_conditional_label',revelation_threshold=self.threshold,
                recency_semantics='last certificate publication; repeated visibility is not recorded',training_source_commit=self.config['source']['commit'])
            out.append(delta)
        self.last_batch_profile=dict(queries=len(contexts),prepare_seconds=prepared_at-start,context_seconds=contexts_at-prepared_at,
            feature_seconds=features_at-contexts_at,network_transfers_probability_seconds=network_at-features_at,
            decode_update_seconds=time.perf_counter()-network_at,update_seconds=update_seconds,total_seconds=time.perf_counter()-start,
            materialized_future_graphs=len(out) if materialize else 0,feature_profiles=feature_profiles)
        return out

    def predict_batch(self,inputs,materialize=True):
        items=list(inputs);groups=OrderedDict();out=[None]*len(items)
        for i,item in enumerate(items):groups.setdefault(id(item['partial_graph']),[]).append(i)
        for ids in groups.values():
            results=self.predict_segments(partial_graph=items[ids[0]]['partial_graph'],trajectories=[items[i]['trajectory']for i in ids],materialize=materialize)
            for i,r in zip(ids,results):out[i]=r
        return out
