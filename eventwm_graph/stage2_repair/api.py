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
from .materialize import future_from_columns
from .decode import decode_columns


class TimePredictor(SegmentPredictor):
    def __init__(self, checkpoint, *, device='cuda:0', threshold=None, cache_size=4,
                 future_format='native', feature_threads=1, tile_size=65536,increment_format='list'):
        super().__init__(CKPT,cache_size=cache_size,execution='optimized',future_format=future_format,device=device,feature_threads=feature_threads)
        saved=torch.load(checkpoint,map_location='cpu')
        if saved['schema']!='stage2_hf_time_1':raise ValueError('unsupported point/recency checkpoint')
        self.model=TimeNet(**saved['model_config']).to(device).eval();self.model.load_state_dict(saved['model'])
        self.config=saved['config'];self.checkpoint_sha256=sha(checkpoint);self.threshold=float(saved.get('threshold',.2) if threshold is None else threshold)
        if not 0<=self.threshold<=1:raise ValueError('invalid revelation threshold')
        self.ablate_time=bool(self.config.get('ablate_time'));self.tile_size=int(tile_size)
        self.device=torch.device(device)
        if increment_format not in ('list','columns'):raise ValueError('unsupported sparse increment format')
        self.increment_format=increment_format
        self.predictor_schema='stage2_hf_time_1'

    def prepare_graph(self,graph):
        prepared=super().prepare_graph(graph)
        # Six queries must fit without evicting their own crops. This remains
        # bounded by eight entries and the existing four-million-cell guard.
        prepared.crop_cache_size=8;prepared.device_cache_size=8
        return prepared

    def _projection(self,prepared,graph,path):
        if self.ablate_time:return None
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

    def _decode(self,values,baseline,keys,no_future):
        return decode_columns(values,baseline,keys,self.threshold,no_future)

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
            if not self.ablate_time:
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

    def _candidate_keys(self,prepared,contexts):
        # Only map UNKNOWN-grid construction is retained between calls.
        # Candidate masks share only exact float32 supplied positions.
        from eventwm_graph.stage2_quality.features import candidate_mask
        if not hasattr(prepared,'unknown_grids'):prepared.unknown_grids=OrderedDict()
        domains={};keysets=[]
        for ctx in contexts:
            gridlo=getattr(ctx,'candidate_lo',ctx.lo);gridshape=getattr(ctx,'candidate_shape',ctx.shape);gridstates=getattr(ctx,'candidate_states',ctx.states)
            gridkey=tuple(gridlo)+tuple(gridshape)
            if gridkey not in prepared.unknown_grids:
                prepared.unknown_grids[gridkey]=np.argwhere(gridstates==0).astype(np.int32)+gridlo
                while len(prepared.unknown_grids)>8:prepared.unknown_grids.popitem(last=False)
            xy=np.unique(ctx.xy,axis=0);domainkey=(gridkey,xy.tobytes())
            if domainkey not in domains:
                unknown=prepared.unknown_grids[gridkey];domains[domainkey]=unknown[candidate_mask(unknown,xy,ctx.resolution)]
            keys=domains[domainkey];ctx._candidate_keys=keys;keysets.append(keys)
        return keysets

    def _forward_batch(self,prepared,contexts,partial_graph):
        xs=[];details=[];feature_profiles=[]
        with torch.inference_mode():
            keysets=self._candidate_keys(prepared,contexts)
            if self.cuda is not None:
                from .features import transform_batch
                xs,baselines,profile=transform_batch(self.cuda,contexts,keysets);feature_profiles.append(profile)
            else:
                baselines=[]
                for ctx,keys in zip(contexts,keysets):
                    x,baseline=transform(ctx,keys);xs.append(torch.from_numpy(x).to(self.device));baselines.append(baseline)
            for ctx,keys,baseline in zip(contexts,keysets,baselines):details.append((keys,baseline,self._projection(prepared,partial_graph,ctx.path)))
            features_at=time.perf_counter();returned=self._network(xs,details)
        return details,returned,features_at,feature_profiles

    def predict_segments(self,*,partial_graph,trajectories,materialize=True):
        start=time.perf_counter();prepared=self.prepare_graph(partial_graph);prepared_at=time.perf_counter()
        # Validation lives in the feature context, including strictly increasing
        # times and starting graph timestamp. Validate before any network call.
        trajectories=list(trajectories)
        if len(trajectories)>1 and hasattr(prepared,'prewarm'):prepared.prewarm(trajectories)
        contexts=[self.feature_class(partial_graph,p,prepared)for p in trajectories]
        contexts_at=time.perf_counter();details,returned,features_at,feature_profiles=self._forward_batch(prepared,contexts,partial_graph)
        network_at=time.perf_counter();out=[];update_seconds=0.
        for ctx,(keys,baseline,_),values in zip(contexts,details,returned):
            prob,heights,states,newkeys,newstates,newheights,unknown=self._decode(values,baseline,keys,len(ctx.path)==1)
            now=float(ctx.path[-1,0])
            if self.increment_format=='columns':
                from .increment import SparseIncrement
                updates=SparseIncrement(newkeys,newstates,newheights,now)
            else:updates=[dict(cell=k,state=s,height=h,timestamp=now,source='predicted_stage2_research')for k,s,h in zip(newkeys.tolist(),newstates.tolist(),newheights.tolist())]
            delta=dict(schema=SCHEMA,predictor_schema=self.predictor_schema,predicted=True,base_graph_identity=prepared.identity,
                base_timestamp=float(partial_graph['timestamp']),timestamp=now,robot_pose=ctx.path[-1,1:].tolist(),static_updates=updates,
                candidate_cells=keys,probabilities=prob,states=states,heights=heights,unknown_cells=unknown,outside_domain='UNKNOWN_NO_UPDATE',checkpoint_sha256=self.checkpoint_sha256,
                increment_format=self.increment_format)
            if materialize:
                t=time.perf_counter();future=future_from_columns(getattr(prepared,'future_base',None),newkeys,newstates,newheights,now,ctx.path[-1,1:].tolist(),self.checkpoint_sha256)
                if future is None:
                    delta=self._with_base(delta,prepared);self._complete(partial_graph,delta,ctx.path)
                else:
                    delta.update(future_partial_graph=future,future_format='native_columns',prediction_task='segment_conditioned_legal_observation_1',
                        observation_scope='initially_UNKNOWN_static_cells_only',probability_columns=['UNKNOWN','FREE','BLOCKED'],
                        provenance=dict(hypothetical=True,input='current_legal_graph_and_supplied_pose_time_segment',checkpoint_sha256=self.checkpoint_sha256,
                            trajectory_sha256=hashlib.sha256(ctx.path.astype('<f8').tobytes()).hexdigest(),temporal_semantics='terminal-time hypothetical additions; real observations never replaced'))
                update_seconds+=time.perf_counter()-t
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
