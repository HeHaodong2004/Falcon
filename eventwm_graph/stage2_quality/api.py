"""CPU API with reusable legal-map preprocessing and unaltered graph semantics."""
from collections import OrderedDict
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .geometry import graph_identity
from .update import apply_graph_delta
from eventwm_graph.stage2.common import SCHEMA as GRAPH_SCHEMA,sha
from .common import SCHEMA
from .features import PreparedGraph,Features
from .model import GraphDeltaNet


def active_columns(mode):
    if mode=='baseline':return np.arange(47)
    if mode=='trajectory':return np.arange(16,67)
    if mode=='map':return np.r_[0:47,67:92]
    if mode=='both':return np.arange(16,92)
    raise ValueError('unsupported checkpoint mode')


class Stage2Predictor:
    def __init__(self,checkpoint,threads=1,cache_size=4):
        if threads is not None:torch.set_num_threads(threads)
        self.checkpoint=Path(checkpoint);saved=torch.load(checkpoint,map_location='cpu')
        if saved.get('schema')!=SCHEMA:raise ValueError('unsupported successor checkpoint')
        model=GraphDeltaNet(**saved['model_config']);model.load_state_dict(saved['model']);model.eval()
        self.mode=saved['mode'];self.config=saved['config'];self.checkpoint_sha256=sha(checkpoint)
        self.active=active_columns(self.mode)
        self.feature_count=saved['model_config'].get('input_features',92)
        if self.feature_count>92:
            if self.mode!='both':raise ValueError('extended features require both mode')
            if type(self) is Stage2Predictor:raise ValueError('extended checkpoints require stage2_segment.api.SegmentPredictor')
            self.active=np.r_[self.active,np.arange(92,self.feature_count)]
        # Inactive input columns are exact zeros by the experiment contract.
        # Remove their multiplies only; preserve all learned active weights.
        self.net=nn.Sequential(nn.Linear(len(self.active),48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,4))
        with torch.no_grad():
            self.net[0].weight.copy_(model.net[0].weight[:,self.active]);self.net[0].bias.copy_(model.net[0].bias)
            for layer in (2,4):self.net[layer].load_state_dict(model.net[layer].state_dict())
        self.net.eval();self.cache_size=cache_size;self.cache=OrderedDict();self.cache_hits=0;self.cache_misses=0;self.last_profile=None;self.last_batch_profile=None

    def clear_cache(self):self.cache.clear()

    def prepare_graph(self,graph):
        if graph.get('schema')!='causal_partial_cspace_graph_1':raise ValueError('unsupported PartialGraph schema')
        key=graph_identity(graph)
        if key in self.cache:
            self.cache_hits+=1;self.cache.move_to_end(key);return self.cache[key]
        self.cache_misses+=1;prepared=PreparedGraph(graph,graph_key=key)
        if self.cache_size:
            self.cache[key]=prepared
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return prepared

    def _predict(self,graph,trajectory,prepared):
        start=time.perf_counter();context=getattr(self,'feature_class',Features)(graph,trajectory,prepared);spatial=time.perf_counter()
        keys=context.candidates(self.mode);candidates=time.perf_counter()
        x,baseline=context.transform(keys,self.mode);features=time.perf_counter()
        # Gather only the exact nonzero columns, preserving full probability API.
        if self.mode=='both':x=np.ascontiguousarray(x[:,16:self.feature_count])
        elif self.mode=='trajectory':x=np.ascontiguousarray(x[:,16:67])
        elif self.mode=='baseline':x=np.ascontiguousarray(x[:,:47])
        else:
            compact=np.empty((len(keys),72),np.float32);compact[:,:47]=x[:,:47];compact[:,47:]=x[:,67:92];x=compact
        p=np.empty((len(keys),3),np.float32);height=baseline.copy()
        with torch.inference_mode():
            for begin in range(0,len(keys),8192):
                end=min(begin+8192,len(keys));values=self.net(torch.from_numpy(x[begin:end]));p[begin:end]=torch.softmax(values[:,:3],-1).numpy();height[begin:end]+=values[:,3].numpy()
        inferred=time.perf_counter()
        if len(context.path)==1:p.fill(0);p[:,0]=1
        states=p.argmax(-1).astype(np.uint8);now=float(context.path[-1,0]);selected=np.flatnonzero(states)
        updates=[dict(cell=keys[i].tolist(),state=int(states[i]),height=float(height[i]),timestamp=now,source='predicted_stage2_research') for i in selected]
        delta=dict(schema=GRAPH_SCHEMA,predictor_schema=SCHEMA,predicted=True,base_graph_identity=prepared.identity,
            base_timestamp=float(graph['timestamp']),timestamp=now,robot_pose=context.path[-1,1:].tolist(),
            static_updates=updates,candidate_cells=keys,probabilities=p,states=states,heights=height,unknown_cells=keys[states==0],
            outside_domain='UNKNOWN_NO_UPDATE',checkpoint_sha256=self.checkpoint_sha256)
        end=time.perf_counter()
        profile=dict(spatial_seconds=spatial-start,candidate_seconds=candidates-spatial,feature_seconds=features-candidates,
            neural_seconds=inferred-features,delta_seconds=end-inferred,total_seconds=end-start)
        profile.update(context.last_transform_profile)
        self.last_profile=profile;return delta

    def predict(self,*,partial_graph,trajectory):
        start=time.perf_counter();prepared=self.prepare_graph(partial_graph);ready=time.perf_counter()
        delta=self._predict(partial_graph,trajectory,prepared)
        self.last_profile['prepare_seconds']=ready-start;self.last_profile['total_seconds']=time.perf_counter()-start
        return delta

    def predict_batch(self,inputs):
        # Same-object reuse is confined to a synchronous invocation: no caller
        # mutation occurs between Query items. Cross-call lookup rehashes content.
        prepared={};out=[];before_hits=self.cache_hits;before_misses=self.cache_misses;spatial_before={}
        for item in inputs:
            graph=item['partial_graph'];key=id(graph)
            if key not in prepared:
                prepared[key]=self.prepare_graph(graph);p=prepared[key]
                if id(p) not in spatial_before:spatial_before[id(p)]=(p,p.crop_hits,p.crop_misses,p.integrals is not None)
            out.append(self._predict(graph,item['trajectory'],prepared[key]))
        self.last_batch_profile=dict(queries=len(out),distinct_input_graph_objects=len(prepared),
            graph_cache_hits=self.cache_hits-before_hits,graph_cache_misses=self.cache_misses-before_misses,
            within_batch_geometry_reuses=len(out)-(self.cache_hits-before_hits+self.cache_misses-before_misses),
            crop_cache_hits=sum(p.crop_hits-h for p,h,m,i in spatial_before.values()),crop_cache_misses=sum(p.crop_misses-m for p,h,m,i in spatial_before.values()),
            map_integral_builds=sum(int(p.integrals is not None and not had) for p,h,m,had in spatial_before.values()))
        return out
