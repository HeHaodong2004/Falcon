"""Complete legal-only spatial Stage2 API, with shared-map multi-path execution."""
from collections import OrderedDict
import hashlib,time
import numpy as np
import torch
from torch.nn import functional as F
from .common import sha
from .spatial import SpatialNet,build,bounds,spatial_probabilities
from .spatial_cuda import QueryKernel,path_arrays
from eventwm_graph.stage2_segment.prepared import content_key,CompiledGraph
from eventwm_graph.stage2_segment.native import Base
from eventwm_graph.stage2_segment.update import apply_graph_delta
from eventwm_graph.stage2.common import SCHEMA as GRAPH_SCHEMA

class Predictor:
 def __init__(self,checkpoint,device='cuda:1',threshold=None,cache_size=4,future_format='native',precision='float32',query_backend='cuda'):
  torch.set_num_threads(1);self.device=torch.device(device);self.checkpoint=str(checkpoint);saved=torch.load(checkpoint,map_location='cpu')
  if saved['schema']not in ('stage2_hf_spatial_1','stage2_hf_hybrid_1'):raise ValueError('unsupported highfidelity checkpoint')
  if future_format not in ('native','snapshot'):raise ValueError('unsupported future graph representation')
  if precision not in ('float32','float16')or(precision=='float16'and self.device.type!='cuda'):raise ValueError('unsupported precision')
  self.hybrid=saved['schema']=='stage2_hf_hybrid_1';self.prior_cuda=None;self.prior=None
  if self.hybrid:
   from .hybrid import HybridNet
   combined=HybridNet(**saved['model_config']);combined.load_state_dict(saved['model']);self.model=combined.spatial.to(self.device).eval();self.train_occupancy=combined.train_occupancy
   if self.device.type=='cuda':
    from eventwm_graph.stage2_segment.cuda import CudaExecution
    self.prior_cuda=CudaExecution(combined.prior,self.device);self.prior=self.prior_cuda.net
   else:self.prior=combined.prior.eval()
  else:self.model=SpatialNet(**saved['model_config']).to(self.device).eval();self.model.load_state_dict(saved['model'])
  self.config=saved['config'];self.checkpoint_sha256=sha(checkpoint);self.pos_weight=float(self.config['pos_weight']);self.calibration=saved.get('calibration')
  self.threshold=float(threshold if threshold is not None else saved.get('threshold',.2));self.dtype=torch.float16 if precision=='float16'else torch.float32
  if not 0<=self.threshold<=1:raise ValueError('invalid revelation threshold')
  self.model.to(dtype=self.dtype);self.future_format=future_format;self.precision=precision;self.query_backend=query_backend;self.kernel=QueryKernel(self.device)if self.device.type=='cuda'and query_backend=='cuda'else None
  torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  self.cache=OrderedDict();self.cache_size=cache_size;self.cache_hits=0;self.cache_misses=0;self.last_profile={};self.last_batch_profile={}
 def clear_cache(self):self.cache.clear()
 def prepare_graph(self,graph):
  key=content_key(graph)
  if key in self.cache:self.cache_hits+=1;self.cache.move_to_end(key);return self.cache[key]
  graph_class=CompiledGraph
  if self.hybrid:
   from eventwm_graph.stage2_segment.batch import SharedCompiledGraph
   graph_class=SharedCompiledGraph
  self.cache_misses+=1;p=graph_class(graph);p.hf_maps=OrderedDict();p.hf_encoded=OrderedDict();p.future_base=None
  if p.native_eligible:
   try:p.future_base=Base(graph,p)
   except(TypeError,ValueError,OverflowError):pass
  if self.cache_size:
   self.cache[key]=p
   while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
  return p
 def _features(self,p,path):
  from eventwm_graph.stage2_quality.features import candidate_mask
  lo,shape=bounds(path,p.resolution);key=tuple(lo)+tuple(shape)+(float(path[0,2]),)
  if key not in p.hf_maps:
   d=build(p.keys,p.values,p.heights,path,np.array([p.resolution,p.height,0.]),make_query=self.kernel is None)
   unknown=np.argwhere((d['fine'][0]+d['fine'][1])==0).astype(np.int32)+lo.astype(np.int32)
   p.hf_maps[key]=dict(maps=d['maps'],fine=d['fine'],unknown=unknown)
   while len(p.hf_maps)>8:p.hf_maps.popitem(last=False)
  else:
   static=p.hf_maps[key];unknown=static['unknown'];keep=candidate_mask(unknown,np.unique(path[:,[1,3]].astype(np.float32),axis=0),p.resolution)
   d=dict(maps=static['maps'],fine=static['fine'],keys=unknown[keep],lo=lo,shape=shape,path=path,resolution=p.resolution)
   if self.kernel is None:
    from .spatial import trajectory_features
    d['query']=trajectory_features(lo,(int(shape[0]//2),int(shape[1]//2)),p.resolution,path,d['maps'][0],d['maps'][1])
  d['map_key']=key;return d
 def _complete(self,graph,p,delta):
  future=p.future_base.update(delta)if p.future_base is not None else None
  if future is None:future=apply_graph_delta(graph,delta,verified_identity=p.identity)
  if self.future_format=='snapshot'and hasattr(future,'to_snapshot'):future=future.to_snapshot()
  delta['future_partial_graph']=future;delta['future_format']='native_columns'if hasattr(future,'cell_columns')else'snapshot'
 def predict(self,*,partial_graph,trajectory,materialize=True):
  result=self.predict_segments(partial_graph=partial_graph,trajectories=[trajectory],materialize=materialize)[0];self.last_profile=dict(self.last_batch_profile);return result
 def predict_segments(self,*,partial_graph,trajectories,materialize=True):
  start=time.perf_counter();p=self.prepare_graph(partial_graph);prepared=time.perf_counter();paths=[]
  for value in trajectories:
   path=np.asarray(value,np.float64)
   if path.ndim!=2 or path.shape[1]!=5 or not len(path)or not np.isfinite(path).all()or np.any(np.diff(path[:,0])<=0)or abs(path[0,0]-float(partial_graph['timestamp']))>1e-7:raise ValueError('invalid supplied pose/time trajectory')
   paths.append(path)
  ds=[self._features(p,path)for path in paths];featured=time.perf_counter();out=[];gpu_ms=0.;network_seconds=0.;output_seconds=0.;graph_seconds=0.;prior_seconds=0.;priors={}
  if self.hybrid:
   from eventwm_graph.stage2_segment.features import SegmentFeatures
   from eventwm_graph.stage2_segment.fused import transform
   tick=time.perf_counter()
   if len(paths)>1:p.prewarm(paths)
   with torch.inference_mode():
    for i,(d,path)in enumerate(zip(ds,paths)):
     context=SegmentFeatures(partial_graph,path,p);context._candidate_keys=d['keys']
     x,baseline=transform(context,d['keys'],cuda=self.prior_cuda,device_output=self.prior_cuda is not None)
     if self.prior_cuda is None:x=torch.from_numpy(x)
     priors[i]=(self.prior(x),torch.from_numpy(baseline).to(self.device))
   prior_seconds=time.perf_counter()-tick
  # Bound GPU working memory; every requested query is still evaluated fully.
  groups=[];group=[];pixels=0
  for i,d in enumerate(ds):
   size=int(np.prod(d['shape']))
   if group and pixels+size>2000000:groups.append(group);group=[];pixels=0
   group.append(i);pixels+=size
  if group:groups.append(group)
  results={}
  with torch.inference_mode():
   for ids in groups:
    batch=[ds[i]for i in ids];h=max(int(d['shape'][0])for d in batch);w=max(int(d['shape'][1])for d in batch);n=len(batch);t=time.perf_counter()
    maps=np.zeros((n,6,h//2,w//2),np.float32);fine=np.zeros((n,3,h,w),np.float32);encoded=[]
    for j,d in enumerate(batch):
     a,b=map(int,d['shape']);maps[j,:,:a//2,:b//2]=d['maps'];fine[j,:,:a,:b]=d['fine']
     key=d['map_key']
     if key not in p.hf_encoded:
      mt=torch.from_numpy(d['maps'][None]).to(self.device,dtype=self.dtype);p.hf_encoded[key]=self.model.encode_map(mt)
      while len(p.hf_encoded)>8:p.hf_encoded.popitem(last=False)
     e=p.hf_encoded[key];encoded.append(F.pad(e,(0,w//2-e.shape[-1],0,h//2-e.shape[-2])))
    mt=torch.from_numpy(maps).to(self.device);ft=torch.from_numpy(fine).to(self.device,dtype=self.dtype)
    if self.kernel:
     arrays=path_arrays([d['path']for d in batch],[d['lo']for d in batch],[d['shape']for d in batch],[d['resolution']for d in batch]);tt={k:torch.from_numpy(v).to(self.device)for k,v in arrays.items()}
     ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True);ev0.record(torch.cuda.current_stream(self.device));query=self.kernel(mt,tt['path'],tt['pre'],tt['globals'],tt['spec'])
    else:
     query=torch.zeros((n,32,h//2,w//2),device=self.device)
     for j,d in enumerate(batch):a,b=map(int,d['shape']);query[j,:,:a//2,:b//2]=torch.from_numpy(d['query']).to(self.device)
    raw=self.model(mt.to(dtype=self.dtype),query.to(dtype=self.dtype),ft,encoded=torch.cat(encoded)).float()
    packed=[]
    for j,d in enumerate(batch):
     ci=d['keys']-d['lo'];ii=torch.from_numpy(ci).to(self.device);values=raw[j,:,ii[:,0],ii[:,1]].T
     if self.hybrid:
      base,baseline=priors[ids[j]];rr=base[:,1:3].logsumexp(-1)-base[:,0]+values[:,0]-np.log(self.pos_weight);bb=base[:,2]-base[:,1]+(values[:,1]if self.train_occupancy else 0);height=base[:,3]+baseline
     else:rr=values[:,0]-np.log(self.pos_weight);bb=values[:,1];height=values[:,2]+float(d['path'][0,2])
     if self.calibration:rr=rr*self.calibration['rev_scale']+self.calibration['rev_bias'];bb=bb*self.calibration['blocked_scale']+self.calibration['blocked_bias']
     r=rr.sigmoid();b=bb.sigmoid();packed.append(torch.stack([1-r,r*(1-b),r*b,height],-1))
    if self.kernel:ev1.record(torch.cuda.current_stream(self.device))
    returned=[v.cpu().numpy()for v in packed]
    if self.kernel:gpu_ms+=ev0.elapsed_time(ev1)
    network_seconds+=time.perf_counter()-t
    for i,values in zip(ids,returned):results[i]=values
  for i,(d,path)in enumerate(zip(ds,paths)):
   t=time.perf_counter();values=results[i];prob=np.ascontiguousarray(values[:,:3]);height=np.ascontiguousarray(values[:,3]);keys=d['keys']
   if len(path)==1:prob.fill(0);prob[:,0]=1
   if not np.isfinite(prob).all()or not np.isfinite(height).all():raise ValueError('nonfinite model output')
   rev=(prob[:,1]+prob[:,2]).astype(np.float64);states=np.where(rev>=self.threshold,np.where(prob[:,1]>=prob[:,2],1,2),0).astype(np.uint8);now=float(path[-1,0]);chosen=np.flatnonzero(states)
   updates=[dict(cell=keys[j].tolist(),state=int(states[j]),height=float(height[j]),timestamp=now,source='predicted_stage2_research')for j in chosen]
   delta=dict(schema=GRAPH_SCHEMA,predictor_schema='high_fidelity_segment_legal_1',prediction_task='segment_conditioned_new_legal_observations',predicted=True,base_graph_identity=p.identity,base_timestamp=float(partial_graph['timestamp']),timestamp=now,robot_pose=path[-1,1:].tolist(),static_updates=updates,candidate_cells=keys,probabilities=prob,states=states,heights=height,unknown_cells=keys[states==0],probability_columns=['UNKNOWN','FREE','BLOCKED'],outside_domain='UNKNOWN_NO_UPDATE',observation_scope='initially_UNKNOWN_static_cells_only',checkpoint_sha256=self.checkpoint_sha256,
    provenance=dict(hypothetical=True,input='current_legal_graph_and_supplied_pose_time_segment',checkpoint_sha256=self.checkpoint_sha256,decision='revelation_threshold_then_conditional_label',revelation_threshold=self.threshold,precision=self.precision,trajectory_sha256=hashlib.sha256(path.astype('<f8').tobytes()).hexdigest(),training_source_commit=self.config['source']['commit'],temporal_semantics='terminal-time hypothetical additions; real observations never replaced'))
   output_seconds+=time.perf_counter()-t
   if materialize:t=time.perf_counter();self._complete(partial_graph,p,delta);graph_seconds+=time.perf_counter()-t
   out.append(delta)
  self.last_batch_profile=dict(queries=len(paths),prepare_seconds=prepared-start,map_candidate_seconds=featured-prepared,prior_features_neural_seconds=prior_seconds,network_transfers_probability_seconds=network_seconds,gpu_stream_milliseconds=gpu_ms,delta_seconds=output_seconds,update_seconds=graph_seconds,total_seconds=time.perf_counter()-start,materialized_future_graphs=len(out)if materialize else 0,cache_hits=self.cache_hits,cache_misses=self.cache_misses,neural_groups=len(groups))
  return out
 def predict_batch(self,inputs,materialize=True):
  groups=OrderedDict();items=list(inputs);out=[None]*len(items)
  for i,item in enumerate(items):groups.setdefault(id(item['partial_graph']),[]).append(i)
  for ids in groups.values():
   ds=self.predict_segments(partial_graph=items[ids[0]]['partial_graph'],trajectories=[items[i]['trajectory']for i in ids],materialize=materialize)
   for i,d in zip(ids,ds):out[i]=d
  return out
