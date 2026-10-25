"""A frozen full-feature prior plus a learned legal spatial revelation residual.

The CNN never sees the locations chosen by the label-stratified sampler.
Sampled positions are used only to gather outputs for weighted supervision.
"""
import argparse,math,random,time
from collections import defaultdict,Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset,DataLoader,get_worker_info
from .common import OUT,PREP,CKPT,CKPT_SHA,SEED,read,save,sha,source
from .spatial import SpatialNet,build
from .packed import load_arrays,get_segment
from .spatial_cuda import QueryKernel,path_arrays

class HybridNet(nn.Module):
 def __init__(self,width=12,train_occupancy=False):
  super().__init__();self.spatial=SpatialNet(width);self.prior=nn.Sequential(nn.Linear(88,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,4));self.train_occupancy=train_occupancy
  for p in self.prior.parameters():p.requires_grad_(False)
 def initialize(self,warm=None):
  saved=torch.load(CKPT,map_location='cpu');weights=saved['model']
  for i in (0,2,4):
   self.prior[i].weight.data.copy_(weights['net.%d.weight'%i][:,16:]if i==0 else weights['net.%d.weight'%i]);self.prior[i].bias.data.copy_(weights['net.%d.bias'%i])
  if warm:self.spatial.load_state_dict(torch.load(warm,map_location='cpu')['model'])
  nn.init.zeros_(self.spatial.detail[-1].weight);nn.init.zeros_(self.spatial.detail[-1].bias);self.spatial.detail[-1].bias.data[0]=math.log(8.)
 def forward(self,maps,query,fine,x,indices):
  field=self.spatial(maps,query,fine);batch=torch.arange(len(x),device=x.device)[:,None];delta=field.permute(0,2,3,1)[batch,indices[:,:,0],indices[:,:,1]]
  with torch.no_grad(),torch.cuda.amp.autocast(enabled=False):base=self.prior(x.float())
  r=base[:,:,1:3].logsumexp(-1)-base[:,:,0]+delta[:,:,0];b=base[:,:,2]-base[:,:,1]+(delta[:,:,1]if self.train_occupancy else 0)
  return r,b,delta

class HybridWindows(IterableDataset):
 def __init__(self,m,split,epoch,rank=0,world=1):self.manifest=m;self.split=split;self.epoch=epoch;self.rank=rank;self.world=world
 def __iter__(self):
  torch.set_num_threads(1);m=self.manifest;info=get_worker_info();worker=info.id if info else 0;nw=info.num_workers if info else 1;rng=np.random.default_rng(SEED+self.epoch);order=rng.permutation(len(m['shards']))[self.rank::self.world][worker::nw]
  # Scene masses come from original metadata only, never outcome counts.
  roots={e['root_id']:e['scene_id']for s in m['shards']for e in s['segments']};counts=Counter(roots.values());scale=m['counts']['candidate_segments']/len(counts)
  for si in order:
   shard=m['shards'][int(si)]
   with np.load(shard['path'],allow_pickle=False)as z:xx=z['x'];yy=z['y'];ww=z['weight']
   groups=defaultdict(list)
   for e in shard['segments']:groups[e['root_id']].append(e)
   for rid in rng.permutation(list(groups)):
    folder=OUT/'packed'/self.split;row=read(folder/(str(rid)+'.json'));a=load_arrays(row);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(row['segments'])}
    for j in rng.permutation(len(groups[rid])):
     e=groups[rid][int(j)];i=lookup[(e['start_node'],e['end_node'])];v=get_segment(a,i);d=build(*v[:4],v[-1],make_query=False,make_candidate=False)
     sl=slice(e['start'],e['end']);x=xx[sl,16:].astype(np.float32);y=yy[sl];weight=ww[sl].astype(np.float32);path=v[3];resolution=float(v[-1][0])
     # Encoded relative cell centers are <0.006m from their F32source,
     # versus >0.01m half-cell throughout the supported resolution range.
     centers=x[:,20:22].astype(np.float64)*10+path[0,[1,3]];keys=np.floor(centers/resolution).astype(np.int64);ij=keys-d['lo']
     if np.any(ij<0)or np.any(ij>=d['shape']):raise ValueError('sampled feature position outside dynamiccrop')
     truth={tuple(k):int(s)for k,s in zip(v[4],v[5])};got=np.array([truth.get(tuple(k),0)for k in keys],np.uint8)
     if not np.array_equal(got,y):raise ValueError('sampled coordinate/label identity mismatch')
     if np.any(d['fine'][0,ij[:,0],ij[:,1]]+d['fine'][1,ij[:,0],ij[:,1]]):raise ValueError('sampled knowncell outsideprediction scope')
     scene_mass=scale/counts[e['scene_id']];d.update(x=x,y=y,sample_ij=ij.astype(np.int64),sample_weight=weight*scene_mass,cell_weight=weight*e['segments_in_root'],window_weight=scene_mass/e['segments_in_root'],candidate_count=e['candidate_cells'])
     yield d

def collate(samples):
 n=len(samples);h=max(int(s['shape'][0])for s in samples);w=max(int(s['shape'][1])for s in samples);ns=max(len(s['y'])for s in samples)
 arrays=dict(maps=np.zeros((n,6,h//2,w//2),np.float32),fine=np.zeros((n,3,h,w),np.float32),x=np.zeros((n,ns,88),np.float32),y=np.zeros((n,ns),np.uint8),sample_ij=np.zeros((n,ns,2),np.int64),sample_weight=np.zeros((n,ns),np.float32),cell_weight=np.zeros((n,ns),np.float32),window_weight=np.array([s['window_weight']for s in samples],np.float32),candidate_count=np.array([s['candidate_count']for s in samples],np.float32))
 for j,s in enumerate(samples):
  a,b=map(int,s['shape']);k=len(s['y']);arrays['maps'][j,:,:a//2,:b//2]=s['maps'];arrays['fine'][j,:,:a,:b]=s['fine']
  for key in ('x','y','sample_ij','sample_weight','cell_weight'):arrays[key][j,:k]=s[key]
 arrays.update(path_arrays([s['path']for s in samples],[s['lo']for s in samples],[s['shape']for s in samples],[s['resolution']for s in samples]))
 return {k:torch.from_numpy(v)for k,v in arrays.items()}

def fit(name,split='pilot_fit',epochs=8,dice=.1,device='cuda:0',workers=2,batch_size=24,warm=None,train_occupancy=False,max_steps=None):
 torch.set_num_threads(1);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 path=PREP/(split+'_mixed_ray')/'manifest.json';m=read(path);dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False);model=HybridNet(train_occupancy=train_occupancy);model.initialize(warm);model.to(device);kernel=QueryKernel(device)
 cfg=dict(schema='stage2_hf_hybrid_1',epochs=epochs,split=split,device=device,workers=workers,batch_size=batch_size,width=12,dice_weight=dice,pos_weight=8.,train_occupancy=train_occupancy,seed=SEED,warm_spatial=warm,prior_checkpoint_sha256=sha(CKPT),source=source(),prepared_manifest_sha256=sha(path),packed_manifest_sha256=sha(OUT/'packed'/split/'manifest.json'),learning_rate=.0003,sampling='Original48revealed/48persistentUNKNOWN inverseinclusioncells; equalScene/Root/window; all originalwindows includingempty; samplerpositions are lossgathers only and neverCNNinputs.',objective='Weighted revelationBCEposweight8 + optionalsoftDice. Conditionaloccupancy frozen bydefault; correctionheadbiaslog8 cancels posteriorodds correction atinitialization.',decision='Selectthreshold onlyonDev; no protectedlabels',max_steps=max_steps)
 save(dest/'config.json',cfg);optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0003,weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler();logs=[];tick=time.perf_counter();seen=0
 for epoch in range(1,epochs+1):
  model.train();loader=DataLoader(HybridWindows(m,split,epoch),batch_size=batch_size,num_workers=workers,collate_fn=collate,pin_memory=True,prefetch_factor=2 if workers else None);sums=np.zeros(3);steps=0;num=0;wait=0.;end=time.perf_counter()
  for batch in loader:
   wait+=time.perf_counter()-end;b={k:v.to(device,non_blocking=True)for k,v in batch.items()};q=kernel(b['maps'],b['path'],b['pre'],b['globals'],b['spec']);optim.zero_grad(set_to_none=True)
   with torch.cuda.amp.autocast():
    rlog,blog,delta=model(b['maps'],q,b['fine'],b['x'],b['sample_ij']);rlog=rlog.float();blog=blog.float();target=(b['y']>0).float();rev=F.binary_cross_entropy_with_logits(rlog,target,reduction='none',pos_weight=rlog.new_tensor(8.));ce=(rev*b['sample_weight']).sum(1);p=(rlog-math.log(8)).sigmoid();w=b['cell_weight'];eps=1/b['candidate_count'].clamp_min(1);dl=1-(2*(p*target*w).sum(1)+eps)/((p*w).sum(1)+(target*w).sum(1)+eps)
    loss=(ce+dice*dl*b['window_weight']).mean()
    if train_occupancy:
     occ=F.binary_cross_entropy_with_logits(blog,(b['y']==2).float(),reduction='none');occ=(occ*target*w).sum(1)/(target*w).sum(1).clamp_min(1e-9);loss+=.1*(occ*b['window_weight']).mean()
   if not torch.isfinite(loss):raise ValueError('nonfinite hybrid objective')
   scaler.scale(loss).backward();scaler.unscale_(optim);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(optim);scaler.update();steps+=1;num+=len(target);seen+=len(target);sums+=[float(ce.mean()),float(dl.mean()),float(loss)];end=time.perf_counter()
   if steps%100==0:print(dict(epoch=epoch,steps=steps,windows=num,seconds=end-tick,wait=wait),flush=True)
   if max_steps and steps>=max_steps:break
  if not max_steps and num!=m['counts']['candidate_segments']:raise ValueError('hybrid full-window pass incomplete')
  log=dict(epoch=epoch,windows=num,steps=steps,seconds=time.perf_counter()-tick,loss_parts=(sums/max(steps,1)).tolist(),data_wait_seconds=wait);logs.append(log);save(dest/'learning.json',logs)
  torch.save(dict(schema=cfg['schema'],config=cfg,model_config=dict(width=12,train_occupancy=train_occupancy),model={k:v.detach().cpu()for k,v in model.state_dict().items()},epoch=epoch),dest/('epoch%02d.pt'%epoch));print(log,flush=True)
 save(dest/'complete.json',dict(seconds=time.perf_counter()-tick,windows_seen=seen,parameters=sum(p.numel()for p in model.parameters()),trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),peak_cuda_allocated=torch.cuda.max_memory_allocated(device)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--split',default='pilot_fit');p.add_argument('--epochs',type=int,default=8);p.add_argument('--dice',type=float,default=.1);p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=2);p.add_argument('--batch-size',type=int,default=24);p.add_argument('--warm');p.add_argument('--train-occupancy',action='store_true');p.add_argument('--max-steps',type=int);fit(**vars(p.parse_args()))
