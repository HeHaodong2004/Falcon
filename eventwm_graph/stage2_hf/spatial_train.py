"""Full-domain two-head spatial training with bounded legal-only streaming."""
import argparse,random,time,math
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import IterableDataset,DataLoader,get_worker_info
from .common import OUT,SEED,read,save,sha,source
from .packed import load_arrays,get_segment
from .spatial import SpatialNet,build

class Windows(IterableDataset):
 def __init__(self,manifest,epoch):self.manifest=manifest;self.epoch=epoch
 def __iter__(self):
  torch.set_num_threads(1);m=self.manifest;info=get_worker_info();worker=info.id if info else 0;workers=info.num_workers if info else 1;rng=np.random.default_rng(SEED+self.epoch);rows=m['rows'];counts=Counter(r['scene_id']for r in rows);order=rng.permutation(len(rows))[worker::workers]
  for ri in order:
   row=rows[int(ri)];a=load_arrays(row);n=len(row['segments']);weight=m['segments']/(len(counts)*counts[row['scene_id']]*n)
   for i in rng.permutation(n):
    values=get_segment(a,int(i));sample=build(*values[:4],values[-1],*values[4:7],make_query=False);sample['weight']=weight
    yield sample

def collate(samples):
 h=max(int(s['shape'][0])for s in samples);w=max(int(s['shape'][1])for s in samples);n=len(samples)
 arrays=dict(maps=np.zeros((n,6,h//2,w//2),np.float32),fine=np.zeros((n,3,h,w),np.float32),candidate=np.zeros((n,h,w),np.float32),labels=np.zeros((n,h,w),np.uint8),height=np.zeros((n,h,w),np.float32),weight=np.array([s['weight']for s in samples],np.float32))
 for i,s in enumerate(samples):
  a,b=map(int,s['shape'])
  arrays['maps'][i,:,:a//2,:b//2]=s['maps']
  for k in ('fine',):arrays[k][i,:,:a,:b]=s[k]
  for k in ('candidate','labels','height'):arrays[k][i,:a,:b]=s[k]
 from .spatial_cuda import path_arrays
 arrays.update(path_arrays([s['path']for s in samples],[s['lo']for s in samples],[s['shape']for s in samples],[s['resolution']for s in samples]))
 return {k:torch.from_numpy(v)for k,v in arrays.items()}

def loss_fn(out,b,pos_weight=8.,dice_weight=.1):
 y=b['labels'];mask=b['candidate'];r=(y>0).float();n=mask.sum((1,2)).clamp_min(1);nr=r.sum((1,2)).clamp_min(1)
 rev=F.binary_cross_entropy_with_logits(out[:,0].float(),r,reduction='none',pos_weight=out.new_tensor(pos_weight).float());rev=(rev*mask).sum((1,2))/n
 occ=F.binary_cross_entropy_with_logits(out[:,1].float(),(y==2).float(),reduction='none');occ=(occ*r).sum((1,2))/nr
 ph=F.smooth_l1_loss(out[:,2].float(),b['height'],reduction='none');ph=(ph*r).sum((1,2))/nr
 p=(out[:,0].float()-math.log(pos_weight)).sigmoid()*mask;dice=1-(2*(p*r).sum((1,2))+1)/(p.sum((1,2))+r.sum((1,2))+1)
 loss=rev+.15*occ+.05*ph+dice_weight*dice
 return (loss*b['weight']).mean(),torch.stack([rev.mean(),occ.mean(),ph.mean(),dice.mean()])

def fit(name,split='pilot_fit',epochs=8,width=12,dice=.1,device='cuda:0',workers=2,batch_size=16,warm=None,max_steps=None):
 torch.set_num_threads(1);torch.set_num_interop_threads(1);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 mpath=OUT/'packed'/split/'manifest.json';m=read(mpath);dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False)
 model=SpatialNet(width).to(device)
 from .spatial_cuda import QueryKernel
 kernel=QueryKernel(device)
 if warm:model.load_state_dict(torch.load(warm,map_location='cpu')['model'])
 config=dict(schema='stage2_hf_spatial_1',name=name,split=split,epochs=epochs,width=width,dice_weight=dice,pos_weight=8.,conditional_weight=.15,height_weight=.05,seed=SEED,device=device,batch_size=batch_size,workers=workers,manifest_sha256=sha(mpath),source=source(),warm=warm,learning_rate=.001,probability_conversion='Revelation logit minus log8, conditional sigmoid; no assumption threshold.5 optimal.',weighting='EqualScene/Root/window, full unchanged candidate cellmean revBCE; conditionalBCE/heightmean on revealedcells; allzero windows included.',precision='AMP FP16training, float32 probabilities; scientific inference initiallyfloat32',max_steps=max_steps)
 save(dest/'config.json',config);optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler();logs=[];tick=time.perf_counter();seen=0
 for epoch in range(1,epochs+1):
  model.train();loader=DataLoader(Windows(m,epoch),batch_size=batch_size,num_workers=workers,collate_fn=collate,pin_memory=True,prefetch_factor=2 if workers else None);total=np.zeros(4);steps=0;num=0;data_end=time.perf_counter();wait=0.
  for batch in loader:
   wait+=time.perf_counter()-data_end;b={k:v.to(device,non_blocking=True)for k,v in batch.items()};optim.zero_grad(set_to_none=True)
   query=kernel(b['maps'],b['path'],b['pre'],b['globals'],b['spec'])
   with torch.cuda.amp.autocast():out=model(b['maps'],query,b['fine']);loss,parts=loss_fn(out,b,8.,dice)
   if not torch.isfinite(loss):raise ValueError('nonfinite spatial training objective')
   scaler.scale(loss).backward();scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optim);scaler.update();total+=parts.detach().cpu().numpy();steps+=1;num+=len(b['weight']);seen+=len(b['weight']);data_end=time.perf_counter()
   if steps%100==0:print(dict(epoch=epoch,steps=steps,windows=num,seconds=time.perf_counter()-tick,data_wait_seconds=wait),flush=True)
   if max_steps and steps>=max_steps:break
  if not max_steps and num!=m['segments']:raise ValueError('full window traversal incomplete '+str((num,m['segments'])))
  log=dict(epoch=epoch,steps=steps,windows=num,seconds=time.perf_counter()-tick,loss_parts=(total/max(steps,1)).tolist(),data_wait_seconds=wait);logs.append(log);save(dest/'learning.json',logs)
  torch.save(dict(schema='stage2_hf_spatial_1',model_config=dict(width=width),model={k:v.detach().cpu()for k,v in model.state_dict().items()},config=config,epoch=epoch),dest/('epoch%02d.pt'%epoch));print(log,flush=True)
 save(dest/'complete.json',dict(seconds=time.perf_counter()-tick,windows_seen=seen,parameters=sum(v.numel()for v in model.parameters()),peak_cuda_allocated=torch.cuda.max_memory_allocated(device),peak_cuda_reserved=torch.cuda.max_memory_reserved(device)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--split',default='pilot_fit');p.add_argument('--epochs',type=int,default=8);p.add_argument('--width',type=int,default=12);p.add_argument('--dice',type=float,default=.1);p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=2);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--warm');p.add_argument('--max-steps',type=int);fit(**vars(p.parse_args()))
