import argparse,random,time
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from .common import OUT,PREP,SEED,CKPT_SHA,read,save,sha,source
from .time_model import TimeNet

def fit(name,split='pilot_fit',epochs=8,train_occupancy=False,device='cuda:0',warm=None,ablate_time=False):
 torch.set_num_threads(1);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False
 path=PREP/(split+'_mixed_ray')/'manifest.json';m=read(path);tp=OUT/'time_prepared'/split/'manifest.json';tm=read(tp)
 if tm['source_manifest_sha256']!=sha(path)or tm['rows']!=m['sampled_cells']:raise ValueError('timefeature/data identity mismatch')
 dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False);model=TimeNet(train_occupancy);model.initialize()
 if warm:model.load_state_dict(torch.load(warm,map_location='cpu')['model'])
 model.to(device);roots={e['root_id']:e['scene_id']for s in m['shards']for e in s['segments']};scenes=Counter(roots.values());scene_factor={s:len(roots)/(len(scenes)*n)for s,n in scenes.items()}
 cfg=dict(schema='stage2_hf_time_1',epochs=epochs,split=split,seed=SEED,device=device,train_occupancy=train_occupancy,width=64,pos_weight=8.,learning_rate=.0003,batch_size=16384,weight_decay=.0001,source=source(),prepared_manifest_sha256=sha(path),time_manifest_sha256=sha(tp),prior_checkpoint_sha256=CKPT_SHA,warm=warm,ablate_time=ablate_time,loss='Inverseinclusion equalScene/Root/window weighted revelationBCEpos8 +8*R*conditionalBCE whenenabled; no focal/Dice; correct revelationodds by8 at inference.',inputs='Frozen legal88features +32causal recencyangular covariates; no absoluteclock, hiddenmap, camera or humanstates',height='Frozen original conditional pointhead')
 save(dest/'config.json',cfg);optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0003,weight_decay=.0001);rng=np.random.default_rng(SEED);scale=m['sampled_cells']/m['roots'];logs=[];tick=time.perf_counter()
 for epoch in range(1,epochs+1):
  total=np.zeros(3);model.train()
  for si in rng.permutation(len(m['shards'])):
   shard=m['shards'][int(si)];tr=tm['shards'][int(si)]
   if tr['source_shard_sha256']!=shard['sha256']:raise ValueError('misaligned original/time shards')
   if epoch==1 and(sha(shard['path'])!=shard['sha256']or sha(tr['path'])!=tr['sha256']):raise ValueError('training source checksum mismatch')
   with np.load(shard['path'],allow_pickle=False)as z:x=z['x'][:,16:].astype(np.float32);y=z['y'].astype(np.int64);w=z['weight'].astype(np.float32)
   with np.load(tr['path'],allow_pickle=False)as z:extra=z['extra'].astype(np.float32)
   if ablate_time:extra.fill(0)
   for e in shard['segments']:w[e['start']:e['end']]*=scene_factor[e['scene_id']]
   order=rng.permutation(len(y))
   for begin in range(0,len(order),16384):
    ids=order[begin:begin+16384];xt=torch.from_numpy(x[ids]).to(device);et=torch.from_numpy(extra[ids]).to(device);yt=torch.from_numpy(y[ids]).to(device);wt=torch.from_numpy(w[ids]).to(device)*scale;r=(yt>0).float();rlog,blog,_=model.raw(xt,et);ce=F.binary_cross_entropy_with_logits(rlog,r,reduction='none',pos_weight=rlog.new_tensor(8.))
    if train_occupancy:ce+=8*r*F.binary_cross_entropy_with_logits(blog,(yt==2).float(),reduction='none')
    loss=(ce*wt).mean()
    if not torch.isfinite(loss):raise ValueError('nonfinite recency loss')
    optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step();total+=[float((ce*wt).sum()),float(wt.sum()),len(ids)]
  log=dict(epoch=epoch,rows=int(total[2]),loss=float(total[0]/total[1]),seconds=time.perf_counter()-tick);logs.append(log);save(dest/'learning.json',logs)
  torch.save(dict(schema=cfg['schema'],config=cfg,model_config=dict(train_occupancy=train_occupancy,width=64),model={k:v.detach().cpu()for k,v in model.state_dict().items()},epoch=epoch),dest/('epoch%02d.pt'%epoch));print(log,flush=True)
 save(dest/'complete.json',dict(seconds=time.perf_counter()-tick,parameters=sum(p.numel()for p in model.parameters()),trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),peak_cuda_allocated=torch.cuda.max_memory_allocated(device)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--split',default='pilot_fit');p.add_argument('--epochs',type=int,default=8);p.add_argument('--train-occupancy',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--warm');p.add_argument('--ablate-time',action='store_true');fit(**vars(p.parse_args()))
