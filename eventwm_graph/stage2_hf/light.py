"""Bounded original-representation objective and factorization pilots."""
import argparse,random,time
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import OUT,PREP,CKPT,SEED,read,save,sha,source

class LightNet(nn.Module):
 def __init__(self,kind='factor'):
  super().__init__();self.kind=kind
  self.net=nn.Sequential(nn.Linear(88,48),nn.SiLU(),nn.Linear(48,32),nn.SiLU(),nn.Linear(32,4 if kind=='softmax' else 3))
 def forward(self,x):return self.net(x)
def initialize(kind):
 model=LightNet(kind);old=torch.load(CKPT,map_location='cpu')['model'];state=model.state_dict()
 for name in ('0.weight','0.bias','2.weight','2.bias'):
  state['net.'+name]=old['net.'+name][:,16:]if name=='0.weight'else old['net.'+name]
 if kind=='softmax':state['net.4.weight']=old['net.4.weight'];state['net.4.bias']=old['net.4.bias']
 else:
  w=old['net.4.weight'];b=old['net.4.bias'];state['net.4.weight']=torch.stack([.5*(w[1]+w[2])-w[0],w[2]-w[1],w[3]])
  state['net.4.bias']=torch.stack([.5*(b[1]+b[2])-b[0]+np.log(2.),b[2]-b[1],b[3]])
 model.load_state_dict(state);return model
def focal_posterior(p,gamma=1.,alpha=3.):
 p=p.clamp(1e-7,1-1e-7);d0=-gamma*p.pow(gamma-1)*torch.log1p(-p)+p.pow(gamma)/(1-p);d1=-gamma*(1-p).pow(gamma-1)*p.log()+(1-p).pow(gamma)/p
 return d0/(d0+alpha*d1)
def probabilities(out,kind,objective):
 if kind=='softmax':return out[:,:3].softmax(-1)
 r=out[:,0].sigmoid();b=out[:,1].sigmoid()
 if objective=='focal':r=focal_posterior(r)
 return torch.stack([1-r,r*(1-b),r*b],-1)
def fit(name,kind='factor',objective='proper',epochs=8,split='pilot_fit',device='cuda:0'):
 torch.set_num_threads(1);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 m=read(PREP/(split+'_mixed_ray')/'manifest.json');dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False)
 config=dict(kind=kind,objective=objective,epochs=epochs,split=split,device=device,seed=SEED,source=source(),manifest_sha256=sha(PREP/(split+'_mixed_ray')/'manifest.json'),warm_checkpoint_sha256=sha(CKPT),loss='inverseinclusionRoot-windowweighted properCE or revelationBCE + revealedconditionalBCE; focalgamma1alpha3 with analytic posterior conversion; heightHuber.2',learning_rate=.001,batch_size=16384)
 save(dest/'config.json',config);model=initialize(kind).to(device);optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);rng=np.random.default_rng(SEED);logs=[];tick=time.perf_counter();scale=m['sampled_cells']/m['roots']
 for epoch in range(1,epochs+1):
  total=np.zeros(4);model.train()
  for at in rng.permutation(len(m['shards'])):
   shard=m['shards'][int(at)]
   if epoch==1 and sha(shard['path'])!=shard['sha256']:raise ValueError('training shard hash mismatch')
   with np.load(shard['path'],allow_pickle=False)as a:x=a['x'][:,16:].astype(np.float32);y=a['y'].astype(np.int64);weight=a['weight'].astype(np.float32);height=a['height'].astype(np.float32)
   order=rng.permutation(len(y))
   for start in range(0,len(order),16384):
    ids=order[start:start+16384];xt=torch.from_numpy(x[ids]).to(device);yt=torch.from_numpy(y[ids]).to(device);w=torch.from_numpy(weight[ids]).to(device)*scale;h=torch.from_numpy(height[ids]).to(device);out=model(xt);r=(yt>0).float()
    if kind=='softmax':cell=F.cross_entropy(out[:,:3],yt,reduction='none');hh=out[:,3]
    else:
     rev=F.binary_cross_entropy_with_logits(out[:,0],r,reduction='none');occup=F.binary_cross_entropy_with_logits(out[:,1],(yt==2).float(),reduction='none');hh=out[:,2]
     if objective=='focal':
      pr=out[:,0].sigmoid();rev=rev*(r*(1-pr)+(1-r)*pr)*(1+2*r)
     cell=rev+r*occup
    if objective=='balanced'and kind=='softmax':cell=cell*(1+7*r)
    huber=F.smooth_l1_loss(hh,h,reduction='none')*r;loss=((cell+.2*huber)*w).mean()
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optim.step();total+=[float((cell*w).sum()),float(w.sum()),len(ids),float(loss)]
  record=dict(epoch=epoch,loss=float(total[0]/total[1]),rows=int(total[2]),seconds=time.perf_counter()-tick);logs.append(record);save(dest/'learning.json',logs)
  torch.save(dict(schema='stage2_hf_light_1',config=config,model_config=dict(kind=kind),model={k:v.detach().cpu()for k,v in model.state_dict().items()},epoch=epoch),dest/('epoch%02d.pt'%epoch))
  print(record,flush=True)
 save(dest/'complete.json',dict(seconds=time.perf_counter()-tick,parameters=sum(v.numel()for v in model.parameters()),epochs=epochs,peak_cuda_allocated=torch.cuda.max_memory_allocated(device)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--kind',choices=['softmax','factor'],default='factor');p.add_argument('--objective',choices=['proper','balanced','focal'],default='proper');p.add_argument('--epochs',type=int,default=8);p.add_argument('--split',default='pilot_fit');p.add_argument('--device',default='cuda:0');fit(**vars(p.parse_args()))
