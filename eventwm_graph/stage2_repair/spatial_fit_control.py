"""One bounded receptive-field control, diagnostic only, same legal examples."""
import argparse
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .diagnose import OUT,HF,read,save
from eventwm_graph.stage2_hf.spatial import SpatialNet,build,spatial_probabilities
from eventwm_graph.stage2_hf.spatial_train import loss_fn
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from scripts.stage2_diagnostic.common import extended_rates


class ContextNet(SpatialNet):
    def __init__(self,width=12):
        super().__init__(width);c=width*2
        self.context=nn.Sequential(nn.Conv2d(c,c,3,padding=2,dilation=2),nn.SiLU(),nn.Conv2d(c,c,3,padding=4,dilation=4),nn.SiLU(),nn.Conv2d(c,c,3,padding=8,dilation=8))
        nn.init.zeros_(self.context[-1].weight);nn.init.zeros_(self.context[-1].bias)
    def forward(self,maps,query,fine,encoded=None):
        e=self.encode_map(maps)if encoded is None else encoded;x=self.mix(torch.cat([e,query],1));z=self.down(F.avg_pool2d(x,2));z=z+self.context(z)
        z=F.interpolate(z,scale_factor=2,mode='nearest');z=self.up(torch.cat([x,z],1));z=F.interpolate(z,scale_factor=2,mode='nearest');return self.detail(torch.cat([z,fine],1))


def fit(steps=1200,device='cuda:1'):
    torch.set_num_threads(1);torch.manual_seed(2526);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    meta=read(OUT/'trace.json')['examples'];data=[];source={}
    for e in meta:
        rid=e['root_id']
        if rid not in source:
            row=read(HF/'packed/pilot_fit'/(rid+'.json'));source[rid]=(row,load_arrays(row))
        row,a=source[rid];v=get_segment(a,e['index']);d=build(*v[:4],v[-1],*v[4:7]);b={k:torch.from_numpy(d[k][None]).to(device)for k in ['maps','query','fine','candidate','labels','height']};b['weight']=torch.ones(1,device=device)
        data.append(b)
    saved=torch.load(HF/'training/spatial_bce/epoch08.pt',map_location='cpu');result={}
    for name,cls in [('original',SpatialNet),('larger_context',ContextNet)]:
        torch.manual_seed(2526);model=cls().to(device);model.load_state_dict(saved['model'],strict=name=='original');optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.);rng=np.random.default_rng(2526);logs=[];tick=time.perf_counter()
        for step in range(steps+1):
            if step%300==0 or step==steps:
                cms=[]
                with torch.inference_mode():
                    for b in data:
                        out=model(b['maps'],b['query'],b['fine']);p=spatial_probabilities(out);hard=torch.where(p[:,1]+p[:,2]>=.2,torch.where(p[:,1]>=p[:,2],1,2),0);valid=b['candidate'].bool();y=b['labels'][valid].long();h=hard[valid];cms.append(torch.bincount(3*y+h,minlength=9).reshape(3,3).cpu().numpy())
                log=dict(step=step,seconds=time.perf_counter()-tick,metrics=extended_rates(np.mean(cms,axis=0)),pooled_confusion=np.sum(cms,axis=0).tolist());logs.append(log);result[name]=logs;save(OUT/'spatial_fit_control.json',result);print(name,step,log['metrics']['label_aware_delta_iou'],flush=True)
            if step==steps:break
            b=data[int(rng.integers(len(data)))];out=model(b['maps'],b['query'],b['fine']);loss,_=loss_fn(out,b,pos_weight=8.,dice_weight=0.);optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step()
        del model,optim


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1200);p.add_argument('--device',default='cuda:1');fit(**vars(p.parse_args()))
