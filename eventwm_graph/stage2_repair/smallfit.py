"""Bounded same-example fitting control, full-domain reconstruction scoring."""
import argparse
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .diagnose import OUT, HF, read, save, CKPT, sha
from eventwm_graph.stage2_hf.light import initialize
from scripts.stage2_diagnostic.common import extended_rates


def prepare(device):
    from eventwm_graph.stage2_segment.features import SegmentFeatures
    from eventwm_graph.stage2_segment.api import SegmentPredictor
    from eventwm_graph.stage2_segment.fused import transform
    m=read(OUT/'trace.json');pred=SegmentPredictor(CKPT,device=device,execution='optimized',future_format='native',feature_threads=1)
    for i,r in enumerate(m['examples'],1):
        p=OUT/'examples'/('%02d.npz'%i);target=p.with_name(p.stem+'_features.npz')
        if target.exists():continue
        with np.load(p,allow_pickle=False)as z:a={k:z[k]for k in z.files}
        v=a['geometry'];g=dict(schema='causal_partial_cspace_graph_1',resolution=float(v[0]),height=float(v[1]),clearance=float(v[2]),timestamp=float(a['path'][0,0]),revision=0,
            cells=[dict(cell=k.tolist(),state=int(s),height=float(h))for k,s,h in zip(a['gkeys'],a['gstates'],a['gheights'])])
        ctx=SegmentFeatures(g,a['path'],pred.prepare_graph(g));x,_=transform(ctx,a['keys'],cuda=pred.cuda,device_output=True)
        np.savez(target,x=x.cpu().numpy(),y=a['y'])


def fit(steps=2000,device='cuda:0',large=False):
    torch.set_num_threads(1);torch.manual_seed(2526);np.random.seed(2526);torch.backends.cuda.matmul.allow_tf32=False
    prepare(device);files=sorted((OUT/'examples').glob('*_features.npz'));data=[]
    for p in files:
        with np.load(p,allow_pickle=False)as z:x=z['x'];y=z['y']
        data.append((torch.from_numpy(x).to(device),torch.from_numpy(y.astype(np.int64)).to(device)))
    model=nn.Sequential(nn.Linear(88,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,4)).to(device)if large else initialize('softmax').to(device)
    if not large:
        with torch.no_grad():model.net[-1].bias[1:3]+=math.log(8.)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.);rng=np.random.default_rng(2526);logs=[];tick=time.perf_counter();name='large_point'if large else 'original_point'
    strata=[(torch.nonzero(y>0).flatten(),torch.nonzero(y==0).flatten())for x,y in data]
    initial={k:v.detach().clone()for k,v in model.state_dict().items()};gradient=[]
    for step in range(steps+1):
        if step%250==0 or step==steps:
            with torch.inference_mode():
                cms=[];preds=[]
                for x,y in data:
                    pp=[]
                    for begin in range(0,len(y),65536):
                        o=model(x[begin:begin+65536]);logits=o[:,:3].clone();logits[:,1:]-=math.log(8.);pp.append(logits.softmax(-1))
                    p=torch.cat(pp);hard=torch.where(p[:,1]+p[:,2]>=.2,torch.where(p[:,1]>=p[:,2],1,2),0)
                    cms.append(torch.bincount(3*y+hard,minlength=9).reshape(3,3).cpu().numpy());preds.append(hard.cpu().numpy())
                cm=np.mean(cms,axis=0);r=dict(step=step,seconds=time.perf_counter()-tick,metrics=extended_rates(cm),pooled_confusion=np.sum(cms,axis=0).tolist(),parameter_update_norm=float(sum((v-initial[k]).square().sum()for k,v in model.state_dict().items()).sqrt()),gradient_norms=gradient[-1:])
                logs.append(r);save(OUT/('smallfit_'+name+'.json'),logs);print(name,step,r['metrics']['label_aware_delta_iou'],r['metrics']['revelation_iou'],flush=True)
                np.savez_compressed(OUT/('smallfit_'+name+'_predictions.npz'),**{'prediction_%02d'%i:p for i,p in enumerate(preds,1)})
        if step==steps:break
        xx=[];yy=[];ww=[]
        for (x,y),st in zip(data,strata):
            for ids in st:
                n=min(128,len(ids))
                if not n:continue
                ix=ids[torch.randperm(len(ids),device=device)[:n]];xx.append(x[ix]);yy.append(y[ix]);ww.append(torch.full((n,),len(ids)/(n*len(y)*len(data)),device=device))
        x=torch.cat(xx);y=torch.cat(yy);w=torch.cat(ww);o=model(x);cell=F.cross_entropy(o[:,:3],y,reduction='none')*(1+7*(y>0));loss=(cell*w).sum()
        optimizer.zero_grad(set_to_none=True);loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%250==0:gradient.append(dict(step=step,gradient_norm=float(grad),first_layer_gradient=float(next(model.parameters()).grad.norm()),loss=float(loss)))
    torch.save(dict(model=model.state_dict(),diagnostic_only=True,steps=steps,examples=len(data),source_sha256=sha(__file__)),OUT/('smallfit_'+name+'.pt'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=2000);p.add_argument('--device',default='cuda:0');p.add_argument('--large',action='store_true');fit(**vars(p.parse_args()))
