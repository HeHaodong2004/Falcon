"""Fresh-cell fitting on the SAME24 original TRAIN diagnostic windows.

Uses the same original fitting feature cache/initialization as the earlier
static96 versus dense test. Legal camera-line queries are derived separately
from the current observation. This is fitting, never development accuracy.
"""
from collections import defaultdict
import argparse
import time
import numpy as np
import torch
from torch.nn import functional as F
from .diagnose import OUT,save,sha
from .range_spatial_eval import full_training_groups
from .ray_residual_model import RayResidualEvidenceNet
from .ray_residual_groups import attach
from .fresh_data import draw
from scripts.stage2_diagnostic.common import extended_rates


def run(steps=1000,device='cuda:1'):
    torch.set_num_threads(1);torch.manual_seed(1927);torch.backends.cuda.matmul.allow_tf32=False;_,groups=full_training_groups();attach(groups,training=True);byroot=defaultdict(list)
    for gi,g in enumerate(groups):
        for e in g['segments']:byroot[g['root_id']].append((gi,g,e))
    chosen=[]
    for rid,entries in sorted(byroot.items()):chosen.extend(entries[int(i)]for i in np.linspace(0,len(entries)-1,6).astype(int))
    arrays=defaultdict(list);maps=[];mapids={};metadata=[];strata=[];cursor=0;rng=np.random.default_rng(1927)
    for gi,g,e in chosen:
        if gi not in mapids:mapids[gi]=len(maps);maps.append(g['maps'])
        sl=slice(e['start'],e['end']);n=sl.stop-sl.start;y=g['y'][sl];near=np.minimum(g['x'][sl,18],g['x'][sl,19])*10<=1.
        for k in ['x','extra','y','indices','ray_origins']:arrays[k].append(g[k][sl])
        arrays['mapids'].append(np.full(n,mapids[gi],np.int64));strata.append(([np.flatnonzero(y>0)+cursor,np.flatnonzero((y==0)&near)+cursor,np.flatnonzero((y==0)&~near)+cursor],n));metadata.append(dict(root_id=g['root_id'],start_node=e['start_node'],end_node=e['end_node'],start=cursor,end=cursor+n,true_counts=np.bincount(y,minlength=3).tolist()));cursor+=n
    def tensor(v,dtype):return torch.from_numpy(np.ascontiguousarray(v,dtype=dtype)).to(device)
    a={k:tensor(np.concatenate(v),np.int64 if k in ['y','indices','mapids']else np.float32)for k,v in arrays.items()};maps=tensor(np.stack(maps),np.float32);checkpoint=OUT/'training/range_spatial64/epoch20.pt';model=RayResidualEvidenceNet().to(device);model.initialize_spatial(torch.load(checkpoint,map_location='cpu')['model']);optimizer=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0005,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler();logs=[];tick=time.perf_counter()
    for step in range(steps+1):
        if step:
            selected=[];weights=[]
            for populations,n in strata:
                ids,w=draw(populations,n,6,rng);selected.append(ids);weights.append(w/len(byroot))
            ids=tensor(np.concatenate(selected),np.int64);weight=tensor(np.concatenate(weights),np.float32);model.train()
            with torch.cuda.amp.autocast():encoded=model.encode(maps);r,b,_=model.raw(a['x'][ids],a['extra'][ids],a['indices'][ids],a['mapids'][ids],encoded,a['ray_origins'][ids])
            y=a['y'][ids];truth=(y>0).float();ce=F.binary_cross_entropy_with_logits(r.float(),truth,pos_weight=r.new_tensor(8.),reduction='none')+8*truth*F.binary_cross_entropy_with_logits(b.float(),(y==2).float(),reduction='none');loss=(ce*weight).sum();optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optimizer);gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);ray_gradient=float(model.ray_map.weight.grad.norm());scaler.step(optimizer);scaler.update()
        if step in [0,100,300,600,steps]:
            model.eval();probabilities=[]
            with torch.inference_mode(),torch.cuda.amp.autocast():
                encoded=model.encode(maps)
                for begin in range(0,cursor,65536):
                    sl=slice(begin,begin+65536);p,_=model.probabilities(a['x'][sl],a['extra'][sl],a['indices'][sl],a['mapids'][sl],encoded,a['ray_origins'][sl]);probabilities.append(p.float().cpu().numpy())
            hard=np.concatenate(probabilities).argmax(1);y=a['y'].cpu().numpy();roots=defaultdict(list);pooled=np.zeros((3,3),np.int64)
            for e in metadata:
                sl=slice(e['start'],e['end']);cm=np.bincount(y[sl]*3+hard[sl],minlength=9).reshape(3,3);roots[e['root_id']].append(cm);pooled+=cm
            cm=np.mean([np.mean(v,0)for v in roots.values()],0);record=dict(step=step,metrics=extended_rates(cm),pooled=extended_rates(pooled),gradient_norm=float(gradient)if step else None,ray_map_gradient=ray_gradient if step else None,seconds=time.perf_counter()-tick);logs.append(record);save(OUT/'ray_residual_fit24.json',dict(role='Same24TRAINwindows fitting only, original fitting feature cache. Fresh128positive/384nearUNKNOWN/128otherUNKNOWN perwindow/step with exact inverse inclusion. Full original cells scored. Not development, not an architecture acceptance result.',checkpoint_sha256=sha(checkpoint),rows=cursor,windows=metadata,steps=steps,logs=logs));print('FIT',record,flush=True)
    torch.save(dict(schema='legal_range_ray_residual_1',model=model.state_dict(),role='Same24TRAINwindows fitting diagnostic only; no deployment config'),OUT/'ray_residual_fit24.pt')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000);p.add_argument('--device',default='cuda:1');run(**vars(p.parse_args()))
