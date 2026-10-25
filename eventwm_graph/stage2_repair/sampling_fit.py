"""Matched static-sample versus full-universe fitting on 24 TRAIN windows."""
import argparse
from collections import defaultdict
import time
import numpy as np
import torch
from torch.nn import functional as F
from .diagnose import OUT,save,sha
from .range_spatial import SpatialEvidenceNet
from .range_spatial_eval import full_training_groups
from scripts.stage2_diagnostic.common import extended_rates


def run(mode='dense',steps=1000,device='cuda:0'):
    torch.set_num_threads(1);torch.manual_seed(1927);torch.backends.cuda.matmul.allow_tf32=False;_,groups=full_training_groups();byroot=defaultdict(list)
    for gi,g in enumerate(groups):
        for e in g['segments']:byroot[g['root_id']].append((gi,g,e))
    chosen=[]
    for rid,entries in sorted(byroot.items()):chosen.extend(entries[int(i)]for i in np.linspace(0,len(entries)-1,6).astype(int))
    arrays=defaultdict(list);maps=[];mapid={};metadata=[];cursor=0;rng=np.random.default_rng(1927);selected=[];sample_weights=[]
    for gi,g,e in chosen:
        if gi not in mapid:mapid[gi]=len(maps);maps.append(g['maps'])
        sl=slice(e['start'],e['end']);n=sl.stop-sl.start;y=g['y'][sl];weight=1/(len(byroot)*6*n)
        for k in ['x','extra','y','indices']:arrays[k].append(g[k][sl])
        arrays['weight'].append(np.full(n,weight,np.float32));arrays['mapids'].append(np.full(n,mapid[gi],np.int64))
        for stratum in [np.flatnonzero(y>0),np.flatnonzero(y==0)]:
            count=min(48,len(stratum))
            if count:selected.extend(cursor+rng.choice(stratum,count,replace=False));sample_weights.extend([len(stratum)*weight/count]*count)
        metadata.append(dict(root_id=g['root_id'],scene_id=g['scene_id'],start=cursor,end=cursor+n,true_counts=np.bincount(y,minlength=3).tolist()));cursor+=n
    def tensor(x,dtype=None):return torch.from_numpy(np.ascontiguousarray(x,dtype=dtype)).to(device)
    a={k:tensor(np.concatenate(v),np.int64 if k in ['y','indices','mapids']else np.float32)for k,v in arrays.items()};maps=tensor(np.stack(maps),np.float32);indices=tensor(selected,np.int64)if mode=='static'else torch.arange(cursor,device=device);weight=tensor(sample_weights,np.float32)if mode=='static'else a['weight'];checkpoint=OUT/'training/range_spatial64/epoch20.pt';model=SpatialEvidenceNet().to(device);model.load_state_dict(torch.load(checkpoint,map_location='cpu')['model']);optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0005,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler();logs=[];tick=time.perf_counter()
    for step in range(steps+1):
        if step:
            model.train()
            with torch.cuda.amp.autocast():encoded=model.encode(maps);rl,bl,_=model.raw(a['x'][indices],a['extra'][indices],a['indices'][indices],a['mapids'][indices],encoded)
            y=a['y'][indices];r=(y>0).float();ce=F.binary_cross_entropy_with_logits(rl.float(),r,pos_weight=rl.new_tensor(8.),reduction='none')+8*r*F.binary_cross_entropy_with_logits(bl.float(),(y==2).float(),reduction='none');loss=(ce*weight).sum();optim.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optim);gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optim);scaler.update()
        if step in [0,100,300,600,steps]:
            model.eval();pp=[]
            with torch.inference_mode(),torch.cuda.amp.autocast():
                encoded=model.encode(maps)
                for begin in range(0,cursor,65536):
                    sl=slice(begin,begin+65536);p,_=model.probabilities(a['x'][sl],a['extra'][sl],a['indices'][sl],a['mapids'][sl],encoded);pp.append(p.float().cpu().numpy())
            p=np.concatenate(pp);hard=p.argmax(1);y=a['y'].cpu().numpy();rootcm=defaultdict(list);pooled=np.zeros((3,3),np.int64)
            for e in metadata:sl=slice(e['start'],e['end']);cm=np.bincount(3*y[sl]+hard[sl],minlength=9).reshape(3,3);rootcm[e['root_id']].append(cm);pooled+=cm
            cm=np.mean([np.mean(v,0)for v in rootcm.values()],0);record=dict(step=step,full_original_cell_training_metrics=extended_rates(cm),pooled=extended_rates(pooled),seconds=time.perf_counter()-tick,gradient_norm_last=float(gradient)if step else None);logs.append(record);save(OUT/('sampling_fit24_%s.json'%mode),dict(role='TRAINING FITTING ONLY. Same24fixed windows, same initialization/optimizer/objective. Only static96point supervision versus ALL original cells differs. Not development or acceptance.',mode=mode,checkpoint_sha256=sha(checkpoint),windows=metadata,rows=cursor,supervised_rows=len(indices),steps=steps,logs=logs));print(record,flush=True)
    torch.save(dict(schema='legal_range_spatial_1',model=model.state_dict(),role='diagnostic same24trainingwindows fitting only'),OUT/('sampling_fit24_%s.pt'%mode))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['dense','static'],default='dense');p.add_argument('--steps',type=int,default=1000);p.add_argument('--device',default='cuda:0');run(**vars(p.parse_args()))
