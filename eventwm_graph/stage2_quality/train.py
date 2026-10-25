"""Fixed-recipe controlled ablations; no development-cohort selection."""
import argparse
import collections
import json
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from eventwm_graph.stage2.evaluate import rates
from .model import GraphDeltaNet
from .common import BASE,OUT,SCHEMA,MODES,identity,dump,sha,snapshot


def masked(x,mode):
    x=x.astype(np.float32,copy=True)
    if mode in ('trajectory','both'):x[:,:16]=0
    else:x[:,47:67]=0
    if mode in ('baseline','trajectory'):x[:,67:92]=0
    return x


def sampled_evaluate(model,mode,manifest,device):
    model.eval();root_matrices=collections.defaultdict(lambda:np.zeros((3,3),np.float64));scene_roots=collections.defaultdict(set)
    total=np.zeros((3,3),np.float64);scores=np.zeros(3,np.float64);queries=[]
    with torch.inference_mode():
        for shard in manifest['shards']:
            assert sha(shard['path'])==shard['sha256']
            with np.load(shard['path'],allow_pickle=False) as a:
                x=masked(a['x'],mode);y=a['y'].astype(np.int64);weight=a['weight'].astype(np.float64)
                probabilities=[]
                for begin in range(0,len(y),8192):
                    logits,_=model(torch.from_numpy(x[begin:begin+8192]).to(device));probabilities.append(torch.softmax(logits,-1).cpu().numpy())
                p=np.concatenate(probabilities);state=p.argmax(-1)
                scores+=np.array([np.sum(-np.log(np.clip(p[np.arange(len(y)),y],1e-9,1))*weight),
                    np.sum((np.square(p).sum(-1)-2*p[np.arange(len(y)),y]+1)*weight),weight.sum()])
                for meta in shard['queries']:
                    lo,hi=meta['start'],meta['end'];n=meta['candidate_cells'];q=meta['queries_in_root'];w=weight[lo:hi]
                    cm=np.bincount(3*y[lo:hi]+state[lo:hi],weights=w*n,minlength=9).reshape(3,3)
                    root_matrices[meta['root_id']]+=cm;total+=cm*q;scene_roots[meta['scene_id']].add(meta['root_id'])
                    counts=(p[lo:hi]*((w*n*q)[:,None])).sum(0)
                    queries.append(dict(**meta,predicted_free=float(counts[1]),predicted_blocked=float(counts[2])))
    scenes=np.array([np.mean([root_matrices[k] for k in ids],axis=0) for ids in scene_roots.values()])
    balanced=rates(scenes.mean(0));rng=np.random.default_rng(2411)
    boots=[rates(scenes[rng.integers(0,len(scenes),len(scenes))].mean(0)) for _ in range(2000)]
    for cls in ('new_free','new_blocked'):
        for metric in ('precision','recall','f1'):
            balanced[cls][metric+'_ci95']=np.quantile([b[cls][metric] for b in boots],[.025,.975]).tolist()
    return dict(scope='inverse-inclusion sampled fitting-only scene holdout estimates; not full-domain original dev results',
        roots=len(root_matrices),scenes=len(scenes),scene_balanced=balanced,pooled=rates(total),
        multiclass_logloss=float(scores[0]/scores[2]),multiclass_brier=float(scores[1]/scores[2]),queries=queries)


def fit(mode,split='pilot_fit',name=None,device='cuda:0'):
    if mode not in MODES:raise ValueError('unknown mode')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);random.seed(2410);np.random.seed(2410);torch.manual_seed(2410);torch.cuda.manual_seed_all(2410)
    name=name or ('pilot_'+mode);dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False)
    path=OUT/'prepared'/split/'manifest.json';manifest=json.loads(path.read_text());assert manifest['source']['split']==split
    actual_source=snapshot('actual fixed fit '+name)
    cfg=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),schema=SCHEMA,mode=mode,split=split,epochs=6,seed=2410,width=48,batch_size=8192,learning_rate=.001,weight_decay=.0001,
        gradient_clip=5.,height_huber_weight=.2,source=identity(),frozen_source=actual_source,dataset_manifest=str(path),dataset_manifest_sha256=sha(path),
        roots=manifest['roots'],scenes=manifest['scenes'],sampled_cells=manifest['sampled_cells'],device=device,selection='fixed sixth epoch, no development tuning')
    dump(dest/'config.json',cfg);model=GraphDeltaNet().to(device);optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(2410);steps=0;start=time.perf_counter();scale=manifest['sampled_cells']/manifest['roots'];ledger=[]
    for epoch in range(1,7):
        model.train();seen=0;weighted=np.zeros(3,np.float64)
        for index in rng.permutation(len(manifest['shards'])):
            shard=manifest['shards'][int(index)];assert sha(shard['path'])==shard['sha256']
            with np.load(shard['path'],allow_pickle=False) as a:x=masked(a['x'],mode);y=a['y'].astype(np.int64);h=a['height'].astype(np.float32);w=a['weight']
            order=rng.permutation(len(y))
            for begin in range(0,len(order),8192):
                at=order[begin:begin+8192];xb=torch.from_numpy(x[at]).to(device);yb=torch.from_numpy(y[at]).to(device)
                hb=torch.from_numpy(h[at]).to(device);wb=torch.from_numpy(w[at]).to(device)
                logits,height=model(xb);ce=F.cross_entropy(logits,yb,reduction='none');hl=F.smooth_l1_loss(height,hb,reduction='none')*(yb>0)
                loss=((ce+.2*hl)*wb).sum()*scale/8192
                if not torch.isfinite(loss):raise RuntimeError('nonfinite weighted objective')
                optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step();steps+=1;seen+=len(at)
                weighted+=np.array([float((ce*wb).sum()),float((hl*wb).sum()),float(wb.sum())])
        assert seen==manifest['sampled_cells']
        row=dict(epoch=epoch,steps=steps,rows=seen,logloss=float(weighted[0]/weighted[2]),height_huber=float(weighted[1]/weighted[2]),seconds=time.perf_counter()-start)
        ledger.append(row);dump(dest/'learning.json',ledger);print(json.dumps(dict(mode=mode,**row)),flush=True)
    torch.save(dict(schema=SCHEMA,mode=mode,model_config={'width':48},model={k:v.cpu() for k,v in model.state_dict().items()},config=cfg,epoch=6,steps=steps),dest/'last.pt')
    dump(dest/'complete.json',dict(checkpoint_sha256=sha(dest/'last.pt'),steps=steps,seconds=time.perf_counter()-start,
        parameters=sum(p.numel() for p in model.parameters()),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    if split.startswith('pilot_fit'):
        holdname='pilot_holdout_corrected' if split.endswith('_corrected') else 'pilot_holdout'
        hold=json.loads((OUT/'prepared'/holdname/'manifest.json').read_text());metrics=sampled_evaluate(model,mode,hold,device)
        dump(dest/'pilot_metrics.json',metrics)
        print(json.dumps(dict(mode=mode,logloss=metrics['multiclass_logloss'],brier=metrics['multiclass_brier'],balanced=metrics['scene_balanced'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--split',choices=['pilot_fit','pilot_fit_corrected','full'],default='pilot_fit');p.add_argument('--name');p.add_argument('--device',default='cuda:0')
    fit(**vars(p.parse_args()))
