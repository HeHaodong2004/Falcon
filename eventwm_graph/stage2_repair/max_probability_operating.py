"""Bounded diagnostic of a Jaccard-aligned operating-point family.

This does not change model inputs, targets or candidate cells. All outcomes
use the same frozen checkpoint and original666-window pilot. No public API
decision rule is added unless this demonstrates a reconstruction benefit.
"""
from collections import defaultdict
import argparse
import time
import numpy as np
import torch
from .diagnose import OUT,sha,save
from .range_spatial import SpatialEvidenceNet
from .range_spatial_train import load_groups
from scripts.stage2_diagnostic.common import extended_rates


def run(checkpoint,device='cuda:1',name='max_probability_epoch02'):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;saved=torch.load(checkpoint,map_location='cpu');model=SpatialEvidenceNet().to(device).eval();model.load_state_dict(saved['model']);manifest,groups=load_groups('dev',saved['config']['dev_feature_manifest']);thresholds=[.20,.25,.30,.35,.40,.45,.50,.55,.60];names=['argmax']+['max_new_%.2f'%t for t in thresholds];roots=defaultdict(list);scenes={};pooled=np.zeros((len(names),3,3),np.int64);tick=time.perf_counter();windows=0
    def tensor(a,dtype):return torch.from_numpy(np.ascontiguousarray(a,dtype=dtype)).to(device)
    with torch.inference_mode():
        for group in groups:
            values=[]
            with torch.cuda.amp.autocast():
                encoded=model.encode(tensor(group['maps'][None],np.float32))
                for begin in range(0,len(group['y']),65536):
                    end=begin+65536;x=tensor(group['x'][begin:end],np.float32);p,_=model.probabilities(x,tensor(group['extra'][begin:end],np.float32),tensor(group['indices'][begin:end],np.int64),torch.zeros(len(x),dtype=torch.long,device=device),encoded);values.append(p.float().cpu().numpy())
            probabilities=np.concatenate(values);scenes[group['root_id']]=group['scene_id']
            for segment in group['segments']:
                start,end=segment['start'],segment['end'];p=probabilities[start:end];y=group['y'][start:end];confidence=np.max(p[:,1:],1).astype(np.float64);cls=np.where(p[:,1]>=p[:,2],1,2);predictions=[p.argmax(1)]+[np.where(confidence>=t,cls,0)for t in thresholds];cm=np.array([np.bincount(y.astype(np.int64)*3+pred,minlength=9).reshape(3,3)for pred in predictions]);pooled+=cm;roots[group['root_id']].append(cm);windows+=1
    scene_values=defaultdict(list)
    for rid,values in roots.items():scene_values[scenes[rid]].append(np.mean(values,0))
    mean=np.mean([np.mean(values,0)for values in scene_values.values()],0);hierarchical={n:extended_rates(c)for n,c in zip(names,mean)};best=max(names,key=lambda n:hierarchical[n]['label_aware_delta_iou']);result=dict(role='One bounded operating-point family on original666pilot windows; not a training/generalization/full-development result. No API change yet. OriginalequalScene/Root/window aggregation and pooled diagnostic both retained.',checkpoint_sha256=sha(checkpoint),feature_manifest_sha256=sha(saved['config']['dev_feature_manifest']),roots=len(roots),scenes=len(scene_values),windows=windows,hierarchical=hierarchical,pooled={n:extended_rates(c)for n,c in zip(names,pooled)},pooled_confusion=pooled.tolist(),best=best,seconds=time.perf_counter()-tick);save(OUT/(name+'.json'),result);print('OPERATING',best,hierarchical[best],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--name',default='max_probability_epoch02');run(**vars(p.parse_args()))
