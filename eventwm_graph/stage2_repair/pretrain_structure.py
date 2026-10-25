"""Pretrain only the map encoder by erasing legally observed certificates.

This is an auxiliary representation exercise. It uses current training maps
only, no future observations or hidden occupancy. The deployed task/output
and its initially-UNKNOWN targets are unchanged. Auxiliary scores are never
reported as increment reconstruction accuracy.
"""
import argparse
from collections import Counter
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset,DataLoader,get_worker_info
from .diagnose import OUT,HF,read,save,sha,source_rows,member
from .shared_map import SharedMapNet,map_canvas


class ErasedMaps(IterableDataset):
    def __init__(self,rows,epoch):
        self.rows=rows;self.epoch=epoch;counts=Counter(r['scene_id']for r in rows);self.factor={s:len(rows)/(len(counts)*n)for s,n in counts.items()}
    def __iter__(self):
        torch.set_num_threads(1);info=get_worker_info();wi=info.id if info else 0;nw=info.num_workers if info else 1;rng=np.random.default_rng(5021+self.epoch)
        for ri in rng.permutation(len(self.rows))[wi::nw]:
            row=self.rows[int(ri)];legal=member(row,'legal_root.json.gz');g=legal['partial_graph'];cells=g['cells'];keys=np.array([c['cell']for c in cells],np.int32).reshape(-1,2);state=np.array([c['state']for c in cells],np.uint8);height=np.array([c['height']for c in cells],np.float32);last=np.array([c['last_seen']for c in cells],np.float64);ego=np.array(g['ego_support'],np.float64)
            # Exact grid-center 90deg rotations/reflections, not new sensor data.
            for _ in range(int(rng.integers(4))):keys=np.column_stack([-keys[:,1]-1,keys[:,0]]).astype(np.int32);ego=np.array([-ego[2],ego[1],ego[0]])
            if rng.random()<.5:keys[:,0]=-keys[:,0]-1;ego[0]=-ego[0]
            erase=np.zeros(len(keys),bool)
            if len(keys)>4:
                centers=[]
                for label in [1,2]:
                    ids=np.flatnonzero(state==label)
                    if len(ids):centers.append(keys[int(rng.choice(ids))])
                centers.extend(keys[rng.integers(len(keys),size=3)])
                for center in centers:
                    radius=int(rng.integers(4,17));erase|=np.max(abs(keys-center),axis=1)<=radius
                if erase.all():erase[rng.choice(len(keys),max(1,len(keys)//3),replace=False)]=False
            observed_state=state.copy();observed_height=height.copy();observed_last=last.copy();observed_state[erase]=0;observed_height[erase]=g['height'];observed_last[erase]=g['timestamp']
            d=map_canvas(keys,observed_state,observed_height,observed_last,g['timestamp'],[g['resolution'],g['height'],g['clearance']],ego,anchor=True,halo=128)
            yield dict(maps=d['maps'],fine=d['fine'],indices=(keys[erase]-d['lo']).astype(np.int64),y=(state[erase]-1).astype(np.int64),weight=self.factor[row['scene_id']])


def collate(items):
    n=len(items);h=max(d['maps'].shape[1]for d in items);w=max(d['maps'].shape[2]for d in items);maps=np.zeros((n,10,h,w),np.float32);fine=np.zeros((n,5,h*2,w*2),np.float32);sizes=[];indices=[];labels=[];ids=[]
    for i,d in enumerate(items):
        a,b=d['maps'].shape[1:];maps[i,:,:a,:b]=d['maps'];fine[i,:,:a*2,:b*2]=d['fine'];sizes.append([a,b]);indices.append(d['indices']);labels.append(d['y']);ids.append(np.full(len(d['y']),i,np.int64))
    values=dict(maps=maps,fine=fine,sizes=np.array(sizes,np.int64),indices=np.concatenate(indices),y=np.concatenate(labels),map_ids=np.concatenate(ids),weight=np.array([d['weight']for d in items],np.float32))
    return {k:torch.from_numpy(v)for k,v in values.items()}


def fit(name='structure_pretrain_full',epochs=2,device='cuda:1',workers=12,batch_size=12,limit=None):
    torch.set_num_threads(1);random.seed(5021);np.random.seed(5021);torch.manual_seed(5021);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    rows,manifest=source_rows('train')
    if limit:rows=rows[:limit]
    dest=OUT/'training'/name;dest.mkdir(parents=True,exist_ok=False);m=SharedMapNet(map_channels=10,fine=True);m.initialize(OUT/'training/point_residual_full/epoch06.pt',HF/'training/spatial_bce/epoch08.pt');m.to(device);aux=nn.Sequential(nn.Linear(40,32),nn.SiLU(),nn.Linear(32,2)).to(device)
    params=[p for n,p in m.named_parameters()if p.requires_grad and not n.startswith('head.')]+list(aux.parameters());optim=torch.optim.AdamW(params,lr=.001,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler();logs=[];tick=time.perf_counter()
    save(dest/'config.json',dict(name=name,epochs=epochs,roots=len(rows),manifest_sha256=sha(manifest),seed=5021,workers=workers,batch_size=batch_size,source_sha256=sha(__file__),map_source_sha256=sha(Path(__file__).with_name('shared_map.py')),
        role='Auxiliary current-map certificate erasure pretraining only. No future/hidden geometry. Not an increment model result; primary legal increment supervision unchanged.',augmentation='Exact90deggridrotations/mirrors; randomcontiguous erasure of previously legal FREE/BLOCKED certificates. Target positions used only for auxiliary loss.',model_config=dict(map_channels=10,fine=True)))
    for epoch in range(1,epochs+1):
        loader=DataLoader(ErasedMaps(rows,epoch),batch_size=batch_size,num_workers=workers,collate_fn=collate,pin_memory=True,prefetch_factor=2);ng=0;steps=0;cm=np.zeros((2,2),np.int64)
        for batch in loader:
            b={k:v.to(device,non_blocking=True)for k,v in batch.items()};optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                local,global_=m.encode(b['maps'],b['sizes'],b['fine']);ii=b['indices'];mi=b['map_ids'];features=torch.cat([local[mi,:,ii[:,0],ii[:,1]],global_[mi]],1);logits=aux(features)
                per=F.cross_entropy(logits.float(),b['y'],reduction='none');sums=torch.zeros(len(b['maps']),device=device).scatter_add_(0,mi,per);counts=torch.bincount(mi,minlength=len(b['maps'])).clamp_min(1);loss=((sums/counts)*b['weight']).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite structure pretraining loss')
            scaler.scale(loss).backward();scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(params,5.);scaler.step(optim);scaler.update();ng+=len(b['maps']);steps+=1
            pred=logits.argmax(1);cm+=torch.bincount(2*b['y']+pred,minlength=4).reshape(2,2).cpu().numpy()
            if steps%100==0:print(dict(epoch=epoch,steps=steps,roots=ng,seconds=time.perf_counter()-tick,auxiliary_loss=float(loss),role='NOTprimaryincrementmetric'),flush=True)
        assert ng==len(rows);logs.append(dict(epoch=epoch,roots=ng,seconds=time.perf_counter()-tick,auxiliary_confusion=cm.tolist()));save(dest/'learning.json',logs)
        torch.save(dict(schema='masked_legal_certificate_pretrain_1',model=m.state_dict(),model_config=dict(map_channels=10,fine=True),auxiliary_head=aux.state_dict(),epoch=epoch),dest/('epoch%02d.pt'%epoch))
    save(dest/'complete.json',dict(epochs=epochs,roots=len(rows),seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='structure_pretrain_full');p.add_argument('--epochs',type=int,default=2);p.add_argument('--device',default='cuda:1');p.add_argument('--workers',type=int,default=12);p.add_argument('--batch-size',type=int,default=12);p.add_argument('--limit',type=int);fit(**vars(p.parse_args()))
