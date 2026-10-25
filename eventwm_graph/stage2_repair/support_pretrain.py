"""Map-completion encoder pretraining with private loss targets only."""
import argparse
from collections import Counter
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset,DataLoader,get_worker_info
from .diagnose import OUT,read,save,sha
from .range_spatial import SpatialEvidenceNet
from .fresh_data import keep


def augment(maps,target,rotation,flip):
    maps=np.rot90(maps,rotation,axes=(-2,-1)).copy()
    target=np.rot90(target,rotation,axes=(-2,-1)).copy()
    for _ in range(rotation):
        x=maps[12].copy();maps[12]=-maps[13];maps[13]=x
    if flip:
        maps=maps[:,:,::-1].copy();target=target[:,:,::-1].copy();maps[13]*=-1
    return maps,target


class CompletionMaps(IterableDataset):
    def __init__(self,manifest):
        self.manifest=read(manifest);self.epoch=mp.Value('i',0);self.groups=[]
        m=self.manifest;counts=Counter(r['scene_id']for r in m['records'])
        for row in m['records']:
            if sha(row['path'])!=row['sha256']:raise ValueError('completion dataset changed')
            with np.load(row['path'])as z:maps=z['maps'];targets=z['support_targets']
            maps.flags.writeable=False;targets.flags.writeable=False
            weight=1/(len(counts)*counts[row['scene_id']]*len(maps))
            for canvas,target in zip(maps,targets):self.groups.append((canvas,target,weight))

    def __iter__(self):
        info=get_worker_info();worker=info.id if info else 0;workers=info.num_workers if info else 1
        epoch=self.epoch.value;rng=np.random.default_rng(7811+epoch)
        order=rng.permutation(len(self.groups))
        for i in order[worker::workers]:
            canvas,target,weight=self.groups[int(i)]
            local=np.random.default_rng(7811+epoch*100003+int(i))
            canvas,target=augment(canvas,target,int(local.integers(4)),bool(local.integers(2)))
            yield dict(maps=canvas,target=target,weight=weight)


def run(manifest,warm,epochs=3,device='cuda:0',name='support_pretrain2048'):
    torch.set_num_threads(1);torch.manual_seed(7811);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.perf_counter();data=CompletionMaps(manifest);print('LOADED',len(data.groups),'seconds',time.perf_counter()-tick,flush=True)
    loader=DataLoader(data,batch_size=32,num_workers=4,collate_fn=keep,persistent_workers=True,multiprocessing_context='fork')
    model=SpatialEvidenceNet().to(device);model.load_state_dict(torch.load(warm,map_location='cpu')['model'])
    head=nn.Conv2d(24,2,1).to(device);saved=torch.load(warm,map_location='cpu')
    if 'training_only_teacher_head'in saved:head.load_state_dict(saved['training_only_teacher_head'])
    parameters=[p for p in model.parameters()if p.requires_grad]+list(head.parameters())
    optim=torch.optim.AdamW(parameters,lr=.0003,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler()
    dest=OUT/'training'/name;dest.mkdir(exist_ok=False)
    cfg=dict(schema='legal_range_spatial_1',name=name,base_checkpoint=saved['config']['base_checkpoint'],warm=str(warm),warm_sha256=sha(warm),data_manifest=str(manifest),data_manifest_sha256=sha(manifest),epochs=epochs,roots=data.manifest['roots'],scenes=data.manifest['scenes'],maps=len(data.groups),role='ENCODER PRETRAINING ONLY, requires increment fine-tuning before deployment',inputs='Legal graph only; absent range channels zero; relative coordinates. D4 image/vector augmentation. Private traversability/distance targets enter loss only.',source_sha256=sha(__file__))
    save(dest/'config.json',cfg);logs=[];tick=time.perf_counter()
    def tensor(x):return torch.from_numpy(np.ascontiguousarray(x,np.float32)).to(device)
    for epoch in range(1,epochs+1):
        data.epoch.value=epoch;loss_sum=0.;batches=0;confusion=np.zeros((2,2),np.int64)
        for items in loader:
            maps=tensor(np.stack([g['maps']for g in items]));target=tensor(np.stack([g['target']for g in items]));weight=tensor([g['weight']for g in items])
            with torch.cuda.amp.autocast():encoded=model.encode(maps);logits=head(encoded[0])
            domain=(maps[:,12].square()+maps[:,13].square()<=1.44).float()*(1+3*(1-maps[:,5]).clamp(0,1))
            ce=F.binary_cross_entropy_with_logits(logits[:,0].float(),target[:,0],reduction='none')+(logits[:,1].float().tanh()-target[:,1]).square()
            per=(ce*domain).sum((1,2))/domain.sum((1,2)).clamp_min(1)
            loss=(per*weight).mean()*len(data.groups)
            if not torch.isfinite(loss):raise ValueError('nonfinite encoder completion loss')
            optim.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optim)
            gradient=torch.nn.utils.clip_grad_norm_(parameters,5.);encoder_gradient=float(model.e0.net[0].weight.grad.norm())
            scaler.step(optim);scaler.update();loss_sum+=float(loss);batches+=1
            with torch.no_grad():
                mask=domain>0;code=(target[:,0][mask]>=.5).long()*2+(logits[:,0][mask]>=0).long();confusion+=torch.bincount(code,minlength=4).reshape(2,2).cpu().numpy()
        rec=dict(epoch=epoch,loss=loss_sum/batches,gradient=float(gradient),encoder_gradient=encoder_gradient,auxiliary_coarse_support_pooled_confusion=confusion.tolist(),auxiliary_coarse_support_pooled_iou=float(confusion[1,1]/max(1,confusion[1,1]+confusion[0,1]+confusion[1,0])),seconds=time.perf_counter()-tick)
        logs.append(rec);save(dest/'learning.json',logs);print(rec,flush=True)
        torch.save(dict(schema=cfg['schema'],config=cfg,epoch=epoch,model=model.state_dict(),training_only_teacher_head=head.state_dict()),dest/('epoch%02d.pt'%epoch))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--warm',required=True);p.add_argument('--epochs',type=int,default=3);p.add_argument('--device',default='cuda:0');p.add_argument('--name',default='support_pretrain2048');run(**vars(p.parse_args()))
