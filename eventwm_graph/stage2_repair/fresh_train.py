"""Fresh boundary-aware draws over complete, immutable training universes."""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .diagnose import OUT,read,save,sha
from .fresh_data import FreshGroups,keep
from .range_spatial import SpatialEvidenceNet


def train(manifest=None,name='range_fresh64',epochs=24,device='cuda:0',warm=None):
    torch.set_num_threads(1);torch.manual_seed(9017);torch.backends.cuda.matmul.allow_tf32=False;manifest=Path(manifest)if manifest else OUT/'range_dense_bev_train/manifest_roots64.json';data=FreshGroups(manifest);m=data.manifest;loader=DataLoader(data,batch_size=8,num_workers=4,collate_fn=keep,persistent_workers=True,multiprocessing_context='fork',prefetch_factor=2);warm=Path(warm)if warm else OUT/'training/range_spatial64/epoch20.pt';model=SpatialEvidenceNet().to(device);model.load_state_dict(torch.load(warm,map_location='cpu')['model']);optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0003,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler();dest=OUT/'training'/name;dest.mkdir(exist_ok=False);logs=[];tick=time.perf_counter()
    cfg=dict(schema='legal_range_spatial_1',name=name,epochs=epochs,seed=9017,warm_checkpoint=str(warm),warm_checkpoint_sha256=sha(warm),data_manifest=str(manifest),data_manifest_sha256=sha(manifest),roots=m['roots'],scenes=m['scenes'],maps=len(data.groups),learning_rate=.0003,base_checkpoint=read(OUT/'training/range_spatial64/config.json')['base_checkpoint'],sampling='Refresh every epoch: up to128positive,384UNKNOWN within1m of known cells,128remainingUNKNOWN. Exact inverse inclusion equalScene/Root/window weights. Every original cell remains eligible.',objective='BCE revelation pos8 +8R conditional BCE +.1sum typed softDice +.25 auxiliary conditional certificate-class BCE. Same objective as prior spatial fit.',inputs='Current legally observed graph, supplied trajectory, legally measured current/past ranges and tracks only. Future certificate UNION is auxiliary training TARGET only; never encoder input.',sources={p:sha(Path(__file__).with_name(p))for p in ['fresh_train.py','fresh_data.py','range_spatial.py','range_bev.py']})
    save(dest/'config.json',cfg)
    def tensor(x,dtype=None):return torch.from_numpy(np.ascontiguousarray(x,dtype=dtype)).to(device)
    for epoch in range(1,epochs+1):
        data.epoch.value=epoch;total=np.zeros(3);batches=0;seen=0;rows=0
        for items in loader:
            x=tensor(np.concatenate([g['x']for g in items]),np.float32);extra=tensor(np.concatenate([g['extra']for g in items]),np.float32);y=tensor(np.concatenate([g['y']for g in items]),np.int64);w=tensor(np.concatenate([g['weight']for g in items]),np.float32);indices=tensor(np.concatenate([g['indices']for g in items]),np.int64);ids=tensor(np.repeat(np.arange(len(items)),[len(g['y'])for g in items]),np.int64)
            with torch.cuda.amp.autocast():encoded=model.encode(tensor(np.stack([g['maps']for g in items]),np.float32));rl,bl,_=model.raw(x,extra,indices,ids,encoded);auxlogit=model.aux(encoded[0])[:,0]
            r=(y>0).float();ce=F.binary_cross_entropy_with_logits(rl.float(),r,pos_weight=rl.new_tensor(8.),reduction='none')+8*r*F.binary_cross_entropy_with_logits(bl.float(),(y==2).float(),reduction='none');main=(ce*w).sum()*len(data.groups)/(m['roots']*len(items));pr=(rl.float()-np.log(8.)).sigmoid();pb=bl.float().sigmoid();dice=0.
            for cls,prob in [(1,pr*(1-pb)),(2,pr*pb)]:target=(y==cls).float();dice+=1-(2*(w*prob*target).sum()+1.e-7)/((w*(prob+target)).sum()+1.e-7)
            target=tensor(np.stack([g['aux']for g in items]),np.float32);mask=tensor(np.stack([g['mask']for g in items]),np.float32);ac=F.binary_cross_entropy_with_logits(auxlogit.float(),target,reduction='none');per=(ac*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1);aw=tensor([g['aux_weight']for g in items],np.float32);aux=(per*aw).mean()*len(data.groups)/m['roots'];loss=main+.1*dice+.25*aux
            if not torch.isfinite(loss):raise ValueError('nonfinite fresh reconstruction objective')
            optim.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optim);gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);encoder_gradient=float(model.e0.net[0].weight.grad.norm());scaler.step(optim);scaler.update();total+=[float(main),float(dice),float(aux)];batches+=1;seen+=len(items);rows+=len(y)
        assert seen==len(data.groups)
        rec=dict(epoch=epoch,groups=seen,sampled_rows=rows,main_loss=float(total[0]/batches),soft_dice_loss=float(total[1]/batches),auxiliary_loss=float(total[2]/batches),gradient_norm_last=float(gradient),encoder_gradient_norm_last=encoder_gradient,seconds=time.perf_counter()-tick);logs.append(rec);save(dest/'learning.json',logs);print(rec,flush=True)
        if epoch%4==0 or epoch==epochs:torch.save(dict(schema=cfg['schema'],config=cfg,epoch=epoch,threshold=.2,model=model.state_dict()),dest/('epoch%02d.pt'%epoch))
    save(dest/'complete.json',dict(epochs=epochs,seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--name',default='range_fresh64');p.add_argument('--epochs',type=int,default=24);p.add_argument('--device',default='cuda:0');p.add_argument('--warm');train(**vars(p.parse_args()))
