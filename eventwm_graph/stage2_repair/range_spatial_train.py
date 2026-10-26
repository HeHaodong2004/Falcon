"""Bounded spatial fit: same legal evidence, original weighted increment loss."""
import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .diagnose import OUT,HF,read,save,sha
from .range_spatial import SpatialEvidenceNet
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment


def load_groups(split,point_manifest=None,map_manifest=None):
    m=read(map_manifest or OUT/('range_bev_'+split)/'manifest.json');counts=Counter(r['scene_id']for r in m['records']);factor={s:m['roots']/(len(counts)*n)for s,n in counts.items()};groups=[]
    override={r['root_id']:r for r in read(point_manifest)['records']}if point_manifest else{}
    for row in m['records']:
        with np.load(row['path'])as z:maps=z['maps'];low=z['lo'];aux=z['aux'];mask=z['mask']
        point=override.get(row['root_id'])
        with np.load(point['path']if point else row['point_path'])as z:extra=z['extra'];y=z['y'];x=z['x']if 'x'in z.files else None;w=z['weight']if split=='train'else None
        if x is None:
            with np.load(point['original_feature_path']if point else row['original_feature_path'])as z:x=z['x']
        packed=read(HF/'packed'/('pilot_fit'if split=='train'else'dev_pilot')/(row['root_id']+'.json'));a=load_arrays(packed);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed['segments'])}
        for g in row['groups']:
            ids=np.array(g['point_ids']);xx=x[ids];mi=g['map_index'];p=get_segment(a,lookup[g['start_node'],g['segments'][0]['end_node']])[3];keys=np.floor((xx[:,20:22].astype(np.float64)*10+p[0,[1,3]])/a['geometry'][0]).astype(np.int32);relative=keys-low[mi];segments=[];cursor=0
            for e in g['segments']:n=e['end']-e['start'];segments.append(dict(start=cursor,end=cursor+n,start_node=e['start_node'],end_node=e['end_node']));cursor+=n
            groups.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],start_node=g['start_node'],low=low[mi],maps=maps[mi],aux=aux[mi],mask=mask[mi],x=xx,extra=extra[ids],y=y[ids],weight=w[ids]*factor[row['scene_id']]if w is not None else None,indices=relative,segments=segments,aux_weight=len(segments)/sum(len(v['segments'])for v in row['groups'])*factor[row['scene_id']]))
    return m,groups


def train(epochs=80,device='cuda:0',name='range_spatial64'):
    torch.set_num_threads(1);torch.manual_seed(9017);torch.backends.cuda.matmul.allow_tf32=False;rng=np.random.default_rng(9017);m,groups=load_groups('train');base=OUT/'training/range_evidence64/selected.pt';model=SpatialEvidenceNet();model.initialize(base);model.to(device);optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0005,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler();dest=OUT/'training'/name;dest.mkdir(exist_ok=False);logs=[];tick=time.perf_counter();batch_size=8
    cfg=dict(schema='legal_range_spatial_1',name=name,epochs=epochs,seed=9017,base_checkpoint=str(base),base_checkpoint_sha256=sha(base),spatial_manifest_sha256=sha(OUT/'range_bev_train/manifest.json'),roots=m['roots'],scenes=m['scenes'],maps=len(groups),point_weighting='Exact original inverse inclusion equalScene/Root/window weights; uniformly shuffled starting-map groups with group/root normalization.',auxiliary='Published future certificate class UNION, auxiliary TARGET only; never encoder inputs, not a physical occupancy relabeling.',objective='BCE revelation pos8 +8R conditional BCE; .2 weighted FREE/BLOCKED soft Dice; .25 conditional auxiliary certificate class BCE.',precision='AMP spatial encoder/head, F32 frozen legal point/range prior; heights preserved.')
    save(dest/'config.json',cfg)
    def tensor(x,dtype=None):return torch.from_numpy(np.ascontiguousarray(x,dtype=dtype)).to(device)
    for epoch in range(1,epochs+1):
        totals=np.zeros(3)
        for start in range(0,len(groups),batch_size):
            if start==0:order=rng.permutation(len(groups))
            chosen=[groups[i]for i in order[start:start+batch_size]];x=tensor(np.concatenate([g['x']for g in chosen]),np.float32);extra=tensor(np.concatenate([g['extra']for g in chosen]),np.float32);y=tensor(np.concatenate([g['y']for g in chosen]),np.int64);w=tensor(np.concatenate([g['weight']for g in chosen]),np.float32);indices=tensor(np.concatenate([g['indices']for g in chosen]),np.int64);mapids=tensor(np.repeat(np.arange(len(chosen)),[len(g['y'])for g in chosen]),np.int64)
            with torch.cuda.amp.autocast():encoded=model.encode(tensor(np.stack([g['maps']for g in chosen]),np.float32));rl,bl,_=model.raw(x,extra,indices,mapids,encoded);auxlogit=model.aux(encoded[0])[:,0]
            r=(y>0).float();ce=F.binary_cross_entropy_with_logits(rl.float(),r,pos_weight=rl.new_tensor(8.),reduction='none')+8*r*F.binary_cross_entropy_with_logits(bl.float(),(y==2).float(),reduction='none');main=(ce*w).sum()*len(groups)/(m['roots']*len(chosen));pr=(rl.float()-np.log(8.)).sigmoid();pb=bl.float().sigmoid();dice=0.
            for cls,prob in [(1,pr*(1-pb)),(2,pr*pb)]:target=(y==cls).float();dice+=1-(2*(w*prob*target).sum()+1.e-7)/((w*(prob+target)).sum()+1.e-7)
            aux_target=tensor(np.stack([g['aux']for g in chosen]),np.float32);aux_mask=tensor(np.stack([g['mask']for g in chosen]),np.float32);ac=F.binary_cross_entropy_with_logits(auxlogit.float(),aux_target,reduction='none');per=(ac*aux_mask).sum((1,2))/aux_mask.sum((1,2)).clamp_min(1);aw=tensor([g['aux_weight']for g in chosen],np.float32);auxloss=(per*aw).mean()*len(groups)/m['roots'];loss=main+.1*dice+.25*auxloss
            if not torch.isfinite(loss):raise ValueError('nonfinite spatial reconstruction loss')
            optim.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optim);grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optim);scaler.update();totals+=[float(main),float(dice),float(auxloss)]
        rec=dict(epoch=epoch,main_loss=float(totals[0]/np.ceil(len(groups)/batch_size)),soft_dice_loss=float(totals[1]/np.ceil(len(groups)/batch_size)),auxiliary_loss=float(totals[2]/np.ceil(len(groups)/batch_size)),gradient_norm_last=float(grad),seconds=time.perf_counter()-tick);logs.append(rec);save(dest/'learning.json',logs);print(rec,flush=True)
        if epoch%20==0 or epoch==epochs:torch.save(dict(schema=cfg['schema'],config=cfg,model=model.state_dict(),epoch=epoch,threshold=.2),dest/('epoch%02d.pt'%epoch))
    save(dest/'complete.json',dict(epochs=epochs,seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=80);p.add_argument('--device',default='cuda:0');p.add_argument('--name',default='range_spatial64');train(**vars(p.parse_args()))
