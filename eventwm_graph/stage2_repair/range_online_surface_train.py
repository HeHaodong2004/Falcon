"""Matched expanded fit with training-only later measured surface completion."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .diagnose import OUT,read,save,sha
from .range_surface_data import SurfaceOnlineGroups
from .range_online_features import LegalFeatureBuilder
from .range_spatial import SpatialEvidenceNet
from .range_bev import SIZE
from .fresh_data import draw,keep
from .full_evaluate import labels


from .range_online_train import make_batch as primary_batch


def make_batch(items,builder,device):
    batch,ledger=primary_batch(items,builder,device)
    for item in items:
        expected=(np.floor(np.asarray(item['ego'])[[0,2]]/item['geometry'][0]).astype(np.int64)//32)*32-SIZE
        np.testing.assert_array_equal(item['surface_low'],expected)
    batch['surface']=torch.from_numpy(np.ascontiguousarray([d['surface_targets']for d in items],np.float32)).to(device)
    return batch,ledger


def run(snapshot,targets,epochs=4,device='cuda:1',name='range_surface_online4096',warm=None,workers=4,batch_size=4,smoke_groups=0,resume=None):
    torch.set_num_threads(1);torch.manual_seed(5511);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.perf_counter();m=read(snapshot);print('INVENTORY',{k:v for k,v in m.items()if k!='records'},flush=True)
    dest=OUT/'training'/name
    if resume:
        if not dest.is_dir()or read(dest/'snapshot.json')!=m:raise ValueError('resume requires the existing unchanged training snapshot')
    else:dest.mkdir(exist_ok=False);save(dest/'snapshot.json',m)
    data=SurfaceOnlineGroups(m,targets,sha(snapshot));loader=DataLoader(data,batch_size=batch_size,num_workers=workers,collate_fn=keep,persistent_workers=True,multiprocessing_context='fork',prefetch_factor=2)
    warm=Path(resume or warm)if(resume or warm)else OUT/'training/support_pretrained256_control/selected.pt'
    saved=torch.load(warm,map_location='cpu');model=SpatialEvidenceNet().to(device);model.load_state_dict(saved['model']);builder=LegalFeatureBuilder(model,device)
    head=torch.nn.Conv2d(24,3,1).to(device);parameters=[p for p in model.parameters()if p.requires_grad]+list(head.parameters());optim=torch.optim.AdamW(parameters,lr=.0002,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler()
    cfg=dict(schema='legal_range_spatial_1',name=name,epochs=epochs,roots=m['roots'],scenes=m['scenes'],windows=m['windows'],groups=m['groups'],base_checkpoint=saved['config']['base_checkpoint'],warm=str(warm),warm_sha256=sha(warm),snapshot_sha256=sha(dest/'snapshot.json'),source_manifest_sha256=m['source_manifest_sha256'],observed_obstacle_schema='capped_2m_complete',device=device,batch_size=batch_size,workers=workers,learning_rate=.0002,sampling='Fresh128positive/384nearUNKNOWN/128remainingUNKNOWN every pass over all original windows; exact inverse inclusion equalScene/Root/window weights. Full candidate construction before sampling, no hard windows dropped.',objective='Same primary pos8revelation BCE +8R conditional BCE +.1typedDice +.25auxiliary published-certificate-class UNION.',inputs='Only current legal graph, supplied trajectory, measured ancestor/current range arrays and legally observed track history/current camera. No private full scene or future truth inputs.',sources={n:sha(Path(__file__).with_name(n))for n in ['range_online_surface_train.py','range_surface_data.py','range_online_train.py','range_online_data.py','range_online_features.py','range_fast.py','range_gpu.py','range_spatial.py']})
    cfg.update(training_only_surface_targets=str(Path(targets).resolve()),target_snapshot_sha256=sha(snapshot),surface_coefficient=.5,surface_semantics='Later measured surface at3heights, only observed initiallyUNKNOWN regions. Auxiliary TARGETS only, not certificate relabeling. Training-only head discarded.',smoke_groups=smoke_groups)
    cfg['dev_feature_manifest']=str((OUT/'range_evidence_dev_v2/manifest.json').resolve())
    cfg['sources']['cuda_module.py']=sha(Path(__file__).with_name('cuda_module.py'));logs=[];updates=0;first_epoch=1;elapsed_base=0.
    if resume:
        if saved['config']['snapshot_sha256']!=sha(dest/'snapshot.json')or saved['config']['epochs']!=epochs or saved['config']['batch_size']!=batch_size:raise ValueError('resume snapshot or optimization schedule changed')
        if saved['config']['training_only_surface_targets']!=cfg['training_only_surface_targets']:raise ValueError('resume surface targets changed')
        head.load_state_dict(saved['training_only_surface_head']);optim.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);updates=int(saved['updates']);first_epoch=int(saved['epoch'])+1
        logs=[r for r in read(dest/'learning.json')if r['epoch']<first_epoch];elapsed_base=float(logs[-1]['seconds'])if logs else 0.
        cfg.update(resume_checkpoint=str(warm.resolve()),resume_sha256=sha(warm),resume_epoch=int(saved['epoch']),original_warm=saved['config']['warm'],resume_note='Exact completed-epoch model, auxiliary head, optimizer, scaler and update state; original data/schedule/config retained. CUDA module-lifetime repair.')
        save(dest/('resume_epoch%02d_config.json'%saved['epoch']),cfg)
    else:save(dest/'config.json',cfg)
    tick=time.perf_counter();total_updates=epochs*int(np.ceil(m['groups']/batch_size))
    for epoch in range(first_epoch,epochs+1):
        data.epoch.value=epoch;totals=np.zeros(4);batches=0;seen_groups=0;seen_roots=set();ledger={}
        for items in loader:
            batch,counts=make_batch(items,builder,device)
            for k,v in counts.items():ledger[k]=ledger.get(k,0)+v
            progress=min(1.,updates/max(1,total_updates));lr=.0002*min(1.,(updates+1)/100)*(.1+.9*.5*(1+np.cos(np.pi*progress)))
            for group in optim.param_groups:group['lr']=lr
            with torch.cuda.amp.autocast():encoded=model.encode(batch['maps']);rl,bl,_=model.raw(batch['x'],batch['extra'],batch['indices'],batch['mapids'],encoded);auxlogit=model.aux(encoded[0])[:,0];surface_logits=head(encoded[0])
            y=batch['y'];w=batch['weight'];r=(y>0).float()
            ce=F.binary_cross_entropy_with_logits(rl.float(),r,pos_weight=rl.new_tensor(8.),reduction='none')+8*r*F.binary_cross_entropy_with_logits(bl.float(),(y==2).float(),reduction='none')
            main=(ce*w).sum()*m['groups']/(m['roots']*len(items));pr=(rl.float()-np.log(8.)).sigmoid();pb=bl.float().sigmoid();dice=0.
            for cls,prob in [(1,pr*(1-pb)),(2,pr*pb)]:truth=(y==cls).float();dice+=1-(2*(w*prob*truth).sum()+1.e-7)/((w*(prob+truth)).sum()+1.e-7)
            ac=F.binary_cross_entropy_with_logits(auxlogit.float(),batch['aux'],reduction='none');per=(ac*batch['mask']).sum((1,2))/batch['mask'].sum((1,2)).clamp_min(1);aux=(per*batch['auxweight']).mean()*m['groups']/m['roots'];surface_target=batch['surface'];surface_mask=surface_target[:,3:]*(1-batch['maps'][:,5:6]).clamp(0,1);sc=F.binary_cross_entropy_with_logits(surface_logits.float(),surface_target[:,:3],pos_weight=surface_logits.new_tensor([8.,8.,8.])[:,None,None],reduction='none');per=(sc*surface_mask).sum((1,2,3))/surface_mask.sum((1,2,3)).clamp_min(1);surface=(per*batch['auxweight']).mean()*m['groups']/m['roots'];loss=main+.1*dice+.25*aux+.5*surface
            if not torch.isfinite(loss):raise ValueError('nonfinite online increment loss')
            optim.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optim);gradient=torch.nn.utils.clip_grad_norm_(parameters,5.);encoder_gradient=float(model.e0.net[0].weight.grad.norm());scaler.step(optim);scaler.update()
            updates+=1;batches+=1;seen_groups+=len(items);seen_roots.update(d['root_id']for d in items);totals+=[float(main),float(dice),float(aux),float(surface)]
            if batches==1 or batches%100==0:print(dict(epoch=epoch,batches=batches,groups=seen_groups,roots=len(seen_roots),loss=float(loss),gradient=float(gradient),encoder_gradient=encoder_gradient,surface_head_gradient=float(head.weight.grad.norm()),surface_loss=float(surface),seconds=time.perf_counter()-tick),flush=True)
            if smoke_groups and seen_groups>=smoke_groups:
                save(dest/'smoke_complete.json',dict(groups=seen_groups,ledger=ledger,loss=float(loss),encoder_gradient=encoder_gradient,surface_head_gradient=float(head.weight.grad.norm()),seconds=time.perf_counter()-tick,role='Training smoke only; not reconstruction/generalization.'));return
        assert seen_groups==m['groups']and len(seen_roots)==m['roots']and ledger['windows']==m['windows']
        record=dict(epoch=epoch,roots=len(seen_roots),groups=seen_groups,losses=(totals/batches).tolist(),ledger=ledger,seconds=elapsed_base+time.perf_counter()-tick);logs.append(record);save(dest/'learning.json',logs)
        torch.save(dict(schema=cfg['schema'],config=cfg,epoch=epoch,model=model.state_dict(),threshold=.2,training_only_surface_head=head.state_dict(),optimizer=optim.state_dict(),scaler=scaler.state_dict(),updates=updates),dest/('epoch%02d.pt'%epoch));print('EPOCH',record,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--targets',required=True);p.add_argument('--epochs',type=int,default=4);p.add_argument('--device',default='cuda:1');p.add_argument('--name',default='range_surface_online4096');p.add_argument('--warm');p.add_argument('--workers',type=int,default=4);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--smoke-groups',type=int,default=0);p.add_argument('--resume');run(**vars(p.parse_args()))
