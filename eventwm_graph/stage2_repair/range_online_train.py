"""Bounded fit on expanded existing recordings, fresh original-cell features."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .diagnose import OUT,read,save,sha
from .range_online_data import inventory,OnlineRangeGroups
from .range_online_features import LegalFeatureBuilder
from .range_spatial import SpatialEvidenceNet
from .range_bev import SIZE
from .fresh_data import draw,keep
from .full_evaluate import labels


def make_batch(items,builder,device):
    arrays={k:[]for k in ['x','extra','y','weight','indices','mapids']}
    maps=[];aux=[];masks=[];auxweights=[];ledger=dict(windows=0,out_of_domain_positives=0,known_terminal_differences=0,original_candidate_rows=0,sampled_rows=0)
    for mi,d in enumerate(items):
        paths=[e['path']for e in d['windows']]
        prepared,contexts=builder.prepare(keys=d['keys'],states=d['states'],heights=d['heights'],geometry=d['geometry'],ego=d['ego'],now=d['now'],paths=paths)
        rng=np.random.default_rng(d['sample_seed']);keysets=[];ys=[];weights=[]
        for context,e in zip(contexts,d['windows']):
            keys=context.candidates('both');target={tuple(k):(int(s),0.)for k,s in zip(e['target_keys'],e['target_states'])}
            y,_,missing=labels(keys,target);ij=keys-context.lo
            near=np.minimum(context.dist[0][ij[:,0],ij[:,1]],context.dist[1][ij[:,0],ij[:,1]])<=1.
            strata=[np.flatnonzero(y>0),np.flatnonzero((y==0)&near),np.flatnonzero((y==0)&~near)]
            selected,w=draw(strata,len(keys),d['windows_in_root'],rng)
            keysets.append(keys[selected]);ys.append(y[selected]);weights.append(w*d['scene_factor'])
            ledger['windows']+=1;ledger['out_of_domain_positives']+=len(missing);ledger['known_terminal_differences']+=e['known_terminal_differences'];ledger['original_candidate_rows']+=len(keys);ledger['sampled_rows']+=len(selected)
        # Builder consumes only named legal fields and supplied paths/keys.
        # Future certificate arrays are attached to losses after extraction.
        xs,extra,canvas,low,_=builder.build(prepared=prepared,contexts=contexts,keysets=keysets,known_keys=d['keys'],known_states=d['states'],known_heights=d['heights'],ego=d['ego'],packet=d['packet'],tracks=d['tracks'])
        for x,ex,keys,y,w in zip(xs,extra,keysets,ys,weights):
            arrays['x'].append(x);arrays['extra'].append(ex);arrays['y'].append(y);arrays['weight'].append(w);arrays['indices'].append(keys-low);arrays['mapids'].append(np.full(len(keys),mi,np.int64))
        maps.append(canvas);counts=np.zeros((2,SIZE,SIZE),np.float32)
        for e in d['windows']:
            ij=(e['target_keys']-low)//2;valid=np.all((ij>=0)&(ij<SIZE),1);ij=ij[valid]
            np.add.at(counts,(e['target_states'][valid].astype(int)-1,ij[:,0],ij[:,1]),1)
        total=counts.sum(0);aux.append(counts[1]/np.maximum(total,1));masks.append(total>0);auxweights.append(len(paths)/d['windows_in_root']*d['scene_factor'])
    batch={k:torch.cat(arrays[k])for k in ['x','extra']}
    for k in ['y','weight','indices','mapids']:
        dtype=np.float32 if k=='weight'else np.int64;batch[k]=torch.from_numpy(np.ascontiguousarray(np.concatenate(arrays[k]),dtype=dtype)).to(device)
    for k,v in [('maps',maps),('aux',aux),('mask',masks),('auxweight',auxweights)]:batch[k]=torch.from_numpy(np.ascontiguousarray(v,np.float32)).to(device)
    return batch,ledger


def run(roots=4096,epochs=4,device='cuda:0',name='range_online4096',warm=None,workers=4,batch_size=4,wait=False,augment_cardinal=False):
    torch.set_num_threads(1);torch.manual_seed(5511);torch.backends.cuda.matmul.allow_tf32=False
    tick=time.perf_counter();m=inventory(roots);print('INVENTORY',{k:v for k,v in m.items()if k!='records'},flush=True)
    dest=OUT/'training'/name;dest.mkdir(exist_ok=False);save(dest/'snapshot.json',m)
    while True:
        missing=[r['source']['root_id']for r in m['records']if not Path(r['sensor_path']).with_suffix('.json').exists()]
        if not missing:break
        save(dest/'waiting.json',dict(ready=m['roots']-len(missing),requested=m['roots'],pending=missing,seconds=time.perf_counter()-tick))
        if not wait:raise FileNotFoundError('sensor packets pending; use --wait or a completed eligible snapshot')
        print('WAITING',m['roots']-len(missing),'/',m['roots'],flush=True);time.sleep(30)
    group_class=OnlineRangeGroups
    if augment_cardinal:
        from .cardinal_augment import CardinalOnlineGroups
        group_class=CardinalOnlineGroups
    data=group_class(m);loader=DataLoader(data,batch_size=batch_size,num_workers=workers,collate_fn=keep,persistent_workers=True,multiprocessing_context='fork',prefetch_factor=2)
    warm=Path(warm)if warm else OUT/'training/support_pretrained256_control/selected.pt'
    saved=torch.load(warm,map_location='cpu');model=SpatialEvidenceNet().to(device);model.load_state_dict(saved['model']);builder=LegalFeatureBuilder(model,device)
    optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0002,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler()
    cfg=dict(schema='legal_range_spatial_1',name=name,epochs=epochs,roots=m['roots'],scenes=m['scenes'],windows=m['windows'],groups=m['groups'],base_checkpoint=saved['config']['base_checkpoint'],warm=str(warm),warm_sha256=sha(warm),snapshot_sha256=sha(dest/'snapshot.json'),source_manifest_sha256=m['source_manifest_sha256'],observed_obstacle_schema='capped_2m_complete',device=device,batch_size=batch_size,workers=workers,learning_rate=.0002,sampling='Fresh128positive/384nearUNKNOWN/128remainingUNKNOWN every pass over all original windows; exact inverse inclusion equalScene/Root/window weights. Full candidate construction before sampling, no hard windows dropped.',objective='Same primary pos8revelation BCE +8R conditional BCE +.1typedDice +.25auxiliary published-certificate-class UNION.',inputs='Only current legal graph, supplied trajectory, measured ancestor/current range arrays and legally observed track history/current camera. No private full scene or future truth inputs.',sources={n:sha(Path(__file__).with_name(n))for n in ['range_online_train.py','range_online_data.py','range_online_features.py','range_fast.py','range_gpu.py','range_spatial.py']})
    cfg['dev_feature_manifest']=str((OUT/'range_evidence_dev_v2/manifest.json').resolve())
    cfg['cardinal_augmentation']=bool(augment_cardinal)
    if augment_cardinal:cfg['sources']['cardinal_augment.py']=sha(Path(__file__).with_name('cardinal_augment.py'))
    save(dest/'config.json',cfg);logs=[];tick=time.perf_counter();updates=0;total_updates=epochs*int(np.ceil(m['groups']/batch_size))
    for epoch in range(1,epochs+1):
        data.epoch.value=epoch;totals=np.zeros(3);batches=0;seen_groups=0;seen_roots=set();ledger={}
        for items in loader:
            batch,counts=make_batch(items,builder,device)
            for k,v in counts.items():ledger[k]=ledger.get(k,0)+v
            progress=min(1.,updates/max(1,total_updates));lr=.0002*min(1.,(updates+1)/100)*(.1+.9*.5*(1+np.cos(np.pi*progress)))
            for group in optim.param_groups:group['lr']=lr
            with torch.cuda.amp.autocast():encoded=model.encode(batch['maps']);rl,bl,_=model.raw(batch['x'],batch['extra'],batch['indices'],batch['mapids'],encoded);auxlogit=model.aux(encoded[0])[:,0]
            y=batch['y'];w=batch['weight'];r=(y>0).float()
            ce=F.binary_cross_entropy_with_logits(rl.float(),r,pos_weight=rl.new_tensor(8.),reduction='none')+8*r*F.binary_cross_entropy_with_logits(bl.float(),(y==2).float(),reduction='none')
            main=(ce*w).sum()*m['groups']/(m['roots']*len(items));pr=(rl.float()-np.log(8.)).sigmoid();pb=bl.float().sigmoid();dice=0.
            for cls,prob in [(1,pr*(1-pb)),(2,pr*pb)]:truth=(y==cls).float();dice+=1-(2*(w*prob*truth).sum()+1.e-7)/((w*(prob+truth)).sum()+1.e-7)
            ac=F.binary_cross_entropy_with_logits(auxlogit.float(),batch['aux'],reduction='none');per=(ac*batch['mask']).sum((1,2))/batch['mask'].sum((1,2)).clamp_min(1);aux=(per*batch['auxweight']).mean()*m['groups']/m['roots'];loss=main+.1*dice+.25*aux
            if not torch.isfinite(loss):raise ValueError('nonfinite online increment loss')
            optim.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(optim);gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);encoder_gradient=float(model.e0.net[0].weight.grad.norm());scaler.step(optim);scaler.update()
            updates+=1;batches+=1;seen_groups+=len(items);seen_roots.update(d['root_id']for d in items);totals+=[float(main),float(dice),float(aux)]
            if batches%100==0:print(dict(epoch=epoch,batches=batches,groups=seen_groups,roots=len(seen_roots),loss=float(loss),gradient=float(gradient),encoder_gradient=encoder_gradient,seconds=time.perf_counter()-tick),flush=True)
        assert seen_groups==m['groups']and len(seen_roots)==m['roots']and ledger['windows']==m['windows']
        record=dict(epoch=epoch,roots=len(seen_roots),groups=seen_groups,losses=(totals/batches).tolist(),ledger=ledger,seconds=time.perf_counter()-tick);logs.append(record);save(dest/'learning.json',logs)
        torch.save(dict(schema=cfg['schema'],config=cfg,epoch=epoch,model=model.state_dict(),threshold=.2,optimizer=optim.state_dict(),scaler=scaler.state_dict(),updates=updates),dest/('epoch%02d.pt'%epoch));print('EPOCH',record,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=int,default=4096);p.add_argument('--epochs',type=int,default=4);p.add_argument('--device',default='cuda:0');p.add_argument('--name',default='range_online4096');p.add_argument('--warm');p.add_argument('--workers',type=int,default=4);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--wait',action='store_true');p.add_argument('--augment-cardinal',action='store_true');run(**vars(p.parse_args()))
