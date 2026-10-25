"""Same18-window fit control for the shared current-map reconstruction model."""
import argparse
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save
from .shared_map import SharedMapNet,map_canvas
from .shared_train import loss_fn
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from scripts.stage2_diagnostic.common import extended_rates


def fit(steps=900,device='cuda:0',dense=False,all_windows=False,anchored_fine=False):
    torch.set_num_threads(1);torch.manual_seed(3011);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    meta=read(OUT/'trace.json')['examples'];data=[];lookup={}
    for i,e in enumerate(meta,1):
        rid=e['root_id']
        if rid not in lookup:
            row=read(HF/'packed/pilot_fit'/(rid+'.json'));a=load_arrays(row);tr=read(HF/'temporal/pilot_fit'/(rid+'.json'))
            with np.load(tr['path'],allow_pickle=False)as z:last=z['last']
            lookup[rid]=(row,a,last)
        row,a,last=lookup[rid];v=get_segment(a,e['index']);gi=int(a['graph_index'][e['index']]);gs=slice(*a['goffsets'][gi:gi+2]);p=v[3];d=map_canvas(v[0],v[1],v[2],last[gs],p[0,0],v[-1],p[0,1:4],anchor=anchored_fine,halo=128 if anchored_fine else 96)
        with np.load(OUT/'examples'/('%02d_features.npz'%i),allow_pickle=False)as z:x=z['x'];y=z['y']
        with np.load(OUT/'examples'/('%02d.npz'%i),allow_pickle=False)as z:keys=z['keys']
        b=dict(maps=torch.from_numpy(d['maps'][None]).to(device),sizes=torch.tensor([d['maps'].shape[1:]],device=device),x=torch.from_numpy(x).to(device),y=torch.from_numpy(y.astype(np.int64)).to(device),indices=torch.from_numpy((keys-d['lo']).astype(np.int64)).to(device),map_ids=torch.zeros(len(x),dtype=torch.long,device=device))
        if anchored_fine:b['fine']=torch.from_numpy(d['fine'][None]).to(device)
        b['strata']=[torch.nonzero(b['y']>0).flatten(),torch.nonzero(b['y']==0).flatten()];data.append(b)
    combined=None
    if all_windows:
        n=len(data);hh=max(b['maps'].shape[2]for b in data);ww=max(b['maps'].shape[3]for b in data);maps=torch.zeros((n,10 if anchored_fine else 8,hh,ww),device=device);fine=torch.zeros((n,5,hh*2,ww*2),device=device)if anchored_fine else None;sizes=[];fields={k:[]for k in ['x','y','indices','map_ids','weight']}
        for i,b in enumerate(data):
            h,w=b['maps'].shape[2:];maps[i,:,:h,:w]=b['maps'][0];sizes.append([h,w])
            if fine is not None:fine[i,:,:h*2,:w*2]=b['fine'][0]
            for k in ['x','y','indices']:fields[k].append(b[k])
            fields['map_ids'].append(torch.full((len(b['y']),),i,device=device,dtype=torch.long));fields['weight'].append(torch.full((len(b['y']),),1/len(b['y']),device=device))
        combined={k:torch.cat(v)for k,v in fields.items()};combined.update(maps=maps,sizes=torch.tensor(sizes,device=device),cell_weight=combined['weight'],window_id=combined['map_ids'],window_weight=torch.ones(n,device=device),eps=torch.tensor([1/len(b['y'])for b in data],device=device))
        if fine is not None:combined['fine']=fine
    model=SharedMapNet(map_channels=10 if anchored_fine else 8,fine=anchored_fine).to(device);model.initialize(OUT/'training/point_residual_full/epoch06.pt',HF/'training/spatial_bce/epoch08.pt');optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.001,weight_decay=0.);scaler=torch.cuda.amp.GradScaler();rng=np.random.default_rng(3011);logs=[];tick=time.perf_counter()
    name='shared_smallfit_all_dense'if all_windows else'shared_smallfit_dense'if dense else'shared_smallfit_sampled_fixed'
    if anchored_fine:name+='_'+('anchored_fine')
    for step in range(steps+1):
        if step%150==0 or step==steps:
            cms=[];preds=[]
            with torch.inference_mode():
                for b in data:
                    enc=model.encode(b['maps'],b['sizes'],b.get('fine'));pp=[]
                    for begin in range(0,len(b['x']),65536):
                        sl=slice(begin,begin+65536);prob,_=model.probabilities(b['x'][sl],b['indices'][sl],b['map_ids'][sl],enc);pp.append(prob)
                    p=torch.cat(pp);hard=torch.where(p[:,1]+p[:,2]>=.2,torch.where(p[:,1]>=p[:,2],1,2),0);cms.append(torch.bincount(3*b['y']+hard,minlength=9).reshape(3,3).cpu().numpy());preds.append(hard.cpu().numpy())
            r=dict(step=step,seconds=time.perf_counter()-tick,metrics=extended_rates(np.mean(cms,axis=0)),pooled_confusion=np.sum(cms,axis=0).tolist());logs.append(r);save(OUT/(name+'.json'),logs);print(name,step,r['metrics']['label_aware_delta_iou'],r['metrics']['revelation_iou'],flush=True)
            np.savez_compressed(OUT/(name+'_predictions.npz'),**{'prediction_%02d'%i:p for i,p in enumerate(preds,1)})
        if step==steps:break
        full=data[int(rng.integers(len(data)))];ids=[];weights=[]
        for s in full['strata']:
            n=min(len(s),256)
            if n:ids.append(s[torch.randperm(len(s),device=device)[:n]]);weights.append(torch.full((n,),len(s)/(n*len(full['y'])),device=device))
        ids=torch.cat(ids);w=torch.cat(weights)
        if dense:ids=torch.arange(len(full['y']),device=device);w=torch.full((len(ids),),1/len(ids),device=device)
        b={k:full[k][ids]for k in ['x','y','indices','map_ids']};b.update(maps=full['maps'],sizes=full['sizes'],weight=w,cell_weight=w,window_id=torch.zeros(len(ids),device=device,dtype=torch.long),window_weight=torch.ones(1,device=device),eps=torch.tensor([1/len(full['y'])],device=device))
        if anchored_fine:b['fine']=full['fine']
        if combined is not None:b=combined
        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():encoded=model.encode(b['maps'],b['sizes'],b.get('fine'))
        r,bl,_=model.raw(b['x'],b['indices'],b['map_ids'],encoded);loss,_=loss_fn(r,bl,b,1.,.2);scaler.scale(loss).backward();scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optim);scaler.update()
    torch.save(dict(schema='shared_smallfit_diagnostic',model=model.state_dict(),steps=steps),OUT/(name+'.pt'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=900);p.add_argument('--device',default='cuda:0');p.add_argument('--dense',action='store_true');p.add_argument('--all-windows',action='store_true');p.add_argument('--anchored-fine',action='store_true');fit(**vars(p.parse_args()))
