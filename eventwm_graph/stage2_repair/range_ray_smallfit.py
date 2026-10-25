"""Full-cell fitting diagnostic for the bounded learned-ray candidate."""
import argparse
from collections import defaultdict
import time
import numpy as np
import torch
from torch.nn import functional as F
from .diagnose import OUT,HF,read,save,source_rows,member
from .range_ray_model import RayEvidenceNet
from .range_ray_data import camera_lines
from .legal_camera import current_extrinsic
from .range_spatial_eval import full_training_groups
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from scripts.stage2_diagnostic.common import extended_rates


def run(steps=300,device='cuda:0'):
    torch.set_num_threads(1);torch.manual_seed(14317);torch.backends.cuda.matmul.allow_tf32=False
    _,groups=full_training_groups();byroot=defaultdict(list)
    for gi,g in enumerate(groups):
        for e in g['segments']:byroot[g['root_id']].append((gi,g,e))
    source,_=source_rows('train');source={r['root_id']:r for r in source};root_data={}
    chosen=[]
    for rid,entries in sorted(byroot.items()):
        chosen.extend(entries[int(i)]for i in np.linspace(0,len(entries)-1,6).astype(int))
        legal=member(source[rid],'legal_root.json.gz');ob=legal['history'][-1]
        translation,_=current_extrinsic(np.array([[ob['timestamp'],*ob['robot_pose']]]),ob['camera_origin'],ob['camera_rotation'])
        packed=read(HF/'packed/pilot_fit'/(rid+'.json'));a=load_arrays(packed);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed['segments'])}
        root_data[rid]=(translation,a,lookup)
    arrays=defaultdict(list);maps=[];mapid={};metadata=[];cursor=0
    for gi,g,e in chosen:
        if gi not in mapid:mapid[gi]=len(maps);maps.append(g['maps'])
        sl=slice(e['start'],e['end']);n=sl.stop-sl.start
        for key in ['x','extra','y','indices']:arrays[key].append(g[key][sl])
        arrays['mapids'].append(np.full(n,mapid[gi],np.int64));arrays['weight'].append(np.full(n,1/(len(byroot)*6*n),np.float32))
        translation,a,lookup=root_data[g['root_id']];p=get_segment(a,lookup[e['start_node'],e['end_node']])[3]
        origins=camera_lines(p,translation,a['geometry'][0],g['low']);arrays['origins'].append(np.broadcast_to(origins,(n,3,2)))
        metadata.append(dict(root_id=g['root_id'],start=cursor,end=cursor+n));cursor+=n
    def tensor(x,dtype):return torch.from_numpy(np.ascontiguousarray(x,dtype=dtype)).to(device)
    a={k:tensor(np.concatenate(v),np.int64 if k in ['y','indices','mapids']else np.float32)for k,v in arrays.items()};maps=tensor(np.stack(maps),np.float32)
    model=RayEvidenceNet().to(device);model.initialize_spatial(OUT/'training/support256_teacher/selected.pt')
    optim=torch.optim.AdamW([p for p in model.parameters()if p.requires_grad],lr=.0005,weight_decay=.0001);scaler=torch.cuda.amp.GradScaler();tick=time.perf_counter();logs=[]
    for step in range(steps+1):
        if step:
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():encoded=model.encode(maps)
            for begin in range(0,cursor,32768):
                sl=slice(begin,begin+32768)
                with torch.cuda.amp.autocast():rl,bl,_=model.raw(a['x'][sl],a['extra'][sl],a['indices'][sl],a['mapids'][sl],encoded,a['origins'][sl])
                y=a['y'][sl];r=(y>0).float();ce=F.binary_cross_entropy_with_logits(rl.float(),r,pos_weight=rl.new_tensor(8.),reduction='none')+8*r*F.binary_cross_entropy_with_logits(bl.float(),(y==2).float(),reduction='none')
                loss=(ce*a['weight'][sl]).sum();scaler.scale(loss).backward(retain_graph=begin+32768<cursor)
            scaler.unscale_(optim);gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);ray_gradient=float(model.ray_map.weight.grad.norm());scaler.step(optim);scaler.update()
        if step in [0,50,100,steps]:
            probability=[]
            with torch.inference_mode(),torch.cuda.amp.autocast():
                encoded=model.encode(maps)
                for begin in range(0,cursor,32768):
                    sl=slice(begin,begin+32768);p,_=model.probabilities(a['x'][sl],a['extra'][sl],a['indices'][sl],a['mapids'][sl],encoded,a['origins'][sl]);probability.append(p.float().cpu().numpy())
            hard=np.concatenate(probability).argmax(1);truth=a['y'].cpu().numpy();cms=defaultdict(list);pooled=np.zeros((3,3),np.int64)
            for e in metadata:
                sl=slice(e['start'],e['end']);cm=np.bincount(truth[sl]*3+hard[sl],minlength=9).reshape(3,3);cms[e['root_id']].append(cm);pooled+=cm
            mean=np.mean([np.mean(v,0)for v in cms.values()],0)
            record=dict(step=step,metrics=extended_rates(mean),pooled=extended_rates(pooled),gradient=float(gradient)if step else None,ray_gradient=ray_gradient if step else None,seconds=time.perf_counter()-tick)
            logs.append(record);save(OUT/'range_ray_smallfit.json',dict(role='TRAINING FIT ONLY, all original cells on fixed24trainingwindows; NOT generalization or acceptance',rows=cursor,windows=metadata,logs=logs));print(record,flush=True)
    torch.save(dict(schema='legal_range_ray_1',model=model.state_dict(),role='TRAINING FIT ONLY'),OUT/'range_ray_smallfit.pt')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=300);p.add_argument('--device',default='cuda:0');run(**vars(p.parse_args()))
