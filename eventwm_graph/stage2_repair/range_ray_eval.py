"""Original-cell pilot scoring of the learned legal viewing-line model."""
import argparse
from collections import defaultdict
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save,sha,source_rows,member
from .range_ray_model import RayEvidenceNet
from .range_ray_data import camera_lines
from .legal_camera import current_extrinsic
from .range_spatial_train import load_groups
from .range_spatial_eval import full_training_groups
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from eventwm_graph.stage2_hf.spatial_eval import score
from eventwm_graph.stage2_hf.light_eval import aggregate
from scripts.stage2_diagnostic.common import extended_rates


def prepare_groups(training=False):
    manifest,groups=full_training_groups()if training else load_groups('dev')
    source,_=source_rows('train'if training else'dev');source={r['root_id']:r for r in source};cache={}
    for g in groups:
        rid=g['root_id']
        if rid not in cache:
            ob=member(source[rid],'legal_root.json.gz')['history'][-1]
            translation,_=current_extrinsic(np.array([[ob['timestamp'],*ob['robot_pose']]]),ob['camera_origin'],ob['camera_rotation'])
            packed=read(HF/'packed'/('pilot_fit'if training else'dev_pilot')/(rid+'.json'));a=load_arrays(packed);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed['segments'])}
            cache[rid]=(translation,a,lookup)
        translation,a,lookup=cache[rid];origins=np.empty((len(g['y']),3,2),np.float32)
        for e in g['segments']:
            v=get_segment(a,lookup[e['start_node'],e['end_node']]);path=v[3]
            origins[e['start']:e['end']]=camera_lines(path,translation,a['geometry'][0],g['low'])
            e['zero_horizon']=len(path)==1
            keys=g['indices'][e['start']:e['end']]+g['low']
            # Independent integer-key inclusion, preserving all missing targets.
            dtype=np.dtype([('x','<i4'),('z','<i4')])
            got=np.ascontiguousarray(keys,np.int32).view(dtype).ravel()
            wanted=np.ascontiguousarray(v[4],np.int32).view(dtype).ravel()
            e['missing_states']=v[5][~np.isin(wanted,got)]
            e['zero_new']=not bool(len(v[4]))
        g['ray_origins']=origins
    return manifest,groups


def evaluate(checkpoint,device='cuda:1',training=False,_prepared=None):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;tick=time.perf_counter()
    saved=torch.load(checkpoint,map_location='cpu');model=RayEvidenceNet().to(device).eval();model.load_state_dict(saved['model'])
    manifest,groups=_prepared if _prepared is not None else prepare_groups(training)
    roots=defaultdict(list);scenes={};pooled=np.zeros((14,3,3),np.int64);count=0
    def tensor(x,dtype):return torch.from_numpy(np.ascontiguousarray(x,dtype=dtype)).to(device)
    for g in groups:
        probabilities=[]
        with torch.inference_mode(),torch.cuda.amp.autocast():
            encoded=model.encode(tensor(g['maps'][None],np.float32))
            for begin in range(0,len(g['y']),32768):
                sl=slice(begin,begin+32768);x=tensor(g['x'][sl],np.float32);extra=tensor(g['extra'][sl],np.float32);indices=tensor(g['indices'][sl],np.int64);ids=torch.zeros(len(x),device=device,dtype=torch.long);origins=tensor(g['ray_origins'][sl],np.float32)
                p,_=model.probabilities(x,extra,indices,ids,encoded,origins);probabilities.append(p.float().cpu().numpy())
        p=np.concatenate(probabilities)
        for e in g['segments']:
            sl=slice(e['start'],e['end'])
            if e['zero_horizon']:p[sl]=np.array([1.,0.,0.],np.float32)
            cm,sc,empty=score(p[sl],g['y'][sl])
            for state in e['missing_states']:cm[:,state,0]+=1
            roots[g['root_id']].append((cm,sc,empty,e['zero_new']));pooled+=cm;count+=1
        scenes[g['root_id']]=g['scene_id']
    records=[]
    for rid,values in roots.items():records.append(dict(root_id=rid,scene_id=scenes[rid],cm=np.mean([v[0]for v in values],0),scores=np.mean([v[1]for v in values],0),empty=np.sum([v[2]for v in values if v[3]],0)if any(v[3]for v in values)else np.zeros(14,int),zero_cases=sum(v[3]for v in values)))
    result=aggregate(records);best=max(result['rules'],key=lambda k:result['rules'][k]['label_aware_delta_iou'])
    result.update(best_rule=best,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),windows=count,roots=len(roots),role='SAME FOUR TRAINING ROOTS FULL-CELL FITTING ONLY'if training else'Original full666-window pilot; learned legal viewing-line context. Not full development acceptance.',pooled={k:extended_rates(c)for k,c in zip(result['rules'],pooled)},seconds=time.perf_counter()-tick)
    name=saved['config']['name'];save(OUT/'pilot'/('%s_%s_epoch%02d.json'%(name,'train_full'if training else'dev',saved['epoch'])),result)
    print('RESULT',saved['epoch'],best,result['rules'][best]['label_aware_delta_iou'],result['rules'][best]['revelation_iou'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--training',action='store_true');evaluate(**vars(p.parse_args()))
