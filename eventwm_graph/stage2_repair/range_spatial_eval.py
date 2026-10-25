"""Same original pilot aggregation for the legal shared-range spatial model."""
import argparse
from collections import defaultdict
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save,sha
from .range_spatial import SpatialEvidenceNet
from .range_spatial_train import load_groups
from eventwm_graph.stage2_hf.spatial_eval import score
from eventwm_graph.stage2_hf.light_eval import aggregate
from scripts.stage2_diagnostic.common import extended_rates
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment


def full_training_groups():
    manifest=read(OUT/'range_evidence_train_fullcells/manifest.json');spatial={r['root_id']:r for r in read(OUT/'range_bev_train/manifest.json')['records']};groups=[]
    for row in manifest['records']:
        context=spatial[row['root_id']];packed=read(HF/'packed/pilot_fit'/(row['root_id']+'.json'));a=load_arrays(packed);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed['segments'])}
        with np.load(context['path'])as z:maps=z['maps'];low=z['lo']
        with np.load(row['path'])as z:x=z['x'];extra=z['extra'];y=z['y']
        bystart=defaultdict(list)
        for e in row['segments']:bystart[e['start_node']].append(e)
        for g in context['groups']:
            entries=bystart[g['start_node']];ids=np.concatenate([np.arange(e['start'],e['end'])for e in entries]);p=get_segment(a,lookup[g['start_node'],entries[0]['end_node']])[3];keys=np.floor((x[ids,20:22].astype(np.float64)*10+p[0,[1,3]])/a['geometry'][0]).astype(np.int32);segments=[];cursor=0
            for e in entries:n=e['end']-e['start'];segments.append(dict(start=cursor,end=cursor+n));cursor+=n
            groups.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],maps=maps[g['map_index']],x=x[ids],extra=extra[ids],y=y[ids],indices=keys-low[g['map_index']],segments=segments))
    return manifest,groups


def evaluate(checkpoint,device='cuda:1',training=False,full_training=False,_prepared=None):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;tick=time.perf_counter();saved=torch.load(checkpoint,map_location='cpu');model=SpatialEvidenceNet().to(device).eval();model.load_state_dict(saved['model']);manifest,groups=_prepared if _prepared is not None else full_training_groups()if full_training else load_groups('train'if training else'dev');roots=defaultdict(list);root_scene={};pooled=np.zeros((14,3,3),np.int64);count=0
    def tensor(x,dtype):return torch.from_numpy(np.ascontiguousarray(x,dtype=dtype)).to(device)
    for g in groups:
        probability=[]
        # Encoder receives ONLY current legal-map/range canvas. Auxiliary
        # future certificate targets g['aux']/g['mask'] are never read here.
        with torch.inference_mode(),torch.cuda.amp.autocast():
            encoded=model.encode(tensor(g['maps'][None],np.float32))
            for begin in range(0,len(g['y']),65536):
                sl=slice(begin,begin+65536);x=tensor(g['x'][sl],np.float32);extra=tensor(g['extra'][sl],np.float32);indices=tensor(g['indices'][sl],np.int64);ids=torch.zeros(len(x),dtype=torch.long,device=device);p,_=model.probabilities(x,extra,indices,ids,encoded);probability.append(p.float().cpu().numpy())
        p=np.concatenate(probability)
        for e in g['segments']:
            sl=slice(e['start'],e['end']);cm,sc,ee=score(p[sl],g['y'][sl]);roots[g['root_id']].append((cm,sc,ee,not bool(np.any(g['y'][sl]))));pooled+=cm;count+=1
        root_scene[g['root_id']]=g['scene_id']
    records=[]
    for rid,values in roots.items():
        records.append(dict(root_id=rid,scene_id=root_scene[rid],cm=np.mean([v[0]for v in values],0),scores=np.mean([v[1]for v in values],0),empty=np.sum([v[2]for v in values if v[3]],0)if any(v[3]for v in values)else np.zeros(14,int),zero_cases=sum(v[3]for v in values)))
    role='FULL ORIGINAL CELLS ON SAME FOUR TRAINING ROOTS; not generalization'if full_training else'SAMPLED SAME TRAINING EXAMPLES; not full-cell reconstruction or generalization'if training else'Original fixed full666-window development pilot; legal current/past sensor extension, no privileged runtime inputs.'
    result=aggregate(records);best=max(result['rules'],key=lambda k:result['rules'][k]['label_aware_delta_iou']);result.update(best_rule=best,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),windows=count,roots=len(roots),role=role,pooled={k:extended_rates(c)for k,c in zip(result['rules'],pooled)},seconds=time.perf_counter()-tick)
    name=saved.get('config',{}).get('name','range_spatial64');prefix='range_spatial'if name=='range_spatial64'else name
    save(OUT/'pilot'/('%s_%s_epoch%02d.json'%(prefix,'train_full'if full_training else'train'if training else'dev',saved['epoch'])),result);print('RESULT',saved['epoch'],best,result['rules'][best]['label_aware_delta_iou'],result['rules'][best]['revelation_iou'],result['role'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--training',action='store_true');p.add_argument('--full-training',action='store_true');evaluate(**vars(p.parse_args()))
