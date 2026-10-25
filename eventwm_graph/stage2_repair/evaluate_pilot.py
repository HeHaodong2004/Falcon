"""Recompute fixed-pilot point models; preserve hierarchy and pooled counts."""
import argparse
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save,sha
from eventwm_graph.stage2_hf.time_model import TimeNet
from eventwm_graph.stage2_hf.time_features import moment_matrix,basis_torch
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment
from eventwm_graph.stage2_hf.spatial_eval import score
from eventwm_graph.stage2_hf.light_eval import aggregate
from scripts.stage2_diagnostic.common import extended_rates


def evaluate(checkpoint,device='cuda:0',name=None):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;tick=time.perf_counter()
    saved=torch.load(checkpoint,map_location='cpu');model=TimeNet(**saved['model_config']).to(device).eval();model.load_state_dict(saved['model'])
    m=read(HF/'packed/dev_pilot/manifest.json');oldrows={r['root_id']:r for r in read(HF/'light_eval_cache/manifest.json')['rows']};records=[];pooled=np.zeros((14,3,3),np.int64);details=[]
    for row in m['rows']:
        a=load_arrays(row);tr=read(HF/'temporal/dev_pilot'/(row['root_id']+'.json'))
        assert tr['packed_sha256']==row['sha256']
        with np.load(tr['path'],allow_pickle=False)as z:last=z['last']
        old=oldrows[row['root_id']];lookup={(e['start_node'],e['end_node']):e for e in old['segments']}
        with np.load(old['path'],allow_pickle=False)as z:x=z['x'];y=z['y']
        cms=[];scores=[];empty=np.zeros(14,int);zero=0;cache={}
        for i,meta in enumerate(row['segments']):
            v=get_segment(a,i);gi=int(a['graph_index'][i]);p=v[3];key=(gi,*p[0,[1,3]]);e=lookup[(meta['start_node'],meta['end_node'])];sl=slice(e['start'],e['end'])
            if key not in cache:
                gs=slice(*a['goffsets'][gi:gi+2]);matrix=moment_matrix(v[0],v[1],last[gs],p[0,0],p[0,[1,3]],v[-1][0])
                if saved['config'].get('ablate_time'):matrix.fill(0)
                with torch.inference_mode():cache[key]=torch.from_numpy(matrix).to(device)@model.correction[0].weight[:,88:].T
            pp=[]
            with torch.inference_mode():
                for start in range(sl.start,sl.stop,65536):
                    xt=torch.from_numpy(x[start:min(start+65536,sl.stop)]).to(device);prob,_=model.probabilities(xt,basis=basis_torch(xt),projection=cache[key]);pp.append(prob.cpu().numpy())
            prob=np.concatenate(pp);gt=y[sl]
            if len(p)==1:prob.fill(0);prob[:,0]=1
            assert np.count_nonzero(gt)==len(v[4])
            cm,sc,ee=score(prob,gt);cms.append(cm);scores.append(sc);pooled+=cm
            details.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],start_node=meta['start_node'],end_node=meta['end_node'],cm=cm.tolist(),zero_new=not bool(np.any(gt)),poses=len(p)))
            if not np.any(gt):zero+=1;empty+=ee
        records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],cm=np.mean(cms,axis=0),scores=np.mean(scores,axis=0),empty=empty,zero_cases=zero))
    r=aggregate(records);r.update(windows=len(details),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        pooled={k:extended_rates(cm)for k,cm in zip(r['rules'],pooled)},pooled_confusion=pooled.tolist(),seconds=time.perf_counter()-tick)
    best=max(r['rules'],key=lambda k:r['rules'][k]['label_aware_delta_iou']);r['best_rule']=best
    dest=OUT/'pilot';save(dest/((name or ('epoch%02d'%saved['epoch']))+'.json'),r)
    save(dest/((name or ('epoch%02d'%saved['epoch']))+'_windows.json'),details)
    print(name,best,r['rules'][best]['label_aware_delta_iou'],r['rules'][best]['revelation_iou'],'seconds',r['seconds'],flush=True)
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--name');evaluate(**vars(p.parse_args()))
