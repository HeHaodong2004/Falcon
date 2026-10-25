"""Pilot scoring on the exact old full candidate universe, never tiny crops."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from .common import OUT,THRESHOLDS,read,save,sha
from .packed import load_arrays,get_segment
from .spatial import SpatialNet,build,spatial_probabilities
from .spatial_cuda import QueryKernel,path_arrays
from .light_eval import aggregate

def score(p,y):
 pr=(p[:,1]+p[:,2]).astype(np.float64);cond=np.where(p[:,1]>=p[:,2],1,2).astype(np.uint8);preds=[p.argmax(-1)]+[np.where(pr>=t,cond,0)for t in THRESHOLDS]
 cms=np.array([np.bincount(3*y.astype(np.int64)+v,minlength=9).reshape(3,3)for v in preds]);truep=p[np.arange(len(y)),y]
 return cms,[-np.log(np.clip(truep,1e-9,1)).mean(),np.mean(np.square(p).sum(1)-2*truep+1)],np.array([not np.any(v)for v in preds],int)

def baseline():
 m=read(OUT/'light_eval_cache/manifest.json');records=[]
 for row in m['rows']:
  with np.load(row['path'],allow_pickle=False)as z:p=z['baseline'];y=z['y']
  cms=[];scores=[];empt=np.zeros(14,int);zero=0
  for ex in row['segments']:
   sl=slice(ex['start'],ex['end']);cm,sc,ee=score(p[sl],y[sl]);cms.append(cm);scores.append(sc)
   if not ex['new_cells']:zero+=1;empt+=ee
  records.append(dict(scene_id=row['scene_id'],cm=np.mean(cms,axis=0),scores=np.mean(scores,axis=0),empty=empt,zero_cases=zero))
 r=aggregate(records);save(OUT/'baseline_pilot.json',r);print(r['rules']['threshold_0.20'],flush=True)

def evaluate(name,epochs,device='cuda:0'):
 torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 m=read(OUT/'packed/dev_pilot/manifest.json');dest=OUT/'training'/name;allresults=read(dest/'pilot_evaluation.json')if(dest/'pilot_evaluation.json').exists()else{};cfg=read(dest/'config.json');kernel=QueryKernel(device);tick=time.perf_counter()
 oldrows={r['root_id']:r for r in read(OUT/'light_eval_cache/manifest.json')['rows']}
 for epoch in epochs:
  cp=dest/('epoch%02d.pt'%epoch);saved=torch.load(cp,map_location='cpu');model=SpatialNet(**saved['model_config']).to(device).eval();model.load_state_dict(saved['model']);records=[];windows=0
  for row in m['rows']:
   a=load_arrays(row);cms=[];scores=[];empt=np.zeros(14,int);zero=0;old=oldrows[row['root_id']]
   with np.load(old['path'],allow_pickle=False)as z:oldy=z['y']
   lookup={(e['start_node'],e['end_node']):e for e in old['segments']}
   for i,meta in enumerate(row['segments']):
    v=get_segment(a,i);d=build(*v[:4],v[-1],*v[4:7],make_query=False);arrays=path_arrays([d['path']],[d['lo']],[d['shape']],[d['resolution']]);t={k:torch.from_numpy(x).to(device)for k,x in arrays.items()};maps=torch.from_numpy(d['maps'][None]).to(device);fine=torch.from_numpy(d['fine'][None]).to(device)
    with torch.inference_mode():
     query=kernel(maps,t['path'],t['pre'],t['globals'],t['spec']);out=model(maps,query,fine);pfull=spatial_probabilities(out,cfg['pos_weight'])[0].permute(1,2,0).cpu().numpy()
    ci=d['keys']-d['lo'];p=pfull[ci[:,0],ci[:,1]];y=d['labels'][ci[:,0],ci[:,1]]
    if len(d['path'])==1:p.fill(0);p[:,0]=1
    previous=lookup[(meta['start_node'],meta['end_node'])];np.testing.assert_array_equal(y,oldy[previous['start']:previous['end']])
    cm,sc,ee=score(p,y);cms.append(cm);scores.append(sc);windows+=1
    if not np.any(y):zero+=1;empt+=ee
   records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],cm=np.mean(cms,axis=0),scores=np.mean(scores,axis=0),empty=empt,zero_cases=zero))
  summary=aggregate(records);best=max(summary['rules'],key=lambda k:summary['rules'][k]['label_aware_delta_iou']);summary.update(best_rule=best,windows=windows,checkpoint_sha256=sha(cp),old_candidate_label_vectors_exact=True);allresults[cp.stem]=summary;save(dest/'pilot_evaluation.json',allresults);print(name,epoch,best,summary['rules'][best]['label_aware_delta_iou'],summary['logloss'],flush=True)
 save(dest/'pilot_evaluation_complete.json',dict(seconds=time.perf_counter()-tick,windows_per_epoch=windows,source_manifest_sha256=sha(OUT/'packed/dev_pilot/manifest.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name');p.add_argument('--epochs',nargs='+',type=int,default=[1]);p.add_argument('--device',default='cuda:0');p.add_argument('--baseline',action='store_true');a=p.parse_args();baseline()if a.baseline else evaluate(a.name,a.epochs,a.device)
