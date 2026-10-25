"""Matched complete-domain development pilot evaluation for legal recency models."""
import argparse,time
import numpy as np
import torch
from .common import OUT,read,save,sha
from .time_model import TimeNet
from .time_features import moment_matrix,basis_torch
from .packed import load_arrays,get_segment
from .spatial_eval import score
from .light_eval import aggregate

def evaluate(name,epochs,device='cuda:0'):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 dest=OUT/'training'/name;m=read(OUT/'packed/dev_pilot/manifest.json');oldrows={r['root_id']:r for r in read(OUT/'light_eval_cache/manifest.json')['rows']};results=read(dest/'pilot_evaluation.json')if(dest/'pilot_evaluation.json').exists()else{};tick=time.perf_counter()
 for epoch in epochs:
  cp=dest/('epoch%02d.pt'%epoch);saved=torch.load(cp,map_location='cpu');model=TimeNet(**saved['model_config']).to(device).eval();model.load_state_dict(saved['model']);records=[];windows=0
  for row in m['rows']:
   a=load_arrays(row);tr=read(OUT/'temporal/dev_pilot'/(row['root_id']+'.json'))
   if tr['packed_sha256']!=row['sha256']:raise ValueError('timestamp/graph identity mismatch')
   with np.load(tr['path'],allow_pickle=False)as z:last=z['last']
   old=oldrows[row['root_id']];lookup={(e['start_node'],e['end_node']):e for e in old['segments']}
   with np.load(old['path'],allow_pickle=False)as z:x=z['x'];y=z['y']
   cms=[];scores=[];zero=0;empt=np.zeros(14,int);cache={}
   for i,meta in enumerate(row['segments']):
    v=get_segment(a,i);gi=int(a['graph_index'][i]);p=v[3];key=(gi,*p[0,[1,3]]);oldex=lookup[(meta['start_node'],meta['end_node'])];sl=slice(oldex['start'],oldex['end'])
    if key not in cache:
     gs=slice(*a['goffsets'][gi:gi+2]);matrix=moment_matrix(v[0],v[1],last[gs],p[0,0],p[0,[1,3]],v[-1][0]);mt=torch.from_numpy(matrix).to(device)
     if saved['config'].get('ablate_time'):mt.zero_()
     cache[key]=mt@model.correction[0].weight[:,88:].T
    pp=[]
    with torch.inference_mode():
     for start in range(sl.start,sl.stop,65536):
      xt=torch.from_numpy(x[start:min(start+65536,sl.stop)]).to(device);prob,_=model.probabilities(xt,basis=basis_torch(xt),projection=cache[key]);pp.append(prob.cpu().numpy())
    prob=np.concatenate(pp);gt=y[sl]
    if len(p)==1:prob.fill(0);prob[:,0]=1
    if int(np.count_nonzero(gt))!=len(v[4]):raise ValueError('candidate/target domain changed')
    cm,sc,ee=score(prob,gt);cms.append(cm);scores.append(sc);windows+=1
    if not np.any(gt):zero+=1;empt+=ee
   records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],cm=np.mean(cms,axis=0),scores=np.mean(scores,axis=0),empty=empt,zero_cases=zero))
  r=aggregate(records);best=max(r['rules'],key=lambda k:r['rules'][k]['label_aware_delta_iou']);r.update(best_rule=best,windows=windows,checkpoint_sha256=sha(cp),full_domain_unchanged=True);results[cp.stem]=r;save(dest/'pilot_evaluation.json',results);print(name,epoch,best,r['rules'][best]['label_aware_delta_iou'],r['logloss'],flush=True)
 save(dest/'pilot_evaluation_complete.json',dict(seconds=time.perf_counter()-tick))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--epochs',nargs='+',type=int,default=[2,4,8]);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(a.name,a.epochs,a.device)
