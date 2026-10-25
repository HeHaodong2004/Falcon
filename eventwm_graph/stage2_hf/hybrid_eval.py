"""Full-domain pilot scores for the spatial residual, with a frozen exact prior.

Cached released probabilities avoid reconstructing88features per cell again.
The residual multiplies revelation odds; conditional labels remain identical.
Final release evaluation uses the complete public API and actual future maps.
"""
import argparse,math,time
import numpy as np
import torch
from .common import OUT,CKPT,read,save,sha
from .hybrid import HybridNet
from .packed import load_arrays,get_segment
from .spatial import build
from .spatial_cuda import QueryKernel,path_arrays
from .spatial_eval import score
from .light_eval import aggregate

def evaluate(name,epochs,device='cuda:0'):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 dest=OUT/'training'/name;m=read(OUT/'packed/dev_pilot/manifest.json');oldrows={r['root_id']:r for r in read(OUT/'light_eval_cache/manifest.json')['rows']};kernel=QueryKernel(device);results=read(dest/'pilot_evaluation.json')if(dest/'pilot_evaluation.json').exists()else{};tick=time.perf_counter();oldweights=torch.load(CKPT,map_location='cpu')['model']
 for epoch in epochs:
  cp=dest/('epoch%02d.pt'%epoch);saved=torch.load(cp,map_location='cpu');model=HybridNet(**saved['model_config']);model.load_state_dict(saved['model']);model.eval()
  for layer in (0,2,4):
   torch.testing.assert_close(model.prior[layer].weight,oldweights['net.%d.weight'%layer][:,16:]if layer==0 else oldweights['net.%d.weight'%layer],rtol=0,atol=0);torch.testing.assert_close(model.prior[layer].bias,oldweights['net.%d.bias'%layer],rtol=0,atol=0)
  model.spatial.to(device);records=[];windows=0;max_delta=0.
  for row in m['rows']:
   a=load_arrays(row);old=oldrows[row['root_id']];lookup={(e['start_node'],e['end_node']):e for e in old['segments']}
   with np.load(old['path'],allow_pickle=False)as z:base=z['baseline'];oldy=z['y']
   cms=[];scores=[];zero=0;empt=np.zeros(14,int)
   for i,meta in enumerate(row['segments']):
    v=get_segment(a,i);d=build(*v[:4],v[-1],*v[4:7],make_query=False);pa=path_arrays([d['path']],[d['lo']],[d['shape']],[d['resolution']]);t={k:torch.from_numpy(v).to(device)for k,v in pa.items()};maps=torch.from_numpy(d['maps'][None]).to(device);fine=torch.from_numpy(d['fine'][None]).to(device)
    with torch.inference_mode():q=kernel(maps,t['path'],t['pre'],t['globals'],t['spec']);field=model.spatial(maps,q,fine)[0].permute(1,2,0).cpu().numpy()
    ci=d['keys']-d['lo'];delta=field[ci[:,0],ci[:,1]].astype(np.float64);previous=lookup[(meta['start_node'],meta['end_node'])];sl=slice(previous['start'],previous['end']);p0=base[sl].astype(np.float64);y=d['labels'][ci[:,0],ci[:,1]];np.testing.assert_array_equal(y,oldy[sl]);factor=np.exp(delta[:,0]-math.log(8));p=p0.copy();p[:,1:]*=factor[:,None];p/=p.sum(1,keepdims=True)
    if model.train_occupancy:
     rr=p[:,1:].sum(1);f=p0[:,1];b=p0[:,2]*np.exp(delta[:,1]);den=np.maximum(f+b,1e-300);p[:,1]=rr*f/den;p[:,2]=rr*b/den
    p=p.astype(np.float32)
    if len(d['path'])==1:p.fill(0);p[:,0]=1
    cm,sc,ee=score(p,y);cms.append(cm);scores.append(sc);windows+=1;max_delta=max(max_delta,float(np.max(abs(delta[:,0]-math.log(8)))))
    if not np.any(y):zero+=1;empt+=ee
   records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],cm=np.mean(cms,axis=0),scores=np.mean(scores,axis=0),empty=empt,zero_cases=zero))
  r=aggregate(records);best=max(r['rules'],key=lambda k:r['rules'][k]['label_aware_delta_iou']);r.update(best_rule=best,windows=windows,checkpoint_sha256=sha(cp),prior_weights_exactly_frozen=True,old_candidate_label_vectors_exact=True,max_absolute_revelation_logodds_residual=max_delta);results[cp.stem]=r;save(dest/'pilot_evaluation.json',results);print(name,epoch,best,r['rules'][best]['label_aware_delta_iou'],r['logloss'],flush=True)
 save(dest/'pilot_evaluation_complete.json',dict(seconds=time.perf_counter()-tick))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--epochs',nargs='+',type=int,default=[1]);p.add_argument('--device',default='cuda:0');a=p.parse_args();evaluate(a.name,a.epochs,a.device)
