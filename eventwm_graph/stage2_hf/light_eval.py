"""Frozen full-domain pilot cache and hierarchical operating-point comparison."""
import argparse,time
from collections import defaultdict
import numpy as np
import torch
from .common import OUT,CKPT,THRESHOLDS,read,save,sha,source
from .light import LightNet,probabilities
from .decisions import observation_probabilities
from eventwm_graph.stage2.data import source_rows
from eventwm_graph.stage2_segment.data import replay
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.fused import transform
from scripts.stage2_diagnostic.common import labels,extended_rates

def cache():
 dest=OUT/'light_eval_cache';dest.mkdir(parents=True,exist_ok=False);protocol=read(OUT/'protocol.json');ids=set(protocol['pilot_selection_roots']);rows,_=source_rows('dev');rows=[r for r in rows if r['root_id']in ids]
 save(dest/'protocol.json',dict(roots=[r['root_id']for r in rows],source=source(),role='Development architectureselection, full originalcandidate domain; no protecteddata',storage='F32features/labels/baseline probabilities separately; never future inputs'))
 model=SegmentPredictor(CKPT,execution='optimized',future_format='native',device='cuda:1',feature_threads=1);reports=[];tick=time.perf_counter()
 for row in rows:
  xs=[];ys=[];ps=[];metas=[];cursor=0
  for ex in replay(row,'mixed'):
   if'audit'in ex:continue
   delta=model.predict(partial_graph=ex['graph'],trajectory=ex['trajectory']);keys=delta['candidate_cells'];y=labels(keys,ex['target'])
   ctx=model.feature_class(ex['graph'],ex['trajectory'],model.prepare_graph(ex['graph']));x,_=transform(ctx,keys,cuda=model.cuda,device_output=True)
   xs.append(x.cpu().numpy());ys.append(y);ps.append(delta['probabilities']);metas.append(dict(start=cursor,end=cursor+len(y),start_node=ex['start_node'],end_node=ex['end_node'],duration=float(ex['trajectory'][-1,0]-ex['trajectory'][0,0]),new_cells=len(ex['target'])));cursor+=len(y)
  path=dest/(row['root_id']+'.npz');np.savez(path,x=np.concatenate(xs),y=np.concatenate(ys),baseline=np.concatenate(ps));r=dict(root_id=row['root_id'],scene_id=row['scene_id'],path=str(path),sha256=sha(path),segments=metas);save(path.with_suffix('.json'),r);reports.append(r);print(dict(roots=len(reports),cells=cursor,seconds=time.perf_counter()-tick),flush=True)
 save(dest/'manifest.json',dict(rows=reports,seconds=time.perf_counter()-tick,cells=sum(r['segments'][-1]['end']for r in reports)))

def aggregate(roots):
 scenes=defaultdict(list)
 for r in roots:scenes[r['scene_id']].append(r)
 cm=np.mean([np.mean([r['cm']for r in rr],axis=0)for rr in scenes.values()],axis=0);scores=np.mean([np.mean([r['scores']for r in rr],axis=0)for rr in scenes.values()],axis=0)
 return dict(rules={name:extended_rates(c)for name,c in zip(['argmax']+['threshold_%.2f'%t for t in THRESHOLDS],cm)},logloss=float(scores[0]),brier=float(scores[1]),zero_new=dict(cases=sum(r['zero_cases']for r in roots),exact_empty=np.sum([r['empty']for r in roots],axis=0).tolist()),roots=len(roots),scenes=len(scenes))

def evaluate(name,epochs=None,device='cuda:0'):
 dest=OUT/'training'/name;cachem=read(OUT/'light_eval_cache/manifest.json');cfg=read(dest/'config.json');results={};tick=time.perf_counter()
 for ckpt in sorted(dest.glob('epoch*.pt')):
  if epochs and int(ckpt.stem[5:])not in epochs:continue
  saved=torch.load(ckpt,map_location='cpu');model=LightNet(saved['model_config']['kind']).to(device);model.load_state_dict(saved['model']);model.eval();records=[]
  for row in cachem['rows']:
   with np.load(row['path'],allow_pickle=False)as z:x=z['x'];y=z['y']
   pp=[]
   with torch.inference_mode():
    for start in range(0,len(y),65536):
     p=probabilities(model(torch.from_numpy(x[start:start+65536]).to(device)),cfg['kind'],cfg['objective'])
     if cfg['kind']=='softmax'and cfg['objective']=='balanced':
      p=p/torch.tensor([1.,8.,8.],device=device);p=p/p.sum(-1,keepdim=True)
     pp.append(p.cpu().numpy())
   p=np.concatenate(pp)
   for ex in row['segments']:
    sl=slice(ex['start'],ex['end'])
    # Replay timestamps are strictly increasing, so zero duration means the
    # starting pose alone. Zero displacement is intentionally not tested.
    p[sl]=observation_probabilities(p[sl],no_future_observations=ex['duration']==0)
   pr=(p[:,1]+p[:,2]).astype(np.float64);cond=np.where(p[:,1]>=p[:,2],1,2).astype(np.uint8);preds=[p.argmax(-1)]+[np.where(pr>=t,cond,0)for t in THRESHOLDS];cms=[];scores=[];empt=np.zeros(len(preds),int);zero=0
   for ex in row['segments']:
    a,b=ex['start'],ex['end'];gt=y[a:b];cms.append([np.bincount(3*gt.astype(np.int64)+v[a:b],minlength=9).reshape(3,3)for v in preds]);truep=p[np.arange(a,b),gt]
    scores.append([-np.log(np.clip(truep,1e-9,1)).mean(),np.mean(np.square(p[a:b]).sum(1)-2*truep+1)])
    if not ex['new_cells']:zero+=1;empt+=np.array([not np.any(v[a:b])for v in preds],int)
   records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],cm=np.mean(cms,axis=0),scores=np.mean(scores,axis=0),empty=empt,zero_cases=zero))
  summary=aggregate(records);best=max(summary['rules'],key=lambda k:summary['rules'][k]['label_aware_delta_iou']);summary.update(best_rule=best,checkpoint_sha256=sha(ckpt));results[ckpt.stem]=summary;save(dest/'pilot_evaluation.json',results);print(name,ckpt.stem,best,summary['rules'][best]['label_aware_delta_iou'],summary['logloss'],flush=True)
 save(dest/'pilot_evaluation_complete.json',dict(seconds=time.perf_counter()-tick,cache_manifest_sha256=sha(OUT/'light_eval_cache/manifest.json')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',action='store_true');p.add_argument('--name');p.add_argument('--epochs',type=int,nargs='*');p.add_argument('--device',default='cuda:0');a=p.parse_args();cache()if a.cache else evaluate(a.name,a.epochs,a.device)
