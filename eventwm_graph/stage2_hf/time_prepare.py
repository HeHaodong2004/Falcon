"""Append causal recency features to frozen original training rows."""
import argparse,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .common import OUT,PREP,read,save,sha,source
from .packed import load_arrays,get_segment
from .time_features import moment_matrix,basis_numpy

DEST=None;SPLIT=None
def annotate(item):
 index,shard=item;path=DEST/('%06d.npz'%index);receipt=path.with_suffix('.json')
 if receipt.exists():
  r=read(receipt)
  if sha(path)!=r['sha256']:raise ValueError('timefeature shard changed')
  return r
 tick=time.perf_counter()
 with np.load(shard['path'],allow_pickle=False)as z:x=z['x']
 extra=np.empty((len(x),32),np.float16);group=defaultdict(list);used=[]
 for e in shard['segments']:group[e['root_id']].append(e)
 for rid,segments in group.items():
  packed=read(OUT/'packed'/SPLIT/(rid+'.json'));a=load_arrays(packed);tp=OUT/'temporal'/SPLIT/(rid+'.json')
  while not tp.exists():
   if time.perf_counter()-tick>1800:raise TimeoutError('awaiting temporal metadata '+rid)
   time.sleep(1)
  tr=read(tp)
  if tr['packed_sha256']!=packed['sha256']:raise ValueError('temporal graph identity mismatch')
  with np.load(tr['path'],allow_pickle=False)as z:last=z['last']
  lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed['segments'])};cache={}
  for e in segments:
   i=lookup[(e['start_node'],e['end_node'])];gi=int(a['graph_index'][i]);v=get_segment(a,i);p=v[3];key=(gi,*p[0,[1,3]])
   if key not in cache:
    gs=slice(*a['goffsets'][gi:gi+2]);cache[key]=moment_matrix(v[0],v[1],last[gs],p[0,0],p[0,[1,3]],v[-1][0])
   sl=slice(e['start'],e['end']);extra[sl]=basis_numpy(x[sl,36:38].astype(np.float32))@cache[key]
  used.append(dict(root_id=rid,temporal_sha256=tr['sha256'],packed_sha256=packed['sha256']))
 if not np.isfinite(extra).all():raise ValueError('nonfinite timestamp-derived feature')
 tmp=path.with_suffix('.tmp.npz');np.savez(tmp,extra=extra);tmp.replace(path);r=dict(index=index,path=str(path),sha256=sha(path),source_shard_sha256=shard['sha256'],roots=used,rows=len(extra),bytes=path.stat().st_size,seconds=time.perf_counter()-tick);save(receipt,r);return r

def prepare(split,workers=2):
 global DEST,SPLIT
 SPLIT=split;path=PREP/(split+'_mixed_ray')/'manifest.json';m=read(path);DEST=OUT/'time_prepared'/split;DEST.mkdir(parents=True,exist_ok=False);save(DEST/'protocol.json',dict(source=source(),prepared_manifest_sha256=sha(path),inputs='Original legal88features plus32soft recencyangular covariates of startinggraph; recentage<=.1 andexponentialtau.5; no target/camera/humaninputs',storage='F16likeoriginalfeatures; allsampleweights/labels unchanged'))
 tick=time.perf_counter();done=[]
 with ProcessPoolExecutor(max_workers=workers)as pool:
  jobs=[pool.submit(annotate,item)for item in enumerate(m['shards'])]
  for f in as_completed(jobs):
   done.append(f.result())
   if len(done)%16==0 or len(done)==len(jobs):print(dict(shards=len(done),total=len(jobs),seconds=time.perf_counter()-tick),flush=True)
 save(DEST/'manifest.json',dict(shards=sorted(done,key=lambda r:r['index']),rows=sum(r['rows']for r in done),seconds=time.perf_counter()-tick,source_manifest_sha256=sha(path),bytes=sum(r['bytes']for r in done)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['pilot_fit','full'],required=True);p.add_argument('--workers',type=int,default=2);prepare(**vars(p.parse_args()))
