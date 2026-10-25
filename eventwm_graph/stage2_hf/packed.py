"""Compact legal-only spatial training cache, deduplicated starting maps.

Only the existing guarded train/dev loaders are used. The current map, path
and future target are stored separately. Features never read target arrays.
"""
import argparse,time,resource
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .common import OUT,read,save,sha,source
from eventwm_graph.stage2.data import source_rows
from eventwm_graph.stage2_segment.data import replay

DEST=None
def pack_root(row):
 path=DEST/(row['root_id']+'.npz');meta=path.with_suffix('.json')
 if meta.exists():
  r=read(meta)
  if sha(path)!=r['sha256']:raise ValueError('packed cache changed')
  return r
 arrays=defaultdict(list);graph_index={};segments=[];offsets={k:[0]for k in ('g','p','t')};audit=None;start=time.perf_counter()
 for ex in replay(row,'mixed'):
  if 'audit'in ex:audit=ex['audit'];continue
  g=ex['graph'];sid=ex['start_node']
  if sid not in graph_index:
   graph_index[sid]=len(graph_index);cells=g['cells']
   arrays['gkeys'].append(np.array([c['cell']for c in cells],np.int32).reshape(-1,2))
   arrays['gstates'].append(np.array([c['state']for c in cells],np.uint8));arrays['gheights'].append(np.array([c['height']for c in cells],np.float32))
   offsets['g'].append(offsets['g'][-1]+len(cells))
  tk=np.array(list(ex['target']),np.int32).reshape(-1,2);tv=list(ex['target'].values())
  arrays['tkeys'].append(tk);arrays['tstates'].append(np.array([v[0]for v in tv],np.uint8));arrays['theights'].append(np.array([v[1]for v in tv],np.float32));offsets['t'].append(offsets['t'][-1]+len(tk))
  arrays['paths'].append(ex['trajectory']);offsets['p'].append(offsets['p'][-1]+len(ex['trajectory']))
  segments.append(dict(start_node=sid,end_node=ex['end_node'],graph_index=graph_index[sid],query_id=ex['query_id'],query_aliases=ex['query_aliases'],known_terminal_differences=ex['known_terminal_differences']))
  geometry=(g['resolution'],g['height'],g['clearance'])
 if audit is None:raise ValueError('incomplete legal replay')
 values={k:np.concatenate(v,axis=0)for k,v in arrays.items()};values.update({k+'offsets':np.array(v,np.int64)for k,v in offsets.items()});values['graph_index']=np.array([s['graph_index']for s in segments],np.int32);values['geometry']=np.array(geometry,np.float64)
 tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**values);tmp.replace(path)
 result=dict(root_id=row['root_id'],scene_id=row['scene_id'],path=str(path),sha256=sha(path),segments=segments,audit=audit,source_members=row['files'],unique_starting_graphs=len(graph_index),bytes=path.stat().st_size,seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 save(meta,result);return result

def prepare(split,workers=8,limit=None):
 global DEST
 protocol=read(OUT/'protocol.json');rows,manifest=source_rows('dev'if split.startswith('dev')else'train')
 if split in ('pilot_fit','pilot_fitting_holdout'):ids=set(protocol[split]);rows=[r for r in rows if r['root_id']in ids]
 if split=='dev_pilot':ids=set(protocol['pilot_selection_roots']);rows=[r for r in rows if r['root_id']in ids]
 if limit:rows=rows[:limit]
 DEST=OUT/'packed'/(split+('_limit'+str(limit)if limit else''));DEST.mkdir(parents=True,exist_ok=True)
 if (DEST/'manifest.json').exists():raise FileExistsError('completed cache preserved')
 identity=dict(split=split,limit=limit,source_manifest_sha256=sha(manifest),source=source(),protocol_sha256=sha(OUT/'protocol.json'),root_ids=[r['root_id']for r in rows])
 ip=DEST/'identity.json'
 if ip.exists()and read(ip)!=identity:raise ValueError('source changed during extraction')
 save(ip,identity);tick=time.perf_counter();done=[];failures=[];ledger=Counter()
 with ProcessPoolExecutor(max_workers=workers)as pool:
  jobs={pool.submit(pack_root,r):r for r in rows}
  for job in as_completed(jobs):
   try:r=job.result()
   except BaseException as err:
    failures.append(dict(root_id=jobs[job]['root_id'],error=repr(err)));save(DEST/'failure_receipts.json',failures);raise
   done.append(r);ledger.update(r['audit'])
   if len(done)%64==0 or len(done)==len(rows):
    report=dict(roots=len(done),requested=len(rows),seconds=time.perf_counter()-tick,bytes=sum(r['bytes']for r in done),counts=dict(ledger));save(DEST/'progress.json',report);print(str(report),flush=True)
 ordered=sorted(done,key=lambda r:r['root_id'])
 save(DEST/'manifest.json',dict(identity=identity,roots=len(done),scenes=len({r['scene_id']for r in done}),segments=sum(len(r['segments'])for r in done),counts=dict(ledger),rows=ordered,failures=failures,seconds=time.perf_counter()-tick,disk_bytes=sum(r['bytes']for r in done),worker_peak_rss_bytes=max(r['peak_rss_bytes']for r in done)))
def load_arrays(row):
 with np.load(row['path'],allow_pickle=False)as z:return {k:z[k]for k in z.files}
def get_segment(a,i):
 gi=int(a['graph_index'][i]);g=slice(*a['goffsets'][gi:gi+2]);p=slice(*a['poffsets'][i:i+2]);t=slice(*a['toffsets'][i:i+2])
 return a['gkeys'][g],a['gstates'][g],a['gheights'][g],a['paths'][p],a['tkeys'][t],a['tstates'][t],a['theights'][t],a['geometry']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['pilot_fit','pilot_fitting_holdout','full','dev_pilot','dev'],required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int);prepare(**vars(p.parse_args()))
