"""Causal observation-time sidecars for the already verified legal graph cache.

Only ancestor observations up to each saved starting node affect its channels.
No original data or ongoing extraction is changed. Future labels are not read
to form these inputs; the file containing DAG nodes is replayed with ancestry
pruning and an undo journal, exactly matching PartialGraph timestamp rules.
"""
import argparse,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from numba import njit
from .common import OUT,read,save,sha,source
from .packed import load_arrays
from eventwm_graph.stage2.data import source_rows
from eventwm_graph.stage2_segment.data import read as read_member

DEST=None;SPLIT=None
@njit(cache=True)
def dense_apply(updates,first,last,marker,occupied,tag,nx,nz,lx,lz,now):
 oldkey=np.empty(len(updates),np.int64);oldfirst=np.empty(len(updates),np.float64);oldlast=np.empty(len(updates),np.float64);used=0;added=0
 for u in updates:
  if u[4]!=now:raise ValueError('noncausal update timestamp')
  x=int(u[0])-lx;z=int(u[1])-lz
  if int(u[2])==2 and 0<=x<nx and 0<=z<nz:occupied[x*nz+z]=tag
 for u in updates:
  if u[3]:continue
  state=int(u[2]);x=int(u[0])-lx;z=int(u[1])-lz
  if not(0<=x<nx and 0<=z<nz):
   is_occupied=False
   if state==1:
    for v in updates:
     if v[2]==2 and v[0]==u[0]and v[1]==u[1]:is_occupied=True;break
   if is_occupied:continue
   raise ValueError('observation outside starting graph key union')
  key=x*nz+z
  if state==1 and occupied[key]==tag:continue
  if state!=1 and state!=2:raise ValueError('invalid observed state')
  if marker[key]!=tag:
   marker[key]=tag;oldkey[used]=key;oldfirst[used]=first[key];oldlast[used]=last[key];used+=1
  if np.isnan(first[key]):first[key]=now;added+=1
  last[key]=now
 return oldkey[:used],oldfirst[:used],oldlast[:used],added

def root_times_fast(row):
 path=DEST/(row['root_id']+'.npz');receipt=path.with_suffix('.json')
 if receipt.exists():
  r=read(receipt)
  if sha(path)!=r['sha256']:raise ValueError('temporal sidecar changed')
  return r
 packed=OUT/'packed'/SPLIT/(row['root_id']+'.json');tick=time.perf_counter()
 while not packed.exists():
  if time.perf_counter()-tick>1800:raise TimeoutError('awaiting owned packed cache '+row['root_id'])
  time.sleep(1)
 meta=read(packed);a=load_arrays(meta)
 if sha(meta['path'])!=meta['sha256']or any(meta['source_members'][k]!=row['files'][k]for k in ('legal_root.json.gz','future_observations.json.gz')):raise ValueError('packed graph identity changed')
 legal=read_member(row,'legal_root.json.gz');nodes=read_member(row,'future_observations.json.gz');root=legal['partial_graph'];initial=float(root['timestamp']);mapping={e['start_node']:e['graph_index']for e in meta['segments']};needed=set()
 for sid in mapping:
  while sid is not None and sid not in needed:needed.add(sid);sid=nodes[sid]['parent']
 keys=a['gkeys'].astype(np.int64);lo=keys.min(0);hi=keys.max(0)+1;shape=hi-lo;nx,nz=map(int,shape)
 if nx*nz>4000000:return root_times(row)
 first=np.full(nx*nz,np.nan);last=np.full(nx*nz,np.nan);marker=np.zeros(nx*nz,np.int32);occupied=np.zeros(nx*nz,np.int32)
 rk=np.array([c['cell']for c in root['cells']],np.int64)-lo;codes=rk[:,0]*nz+rk[:,1];first[codes]=[c['first_seen']for c in root['cells']];last[codes]=[c['last_seen']for c in root['cells']];present=len(codes);gc=(keys[:,0]-lo[0])*nz+keys[:,1]-lo[1];firstout=np.empty(len(keys));lastout=np.empty(len(keys));written=set();children=defaultdict(list);frames=0
 for sid in needed:
  n=nodes[sid]
  if n['node_id']!=sid or(n['parent']is not None and not 0<=n['parent']<sid):raise ValueError('invalid ancestry')
  children[n['parent']].append(sid)
 def emit(sid,now):
  if sid not in mapping:return
  gi=mapping[sid];sl=slice(*a['goffsets'][gi:gi+2]);cc=gc[sl];ff=first[cc];ll=last[cc]
  if len(cc)!=present or not np.isfinite(ff).all()or not np.isfinite(ll).all()or np.any(ff>ll)or np.any(ll>now+1e-8):raise ValueError('causal timestamp/key identity mismatch')
  firstout[sl]=ff;lastout[sl]=ll;written.add(gi)
 def visit(sid,parent_time):
  nonlocal present,frames
  ob=nodes[sid]['observation'];now=float(ob['timestamp'])
  if now<=parent_time:raise ValueError('noncausal ancestry timestamp')
  rows=np.array([(u['cell'][0],u['cell'][1],u['state'],u['source']=='occupied_ego_base_support',u['timestamp'])for u in ob['static_updates']],np.float64).reshape(-1,5)
  old,ff,ll,added=dense_apply(rows,first,last,marker,occupied,sid+1,nx,nz,int(lo[0]),int(lo[1]),now);present+=added;frames+=1;emit(sid,now)
  for child in children[sid]:visit(child,now)
  first[old]=ff;last[old]=ll;present-=added
 emit(None,initial)
 for sid in children[None]:visit(sid,initial)
 if len(written)!=len(a['goffsets'])-1:raise ValueError('missing timestamp snapshot')
 tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,first=firstout,last=lastout);tmp.replace(path)
 result=dict(root_id=row['root_id'],scene_id=row['scene_id'],path=str(path),sha256=sha(path),packed_sha256=meta['sha256'],source_member_sha256={k:row['files'][k]for k in ('legal_root.json.gz','future_observations.json.gz')},starting_graphs=len(written),ancestor_frames=frames,all_timestamps_causal=True,algorithm='Densevector timestamp journal, identical recordedupdate rules, exact reference check separate',bytes=path.stat().st_size,seconds=time.perf_counter()-tick);save(receipt,result);return result
def root_times(row):
 path=DEST/(row['root_id']+'.npz');receipt=path.with_suffix('.json')
 if receipt.exists():
  old=read(receipt)
  if sha(path)!=old['sha256']:raise ValueError('temporal sidecar changed')
  return old
 packed=OUT/'packed'/SPLIT/(row['root_id']+'.json')
 tick=time.perf_counter()
 while not packed.exists():
  if time.perf_counter()-tick>1800:raise TimeoutError('awaiting owned packed cache '+row['root_id'])
  time.sleep(1)
 meta=read(packed);a=load_arrays(meta)
 if sha(meta['path'])!=meta['sha256']:raise ValueError('packed graph hash mismatch')
 legal=read_member(row,'legal_root.json.gz');nodes=read_member(row,'future_observations.json.gz');root=legal['partial_graph'];initial=float(root['timestamp']);times={tuple(c['cell']):(float(c['first_seen']),float(c['last_seen']))for c in root['cells']}
 mapping={e['start_node']:e['graph_index']for e in meta['segments']};needed=set()
 for sid in mapping:
  while sid is not None and sid not in needed:
   needed.add(sid);sid=nodes[sid]['parent']
 children=defaultdict(list)
 for sid in needed:
  node=nodes[sid]
  if node['node_id']!=sid or(node['parent']is not None and not 0<=node['parent']<sid):raise ValueError('invalid causal timestamp ancestry')
  children[node['parent']].append(sid)
 first=np.empty(len(a['gkeys']),np.float64);last=np.empty_like(first);written=set();frames=0
 def emit(sid,now):
  if sid not in mapping:return
  gi=mapping[sid];sl=slice(*a['goffsets'][gi:gi+2]);keys=a['gkeys'][sl]
  if len(keys)!=len(times)or any(tuple(k)not in times for k in keys):raise ValueError('starting graph key identity mismatch in timestamp replay')
  values=np.array([times[tuple(k)]for k in keys],np.float64).reshape(-1,2)
  if not np.isfinite(values).all()or np.any(values[:,0]>values[:,1])or np.any(values[:,1]>now+1e-8):raise ValueError('future/stale/invalid timestamp channel')
  first[sl]=values[:,0];last[sl]=values[:,1];written.add(gi)
 def visit(sid,parent_time):
  nonlocal frames
  ob=nodes[sid]['observation'];now=float(ob['timestamp'])
  if now<=parent_time:raise ValueError('noncausal timestamp')
  occupied={tuple(u['cell'])for u in ob['static_updates']if u['state']==2};old={}
  for u in ob['static_updates']:
   if u['timestamp']!=now:raise ValueError('noncausal legal update')
   key=tuple(u['cell']);state=int(u['state'])
   if u['source']=='occupied_ego_base_support'or(state==1 and key in occupied):continue
   if state not in (1,2):raise ValueError('invalid static observation')
   if key not in old:old[key]=times.get(key)
   times[key]=(times[key][0]if key in times else now,now)
  frames+=1;emit(sid,now)
  for child in children[sid]:visit(child,now)
  for key,value in old.items():
   if value is None:times.pop(key)
   else:times[key]=value
 emit(None,initial)
 for sid in children[None]:visit(sid,initial)
 if len(written)!=len(a['goffsets'])-1:raise ValueError('missing starting timestamp graph')
 tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,first=first,last=last);tmp.replace(path)
 r=dict(root_id=row['root_id'],scene_id=row['scene_id'],path=str(path),sha256=sha(path),packed_sha256=meta['sha256'],source_member_sha256={k:row['files'][k]for k in ('legal_root.json.gz','future_observations.json.gz')},starting_graphs=len(written),ancestor_frames=frames,all_timestamps_causal=True,bytes=path.stat().st_size,seconds=time.perf_counter()-tick);save(receipt,r);return r

def prepare(split,workers=2,limit=None):
 global DEST,SPLIT
 SPLIT=split;protocol=read(OUT/'protocol.json');rows,manifest=source_rows('dev'if split=='dev_pilot'else'train')
 if split in ('pilot_fit','dev_pilot'):
  ids=set(protocol['pilot_fit'if split=='pilot_fit'else'pilot_selection_roots']);rows=[r for r in rows if r['root_id']in ids]
 if limit:rows=rows[:limit]
 DEST=OUT/'temporal'/(split+('_limit'+str(limit)if limit else''));DEST.mkdir(parents=True,exist_ok=True)
 if(DEST/'manifest.json').exists():raise FileExistsError('completed temporal cache preserved')
 save(DEST/'identity.json',dict(source=source(),split=split,limit=limit,source_manifest_sha256=sha(manifest),roots=[r['root_id']for r in rows],channels='Startinggraph first/lastobserved timestamps only; no camera/human/futureinputs; no interpolation; ancestor-only replay and siblingundo; legalGkeyhash binding.'))
 start=time.perf_counter();done=[]
 with ProcessPoolExecutor(max_workers=workers)as pool:
  jobs={pool.submit(root_times_fast,row):row['root_id']for row in rows}
  for future in as_completed(jobs):
   try:done.append(future.result())
   except BaseException as e:save(DEST/'failure.json',dict(root_id=jobs[future],error=repr(e)));raise
   if len(done)%64==0 or len(done)==len(rows):
    r=dict(roots=len(done),requested=len(rows),seconds=time.perf_counter()-start,bytes=sum(r['bytes']for r in done));save(DEST/'progress.json',r);print(r,flush=True)
 save(DEST/'manifest.json',dict(rows=sorted(done,key=lambda r:r['root_id']),roots=len(done),seconds=time.perf_counter()-start,disk_bytes=sum(r['bytes']for r in done),all_timestamps_causal=True,failures=[]))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['pilot_fit','full','dev_pilot'],required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int);prepare(**vars(p.parse_args()))
