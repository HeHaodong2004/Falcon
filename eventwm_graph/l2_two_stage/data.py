"""Offline native-anchor preparation. Target identities never enter features."""
import argparse
import collections
import copy
import gzip
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .common import OUT, OLD, dump, sha
from .types import LegalContext, validate_frame
from .features import extract, positives,OfflineViewFrame
from .types import SealedObservation
from .jsonio import loads

TARGET_TIMES=np.array([.25,.5,.75,1.,1.5,2.,2.5,3.])


def read(path):
    with gzip.open(path,'rb') as f:return loads(f.read())


def selected_branch(folder):
    queries=read(folder/'queries.json.gz');outcomes=read(folder/'outcomes.json.gz')
    qi=max(range(len(queries)),key=lambda i:(len(queries[i]['compiled']['actions']),queries[i]['query_id']))
    outcome=outcomes[qi]
    # Schema-aware lossless selected-node decoding. Irrelevant sibling static
    # observations need not become Python objects. Fail closed to the complete
    # decoder if serializer structure differs; no field values are invented.
    import re
    with gzip.open(folder/'future_observations.json.gz','rb') as f:raw=f.read()
    starts=[m.start() for m in re.finditer(rb'\{"action"\s*:',raw)]
    ids=[int(m.group(1)) for m in re.finditer(rb'"node_id"\s*:\s*(\d+)',raw)]
    if len(starts)==len(ids) and ids==list(range(len(ids))):
        ends=starts[1:]+[len(raw)]
        nodes={i:loads(raw[starts[i]:ends[i]].rstrip(b' \n\r\t,]')) for i in outcome['node_ids']}
    else:nodes=loads(raw)
    assert outcome['query_id']==queries[qi]['query_id']
    branch=[];parent=None
    for node_id in outcome['node_ids']:
        node=nodes[node_id]
        assert node['node_id']==node_id and node['parent']==parent
        branch.append(node);parent=node_id
    return queries[qi],outcome,branch


def native_index(times,t,tolerance=1e-7):
    at=int(np.searchsorted(times,t));possible=[i for i in (at-1,at) if 0<=i<len(times) and abs(times[i]-t)<=tolerance]
    if len(possible)!=1:raise ValueError('timestamp is not uniquely native')
    return possible[0]


def advance_graph(graph,ob,delta,cells=None,materialize=True):
    if cells is None:cells={tuple(c['cell']):dict(c) for c in graph['cells']}
    occupied={tuple(u['cell']) for u in ob['static_updates'] if u['state']==2}
    for update in ob['static_updates']:
        if update['source']=='occupied_ego_base_support':continue
        key=tuple(update['cell']);old=cells.get(key,{})
        if update['state']==1 and key in occupied:continue
        cells[key]=dict(cell=list(key),height=update['height'],state=update['state'],
            source=update['source'],first_seen=old.get('first_seen',ob['timestamp']),last_seen=ob['timestamp'])
    result=dict(graph,timestamp=ob['timestamp'],revision=delta['revision'],ego_support=ob['robot_pose'][:3])
    if delta.get('invalidated_free_cells'):
        result['global_edges']=[]
        invalid={tuple(k) for k in delta['invalidated_free_cells']}
        result['global_nodes']={k:n for k,n in result['global_nodes'].items() if tuple(n['cell']) not in invalid}
    if not materialize:return result
    result['cells']=list(cells.values())
    # Retain only legal old abstraction. No future global rebuild is imported.
    result['global_nodes']={k:n for k,n in graph['global_nodes'].items() if cells.get(tuple(n['cell']),{}).get('state')==1}
    # Global links are not needed by this local encoder beyond their count.
    # A new obstruction invalidates all retained topology rather than allowing
    # an uncertified old edge across a changed path. Full production rebuilding
    # is deliberately not fabricated from future graph snapshots.
    if delta.get('invalidated_free_cells'):
        result['global_edges']=[]
    else:
        result['global_edges']=[e for e in graph['global_edges'] if e['source'] in result['global_nodes'] and e['target'] in result['global_nodes']]
    return result


def one_root(job):
    folder,row=job;folder=Path(folder)
    legal=read(folder/'legal_root.json.gz');query,outcome,branch=selected_branch(folder)
    with np.load(folder/'privileged_future.npz',allow_pickle=False) as z:
        times=z['timestamps'].copy();truth_xyz=z['positions'].copy();truth=truth_xyz[...,[0,2]]
    assert len(times)==len(truth) and np.all(np.diff(times)>0) and np.isfinite(truth).all()
    root_time=float(legal['history'][-1]['timestamp']);assert abs(root_time-times[0])<1e-7
    for ob in legal['history']:validate_frame(ob)
    history=list(legal['history']);actions=list(legal['history_actions']);graph=legal['partial_graph']
    graph_cells={tuple(c['cell']):dict(c) for c in graph['cells']}
    # Offline preparation owns these decoded records. Validate once and replay
    # the actual production partial-map operation, avoiding an unnecessary
    # second deep immutable online conversion of every old static certificate.
    from eventwm_graph.partial.map import PartialGraph
    replay=PartialGraph(graph['resolution'],graph['height'],graph['clearance'])
    for frame in history:replay.update(frame)
    restored={tuple(c['cell']):{k:v for k,v in c.items() if k!='cell'} for c in graph['cells']}
    assert replay.cells==restored and replay.revision==graph['revision'] and replay.timestamp==graph['timestamp']
    sealed_history=[OfflineViewFrame(ob) for ob in history]
    frames=[history[-1]]+[n['observation'] for n in branch]
    # Fixed anchor schedule plus genuine missing/return events. Eligibility is
    # established from the already-arrived frame at that anchor, not its future.
    schedule={0}
    ft=np.array([ob['timestamp'] for ob in frames])
    for h in (.5,1.,2.,3.):
        if ft[-1]>=root_time+h-1e-7:schedule.add(int(np.abs(ft-root_time-h).argmin()))
    source_to_person={};person_last={};source_last={};previous_positive=set()
    gap_added=False;return_added=False;records=[];ledger=collections.Counter()
    for j,ob in enumerate(frames):
        if j:
            node=branch[j-1];history.append(ob);actions.append(node['action']);graph=advance_graph(graph,ob,node['graph_delta'],graph_cells,False)
            validate_frame(ob);sealed_history.append(OfflineViewFrame(ob))
        ti=native_index(times,float(ob['timestamp']));y=truth[ti];clean=True;mapping={};current=set()
        for tr in positives(ob):
            z=np.asarray(tr['position']);close=np.flatnonzero(np.linalg.norm(truth_xyz[ti]-z,axis=-1)<2e-4)
            if len(close)!=1 or int(close[0]) in current:
                clean=False;ledger['ambiguous_positive_frames']+=1;continue
            p=int(close[0]);current.add(p);mapping[str(tr['track_id'])]=p
        returning=[]
        if not clean or not ob.get('observation_valid',True) or not ob.get('human_sensor_available',True):
            # Break physical certification across ambiguous/unavailable frames.
            source_to_person={};person_last={};source_last={};previous_positive=set()
        else:
            returning=[p for p in current if p in person_last and p not in previous_positive]
            if returning and not return_added:schedule.add(j);return_added=True
            for key,p in mapping.items():
                source_to_person[key]=p;source_last[key]=float(ob['timestamp']);person_last[p]=float(ob['timestamp'])
            aged=[p for p,t in person_last.items() if p not in current and .75-1e-7<=ob['timestamp']-t<=1.25+1e-7]
            if aged and not gap_added:schedule.add(j);gap_added=True
            previous_positive=current
        ledger['frames']+=1
        ledger['valid_empty_frames']+=int(clean and not current and ob.get('observation_valid',True))
        if j not in schedule:continue
        graph=dict(graph,cells=list(graph_cells.values()))
        context=LegalContext(tuple(sealed_history),tuple(actions),graph)
        features=extract(context);keys=features['keys'];n=len(keys)
        target=np.zeros((n,len(TARGET_TIMES),2),np.float32);target_mask=np.zeros((n,len(TARGET_TIMES)),np.float32)
        presence=np.zeros(n,np.float32);presence_mask=np.zeros(n,np.float32);physical=np.full(n,-1,int)
        # Multiple anonymous IDs that match one physical person are explicitly
        # counted. Only most recently observed episode gets existence target1.
        owners={}
        for key,p in source_to_person.items():
            if key in keys and (p not in owners or source_last[key]>source_last[owners[p]]):owners[p]=key
        for i,key in enumerate(keys):
            if key in source_to_person:
                p=source_to_person[key];presence_mask[i]=1;presence[i]=owners[p]==key
                if presence[i]:physical[i]=p
            else:ledger['unresolved_prefix_episodes']+=1
        actual_times=np.zeros(len(TARGET_TIMES),np.float64)
        for hidx,h in enumerate(TARGET_TIMES):
            requested=float(ob['timestamp'])+h
            if requested>times[-1]+1e-7:continue
            targetidx=int(np.abs(times-requested).argmin())
            if abs(times[targetidx]-requested)>.0280001:
                ledger['native_horizon_nearest_tolerance_censored']+=1;continue
            actual_times[hidx]=times[targetidx]-ob['timestamp']
            for i,p in enumerate(physical):
                if p>=0:target[i,hidx]=truth[targetidx,p];target_mask[i,hidx]=1
        represented=set(physical[physical>=0]);unseen=[p for p in range(truth.shape[1]) if p not in represented]
        # This residual target includes physical people whose old prefix identity
        # is unresolvable. The approximation is recorded, never called exact.
        feature_arrays={k:features[k] for k in ('x','graph','global_graph','origin','velocity','global_features','robotxy','age','minimum_count')}
        remaining=list(range(len(y)));set_valid=True
        for i in np.flatnonzero(features['age']==0):
            close=[p for p in remaining if np.linalg.norm(y[p]-features['origin'][i])<2e-4]
            if not close or (len(close)>1 and np.max(np.linalg.norm(y[close]-y[close[0]],axis=-1))>1e-7):set_valid=False;break
            remaining.remove(close[0])
        unseen_velocity=(truth[ti+1,unseen]-y[unseen])/(times[ti+1]-times[ti]) if ti+1<len(times) else np.zeros((len(unseen),2))
        records.append(dict(**feature_arrays,keys=keys,timestamp=float(ob['timestamp']),times=actual_times,
            target=target,target_mask=target_mask,presence=presence,presence_mask=presence_mask,
            unseen_positions=y[unseen].astype(np.float32),unseen_velocity=unseen_velocity.astype(np.float32),
            set_positions=y[remaining].astype(np.float32),set_valid=float(set_valid),
            unseen_velocity_valid=np.full(len(unseen),float(ti+1<len(times)),np.float32),physical=physical,
            root_id=row['root_id'],scene_id=row['scene_id'],anchor_index=j,
            reappearance=bool(returning),
            unresolved_prefix=int(sum(key not in source_to_person for key in keys)),
            query_id=query['query_id'],terminal=outcome['summary']['termination'],
            censored_times=int(np.count_nonzero(actual_times==0))))
    for r in records:r['weight']=1./len(records)
    ledger.update(roots=1,anchors=len(records),tracks=sum(len(r['keys']) for r in records),
                  supervised_paths=sum(int((r['target_mask'].sum(-1)>0).sum()) for r in records),
                  true_people=truth.shape[1],native_horizon_milliseconds=round((times[-1]-times[0])*1000))
    return records,dict(ledger)


def prepared_root_ledger(manifest_path):
    """Recover exact Root accounting without reopening prepared payload shards.

    Old full caches own a report beside each shard. Extended caches reference
    their immutable parent, so their Root accounting must include that parent
    recursively rather than just the local delta reports.
    """
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('root_ledger'):
        item=manifest['root_ledger'];path=OUT/item['path']
        if sha(path)!=item['sha256']:raise ValueError('prepared Root ledger hash mismatch')
        with gzip.open(path,'rt') as stream:roots=json.load(stream)
    else:
        roots=[];inherited=0
        if manifest['schema']=='two_stage_extended_native_anchor_cache_1':
            parent=Path(manifest['parent_manifest'])
            if sha(parent)!=manifest['identity']['parent_manifest_sha256']:
                raise ValueError('prepared parent manifest hash mismatch')
            parent_manifest=json.loads(parent.read_text())
            inherited=len(parent_manifest['shards'])
            if manifest['shards'][:inherited]!=parent_manifest['shards']:
                raise ValueError('prepared inherited shard identity mismatch')
            roots=prepared_root_ledger(parent)
        elif manifest['schema']!='two_stage_full_native_anchor_cache_1':
            raise ValueError('cache has no exact metadata Root ledger')
        for shard in manifest['shards'][inherited:]:
            report=OUT/Path(shard['path']).with_suffix('.json')
            item=json.loads(report.read_text())
            if item['sha256']!=shard['sha256'] or item['anchors']!=shard['anchors']:
                raise ValueError('prepared shard report identity mismatch')
            roots.extend(item['roots'])
    if len(roots)!=manifest['roots'] or len({r['root_id'] for r in roots})!=len(roots):
        raise ValueError('prepared Root ledger cardinality or duplicate mismatch')
    if len({r['scene_id'] for r in roots})!=manifest['scenes']:
        raise ValueError('prepared Root ledger scene mismatch')
    if sum(r['anchors'] for r in roots)!=sum(s['anchors'] for s in manifest['shards']):
        raise ValueError('prepared Root ledger anchor mismatch')
    return roots


def jobs_for(split,snapshot=None):
    """Resolve sources from an explicit snapshot or a named prepared cache.

    The historical train/dev defaults are retained. New fitting campaigns pass
    their cache name (or ``snapshot=...``), never a moving latest-data alias.
    """
    if snapshot is not None:
        if split!='train':raise ValueError('explicit fitting snapshot requires train split')
        path=Path(snapshot)
        if not path.is_absolute():
            if path.name!=str(path):raise ValueError('snapshot must be a name or absolute manifest path')
            path=OUT/path/'manifest.json'
        if path.is_dir():path=path/'manifest.json'
        m=json.loads(path.read_text())
        if any(r['split']!='train' or r['source_split']!='train' for r in m['rows']):
            raise ValueError('fitting snapshot contains nontraining Roots')
        return [(r['folder'],r) for r in m['rows']]
    if split=='train':
        latest=OUT/'latest_snapshot/manifest.json'
        if latest.exists():
            m=json.loads(latest.read_text());return [(r['folder'],r) for r in m['rows']]
        jobs=[]
        for kind in ('snapshot','supplement'):
            m=json.loads((OUT/kind/'manifest.json').read_text())
            jobs.extend((str(OUT/kind/'roots'/r['root_id']),r) for r in m['rows'])
        return jobs
    if split=='dev':
        m=json.loads((OLD/'snapshot/manifest.json').read_text())
        return [(str(OLD/'snapshot/roots'/r['root_id']),r) for r in m['rows'] if r['split']=='dev']
    if Path(split).name==split:
        path=OUT/'prepared'/split/'manifest.json'
        if path.exists():
            cache=json.loads(path.read_text())
            if cache['split']=='dev':return jobs_for('dev')
            if cache['split']!='train':raise ValueError('protected split cannot be prepared')
            if 'snapshot' not in cache:raise ValueError('named cache lacks an explicit frozen snapshot')
            jobs=jobs_for('train',snapshot=cache['snapshot'])
            expected=cache['identity']['snapshot_sha256']
            if sha(cache['snapshot'])!=expected:raise ValueError('named cache snapshot hash mismatch')
            roots={r['root_id'] for r in prepared_root_ledger(path)}
            result=[job for job in jobs if job[1]['root_id'] in roots]
            if len(result)!=len(roots):raise ValueError('named cache Root source mapping mismatch')
            return result
    raise ValueError('protected split cannot be prepared')


def prepare(split,workers=2,limit=0,name=None):
    jobs=jobs_for(split)
    if limit:jobs=jobs[:limit]
    name=name or split+('_pilot' if limit else '')
    if Path(name).name!=name:raise ValueError('cache name must be a single path component')
    dest=OUT/'prepared'/name;dest.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();counts=collections.Counter();scenes=set();failures=[];shards=[];batch=[]
    def flush():
        if not batch:return
        path=dest/('%05d.npy'%len(shards));np.save(path,np.asarray(batch,dtype=object),allow_pickle=True)
        shards.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path),anchors=len(batch)))
        batch.clear()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # map propagates data integrity errors rather than dropping failed roots.
        for i,(records,ledger) in enumerate(pool.map(one_root,jobs,chunksize=16)):
            counts.update(ledger);scenes.add(jobs[i][1]['scene_id']);batch.extend(records)
            if len(batch)>=1000:flush()
            if (i+1)%1000==0:print(json.dumps(dict(split=name,roots=i+1,anchors=counts['anchors'],seconds=time.perf_counter()-start)),flush=True)
    flush()
    dump(dest/'manifest.json',dict(schema='two_stage_native_anchor_cache_1',split=split,roots=len(jobs),scenes=len(scenes),
        counts=dict(counts),shards=shards,seconds=time.perf_counter()-start,failures=failures,source_sha256=sha(__file__),
        snapshot_sha256=sha(OUT/'snapshot/manifest.json') if split=='train' else sha(OLD/'snapshot/manifest.json')))


def load_records(split):
    manifest=json.loads((OUT/'prepared'/split/'manifest.json').read_text());records=[]
    for shard in manifest['shards']:
        path=OUT/shard['path'];assert sha(path)==shard['sha256']
        records.extend(np.load(path,allow_pickle=True).tolist())
    return records


def collate(records,device='cpu'):
    import torch
    n=max(1,max(len(r['keys']) for r in records));u=max(1,max(len(r['unseen_positions']) for r in records));b=len(records)
    result={}
    for key in ('x','graph','origin','velocity','target','target_mask','presence','presence_mask','age'):
        sample=records[0][key];shape=(b,n)+sample.shape[1:];a=np.zeros(shape,np.float32)
        for i,r in enumerate(records):a[i,:len(r['keys'])]=r[key]
        result[key]=torch.as_tensor(a,device=device)
    for key in ('global_features','global_graph','robotxy','times','weight'):
        result[key]=torch.as_tensor(np.asarray([r[key] for r in records],np.float32),device=device)
    a=np.zeros((b,n),np.float32);positions=np.zeros((b,u,2),np.float32);umask=np.zeros((b,u),np.float32)
    velocities=np.zeros((b,u,2),np.float32);velocity_valid=np.zeros((b,u),np.float32)
    for i,r in enumerate(records):
        a[i,:len(r['keys'])]=1;positions[i,:len(r['unseen_positions'])]=r['unseen_positions'];umask[i,:len(r['unseen_positions'])]=1
        velocities[i,:len(r['unseen_positions'])]=r['unseen_velocity'];velocity_valid[i,:len(r['unseen_positions'])]=r['unseen_velocity_valid']
    result['mask']=torch.as_tensor(a,device=device);result['unseen_positions']=torch.as_tensor(positions,device=device);result['unseen_mask']=torch.as_tensor(umask,device=device)
    result['unseen_velocity']=torch.as_tensor(velocities,device=device);result['unseen_velocity_valid']=torch.as_tensor(velocity_valid,device=device)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['train','dev']);p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int,default=0);p.add_argument('--name');a=p.parse_args()
    prepare(a.split,a.workers,a.limit,a.name)
