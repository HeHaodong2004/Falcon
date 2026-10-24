"""Legal observation-event metadata, without permanent physical identity.

Anonymous raw keys are used transiently to recover the exact last observation
that supplied each legacy source feature. Stored indices identify observations
in the positive-set witness, never people. Raw tracker velocities are soft
association evidence, not ground-truth motion or hard identity constraints.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from .common import OUT,OLD,dump,sha
from .positive_history import FILES,compact_decode,selected_frames,positive_xyz,immutable_array

SCHEMA='legal_last_observation_event_metadata_1'
JOBS=None
DEST=None
IDENTITY=None
PRIMARY=None


def key_order_hash(keys):return hashlib.sha256(json.dumps(list(map(str,keys)),separators=(',',':')).encode()).hexdigest()


def pack_events(history,branch,anchors,primary):
    frames=list(history)+[node['observation'] for node in branch];count=len(primary['positive_xyz'])
    velocities=np.zeros((count,3),np.float64);velocity_valid=np.zeros(count,bool);velocity_present=np.zeros(count,bool)
    events=np.zeros(count,np.int64);last={};by_end={len(history)+int(a['anchor_index']):a for a in anchors};packed=[]
    if len(by_end)!=len(anchors):raise ValueError('duplicate event-metadata anchor')
    for fi,frame in enumerate(frames):
        if float(frame['timestamp'])!=float(primary['timestamps'][fi]):raise ValueError('event and witness frame clocks differ')
        tracks=[]
        if frame.get('observation_valid',True) and frame.get('human_sensor_available',True):
            tracks=[tr for tr in frame['tracks'] if tr.get('visible_now',False) and tr.get('valid',True) and abs(float(tr['last_seen_time'])-float(frame['timestamp']))<1e-7]
        points=np.asarray([tr['position'] for tr in tracks],np.float64).reshape(-1,3)
        order=np.lexsort((points[:,2],points[:,1],points[:,0])) if len(points) else np.empty(0,np.int64)
        begin,end=map(int,primary['positive_offsets'][fi:fi+2])
        if not np.array_equal(points[order],primary['positive_xyz'][begin:end]):raise ValueError('last-event observations differ from exact witness multiset')
        for offset,index in enumerate(order):
            tr=tracks[int(index)];event=begin+offset;value=tr.get('estimated_velocity');present=value is not None
            if present:
                estimate=np.asarray(value,np.float64)
                if estimate.shape!=(3,) or not np.isfinite(estimate).all():raise ValueError('invalid raw legal velocity estimate')
                velocities[event]=estimate
            valid=bool(tr.get('velocity_valid',False))
            if valid and not present:raise ValueError('raw legal velocity validity contradicts missing estimate')
            velocity_valid[event]=valid;velocity_present[event]=present;events[event]=fi;last[str(tr['track_id'])]=event
        anchor=by_end.get(fi+1)
        if anchor is not None:
            if anchor['query_id']!=primary['canonical_query_id'] or anchor['timestamp']!=frame['timestamp']:raise ValueError('event-metadata anchor identity differs')
            try:indices=np.asarray([last[str(key)] for key in anchor['keys']],np.int64)
            except KeyError as error:raise ValueError('legacy source lacks actual causal observation event') from error
            if np.any(indices>=end):raise ValueError('future observation event leaked into earlier anchor')
            packed.append(dict(anchor_index=int(anchor['anchor_index']),timestamp=float(anchor['timestamp']),positive_event_prefix_end=end,source_key_order_sha256=key_order_hash(anchor['keys']),
                source_last_event_indices=indices,source_last_raw_xyz=primary['positive_xyz'][indices].copy(),
                source_last_timestamps=primary['timestamps'][events[indices]].copy(),source_last_estimated_velocity=velocities[indices].copy(),source_last_velocity_valid=velocity_valid[indices].copy()))
    if len(packed)!=len(anchors):raise ValueError('event metadata omitted an unchanged anchor')
    return dict(schema=SCHEMA,root_id=primary['root_id'],scene_id=primary['scene_id'],canonical_query_id=primary['canonical_query_id'],
        event_frame_indices=events,event_estimated_velocity=velocities,event_velocity_valid=velocity_valid,event_velocity_present=velocity_present,anchors=packed)


def root_record(row,anchors,primary):
    raw={}
    for name in FILES:
        data=(Path(row['folder'])/name).read_bytes();digest=hashlib.sha256(data).hexdigest()
        if digest!=row['files'][name] or digest!=primary['source_hashes'][name]:raise ValueError('event metadata source/witness hash mismatch')
        raw[name]=gzip.decompress(data) if name.endswith('.gz') else data
    history=compact_decode(raw['legal_root.json.gz'],True);query,branch=selected_frames(json.loads(raw['queries.json.gz']),json.loads(raw['outcomes.json.gz']),raw['future_observations.json.gz'])
    if query['query_id']!=primary['canonical_query_id']:raise ValueError('event metadata canonical branch differs')
    return pack_events(history,branch,anchors,primary)


def chunk(bounds):
    begin,end=bounds;tag='%06d_%06d'%(begin,end);report=DEST/(tag+'.json')
    if report.exists():
        value=json.loads(report.read_text())
        if value['identity']!=IDENTITY or sha(OUT/value['path'])!=value['sha256']:raise ValueError('event metadata resume identity mismatch')
        return value
    primary_report=PRIMARY/(tag+'.json')
    while not primary_report.exists():time.sleep(5)
    pr=json.loads(primary_report.read_text());data=(OUT/pr['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=pr['sha256']:raise ValueError('primary witness shard hash mismatch')
    roots=list(np.load(io.BytesIO(data),allow_pickle=True))
    if [r['root_id'] for r in roots]!=[job[0]['root_id'] for job in JOBS[begin:end]]:raise ValueError('primary witness Root ordering differs')
    rows=[];counts=collections.Counter()
    for (row,anchors),primary in zip(JOBS[begin:end],roots):
        value=root_record(row,anchors,primary);rows.append(value)
        counts.update(roots=1,anchors=len(value['anchors']),events=len(value['event_frame_indices']),sources=sum(len(a['source_last_event_indices']) for a in value['anchors']))
    path=DEST/(tag+'.npy');temp=DEST/(tag+'.tmp.npy');np.save(temp,np.asarray(rows,object),allow_pickle=True);temp.replace(path)
    value=dict(identity=IDENTITY,path=str(path.relative_to(OUT)),sha256=sha(path),primary_shard_path=pr['path'],primary_shard_sha256=pr['sha256'],
        roots=[dict(root_id=r['root_id'],scene_id=r['scene_id'],anchors=len(r['anchors'])) for r in rows],counts=dict(counts));dump(report,value);return value


def run(cache='train_campaign_20261023',primary_cache=None,workers=2):
    global JOBS,DEST,IDENTITY,PRIMARY
    if Path(cache).name!=cache or not 1<=workers<=4:raise ValueError('invalid last-event data arguments')
    primary_cache=primary_cache or cache;PRIMARY=OUT/'positive_history'/primary_cache
    pp=PRIMARY/'preparation_identity.json'
    if not pp.exists():raise ValueError('primary witness campaign must exist before event metadata')
    prepared_path=OUT/'prepared'/cache/'manifest.json';prepared=json.loads(prepared_path.read_text());anchors=collections.defaultdict(list)
    for shard in prepared['shards']:
        data=(OUT/shard['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=shard['sha256']:raise ValueError('prepared source-key shard hash mismatch')
        for row in np.load(io.BytesIO(data),allow_pickle=True):anchors[row['root_id']].append({key:row[key] for key in ('anchor_index','timestamp','query_id','keys')})
    if prepared['split']=='train':
        sp=Path(prepared['snapshot']);source_rows=json.loads(sp.read_text())['rows']
        if sha(sp)!=prepared['identity']['snapshot_sha256'] or any(r['split']!='train' or r['source_split']!='train' for r in source_rows):raise ValueError('altered or nontraining event source')
    elif prepared['split']=='dev':
        sp=OLD/'snapshot/manifest.json';source_rows=[dict(r,folder=str(OLD/'snapshot/roots'/r['root_id'])) for r in json.loads(sp.read_text())['rows'] if r['split']=='dev']
    else:raise ValueError('protected event metadata cannot be prepared')
    if len({r['root_id'] for r in source_rows})!=len(source_rows) or set(anchors)!={r['root_id'] for r in source_rows}:raise ValueError('event metadata Root set mismatch')
    JOBS=[(row,anchors[row['root_id']]) for row in source_rows];DEST=OUT/'last_event_metadata'/cache;DEST.mkdir(parents=True,exist_ok=True)
    IDENTITY=dict(schema=SCHEMA,prepared_manifest_sha256=sha(prepared_path),snapshot_manifest_sha256=sha(sp),primary_identity_sha256=sha(pp),primary_cache=primary_cache,source_sha256=sha(__file__),
        identity_semantics='Indices identify actual legal positive observation events, not physical people; anonymous keys are used only transiently for exact legacy feature order',velocity_semantics='Raw legally reported estimated velocity and validity, soft evidence only')
    ip=DEST/'preparation_identity.json'
    if ip.exists() and json.loads(ip.read_text())!=IDENTITY:raise ValueError('last-event preparation identity changed')
    dump(ip,IDENTITY);counts=collections.Counter();reports=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for report in pool.map(chunk,[(i,min(i+200,len(JOBS))) for i in range(0,len(JOBS),200)]):
            counts.update(report['counts']);reports.append(report);dump(DEST/'progress.json',dict(counts=counts,total_roots=len(JOBS),seconds=time.perf_counter()-start))
            print(json.dumps(dict(roots=counts['roots'],total_roots=len(JOBS),anchors=counts['anchors'],seconds=time.perf_counter()-start)),flush=True)
    if counts['roots']!=prepared['roots'] or counts['anchors']!=prepared['counts']['anchors']:raise ValueError('last-event complete accounting mismatch')
    dump(DEST/'manifest.json',dict(schema=SCHEMA,split=prepared['split'],roots=counts['roots'],scenes=prepared['scenes'],counts=dict(counts),identity=IDENTITY,prepared_manifest=str(prepared_path),
        shards=[{k:r[k] for k in ('path','sha256','primary_shard_path','primary_shard_sha256','roots','counts')} for r in reports],failures=[],seconds=time.perf_counter()-start))
    for p in DEST.iterdir():
        if p.is_file():os.chmod(p,0o444)
    os.chmod(DEST,0o555)


def attach_metadata(records,cache='train_campaign_20261023'):
    path=OUT/'last_event_metadata'/cache/'manifest.json';manifest=json.loads(path.read_text())
    if manifest['schema']!=SCHEMA or sha(manifest['prepared_manifest'])!=manifest['identity']['prepared_manifest_sha256']:raise ValueError('last-event cache identity mismatch')
    needed=collections.defaultdict(dict)
    for row in records:
        idx=int(row['anchor_index'])
        if idx in needed[row['root_id']]:raise ValueError('duplicate requested event-metadata anchor')
        needed[row['root_id']][idx]=row
    attached=0
    for shard in manifest['shards']:
        if not any(root['root_id'] in needed for root in shard['roots']):continue
        data=(OUT/shard['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=shard['sha256']:raise ValueError('event metadata shard hash mismatch')
        for packed in np.load(io.BytesIO(data),allow_pickle=True):
            rid=packed['root_id']
            if rid not in needed:continue
            for anchor in packed['anchors']:
                idx=int(anchor['anchor_index'])
                if idx not in needed[rid]:continue
                row=needed[rid][idx]
                if row['timestamp']!=anchor['timestamp'] or row['query_id']!=packed['canonical_query_id'] or row['scene_id']!=packed['scene_id'] or key_order_hash(row['keys'])!=anchor['source_key_order_sha256']:raise ValueError('legacy source order or event anchor differs')
                for key in ('source_last_event_indices','source_last_raw_xyz','source_last_timestamps','source_last_estimated_velocity','source_last_velocity_valid'):
                    values=anchor[key];row[key]=immutable_array(values,np.int64 if key.endswith('indices') else bool if key.endswith('valid') else np.float64)
                end=int(anchor['positive_event_prefix_end'])
                for source,key,dtype in (('event_estimated_velocity','positive_event_estimated_velocity',np.float64),
                                         ('event_velocity_valid','positive_event_velocity_valid',bool),('event_velocity_present','positive_event_velocity_present',bool)):
                    row[key]=immutable_array(packed[source][:end],dtype)
                if 'positive_history' in row and np.any(row['source_last_event_indices']>=len(row['positive_history'].positive_xyz)):raise ValueError('future source observation event leaked')
                attached+=1
    if attached!=len(records):raise ValueError('last-event cache omitted requested anchors')
    return dict(schema=SCHEMA,anchors=attached,roots=len(needed),manifest_sha256=sha(path),role='LEGAL_OBSERVATION_EVENT_BOOKKEEPING_ONLY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default='train_campaign_20261023');p.add_argument('--primary-cache');p.add_argument('--workers',type=int,default=2);a=p.parse_args();run(a.cache,a.primary_cache,a.workers)
