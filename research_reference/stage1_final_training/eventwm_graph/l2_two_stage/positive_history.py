"""Lossless causal positive-set histories for association; no physical IDs.

Only recorded legal observation sets become witness coordinates. Each training
anchor receives an immutable prefix. The backing Root cache can serve later
anchors, but the returned witness never contains their observations.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import numpy as np
from .common import OUT,OLD,dump,sha

SCHEMA='causal_actual_positive_set_history_1'
SENSOR='ideal_frustum_observed_cspace_dynamic_occlusion_1'
FILES=('legal_root.json.gz','queries.json.gz','outcomes.json.gz','future_observations.json.gz','manifest.json')
STATIC=re.compile(rb'("static_updates"\s*:\s*)\[(?:\s*\{[^{}]*\}\s*,?)*\s*\]')
JOBS=None
DEST=None
IDENTITY=None


def compact_decode(raw,history=False):
    """Strip only irrelevant static arrays, with fail-closed full decoding."""
    compact,removed=STATIC.subn(rb'\1[]',raw)
    if removed!=raw.count(b'"static_updates"'):compact=raw
    if history:
        match=re.search(rb'"history"\s*:\s*',compact)
        if match is None:raise ValueError('legal history missing')
        rows,_=json.JSONDecoder().raw_decode(compact[match.end():].decode())
        return rows
    return json.loads(compact)


def selected_frames(queries,outcomes,raw):
    index=max(range(len(queries)),key=lambda i:(len(queries[i]['compiled']['actions']),queries[i]['query_id']))
    query,outcome=queries[index],outcomes[index]
    if query['query_id']!=outcome['query_id']:raise ValueError('canonical Query/outcome mismatch')
    starts=[m.start() for m in re.finditer(rb'\{"action"\s*:',raw)]
    ids=[int(m.group(1)) for m in re.finditer(rb'"node_id"\s*:\s*(\d+)',raw)]
    parent=None;branch=[]
    if len(starts)==len(ids) and ids==list(range(len(ids))):
        ends=starts[1:]+[len(raw)]
        selected=(compact_decode(raw[starts[i]:ends[i]].rstrip(b' \n\r\t,]')) for i in outcome['node_ids'])
    else:
        nodes=compact_decode(raw);selected=(nodes[i] for i in outcome['node_ids'])
    for index,node in zip(outcome['node_ids'],selected):
        if node['node_id']!=index or node['parent']!=parent:raise ValueError('noncausal canonical branch')
        branch.append(node);parent=index
    return query,branch


def positive_xyz(frame):
    if not frame.get('observation_valid',True) or not frame.get('human_sensor_available',True):
        return np.empty((0,3),np.float64)
    now=float(frame['timestamp']);rows=[]
    for track in frame['tracks']:
        if track.get('visible_now',False) and track.get('valid',True) and abs(float(track['last_seen_time'])-now)<1e-7:
            position=np.asarray(track['position'],np.float64)
            if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('invalid actual positive position')
            rows.append(position)
    # A multiset: identical co-located detections retain their multiplicity.
    points=np.asarray(rows,np.float64).reshape(-1,3)
    if len(points):points=points[np.lexsort((points[:,2],points[:,1],points[:,0]))]
    return points


def pack_frames(history,branch,anchors,query_id,root_id,scene_id):
    frames=list(history)+[node['observation'] for node in branch]
    times=np.asarray([frame['timestamp'] for frame in frames],np.float64)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('noncausal legal frame clocks')
    if any(frame.get('sensor_protocol')!=SENSOR for frame in frames):raise ValueError('unsupported legal sensor protocol')
    observed=np.asarray([frame.get('observation_valid',True) for frame in frames],bool)
    available=np.asarray([frame.get('human_sensor_available',True) for frame in frames],bool)
    points=[positive_xyz(frame) for frame in frames]
    offsets=np.r_[0,np.cumsum([len(p) for p in points])].astype(np.int64)
    ends=[];indices=[];anchor_times=[]
    for anchor in sorted(anchors,key=lambda a:a['anchor_index']):
        j=int(anchor['anchor_index']);end=len(history)+j
        if j<0 or end>len(frames) or anchor['query_id']!=query_id or anchor['scene_id']!=scene_id or abs(float(anchor['timestamp'])-times[end-1])>1e-7:
            raise ValueError('prepared anchor differs from its exact causal branch')
        ends.append(end);indices.append(j);anchor_times.append(float(anchor['timestamp']))
    if len(set(indices))!=len(indices):raise ValueError('duplicate prepared anchor identity')
    return dict(schema=SCHEMA,root_id=root_id,scene_id=scene_id,canonical_query_id=query_id,
        timestamps=times,positive_offsets=offsets,positive_xyz=np.concatenate(points,axis=0),
        observation_valid=observed,human_sensor_available=available,
        anchor_indices=np.asarray(indices,np.int64),anchor_prefix_ends=np.asarray(ends,np.int64),anchor_timestamps=np.asarray(anchor_times,np.float64),
        horizontal_motion_bound_m_s=3.0,bound_contract='native_endpoint_horizontal_bound_3m_s; exact float64 coordinates, no finite position quantization')


def immutable_array(values,dtype):
    source=np.asarray(values,dtype=dtype)
    return np.frombuffer(source.tobytes(),dtype=source.dtype).reshape(source.shape)


@dataclass(frozen=True)
class PositiveSetHistory:
    root_id:str
    scene_id:str
    canonical_query_id:str
    anchor_index:int
    timestamps:np.ndarray
    positive_offsets:np.ndarray
    positive_xyz:np.ndarray
    observation_valid:np.ndarray
    human_sensor_available:np.ndarray
    horizontal_motion_bound_m_s:float=3.0

    def __post_init__(self):
        for key,dtype in (('timestamps',np.float64),('positive_offsets',np.int64),('positive_xyz',np.float64),
                          ('observation_valid',bool),('human_sensor_available',bool)):
            object.__setattr__(self,key,immutable_array(getattr(self,key),dtype))
        n=len(self.timestamps)
        if (not n or self.timestamps.ndim!=1 or not np.isfinite(self.timestamps).all() or np.any(np.diff(self.timestamps)<=0)
                or self.positive_offsets.shape!=(n+1,) or self.positive_offsets[0]!=0 or np.any(np.diff(self.positive_offsets)<0)
                or self.positive_xyz.shape!=(int(self.positive_offsets[-1]),3) or not np.isfinite(self.positive_xyz).all()
                or self.observation_valid.shape!=(n,) or self.human_sensor_available.shape!=(n,)
                or self.horizontal_motion_bound_m_s!=3.0):
            raise ValueError('invalid causal positive-set history')
        if np.any((np.diff(self.positive_offsets)>0)&~(self.observation_valid&self.human_sensor_available)):
            raise ValueError('invalid/unavailable frame cannot assert actual positive detections')

    @property
    def minimum_count(self):return int(np.diff(self.positive_offsets).max(initial=0))


def history_at_anchor(record,anchor_index):
    """Return copied immutable causal arrays; later Root frames are unreachable."""
    at=np.flatnonzero(np.asarray(record['anchor_indices'])==int(anchor_index))
    if len(at)!=1:raise ValueError('anchor missing or duplicated in positive-set history')
    end=int(record['anchor_prefix_ends'][at[0]]);offsets=np.asarray(record['positive_offsets'])[:end+1]
    return PositiveSetHistory(record['root_id'],record['scene_id'],record['canonical_query_id'],int(anchor_index),
        np.asarray(record['timestamps'])[:end],offsets,np.asarray(record['positive_xyz'])[:int(offsets[-1])],
        np.asarray(record['observation_valid'])[:end],np.asarray(record['human_sensor_available'])[:end],
        float(record['horizontal_motion_bound_m_s']))


def root_record(job):
    row,anchors=job;folder=Path(row['folder']);raw={};hashes={}
    for name in FILES:
        data=(folder/name).read_bytes();digest=hashlib.sha256(data).hexdigest()
        if digest!=row['files'][name]:raise ValueError('legal source hash mismatch: '+name)
        hashes[name]=digest;raw[name]=gzip.decompress(data) if name.endswith('.gz') else data
    metadata=json.loads(raw['manifest.json'])
    if (metadata['root_id']!=row['root_id'] or metadata['scene_id']!=row['scene_id']
            or metadata['collection_config']['human_horizontal_motion_bound_m_s']!=3.0):
        raise ValueError('Root identity or certified native motion bound differs')
    history=compact_decode(raw['legal_root.json.gz'],True)
    query,branch=selected_frames(json.loads(raw['queries.json.gz']),json.loads(raw['outcomes.json.gz']),raw['future_observations.json.gz'])
    record=pack_frames(history,branch,anchors,query['query_id'],row['root_id'],row['scene_id'])
    record['source_hashes']=hashes
    record['bound_version']={name:metadata['version'][name] for name in ('collect.py','query.py')}
    return record


def attach_histories(records,cache='train_campaign_20261023'):
    """Attach only immutable anchor prefixes to existing training records."""
    if Path(cache).name!=cache:raise ValueError('cache name must be a path component')
    path=OUT/'positive_history'/cache/'manifest.json';manifest=json.loads(path.read_text())
    if manifest['schema']!=SCHEMA:raise ValueError('unsupported positive history schema')
    if sha(manifest['prepared_manifest'])!=manifest['identity']['prepared_manifest_sha256']:
        raise ValueError('positive history prepared-cohort identity mismatch')
    needed=collections.defaultdict(list)
    for record in records:needed[record['root_id']].append(record)
    attached=0;seen=set()
    for shard in manifest['shards']:
        if not any(row['root_id'] in needed for row in shard['roots']):continue
        data=(OUT/shard['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=shard['sha256']:raise ValueError('positive history shard hash mismatch')
        for packed in np.load(io.BytesIO(data),allow_pickle=True):
            rid=packed['root_id']
            if rid not in needed:continue
            if rid in seen:raise ValueError('duplicate positive history Root')
            seen.add(rid)
            for record in needed[rid]:
                witness=history_at_anchor(packed,record['anchor_index'])
                if (witness.scene_id!=record['scene_id'] or witness.canonical_query_id!=record['query_id']
                        or float(witness.timestamps[-1])!=float(record['timestamp'])):
                    raise ValueError('positive history anchor identity differs')
                record['positive_history']=witness;attached+=1
    if set(needed)!=seen or attached!=len(records):raise ValueError('positive histories omitted requested anchors')
    return dict(schema=SCHEMA,records=attached,roots=len(seen),manifest_sha256=sha(path),
                information_boundary='Only actual positive XYZ multisets through each anchor; no person IDs or privileged labels')


def chunk(bounds):
    begin,end=bounds;tag='%06d_%06d'%(begin,end);report=DEST/(tag+'.json')
    if report.exists():
        value=json.loads(report.read_text())
        if value['identity']!=IDENTITY or sha(OUT/value['path'])!=value['sha256']:raise ValueError('positive history resume identity mismatch')
        return value
    rows=[];counts=collections.Counter();roots=[]
    for job in JOBS[begin:end]:
        row=root_record(job);rows.append(row);sizes=np.diff(row['positive_offsets'])
        valid=row['observation_valid']&row['human_sensor_available']
        counts.update(roots=1,anchors=len(row['anchor_indices']),frames=len(row['timestamps']),positive_points=len(row['positive_xyz']),
            valid_empty_frames=int(np.count_nonzero(valid&(sizes==0))),whole_frame_invalid=int(np.count_nonzero(~row['observation_valid'])),
            human_sensor_unavailable=int(np.count_nonzero(~row['human_sensor_available'])))
        roots.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],anchors=len(row['anchor_indices'])))
    path=DEST/(tag+'.npy');temp=DEST/(tag+'.tmp.npy');np.save(temp,np.asarray(rows,object),allow_pickle=True);temp.replace(path)
    value=dict(identity=IDENTITY,path=str(path.relative_to(OUT)),sha256=sha(path),roots=roots,counts=dict(counts))
    dump(report,value);return value


def run(cache='train_campaign_20261023',workers=2,name=None):
    global JOBS,DEST,IDENTITY
    if Path(cache).name!=cache or workers<1 or workers>4:raise ValueError('invalid data campaign arguments')
    prepared_path=OUT/'prepared'/cache/'manifest.json';prepared=json.loads(prepared_path.read_text());anchors=collections.defaultdict(list)
    for shard in prepared['shards']:
        data=(OUT/shard['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=shard['sha256']:raise ValueError('prepared anchor shard hash mismatch')
        for row in np.load(io.BytesIO(data),allow_pickle=True):
            anchors[row['root_id']].append({key:row[key] for key in ('anchor_index','timestamp','query_id','scene_id')})
    if prepared['split']=='train':
        snapshot_path=Path(prepared['snapshot']);snapshot=json.loads(snapshot_path.read_text());source_rows=snapshot['rows']
        if sha(snapshot_path)!=prepared['identity']['snapshot_sha256']:raise ValueError('fitting snapshot identity mismatch')
        if any(row['split']!='train' or row['source_split']!='train' for row in source_rows):
            raise ValueError('fitting positive histories cannot open nontraining sources')
    elif prepared['split']=='dev':
        snapshot_path=OLD/'snapshot/manifest.json';snapshot=json.loads(snapshot_path.read_text())
        source_rows=[dict(row,folder=str(OLD/'snapshot/roots'/row['root_id'])) for row in snapshot['rows'] if row['split']=='dev']
    else:raise ValueError('protected positive histories cannot be prepared')
    rows={row['root_id']:row for row in source_rows}
    if len(rows)!=len(source_rows):raise ValueError('duplicate positive-history source Root')
    if set(anchors)!=set(rows):raise ValueError('positive history Root set differs from unchanged prepared cohort')
    JOBS=[(row,anchors[row['root_id']]) for row in source_rows]
    name=name or cache
    if Path(name).name!=name:raise ValueError('cache name must be a path component')
    DEST=OUT/'positive_history'/name;DEST.mkdir(parents=True,exist_ok=True)
    IDENTITY=dict(schema=SCHEMA,prepared_manifest_sha256=sha(prepared_path),snapshot_manifest_sha256=sha(snapshot_path),source_sha256=sha(__file__),
        source_files=list(FILES),position_contract='exact serialized float64 XYZ multiset; no track IDs or simulator actor columns',
        horizontal_motion_bound_m_s=3.0,sensor_protocol=SENSOR,anchor_policy='all unchanged prepared anchors; only arrived same-branch history through anchor')
    identity_path=DEST/'preparation_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=IDENTITY:raise ValueError('positive history preparation identity changed')
    dump(identity_path,IDENTITY);reports=[];counts=collections.Counter();start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for value in pool.map(chunk,[(i,min(i+200,len(JOBS))) for i in range(0,len(JOBS),200)]):
            reports.append(value);counts.update(value['counts']);dump(DEST/'progress.json',dict(counts=counts,total_roots=len(JOBS),seconds=time.perf_counter()-start))
            print(json.dumps(dict(roots=counts['roots'],total_roots=len(JOBS),anchors=counts['anchors'],seconds=time.perf_counter()-start)),flush=True)
    if counts['roots']!=prepared['roots'] or counts['anchors']!=prepared['counts']['anchors']:raise ValueError('positive history complete accounting mismatch')
    dump(DEST/'manifest.json',dict(schema=SCHEMA,split=prepared['split'],roots=counts['roots'],scenes=prepared['scenes'],counts=dict(counts),
        identity=IDENTITY,prepared_manifest=str(prepared_path),shards=[{key:value[key] for key in ('path','sha256','roots','counts')} for value in reports],
        failures=[],seconds=time.perf_counter()-start))
    for path in DEST.iterdir():
        if path.is_file():os.chmod(path,0o444)
    os.chmod(DEST,0o555)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',default='train_campaign_20261023');parser.add_argument('--workers',type=int,default=2);parser.add_argument('--name');args=parser.parse_args()
    run(args.cache,args.workers,args.name)
