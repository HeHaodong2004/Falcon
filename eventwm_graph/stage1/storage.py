"""Atomic structured root shards. Raw evidence and oracle futures stay separate."""
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from ..config import digest,file_digest,write_json,read_json
from . import DATASET_SCHEMA


def write_gzip(path,value):
    with Path(path).open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0,compresslevel=3) as f:
            f.write(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def read_gzip(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def scene_split(scene_id):
    # Assigned before outcomes, immutable across branch/root counts.
    value=int(digest({'split_version':'scene_disjoint_v1','scene':scene_id})[:8],16)%100
    return 'test' if value<10 else 'validation' if value<20 else 'train'


def validate_outcome(query,outcome,root_timestamp,future_times):
    rows=outcome['observations'];n=len(rows)
    if n!=outcome['executed_actions'] or n>len(query['actions']):raise ValueError('Truncated/malformed outcome')
    if not rows and query['actions']:raise ValueError('Missing along-path observation sequence')
    for i,row in enumerate(rows):
        if not row['observation_valid']:raise ValueError('Invalid observation cannot masquerade as negative evidence')
        if abs(row['timestamp']-future_times[i+1])>1e-8 or row['timestamp']<=root_timestamp:
            raise ValueError('Observation/future clock mismatch')
        if row['episode_over'] and i!=n-1:raise ValueError('Post-termination observation')
        for track in row['tracks']:
            if track['last_seen_time']>row['timestamp']+1e-9:raise ValueError('Future observation leaked into belief')
    if n<len(query['actions']) and not rows[-1]['episode_over']:raise ValueError('Unexplained sequence truncation')
    if not np.isfinite(np.asarray([r['robot_pose'] for r in rows])).all():raise ValueError('Nonfinite robot state')
    return {'frames':n,'detections':sum(t['visible_now'] for r in rows for t in r['tracks']),
        'negative_frames':sum(not any(t['visible_now'] for t in r['tracks']) for r in rows),
        'remembered_lost_track_frames':sum(any(not t['visible_now'] for t in r['tracks']) for r in rows),
        'human_collision':any(r['human_collision'] for r in rows)}


def seal_root(output,record,truth,times,queries,outcomes,dependencies=None,min_free_bytes=100*1024**3):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<min_free_bytes:raise RuntimeError('Disk reserve reached')
    identity={'schema':DATASET_SCHEMA,'scene_id':record['scene_id'],'map_id':record['map_id'],
        'seed':record['seed'],'episode':record['episode'],'sampling':record['sampling']}
    future_id=digest({'scene_id':record['scene_id'],'map_id':record['map_id'],
        'seed':record['seed'],'episode':record['episode'],'prefix_actions':record['legal_root'].get('prefix_actions',[])})
    root_id=digest(identity);destination=output/'roots'/root_id[:2]/root_id
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        existing=validate_root(destination)
        if dependencies:register_dependencies(destination,dependencies)
        return existing
    temporary=Path(tempfile.mkdtemp(prefix='.partial_',dir=destination.parent))
    began=time.perf_counter();stats=[]
    try:
        legal={'schema':DATASET_SCHEMA,'root_id':root_id,'scene_id':record['scene_id'],'map_id':record['map_id'],
            'robot_configuration_id':record['robot_configuration_id'],'legal_root':record['legal_root'],
            'sensor':record['sensor'],'split':scene_split(record['scene_id'])}
        write_gzip(temporary/'legal_root.json.gz',legal)
        write_gzip(temporary/'oracle_initialization.json.gz',{'identity':identity,'episode':record['episode'],
            'private':True,'online_input':False,'prefix_truth':record.get('prefix_truth')})
        h=len(truth[0]);t=len(truth)
        positions=np.asarray([[r['position'] for r in frame] for frame in truth],dtype=np.float32).reshape(t,h,3)
        transforms=np.asarray([[r['transform'] for r in frame] for frame in truth],dtype=np.float32).reshape(t,h,4,4)
        offsets=np.asarray([[r['offset_transform'] for r in frame] for frame in truth],dtype=np.float32).reshape(t,h,4,4)
        joints=np.asarray([[r['joints'] for r in frame] for frame in truth],dtype=np.float32)
        np.savez_compressed(temporary/'oracle_future.npz',timestamps=np.asarray(times,dtype=np.float64),
            positions=positions,transforms=transforms,offset_transforms=offsets,joints=joints,oracle_ids=np.arange(1,h+1,dtype=np.int16))
        write_gzip(temporary/'queries.json.gz',queries)
        for i,(query,outcome) in enumerate(zip(queries,outcomes)):
            stats.append(validate_outcome(query,outcome,record['legal_root']['timestamp'],times))
            # JSON+gzip is lossless, handles ragged tracks and permits future
            # estimators. Numeric shared futures dominate bytes, stored once.
            write_gzip(temporary/f'outcome_{i:03d}.json.gz',outcome)
        if len(queries)!=len(outcomes):raise ValueError('Query/outcome cardinality mismatch')
        # Offline quality metadata, never an input to candidate generation.
        # Compare only common realized timestamps; different durations alone
        # do not count as distinct counterfactual observations.
        aligned=[set() for _ in range(max(len(o['observations']) for o in outcomes))]
        for o in outcomes:
            for j,row in enumerate(o['observations']):
                aligned[j].add(tuple(sorted(t['track_id'] for t in row['tracks'] if t['visible_now'])))
        quality={'aligned_detection_difference':any(len(values)>1 for values in aligned),
            'query_types':sorted({q['kind'] for q in queries}),
            'distinct_targets':len({q['target_node'] for q in queries}),
            'maximum_planned_duration':max(q['duration'] for q in queries)}
        files={p.name:file_digest(p) for p in sorted(temporary.iterdir())}
        manifest={'schema':DATASET_SCHEMA,'root_id':root_id,'scene_id':record['scene_id'],'map_id':record['map_id'],
            'robot_configuration_id':record['robot_configuration_id'],'split':scene_split(record['scene_id']),
            'queries':len(queries),'independent_futures':1,'future_frames':len(times),'human_count':h,
            'future_realization_id':future_id,
            'sampling':record['sampling'],
            'counterfactual_quality':quality,
            'statistics':stats,'performance':record.get('performance',{}),
            'bytes':sum(p.stat().st_size for p in temporary.iterdir()),'files':files,
            'query_ids':[q['query_id'] for q in queries],'edge_dependencies':sorted({e for q in queries for e in q['edge_ids']}),
            'outcome_ids':[root_id+':'+q['query_id'] for q in queries],
            'serialization_seconds':time.perf_counter()-began,'sealed':True}
        write_json(temporary/'manifest.json',manifest)
        # Durability before publishing: no visible root has partial children.
        for p in temporary.iterdir():
            with p.open('rb') as f:os.fsync(f.fileno())
        os.replace(temporary,destination)
        fd=os.open(destination.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
        if dependencies:
            for q in queries:dependencies.register(root_id+':'+q['query_id'],record['map_id'],q['edge_ids']+["@node:"+record['legal_root']['start_node']],destination)
        return manifest
    except Exception:
        # Retain failed staging for diagnosis; it is never counted as sealed.
        write_json(temporary/'FAILED.json',{'root_id':root_id,'time':time.time()})
        raise


def validate_root(folder):
    folder=Path(folder);m=read_json(folder/'manifest.json')
    if not m['sealed'] or m['schema']!=DATASET_SCHEMA:raise ValueError('Unsealed/unknown root schema')
    for name,sha in m['files'].items():
        if file_digest(folder/name)!=sha:raise ValueError('Corrupt root member '+name)
    legal=read_gzip(folder/'legal_root.json.gz');queries=read_gzip(folder/'queries.json.gz')
    if legal['map_id']!=m['map_id'] or legal['split']!=scene_split(m['scene_id']):raise ValueError('Identity/split mismatch')
    with np.load(folder/'oracle_future.npz',allow_pickle=False) as f:
        times=f['timestamps']
        if len(times)!=m['future_frames'] or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Invalid shared future clock')
        if f['positions'].shape!=(len(times),m['human_count'],3) or not np.isfinite(f['positions']).all():raise ValueError('Invalid human supervision')
    if abs(times[0]-legal['legal_root']['timestamp'])>1e-9:raise ValueError('Future/root clock mismatch')
    if len(queries)!=m['queries'] or len({q['query_id'] for q in queries})!=len(queries):raise ValueError('Duplicated/missing queries')
    for i,q in enumerate(queries):
        if q['map_id']!=m['map_id'] or q['robot_configuration_id']!=m['robot_configuration_id']:raise ValueError('Query version mismatch')
        if digest({k:v for k,v in q.items() if k!='query_id'})!=q['query_id']:raise ValueError('Query checksum mismatch')
        statistic=validate_outcome(q,read_gzip(folder/f'outcome_{i:03d}.json.gz'),legal['legal_root']['timestamp'],times)
        if statistic!=m['statistics'][i]:raise ValueError('Manifest supervision statistics mismatch')
    return m


def online_input(folder,query_index):
    """Whitelisted loader intentionally never opens an oracle file."""
    folder=Path(folder);legal=read_gzip(folder/'legal_root.json.gz');q=read_gzip(folder/'queries.json.gz')[query_index]
    from scripts.collector.observation import legal_tracks
    root=legal['legal_root']
    history=[{'timestamp':r['timestamp'],'robot_pose':copy_pose(r['robot_pose']),
        'tracks':legal_tracks(r['tracks']),'observation_valid':r['observation_valid'],
        'camera_origin':copy_pose(r['camera_origin']),'camera_rotation':[copy_pose(v) for v in r['camera_rotation']]}
        for r in root['history']]
    return {'map_id':legal['map_id'],'robot_configuration_id':legal['robot_configuration_id'],
        'history':history,'sensor':legal['sensor'],
        'query':{k:q[k] for k in ('kind','target_node','edge_ids','actions','planned_timed_poses','target_heading','duration')}}


def copy_pose(x):return [float(v) for v in x]


def register_dependencies(folder,index):
    """Recover the small rename-to-index crash window without rewriting data."""
    folder=Path(folder);m=read_json(folder/'manifest.json')
    root=read_gzip(folder/'legal_root.json.gz')['legal_root']
    for q in read_gzip(folder/'queries.json.gz'):
        index.register(m['root_id']+':'+q['query_id'],m['map_id'],
            q['edge_ids']+['@node:'+root['start_node']],folder.resolve())
