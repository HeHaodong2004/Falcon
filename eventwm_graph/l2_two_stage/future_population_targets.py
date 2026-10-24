"""OFFLINE ONLY: complete native population tapes and aligned future targets.

This namespace is supervision, never a predictor input or positive-history
witness. Actor columns remain coherent through time inside each Root tape.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import time
import numpy as np
from .common import OUT,OLD,dump,sha

SCHEMA='offline_native_population_future_targets_1'
JOBS=None
DEST=None
IDENTITY=None


def native_index(times,value):
    at=int(np.searchsorted(times,value));valid=[i for i in (at-1,at) if 0<=i<len(times) and abs(float(times[i])-value)<1e-7]
    if len(valid)!=1:raise ValueError('population target timestamp is not uniquely native')
    return valid[0]


def current_measurement_overlay(native_positions,actual_positive_positions,tolerance=2e-4):
    """OFFLINE numerical alignment for the current conditional set target.

    Native current coordinates are retained separately as supervision. Exact
    legal current measurements are authoritative Dirac atoms. Injective
    position matching removes their native counterparts without dropping any
    co-located duplicates; the remaining native coordinates stay float64. No
    actor identity enters this operation. Future targets are never overlaid.

    Returns (full_current_set_with_measurements, unobserved_native_set).
    The full set is for current-set feasibility/diagnostics, not an additional
    privileged predictor input. ``attach_targets`` intentionally exposes the
    unmodified native current target; callers must explicitly request this
    overlay before strict dt=0 checks against the measurement-conditioned law.
    """
    from scipy.optimize import linear_sum_assignment
    native=np.asarray(native_positions,np.float64)
    observed=np.asarray(actual_positive_positions,np.float64)
    if native.ndim!=2 or observed.ndim!=2 or native.shape[-1]!=2 or observed.shape[-1]!=2 or not np.isfinite(native).all() or not np.isfinite(observed).all():
        raise ValueError('current overlay requires finite float64 XZ pointsets')
    if len(observed)>len(native):raise ValueError('more actual positives than native current population')
    overlay=native.copy();remaining=np.ones(len(native),bool)
    if len(observed):
        distances=np.linalg.norm(observed[:,None]-native[None],axis=-1)
        rows,columns=linear_sum_assignment(distances)
        if len(rows)!=len(observed) or np.any(distances[rows,columns]>=tolerance):
            raise ValueError('current native/measurement multiset alignment unsupported')
        overlay[columns]=observed[rows];remaining[columns]=False
    return overlay,native[remaining].copy()


def chunk(bounds):
    begin,end=bounds;tag='%06d_%06d'%(begin,end);report=DEST/(tag+'.json')
    if report.exists():
        value=json.loads(report.read_text())
        if value['identity']!=IDENTITY or sha(OUT/value['path'])!=value['sha256']:raise ValueError('offline target resume identity mismatch')
        return value
    arrays=collections.defaultdict(list);frame_offsets=[0];point_offsets=[0];roots=[];counts=collections.Counter()
    for local,(row,anchors) in enumerate(JOBS[begin:end]):
        data=(Path(row['folder'])/'privileged_future.npz').read_bytes();digest=hashlib.sha256(data).hexdigest()
        if digest!=row['files']['privileged_future.npz']:raise ValueError('privileged target tape hash mismatch')
        with np.load(io.BytesIO(data),allow_pickle=False) as value:times=value['timestamps'].copy();positions=value['positions'].copy()
        if times.ndim!=1 or positions.ndim!=3 or positions.shape[0]!=len(times) or positions.shape[-1]!=3 or not np.isfinite(times).all() or not np.isfinite(positions).all() or np.any(np.diff(times)<=0):raise ValueError('invalid native target tape')
        people=positions.shape[1];arrays['root_ids'].append(row['root_id']);arrays['scene_ids'].append(row['scene_id']);arrays['source_future_sha256'].append(digest)
        arrays['timestamps'].append(times);arrays['positions_xyz'].append(positions.reshape(-1,3));arrays['people_per_root'].append(people)
        frame_offsets.append(frame_offsets[-1]+len(times));point_offsets.append(point_offsets[-1]+len(times)*people)
        for anchor in sorted(anchors,key=lambda a:a['anchor_index']):
            target_indices=np.full(8,-1,np.int64);offsets=np.asarray(anchor['times'],np.float64)
            if offsets.shape!=(8,) or np.any(offsets<0):raise ValueError('invalid unchanged future horizon vector')
            current=native_index(times,float(anchor['timestamp']))
            for h in np.flatnonzero(offsets>0):target_indices[h]=native_index(times,float(anchor['timestamp'])+float(offsets[h]))
            arrays['anchor_root_indices'].append(local);arrays['anchor_indices'].append(int(anchor['anchor_index']));arrays['anchor_timestamps'].append(float(anchor['timestamp']));arrays['anchor_current_native_indices'].append(current)
            arrays['anchor_future_native_indices'].append(target_indices);arrays['anchor_query_ids'].append(anchor['query_id'])
            counts['supported_anchor_horizons']+=int(np.count_nonzero(target_indices>=0));counts['censored_anchor_horizons']+=int(np.count_nonzero(target_indices<0))
        roots.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],anchors=len(anchors),people=people,frames=len(times),future_sha256=digest))
        counts.update(roots=1,anchors=len(anchors),native_frames=len(times),native_person_frames=len(times)*people)
    payload={key:np.asarray(values) for key,values in arrays.items() if key not in ('timestamps','positions_xyz')}
    payload.update(timestamps=np.concatenate(arrays['timestamps']),positions_xyz=np.concatenate(arrays['positions_xyz'],axis=0),root_frame_offsets=np.asarray(frame_offsets,np.int64),root_point_offsets=np.asarray(point_offsets,np.int64))
    path=DEST/(tag+'.npz');temp=DEST/(tag+'.tmp.npz');np.savez_compressed(temp,**payload);temp.replace(path)
    value=dict(identity=IDENTITY,path=str(path.relative_to(OUT)),sha256=sha(path),roots=roots,counts=dict(counts));dump(report,value);return value


def run(cache='train_campaign_20261023',workers=2):
    global JOBS,DEST,IDENTITY
    if Path(cache).name!=cache or not 1<=workers<=4:raise ValueError('invalid offline target arguments')
    prepared_path=OUT/'prepared'/cache/'manifest.json';prepared=json.loads(prepared_path.read_text());anchors=collections.defaultdict(list)
    for shard in prepared['shards']:
        data=(OUT/shard['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=shard['sha256']:raise ValueError('prepared target-anchor shard hash mismatch')
        for row in np.load(io.BytesIO(data),allow_pickle=True):anchors[row['root_id']].append({key:row[key] for key in ('anchor_index','timestamp','times','query_id')})
    if prepared['split']=='train':
        snapshot_path=Path(prepared['snapshot']);source_rows=json.loads(snapshot_path.read_text())['rows']
        if sha(snapshot_path)!=prepared['identity']['snapshot_sha256'] or any(row['split']!='train' or row['source_split']!='train' for row in source_rows):raise ValueError('nontraining or altered fitting target snapshot')
    elif prepared['split']=='dev':
        snapshot_path=OLD/'snapshot/manifest.json';source_rows=[dict(row,folder=str(OLD/'snapshot/roots'/row['root_id'])) for row in json.loads(snapshot_path.read_text())['rows'] if row['split']=='dev']
    else:raise ValueError('protected offline targets cannot be prepared')
    if len({row['root_id'] for row in source_rows})!=len(source_rows) or set(anchors)!={row['root_id'] for row in source_rows}:raise ValueError('target Root set mismatch')
    JOBS=[(row,anchors[row['root_id']]) for row in source_rows];DEST=OUT/'population_future_targets'/cache;DEST.mkdir(parents=True,exist_ok=True)
    IDENTITY=dict(schema=SCHEMA,prepared_manifest_sha256=sha(prepared_path),snapshot_manifest_sha256=sha(snapshot_path),source_sha256=sha(__file__),
        information_boundary='OFFLINE privileged supervision only; never positive-history inputs',target_clock='exact unchanged anchor actual8 native times with explicit -1 censor indices',joint_truth='same Root actor columns at every supported native horizon')
    identity_path=DEST/'preparation_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=IDENTITY:raise ValueError('offline target preparation identity mismatch')
    dump(identity_path,IDENTITY);reports=[];counts=collections.Counter();start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for report in pool.map(chunk,[(i,min(i+200,len(JOBS))) for i in range(0,len(JOBS),200)]):
            reports.append(report);counts.update(report['counts']);dump(DEST/'progress.json',dict(counts=counts,total_roots=len(JOBS),seconds=time.perf_counter()-start))
            print(json.dumps(dict(roots=counts['roots'],total_roots=len(JOBS),anchors=counts['anchors'],seconds=time.perf_counter()-start)),flush=True)
    if counts['roots']!=prepared['roots'] or counts['anchors']!=prepared['counts']['anchors']:raise ValueError('offline target full accounting mismatch')
    dump(DEST/'manifest.json',dict(schema=SCHEMA,split=prepared['split'],roots=counts['roots'],scenes=prepared['scenes'],counts=dict(counts),identity=IDENTITY,prepared_manifest=str(prepared_path),
        shards=[{key:r[key] for key in ('path','sha256','roots','counts')} for r in reports],failures=[],seconds=time.perf_counter()-start))
    for path in DEST.iterdir():
        if path.is_file():os.chmod(path,0o444)
    os.chmod(DEST,0o555)


def attach_targets(records,cache='train_campaign_20261023'):
    """Attach explicitly named OFFLINE targets, never legal input columns."""
    path=OUT/'population_future_targets'/cache/'manifest.json';manifest=json.loads(path.read_text())
    if manifest['schema']!=SCHEMA or sha(manifest['prepared_manifest'])!=manifest['identity']['prepared_manifest_sha256']:raise ValueError('offline population target identity mismatch')
    needed=collections.defaultdict(dict)
    for record in records:
        key=int(record['anchor_index'])
        if key in needed[record['root_id']]:raise ValueError('duplicate requested target anchor')
        needed[record['root_id']][key]=record
    attached=0
    for shard in manifest['shards']:
        if not any(root['root_id'] in needed for root in shard['roots']):continue
        data=(OUT/shard['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=shard['sha256']:raise ValueError('offline target shard hash mismatch')
        with np.load(io.BytesIO(data),allow_pickle=False) as archive:
            z={key:archive[key] for key in archive.files}
            for a,local in enumerate(z['anchor_root_indices']):
                rid=str(z['root_ids'][local]);idx=int(z['anchor_indices'][a])
                if rid not in needed or idx not in needed[rid]:continue
                record=needed[rid][idx];begin,end=z['root_frame_offsets'][local:local+2];pb,pe=z['root_point_offsets'][local:local+2];people=int(z['people_per_root'][local])
                tape=z['positions_xyz'][pb:pe].reshape(int(end-begin),people,3);indices=z['anchor_future_native_indices'][a];valid=indices>=0
                if record['timestamp']!=z['anchor_timestamps'][a] or record['query_id']!=str(z['anchor_query_ids'][a]) or record['scene_id']!=str(z['scene_ids'][local]):raise ValueError('offline target anchor differs')
                target=np.zeros((8,people,2),np.float64);target[valid]=tape[indices[valid]][...,[0,2]]
                times=np.zeros(8,np.float64);times[valid]=z['timestamps'][begin+indices[valid]]-float(record['timestamp'])
                if not np.allclose(times,record['times'],rtol=0,atol=1e-7):raise ValueError('offline target native horizons differ')
                current=int(z['anchor_current_native_indices'][a])
                record['population_current_positions']=tape[current][...,[0,2]].copy()
                record['population_future_positions']=target;record['population_future_valid']=valid.copy();record['population_future_times']=times
                attached+=1
    if attached!=len(records):raise ValueError('offline targets omitted requested anchors')
    return dict(schema=SCHEMA,anchors=attached,roots=len(needed),manifest_sha256=sha(path),role='PRIVILEGED_OFFLINE_SUPERVISION_ONLY')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default='train_campaign_20261023');p.add_argument('--workers',type=int,default=2);a=p.parse_args();run(a.cache,a.workers)
