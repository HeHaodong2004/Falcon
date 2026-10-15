"""Immutable root membership and compact physical trajectory extraction.

The branch with longest precontact exposure is a target-source choice, never an
input feature. One physical future per root; later windows use only observations
actually acquired before that window. No new simulator rollout is performed.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import torch
from eventbeliefwm.data import (sha256, read, compact_truth, planar_frame,
                               interpolate, associate_tracks, first_contact)
from eventbeliefwm.experiment import save_json
from eventquerywm.inputs import tensorize, collate

VERSION = 'physical-root-windows-v1'
PROJECT = Path('/home/haodong/workspace/projects/eventwm-falcon')
SNAPSHOT_HASH = '5836bd381a206ecd3200134fa7de392317884a57e40397abf37c48c417255c27'
SOURCE_HASH = sha256(__file__)


def order_key(rid):
    return hashlib.sha256(('forecast-20261014/'+rid).encode()).hexdigest()


def prepare(snapshot, output):
    output = Path(output)
    if (output/'protocol.json').exists():
        raise FileExistsError('Existing frozen membership must not change')
    digest = sha256(snapshot)
    if digest != SNAPSHOT_HASH:
        raise ValueError('This protocol requires the specified frozen snapshot')
    source = read(snapshot)
    old = read(PROJECT/'output/eventwm_350k_belief_20261014/cache_pilot128_trainready/manifest.json')
    old_ids = {r['root_id'] for r in old['roots'] if r['split']=='train'}
    pool = [b for b in source['blocks'] if b['split']=='train']
    pool.sort(key=lambda b: order_key(b['root']['root_id']))
    dev = {b['root']['root_id'] for b in pool if b['root']['root_id'] not in old_ids}
    dev = set(sorted(dev, key=order_key)[:256])
    train = [b['root']['root_id'] for b in pool if b['root']['root_id'] not in dev]
    # Explicit nested memberships persist when the production pool grows.
    subsets = {str(n):train[:n] for n in (128,1024,4096)}
    subsets['full']=train
    subsets['refit_all']=[b['root']['root_id'] for b in pool]
    output.mkdir(parents=True,exist_ok=True)
    blocks=[]
    for b in pool:
        root=b['root']; rows=b['rows']; r=rows[0]
        blocks.append(dict(root_id=root['root_id'],split='dev' if root['root_id'] in dev else 'fit',
            scene=r['scene'],category=r['category'],episode_id=r['episode_id'],
            trajectory_id=r['trajectory_id'],archive_path=b['archive_path'],
            seal_path=str(Path(r['dataset'])/root['seal_file']),seal_sha256=b['seal_sha256'],
            complete_sha256=b['complete_sha256'],evidence=b['evidence'],
            rows=[{k:r[k] for k in ('branch_id','query_id','sr_valid','sr_label',
                    'observed_duration_s','human_collision_time_valid','human_collision_time_s')} for r in rows]))
    save_json(output/'blocks.json',blocks)
    p=dict(version=VERSION,snapshot=str(Path(snapshot).resolve()),snapshot_sha256=digest,
        blocks_sha256=sha256(output/'blocks.json'),eligible_roots=len(pool),queries=sum(len(b['rows']) for b in pool),
        scenes=len({b['scene'] for b in blocks}),development=sorted(dev),subsets=subsets,
        previous_pilot_roots=sorted(old_ids),seed=20261014,
        selection='train-internal dev only; legacy106 and exploratory16 are diagnostic, not untouched',
        final_refit='all9221 only after selecting recipe and epoch budget; no selection on refit dev',
        reservations='original validation/natural/unseen memberships unchanged; no reserved raw extraction',
        windows='root origin plus actual-arrived origins approximately1s apart, at least.25s exposure',
        targets='single longest precontact canonical branch; all physical identities targets only',
        exogeneity='verified active simulator factorization and prior250-root cross-query audit; no independent query replication')
    save_json(output/'protocol.json',p)
    for name in ('blocks.json','protocol.json'):
        (output/name).chmod(0o444)
    print(json.dumps({k:v for k,v in p.items() if k not in ('subsets','development','previous_pilot_roots')}),flush=True)
    return p


def nominal_query(history):
    h=history[-1]
    return dict(nominal_timed_trajectory=[[0.,*h['robot_position'],h['robot_yaw']],
                                          [5.,*h['robot_position'],h['robot_yaw']]])


def window(history, frames, origin_time, end, horizon=5.):
    history=[h for h in history if h['timestamp'] <= origin_time+1e-8]
    origin,yaw,rotation=planar_frame(history)
    t0=float(history[-1]['timestamp'])
    truth=compact_truth(frames,origin,yaw,rotation)
    track_ids,mapping,audit=associate_tracks(history,truth,origin,rotation)
    times=np.arange(1,21)*.25
    xy,valid,_,support=interpolate(truth,t0+times)
    support &= t0+times < end-1e-8
    valid &= support[:,None]
    pindex={identity:i for i,identity in enumerate(truth['ids'])}
    positions=np.zeros((20,len(track_ids),2),np.float32)
    track_valid=np.zeros((20,len(track_ids)),bool)
    for i,identity in enumerate(track_ids):
        j=pindex.get(mapping.get(identity))
        if j is not None:
            positions[:,i]=xy[:,j];track_valid[:,i]=valid[:,j]
    base=tensorize(dict(history=history,history_images=None,query=nominal_query(history)))
    acquisition=np.zeros((len(track_ids),len(history)),np.float64)
    for j,h in enumerate(history):
        for tr in h['human_tracks']:
            acquisition[track_ids.index(tr['track_id']),j]=tr['last_seen_time']
    return dict(base=base,history_timestamps=torch.tensor([h['timestamp'] for h in history],dtype=torch.float64),
        track_acquisition_times=torch.from_numpy(acquisition),
        targets=dict(track_positions=torch.from_numpy(positions),track_valid=torch.from_numpy(track_valid),
            all_human_positions=torch.from_numpy(xy),all_human_valid=torch.from_numpy(valid),
            physical_exposure=torch.from_numpy(support)),
        geometry=torch.zeros(2,32,32),geometry_support=torch.zeros(32,32,dtype=torch.bool),
        metadata=dict(timestamp=t0,track_ids=track_ids,association=audit,
            physical_humans_target_only=len(truth['ids']),origin=origin.tolist(),yaw=yaw,
            history_span_s=t0-history[0]['timestamp'],history_steps=len(history)))


def extract(block):
    path=Path(block['archive_path'])
    if sha256(block['seal_path'])!=block['seal_sha256'] or sha256(path/'COMPLETE.json')!=block['complete_sha256']:
        raise ValueError('Root provenance changed')
    seals=read(block['seal_path'])
    names=('legal_observations/history.json','roots/prefix_truth.json.gz',
        'legal_observations/observation_config.json','labels/labels.json','branch_truth/branches.jsonl.gz')
    for name in names:
        if sha256(path/name)!=seals[name]:raise ValueError('Raw hash mismatch: '+name)
    labels={(l['branch_id'],l['query_id']):l for l in read(path/'labels/labels.json')}
    for row in block['rows']:
        label=labels[row['branch_id'],row['query_id']]
        if (label['sr_valid'],label['sr_label'])!=(row['sr_valid'],row['sr_label']):
            raise ValueError('Canonical label join mismatch')
    selected=max(block['rows'],key=lambda r:min(float(labels[r['branch_id'],r['query_id']]['observed_duration']),
                                               first_contact(labels[r['branch_id'],r['query_id']])))
    key=selected['branch_id'],selected['query_id']
    branch=None
    # Skip JSON decoding of duplicate futures. Full compressed SHA256 validation
    # above already authenticates the entire stream; decode only to our branch.
    with gzip.open(path/'branch_truth/branches.jsonl.gz','rt') as f:
        for line in f:
            ids=re.match(r'\{"branch_id":\s*"([^"]+)",\s*"query_id":\s*"([^"]+)"',line)
            if ids is None:raise ValueError('Unexpected canonical branch serialization')
            if tuple(ids.groups())==key:
                if branch is not None:raise ValueError('Duplicate canonical branch')
                branch=json.loads(line)
                break
    if branch is None or branch['query_hash_before']!=branch['query_hash_after']:
        raise ValueError('Missing or mutated source branch')
    history=read(path/'legal_observations/history.json')
    sensor=read(path/'legal_observations/observation_config.json')
    t0=history[-1]['timestamp']; label=labels[key]
    if abs(branch['truth'][0]['timestamp']-t0)>1e-7 or abs(branch['truth'][-1]['timestamp']-t0-label['observed_duration'])>1e-6:
        raise ValueError('Native physical clock/exposure mismatch')
    # Strict contact cutoff; query completion's last native frame remains usable.
    end=min(t0+label['observed_duration']+1e-7,t0+first_contact(label))
    with gzip.open(path/'roots/prefix_truth.json.gz','rt') as f:prefix=json.load(f)
    frames=[f for f in prefix+branch['truth'] if f['timestamp']<end-1e-8]
    arrived={h['timestamp']:h for h in history+branch['legal_observations'] if h['timestamp']<end-1e-8}
    history=[arrived[t] for t in sorted(arrived)]
    origins=[t0];last=t0
    for h in history:
        if h['timestamp']>=last+1.-1e-8 and h['timestamp']+.25<end:
            origins.append(h['timestamp']);last=h['timestamp']
    windows=[window(history,frames,t,end) for t in origins]
    for w in windows:
        w['sensor_config']=sensor;w['robot_geometry']={'radius':.25}
        w['metadata'].update(root_id=block['root_id'],scene=block['scene'],category=block['category'])
    return dict(version=VERSION,source_hash=SOURCE_HASH,root_id=block['root_id'],windows=windows,
        provenance=dict(source_key=key,branch_truth_sha256=seals['branch_truth/branches.jsonl.gz'],
            seal_sha256=block['seal_sha256'],complete_sha256=block['complete_sha256'],
            episode_id=block['episode_id'],trajectory_id=block['trajectory_id']),
        physical_exposure_s=end-t0,scene=block['scene'],split=block['split'])


def coverage(value):
    result=dict(windows=len(value['windows']),horizons={})
    for h in (1,3,5):
        l=4*h
        result['horizons'][str(h)]=dict(
            physical_windows=sum(bool(w['targets']['physical_exposure'][:l].all()) for w in value['windows']),
            trajectory_windows=sum(bool(w['targets']['track_valid'][:l].all(0).any()) for w in value['windows']),
            trajectories=sum(int(w['targets']['track_valid'][:l].all(0).sum()) for w in value['windows']))
    return result


def cache_one(block,output):
    output=Path(output);rid=block['root_id'];dest=output/(rid+'.pt')
    if dest.exists():v=torch.load(dest,map_location='cpu')
    else:
        v=extract(block);torch.save(v,dest.with_suffix('.tmp'));os.replace(dest.with_suffix('.tmp'),dest)
    if v['source_hash']!=SOURCE_HASH or v['root_id']!=rid:raise ValueError('Incompatible root cache')
    return dict(root_id=rid,file=dest.name,sha256=sha256(dest),scene=v['scene'],split=v['split'],**coverage(v))


def build(protocol_dir,output,limit=0,workers=1):
    from eventforecastwm.guard import Guard
    from concurrent.futures import ProcessPoolExecutor
    if workers not in (1,2):raise ValueError('Collector-concurrent extraction is capped at2 CPU workers')
    protocol_dir,output=Path(protocol_dir),Path(output)
    p=read(protocol_dir/'protocol.json');blocks=read(protocol_dir/'blocks.json')
    assert sha256(protocol_dir/'blocks.json')==p['blocks_sha256']
    byid={b['root_id']:b for b in blocks}
    # Development first permits early diagnostic curves without changing fit membership.
    ids=p['development']+p['subsets']['full']
    if limit:ids=ids[:limit]
    output.mkdir(parents=True,exist_ok=True)
    config=dict(version=VERSION,source_hash=SOURCE_HASH,protocol_sha256=sha256(protocol_dir/'protocol.json'),
        snapshot_sha256=p['snapshot_sha256'],root_ids=ids)
    if (output/'config.json').exists() and read(output/'config.json')!=config:raise ValueError('Cache resume mismatch')
    save_json(output/'config.json',config)
    guard=Guard(output);start=time.time();entries=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={i:pool.submit(cache_one,byid[rid],str(output)) for i,rid in enumerate(ids[:workers])}
        for i,rid in enumerate(ids):
            guard.check()
            if (output/'STOP_RESEARCH').exists():raise RuntimeError('Research-only stop requested')
            row=pending.pop(i).result();entries.append(row)
            if i+workers<len(ids):pending[i+workers]=pool.submit(cache_one,byid[ids[i+workers]],str(output))
            save_json(output/'progress.json',dict(completed=i+1,total=len(ids),elapsed_s=time.time()-start,root_id=rid))
            if i%16==0:print(json.dumps(dict(completed=i+1,total=len(ids),elapsed_s=time.time()-start,coverage=row)),flush=True)
            if i+1 in (256+128,256+1024,256+4096):
                stage=output/f'manifest_stage{i+1}.json'
                if not stage.exists():
                    save_json(stage,dict(**config,roots=entries,created_at=time.time(),complete=False))
                    stage.chmod(0o444)
    save_json(output/'manifest.json',dict(**config,roots=entries,created_at=time.time()))


def batch_windows(windows,device='cpu'):
    base=collate([w['base'] for w in windows]).to(device)
    b,n,t=base.humans.shape[:3]
    m=max(1,max(w['targets']['all_human_positions'].shape[1] for w in windows))
    clocks=torch.zeros(b,t,dtype=torch.float64,device=device)
    acquisition=torch.zeros(b,n,t,dtype=torch.float64,device=device)
    targets={}
    for i,w in enumerate(windows):
        nt,tt=w['track_acquisition_times'].shape
        clocks[i,:tt]=w['history_timestamps'].to(device)
        acquisition[i,:nt,:tt]=w['track_acquisition_times'].to(device)
    for name in windows[0]['targets']:
        values=[]
        for w in windows:
            x=w['targets'][name]
            if x.ndim>1:
                count=n if name.startswith('track_') else m
                value=x.new_zeros(x.shape[0],count,*x.shape[2:]);value[:,:x.shape[1]]=x
            else:value=x
            values.append(value)
        targets[name]=torch.stack(values).to(device)
    return dict(base=base,history_timestamps=clocks,track_acquisition_times=acquisition,targets=targets,
        geometry=torch.stack([w['geometry'] for w in windows]).to(device),
        geometry_support=torch.stack([w['geometry_support'] for w in windows]).to(device),
        sensor_config=[w['sensor_config'] for w in windows],robot_geometry=[w['robot_geometry'] for w in windows],
        metadata=[w['metadata'] for w in windows])


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--snapshot',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('build');a.add_argument('--protocol-dir',required=True);a.add_argument('--output',required=True);a.add_argument('--limit',type=int,default=0);a.add_argument('--workers',type=int,default=1)
    a=p.parse_args();kwargs=vars(a);command=kwargs.pop('command')
    (prepare if command=='prepare' else build)(**kwargs)


if __name__=='__main__':main()
