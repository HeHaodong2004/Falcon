"""OFFLINE training-only surface targets for the immutable online snapshot."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,read,save,sha,member
from .range_bev import RangeCanvas,SIZE
from .range_history import ancestor_frames
from .legal_camera import proxy_cameras
from .surface_teacher import project_targets
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment


def run(snapshot=None,device='cuda:1',shard=0,shards=1):
    torch.set_num_threads(1);snapshot=Path(snapshot)if snapshot else OUT/'training/range_online4096/snapshot.json';m=read(snapshot)
    if not 0<=shard<shards:raise ValueError('invalid target shard')
    dest=OUT/'surface_online_targets';dest.mkdir(exist_ok=True);builder=RangeCanvas(device);records=[];tick=time.perf_counter();snapshot_sha=sha(snapshot)
    for record in m['records'][shard::shards]:
        row=record['source'];packed=record['packed'];rid=row['root_id'];sp=Path(record['sensor_path']);sensor_receipt=read(sp.with_suffix('.json'));path=dest/(rid+'.npz');receipt=path.with_suffix('.json')
        if sensor_receipt['identity']['files']!=row['files']or sha(sp)!=sensor_receipt['sha256']:raise ValueError('sensor source mismatch')
        identity=dict(snapshot_sha256=snapshot_sha,packed_sha256=packed['sha256'],sensor_sha256=sensor_receipt['sha256'],sources={n:sha(Path(__file__).with_name(n))for n in ['surface_online_targets.py','surface_teacher.py','range_bev.py','legal_camera.py']})
        if receipt.exists():
            result=read(receipt)
            if result['identity']!=identity or result['sha256']!=sha(path):raise ValueError('stale online surface targets')
        else:
            if sha(packed['path'])!=packed['sha256']:raise ValueError('packed source mismatch')
            a=load_arrays(packed);legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz');tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
            with np.load(sp)as z:s={k:z[k]for k in z.files}
            grouped=defaultdict(list)
            for i,e in enumerate(packed['segments']):grouped[e['start_node']].append(i)
            targets=[];lows=[];start_nodes=[]
            for sid,entries in grouped.items():
                known=set(ancestor_frames(s['parents'],s['history_count'],sid));current=max(known);byframe={};p0=get_segment(a,entries[0])[3];resolution=a['geometry'][0]
                low=(np.floor(p0[0,[1,3]]/resolution).astype(np.int64)//32)*32-SIZE
                for index in entries:
                    e=packed['segments'][index];p=get_segment(a,index)[3];frames=[j for j in ancestor_frames(s['parents'],s['history_count'],e['end_node'])if j not in known]
                    if len(frames)!=len(p)-1:raise ValueError('target frame/path alignment')
                    origins,rotations=proxy_cameras(p,s['origins'][current],s['rotations'][current])
                    for i,j in enumerate(frames,1):
                        if abs(p[i,0]-s['timestamps'][j])>1.e-7:raise ValueError('target clock mismatch')
                        byframe[j]=(origins[i],rotations[i])
                frames=sorted(byframe);origins=np.array([byframe[j][0]for j in frames]).reshape(-1,3);rotations=np.array([byframe[j][1]for j in frames]).reshape(-1,3,3)
                targets.append(project_targets(builder,s['depth'][frames],origins,rotations,[tracks[j]for j in frames],low,resolution,p0[0,2]));lows.append(low);start_nodes.append(-1 if sid is None else sid)
            temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,surface_targets=np.array(targets),lo=np.array(lows),start_nodes=np.array(start_nodes,np.int64));temp.replace(path)
            result=dict(root_id=rid,scene_id=row['scene_id'],path=str(path),sha256=sha(path),identity=identity,maps=len(targets),target_only=True);save(receipt,result)
        records.append(result)
        if len(records)%16==0:print(dict(completed=len(records),total=len(m['records'][shard::shards]),seconds=time.perf_counter()-tick),flush=True)
    save(dest/('shard%d.json'%shard),dict(snapshot_sha256=snapshot_sha,records=records,complete=True,role='Training loss targets only; later measured surfaces and legal-rig-derived supplied-path cameras; unchanged certificate supervision.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot');p.add_argument('--device',default='cuda:1');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);run(**vars(p.parse_args()))
