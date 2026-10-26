"""TRAINING TARGET producer: later measured static surfaces, loss-only.

This does not change the original FREE/BLOCKED certificate labels. Depth from
later recorded observations is projected to a three-height completion target;
the encoder continues receiving only its original legal ancestor/current map.
No simulator, hidden scene, navmesh or recorded future camera is consulted.
"""
import argparse
import ctypes as ct
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save,sha,source_rows,member
from .range_bev import RangeCanvas,SIZE
from .range_history import ancestor_frames
from .legal_camera import proxy_cameras
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment


def project_targets(builder,ranges,origins,rotations,tracks,low,resolution,floor):
    """Loss target only; no history-clock interpretation of descendant unions."""
    positions=[];offsets=[0]
    for rows in tracks:
        for t in rows:
            if not t.get('valid',True):continue
            p=np.asarray(t['position'],np.float64)
            if t.get('velocity_valid')and t.get('estimated_velocity')is not None:p=p+np.asarray(t['estimated_velocity'])*min(float(t.get('observation_age',0)),.5)
            positions.append(p)
        offsets.append(len(positions))
    n,h,w=ranges.shape;out=torch.zeros((6,SIZE,SIZE),device=builder.device)
    if n:
        builder.module.launch('measured_context',SIZE*SIZE,[builder.tensor(ranges,np.float32),builder.tensor(origins,np.float64),builder.tensor(rotations,np.float64),ct.c_int(n),ct.c_int(h),ct.c_int(w),builder.tensor(np.array(positions).reshape(-1,3),np.float64),builder.tensor(offsets,np.int32),ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(SIZE),ct.c_double(resolution),ct.c_double(float(floor)),out])
    a=out.cpu().numpy();return np.concatenate([a[3:],np.maximum(a[:3],a[3:])]).astype(np.uint8)


def run(manifest=None,device='cuda:1'):
    torch.set_num_threads(1);manifest=Path(manifest)if manifest else OUT/'range_dense_bev_train/manifest_roots256.json';m=read(manifest);source,_=source_rows('train');source={r['root_id']:r for r in source};builder=RangeCanvas(device);dest=OUT/'surface_teacher';dest.mkdir(exist_ok=True);records=[];tick=time.perf_counter()
    for row in m['records']:
        rid=row['root_id'];sensor_path=OUT/'train_range_branches'/(rid+'.npz');packed_path=HF/'packed/pilot_fit'/(rid+'.json');path=dest/(rid+'.npz');receipt=path.with_suffix('.json')
        identity=dict(producer_sha256=sha(__file__),projection_sha256=sha(Path(__file__).with_name('range_bev.py')),sensor_sha256=sha(sensor_path),packed_sha256=sha(packed_path),source_map_sha256=row['map_sha256'])
        if receipt.exists():
            record=read(receipt)
            if record['identity']!=identity or record['sha256']!=sha(path):raise ValueError('stale surface targets')
        else:
            packed=read(packed_path);a=load_arrays(packed);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed['segments'])}
            legal=member(source[rid],'legal_root.json.gz');nodes=member(source[rid],'future_observations.json.gz');tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
            with np.load(sensor_path)as z:s={k:z[k]for k in z.files}
            with np.load(row['map_path'])as z:low=z['lo']
            targets=np.zeros((len(low),6,SIZE,SIZE),np.uint8)
            for group in row['groups']:
                sid=group['start_node'];known=set(ancestor_frames(s['parents'],s['history_count'],sid));current=max(known);byframe={};floor=None
                for e in group['segments']:
                    p=get_segment(a,lookup[sid,e['end_node']])[3];floor=p[0,2]
                    frames=[j for j in ancestor_frames(s['parents'],s['history_count'],e['end_node'])if j not in known]
                    if len(frames)!=len(p)-1:raise ValueError('future target trajectory/frame mismatch')
                    origins,rotations=proxy_cameras(p,s['origins'][current],s['rotations'][current])
                    for i,j in enumerate(frames,1):
                        if abs(p[i,0]-s['timestamps'][j])>1.e-7:raise ValueError('future target timestamp mismatch')
                        byframe[j]=(origins[i],rotations[i])
                frames=sorted(byframe);origins=np.array([byframe[j][0]for j in frames]).reshape(-1,3);rotations=np.array([byframe[j][1]for j in frames]).reshape(-1,3,3)
                targets[group['map_index']]=project_targets(builder,s['depth'][frames],origins,rotations,[tracks[j]for j in frames],low[group['map_index']],a['geometry'][0],floor)
            np.savez_compressed(path,surface_targets=targets)
            record=dict(root_id=rid,scene_id=row['scene_id'],path=str(path),sha256=sha(path),identity=identity,source_map_sha256=row['map_sha256'],maps=len(low),target_only=True)
            save(receipt,record)
        records.append(record);print('ROOT',len(records),'maps',record['maps'],'seconds',time.perf_counter()-tick,flush=True)
    save(dest/('manifest_roots%d.json'%len(records)),dict(schema='training_only_later_measured_surface_1',source_manifest_sha256=sha(manifest),records=records,semantics='Three binary near-surface flags followed by three observed(clear OR hit) masks. Future depths are TARGETS only. Current legal extrinsic + supplied recorded base path derives projection cameras. Human hits masked; original certificate labels unchanged.',seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--device',default='cuda:1');run(**vars(p.parse_args()))
