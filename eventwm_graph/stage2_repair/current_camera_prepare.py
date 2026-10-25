"""Append legal current-calibration view features to valid sampled rows.

Reads legal_root history only for calibration; never future_observations,
privileged initialization, scene assets, hidden maps or future camera poses.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import time
import numpy as np
from .diagnose import OUT,HF,PREP,read,save,sha,source_rows,member
from .legal_camera import current_extrinsic
from .legal_camera_features import parameters,cell_features
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment

DEST=None;SPLIT=None;ROWS=None


def annotate(item):
    si,shard=item;path=DEST/('%06d.npz'%si);receipt=path.with_suffix('.json')
    if receipt.exists():
        r=read(receipt)
        if r['source_shard_sha256']!=shard['sha256']or sha(path)!=r['sha256']:raise ValueError('stale current-camera shard')
        return r
    with np.load(shard['path'],allow_pickle=False)as z:xx=z['x']
    geometry=np.empty((len(xx),8),np.float32);owner=np.empty(len(xx),np.uint16);calibration=[];groups=defaultdict(list);bindings=[];maxerr=0.
    for e in shard['segments']:groups[e['root_id']].append(e)
    if len(groups)>65535:raise ValueError('too many Roots in one current-camera shard')
    for ri,(rid,entries)in enumerate(groups.items()):
        legal=member(ROWS[rid],'legal_root.json.gz');current=legal['history'][-1];start=np.array([[current['timestamp'],*current['robot_pose']]],np.float64);translation,rotation=current_extrinsic(start,current['camera_origin'],current['camera_rotation'])
        row=read(HF/'packed'/SPLIT/(rid+'.json'));a=load_arrays(row);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(row['segments'])};signature=None
        for e in entries:
            v=get_segment(a,lookup[e['start_node'],e['end_node']]);p=v[3];assert p[0,0]>=current['timestamp']-1.e-7;sl=slice(e['start'],e['end']);x=xx[sl,16:].astype(np.float32);centers=x[:,20:22].astype(np.float64)*10+p[0,[1,3]];keys=np.floor(centers/v[-1][0]).astype(np.int32);error=float(np.max(abs(centers-(keys+.5)*v[-1][0])))if len(keys)else 0.;maxerr=max(maxerr,error)
            if error>=v[-1][0]*.1:raise ValueError('uncertified quantized key recovery')
            params,signature=parameters(p,translation,rotation,float(v[-1][0]));geometry[sl]=cell_features(keys,p,float(v[-1][0]),x,params);owner[sl]=ri
        calibration.append(signature);bindings.append(dict(root_id=rid,legal_root_sha256=ROWS[rid]['files']['legal_root.json.gz'],packed_sha256=row['sha256']))
    np.savez(path,geometry=geometry,calibration=np.asarray(calibration),owner=owner);r=dict(index=si,path=str(path),sha256=sha(path),source_shard_sha256=shard['sha256'],rows=len(xx),sources=bindings,max_recovered_center_error_m=maxerr);save(receipt,r);return r


def prepare(split='pilot_fit',workers=4):
    global DEST,SPLIT,ROWS
    SPLIT=split;rows,source=source_rows('train');ROWS={r['root_id']:r for r in rows};path=PREP/(split+'_mixed_ray/manifest.json');m=read(path);DEST=OUT/'current_camera_prepared'/split;DEST.mkdir(parents=True,exist_ok=True)
    identity=dict(schema='legal_current_camera20_1',source_manifest_sha256=sha(source),sample_manifest_sha256=sha(path),sources={n:sha(Path(__file__).with_name(n))for n in ['legal_camera.py','legal_camera_features.py','current_camera_prepare.py']},split=split,input_boundary='Calibration from LEGAL Root history last observed camera/base pose; supplied trial base paths only. No future camera pose or hidden map is read.')
    ip=DEST/'identity.json'
    if ip.exists()and read(ip)!=identity:raise ValueError('current-camera features invalidated')
    save(ip,identity);done=[];tick=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers)as pool:
        jobs=[pool.submit(annotate,item)for item in enumerate(m['shards'])]
        for f in as_completed(jobs):
            done.append(f.result())
            if len(done)%16==0 or len(done)==len(jobs):print(dict(shards=len(done),total=len(jobs),seconds=time.perf_counter()-tick),flush=True)
    save(DEST/'manifest.json',dict(identity=identity,shards=sorted(done,key=lambda r:r['index']),rows=sum(r['rows']for r in done),seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='pilot_fit');p.add_argument('--workers',type=int,default=4);prepare(**vars(p.parse_args()))
