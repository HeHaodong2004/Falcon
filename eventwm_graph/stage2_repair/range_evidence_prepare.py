"""Prepare bounded training range evidence with original inclusion weights."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,HF,PREP,read,save,sha,source_rows,member
from .range_history import ancestor_frames
from .range_gpu import GpuObservedRangeMesh
from .range_evidence import features
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment


def prepare(roots=64,device='cuda:0',shard=0,shards=1):
    part=shard
    torch.set_num_threads(1);source,manifest_path=source_rows('train');eligible={r['root_id']:r for r in read(HF/'packed/pilot_fit/manifest.json')['rows']}
    selected=[r for r in source if r['root_id']in eligible][:roots]
    missing=[r['root_id']for r in selected if not(OUT/'train_range_branches'/(r['root_id']+'.json')).exists()]
    if missing:raise ValueError('fixed training sensor subset incomplete: %d Roots'%len(missing))
    selected=selected[part::shards]
    wanted={r['root_id']:r for r in selected};samples=read(PREP/'pilot_fit_mixed_ray/manifest.json');dest=OUT/'range_evidence_train';dest.mkdir(exist_ok=True);records=[];tick=time.perf_counter()
    for shard in samples['shards']:
        groups=defaultdict(list)
        for e in shard['segments']:
            if e['root_id']in wanted:groups[e['root_id']].append(e)
        if not groups:continue
        with np.load(shard['path'],allow_pickle=False)as z:xx=z['x'][:,16:].astype(np.float32);yy=z['y'];ww=z['weight']
        for rid,entries in groups.items():
            path=dest/(rid+'.npz');receipt=path.with_suffix('.json')
            identity=dict(schema='legal_range_evidence16_1',sample_shard_sha256=shard['sha256'],packed_sha256=eligible[rid]['sha256'],sensor_sha256=sha(OUT/'train_range_branches'/(rid+'.npz')),sources={n:sha(Path(__file__).with_name(n))for n in ['range_evidence.py','range_gpu.py','observed_range_mesh.py','range_history.py','geometry_cuda.py','legal_camera_features.py']})
            if receipt.exists():
                record=read(receipt)
                if record['identity']!=identity or record['sha256']!=sha(path):raise ValueError('range features invalidated')
                records.append(record);continue
            a=load_arrays(eligible[rid]);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(eligible[rid]['segments'])};legal=member(wanted[rid],'legal_root.json.gz');nodes=member(wanted[rid],'future_observations.json.gz');all_tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
            with np.load(OUT/'train_range_branches'/(rid+'.npz'))as z:sensors={k:z[k]for k in z.files}
            meshes={};arrays=defaultdict(list);count=0
            for e in entries:
                v=get_segment(a,lookup[e['start_node'],e['end_node']]);path_=v[3];sl=slice(e['start'],e['end']);x=xx[sl];resolution=float(v[-1][0]);keys=np.floor((x[:,20:22].astype(np.float64)*10+path_[0,[1,3]])/resolution).astype(np.int32)
                target={tuple(k):int(y)for k,y in zip(v[4],v[5])}
                np.testing.assert_array_equal(yy[sl],[target.get(tuple(k),0)for k in keys])
                # Verify recovered coordinates, not just their labels.
                error=np.max(abs(x[:,20:22].astype(np.float64)*10+path_[0,[1,3]]-(keys+.5)*resolution))
                if error>resolution*.1:raise ValueError('sample cell identity quantization')
                sid=e['start_node']
                if sid not in meshes:
                    take=ancestor_frames(sensors['parents'],sensors['history_count'],sid)
                    meshes[sid]=GpuObservedRangeMesh(sensors['depth'][take],sensors['origins'][take],sensors['rotations'][take],sensors['timestamps'][take],path_[0,0],resolution,device,tracks=[all_tracks[j]for j in take]);meshes[sid].latest_sensor_time=float(sensors['timestamps'][take[-1]])
                current=legal['history'][-1]if sid is None else nodes[sid]['observation']
                extra=features(meshes[sid],keys,x,path_,v[0],current['camera_origin'],current['camera_rotation'])
                arrays['x'].append(x);arrays['extra'].append(extra);arrays['y'].append(yy[sl]);arrays['weight'].append(ww[sl]);count+=1
            temp=path.with_suffix('.tmp.npz');np.savez(temp,**{k:np.concatenate(v)for k,v in arrays.items()});temp.replace(path)
            record=dict(root_id=rid,scene_id=wanted[rid]['scene_id'],identity=identity,path=str(path),sha256=sha(path),rows=sum(len(v)for v in arrays['y']),windows=count);save(receipt,record);records.append(record);print('ROOT',len(records),rid,'windows',count,'seconds',time.perf_counter()-tick,flush=True)
    assert len(records)==len(selected)
    save(dest/('manifest.json'if shards==1 else'manifest_part%d.json'%part),dict(roots=len(records),scenes=len({r['scene_id']for r in records}),rows=sum(r['rows']for r in records),windows=sum(r['windows']for r in records),source_manifest_sha256=sha(manifest_path),preparation_source_sha256=sha(__file__),records=records,seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=int,default=64);p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);prepare(**vars(p.parse_args()))
