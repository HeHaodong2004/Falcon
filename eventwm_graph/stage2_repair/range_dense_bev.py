"""Bind complete candidate universes to independently prepared legal canvases."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,HF,read,save,sha,source_rows,member
from .range_history import ancestor_frames
from .range_bev import RangeCanvas,SIZE
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment


def prepare(manifest=None,device='cuda:0'):
    torch.set_num_threads(1);manifest=Path(manifest)if manifest else OUT/'range_dense_train/manifest_roots64.json';m=read(manifest);old={r['root_id']:r for r in read(OUT/'range_bev_train/manifest_roots64.json')['records']};raw,_=source_rows('train');raw={r['root_id']:r for r in raw};dest=OUT/'range_dense_bev_train';dest.mkdir(exist_ok=True);builder=RangeCanvas(device);records=[];tick=time.perf_counter()
    for row in m['records']:
        rid=row['root_id'];packed=read(HF/'packed/pilot_fit'/(rid+'.json'));a=load_arrays(packed);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed['segments'])};by=defaultdict(list)
        for e in row['segments']:by[e['start_node']].append(e)
        if rid in old:
            context=old[rid]
            assert context['identity']['sensor_sha256']==row['identity']['sensor_sha256']and context['identity']['packed_sha256']==row['identity']['packed_sha256']
            assert context['identity']['builder_sha256']==sha(Path(__file__).with_name('range_bev.py'))
            expected={(e['start_node'],e['end_node'])for g in context['groups']for e in g['segments']};actual={(e['start_node'],e['end_node'])for e in row['segments']};assert expected==actual
            indices={g['start_node']:g['map_index']for g in context['groups']};map_path=context['path'];map_sha=context['sha256']
        else:
            legal=member(raw[rid],'legal_root.json.gz');nodes=member(raw[rid],'future_observations.json.gz');tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
            with np.load(OUT/'train_range_branches'/(rid+'.npz'))as z:sensors={k:z[k]for k in z.files}
            maps=[];low=[];aux=[];masks=[];indices={}
            for sid,entries in by.items():
                v=get_segment(a,lookup[sid,entries[0]['end_node']]);p=v[3];take=ancestor_frames(sensors['parents'],sensors['history_count'],sid);assert abs(sensors['timestamps'][take[-1]]-p[0,0])<1.e-7
                canvas,lo=builder.build(v[0],v[1],v[2],p[0,1:4],v[-1][0],sensors['depth'][take],sensors['origins'][take],sensors['rotations'][take],sensors['timestamps'][take],[tracks[j]for j in take]);indices[sid]=len(maps);maps.append(canvas.astype(np.float16));low.append(lo);counts=np.zeros((2,SIZE,SIZE),np.float32)
                for e in entries:
                    vv=get_segment(a,lookup[sid,e['end_node']]);ij=(vv[4]-lo)//2;ok=np.all((ij>=0)&(ij<SIZE),1);ij=ij[ok];np.add.at(counts,(vv[5][ok].astype(int)-1,ij[:,0],ij[:,1]),1)
                total=counts.sum(0);aux.append((counts[1]/np.maximum(total,1)).astype(np.float16));masks.append(total>0)
            map_path=str(dest/(rid+'.npz'));temp=Path(map_path).with_suffix('.tmp.npz');np.savez_compressed(temp,maps=np.array(maps),lo=np.array(low),aux=np.array(aux),mask=np.array(masks));temp.replace(map_path);map_sha=sha(map_path)
        record=dict(root_id=rid,scene_id=row['scene_id'],point_path=row['path'],point_sha256=row['sha256'],map_path=map_path,map_sha256=map_sha,rows=row['rows'],groups=[dict(start_node=sid,map_index=indices[sid],segments=entries)for sid,entries in by.items()]);save(dest/(rid+'.json'),record);records.append(record);print('ROOT',len(records),rid,'maps',len(by),'seconds',time.perf_counter()-tick,flush=True)
    value=dict(schema='legal_range_dense_spatial_1',roots=len(records),scenes=len({r['scene_id']for r in records}),maps=sum(len(r['groups'])for r in records),source_manifest_sha256=sha(manifest),records=records,seconds=time.perf_counter()-tick);save(dest/('manifest_roots%d.json'%len(records)),value);save(dest/'manifest.json',value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--device',default='cuda:0');prepare(**vars(p.parse_args()))
