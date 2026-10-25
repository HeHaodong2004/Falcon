"""Legal-only spatial inputs; future certificate UNION is auxiliary TARGET."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,HF,PREP,read,save,sha,source_rows,member
from .range_history import ancestor_frames
from .range_bev import RangeCanvas,SIZE
from eventwm_graph.stage2_hf.packed import load_arrays,get_segment


def prepare(split='train',device='cuda:0'):
    torch.set_num_threads(1);builder=RangeCanvas(device);source,_=source_rows(split);raw={r['root_id']:r for r in source};training=split=='train'
    packed=read(HF/'packed'/('pilot_fit'if training else'dev_pilot')/'manifest.json');packed={r['root_id']:r for r in packed['rows']}
    if training:
        selected=read(OUT/'range_evidence_train/manifest.json')['records'];samples=read(PREP/'pilot_fit_mixed_ray/manifest.json');entries=defaultdict(list)
        wanted={r['root_id']for r in selected}
        for s in samples['shards']:
            for e in s['segments']:
                if e['root_id']in wanted:entries[e['root_id']].append(e)
    else:selected=read(OUT/'range_evidence_dev/manifest.json')['records'];entries={r['root_id']:r['segments']for r in selected}
    dest=OUT/('range_bev_'+split);dest.mkdir(exist_ok=True);records=[];tick=time.perf_counter()
    for row in selected:
        rid=row['root_id'];rp=OUT/('train_range_branches'if training else'range_branches')/(rid+'.npz');path=dest/(rid+'.npz');receipt=path.with_suffix('.json');identity=dict(schema='legal_range_bev14_1',sensor_sha256=sha(rp),packed_sha256=packed[rid]['sha256'],point_evidence_sha256=row['sha256'],builder_sha256=sha(Path(__file__).with_name('range_bev.py')),preparation_sha256=sha(__file__))
        if receipt.exists():
            prior=read(receipt)
            if prior['identity']!=identity or prior['sha256']!=sha(path):raise ValueError('range spatial inputs changed')
            records.append(prior);continue
        legal=member(raw[rid],'legal_root.json.gz');nodes=member(raw[rid],'future_observations.json.gz');tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes];a=load_arrays(packed[rid]);lookup={(e['start_node'],e['end_node']):i for i,e in enumerate(packed[rid]['segments'])}
        with np.load(rp)as z:sensors={k:z[k]for k in z.files}
        bystart=defaultdict(list);cursor=0
        for e in entries[rid]:
            count=e['end']-e['start'];bystart[e['start_node']].append((e,cursor,cursor+count));cursor+=count
        maps=[];lows=[];aux=[];masks=[];groups=[]
        for sid,ee in bystart.items():
            v=get_segment(a,lookup[sid,ee[0][0]['end_node']]);p=v[3];take=ancestor_frames(sensors['parents'],sensors['history_count'],sid)
            canvas,lo=builder.build(v[0],v[1],v[2],p[0,1:4],v[-1][0],sensors['depth'][take],sensors['origins'][take],sensors['rotations'][take],sensors['timestamps'][take],[tracks[j]for j in take]);maps.append(canvas.astype(np.float16));lows.append(lo)
            # Offline auxiliary target only: published future certificate
            # classes at currently UNKNOWN cells, across the supplied training
            # windows. Main initially-UNKNOWN increment labels are unchanged.
            counts=np.zeros((2,SIZE,SIZE),np.float32);sample_ids=[];descriptors=[]
            for e,begin,end in ee:
                vv=get_segment(a,lookup[sid,e['end_node']]);ij=(vv[4]-lo)//2;ok=np.all((ij>=0)&(ij<SIZE),1);ij=ij[ok];states=vv[5][ok];np.add.at(counts,(states.astype(int)-1,ij[:,0],ij[:,1]),1);sample_ids.extend(range(begin,end));descriptors.append(dict(start_node=sid,end_node=e['end_node'],start=begin,end=end))
            total=counts.sum(0);aux.append((counts[1]/np.maximum(total,1)).astype(np.float16));masks.append(total>0);groups.append(dict(map_index=len(maps)-1,start_node=sid,point_ids=sample_ids,segments=descriptors))
        temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,maps=np.array(maps),lo=np.array(lows),aux=np.array(aux),mask=np.array(masks));temp.replace(path);record=dict(root_id=rid,scene_id=row['scene_id'],path=str(path),sha256=sha(path),identity=identity,point_path=row['path'],original_feature_path=row.get('original_feature_path'),rows=cursor,groups=groups);save(receipt,record);records.append(record);print('ROOT',len(records),rid,'maps',len(maps),'seconds',time.perf_counter()-tick,flush=True)
    save(dest/'manifest.json',dict(schema='legal_range_bev14_1',split=split,roots=len(records),scenes=len({r['scene_id']for r in records}),maps=sum(len(r['groups'])for r in records),records=records,seconds=time.perf_counter()-tick))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','dev'],default='train');p.add_argument('--device',default='cuda:0');prepare(**vars(p.parse_args()))
