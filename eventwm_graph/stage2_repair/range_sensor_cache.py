"""New sensor input requires BOTH new viewing features and new map canvases.

Capture from the legal public API, verify original keys/labels, reuse exact
unchanged88features, attach future supervision only after legal extraction.
The existing128sensor caches are never overwritten.
"""
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,read,save,sha,source_rows,member
from .range_dev_v2 import Capture
from .range_history import ancestor_frames
from .range_bev import CHANNELS,SIZE
from .full_evaluate import labels
from eventwm_graph.stage2_segment.data import replay


def run(device='cuda:0',name='sensor256',sensor_directory='range_branches_256'):
    if any(Path(v).name!=v for v in [name,sensor_directory]):raise ValueError('simple artifact names required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;tick=time.perf_counter()
    points=read(OUT/'range_evidence_dev_v2/manifest.json');maps={r['root_id']:r for r in read(OUT/'range_bev_dev/manifest.json')['records']};raw,_=source_rows('dev');raw={r['root_id']:r for r in raw}
    pointdest=OUT/('sensor_cache_dev_'+name+'_point');mapdest=OUT/('sensor_cache_dev_'+name+'_maps');pointdest.mkdir(exist_ok=True);mapdest.mkdir(exist_ok=True)
    model=Capture(OUT/'training/range_online16384_streamed/selected_epoch01_argmax.pt',device=device);model.config['observed_obstacle_schema']='capped_2m_complete';pointrecords=[];maprecords=[]
    sources={n:sha(Path(__file__).with_name(n))for n in ['range_sensor_cache.py','range_dev_v2.py','range_bev.py','range_fast.py','range_planar.py','range_api.py','global_features.py','features.py']}
    for row in points['records']:
        rid=row['root_id'];oldmap=maps[rid];sp=OUT/sensor_directory/(rid+'.npz');sr=read(sp.with_suffix('.json'))
        if sr['sha256']!=sha(sp)or sr['identity']['files']!=raw[rid]['files']:raise ValueError('sensor source changed')
        pointpath=pointdest/(rid+'.npz');mappath=mapdest/(rid+'.npz');identity=dict(schema='legal_sensor_features88_16_maps14_1',sensor_sha256=sr['sha256'],original_point_sha256=row['sha256'],original_map_sha256=oldmap['sha256'],sources=sources)
        if pointpath.with_suffix('.json').exists()and mappath.with_suffix('.json').exists():
            pr=read(pointpath.with_suffix('.json'));mr=read(mappath.with_suffix('.json'))
            if pr['identity']!=identity or mr['identity']!=identity or pr['sha256']!=sha(pointpath)or mr['sha256']!=sha(mappath):raise ValueError('new sensor cache identity changed')
            pointrecords.append(pr);maprecords.append(mr);continue
        with np.load(row['path'])as z:
            y=z['y'];oldkeys=z['keys'];x=z['x']if'x'in z.files else None
        featurepath=row['path']if x is not None else row['original_feature_path']
        if x is None:
            with np.load(featurepath)as z:x=z['x']
        extra=np.empty((len(y),16),np.float32);canvases=np.empty((len(oldmap['groups']),CHANNELS,SIZE,SIZE),np.float16);lows=np.empty((len(oldmap['groups']),2),np.int64);mapids={g['start_node']:g['map_index']for g in oldmap['groups']};segments={(e['start_node'],e['end_node']):e for e in row['segments']}
        legal=member(raw[rid],'legal_root.json.gz');nodes=member(raw[rid],'future_observations.json.gz');tracks=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
        with np.load(sp)as z:sensors={k:z[k]for k in z.files}
        if sensors['depth'].shape[1:]!=(256,256):raise ValueError('this bounded sensor cache requires256pixels')
        groups=defaultdict(list)
        for e in replay(raw[rid],'mixed'):
            if'audit'not in e:groups[e['start_node']].append(e)
        for sid,examples in groups.items():
            take=ancestor_frames(sensors['parents'],sensors['history_count'],sid);packet=dict(ranges=sensors['depth'][take],origins=sensors['origins'][take],rotations=sensors['rotations'][take],timestamps=sensors['timestamps'][take],sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1');tr=[tracks[j]for j in take]
            outputs=model.predict_segments(partial_graph=examples[0]['graph'],trajectories=[e['trajectory']for e in examples],range_observations=packet,track_history=tr,materialize=False)
            g=model.active_graph;ego=examples[0]['trajectory'][0,1:4];canvas,low=model.canvas.build(g.keys,g.values,g.heights,ego,g.resolution,packet['ranges'],packet['origins'],packet['rotations'],packet['timestamps'],tr);canvases[mapids[sid]]=canvas.astype(np.float16);lows[mapids[sid]]=low
            # Supervision attaches AFTER public prediction/canvas extraction.
            for ex,delta,(xx,ee)in zip(examples,outputs,model.features):
                e=segments[sid,ex['end_node']];sl=slice(e['start'],e['end']);np.testing.assert_array_equal(delta['candidate_cells'],oldkeys[sl]);np.testing.assert_array_equal(xx,x[sl]);actual,_,_=labels(oldkeys[sl],ex['target']);np.testing.assert_array_equal(actual,y[sl]);extra[sl]=ee
        with np.load(oldmap['path'])as z:
            np.testing.assert_array_equal(lows,z['lo']);aux=z['aux'];mask=z['mask']
        np.savez(pointpath,extra=extra,y=y,keys=oldkeys);np.savez_compressed(mappath,maps=canvases,lo=lows,aux=aux,mask=mask)
        pr=dict(root_id=rid,scene_id=row['scene_id'],path=str(pointpath),sha256=sha(pointpath),identity=identity,original_feature_path=featurepath,segments=row['segments'],rows=len(y),reuses_original_x88=True)
        mr=dict(root_id=rid,scene_id=row['scene_id'],path=str(mappath),sha256=sha(mappath),identity=identity,point_path=str(pointpath),original_feature_path=featurepath,rows=len(y),groups=oldmap['groups'])
        save(pointpath.with_suffix('.json'),pr);save(mappath.with_suffix('.json'),mr);pointrecords.append(pr);maprecords.append(mr);model.clear_cache();print('ROOT',len(pointrecords),'rows',len(y),'seconds',time.perf_counter()-tick,flush=True)
    assert len(pointrecords)==31 and sum(len(r['segments'])for r in pointrecords)==666
    common=dict(sensor_size=256,sensor_directory=sensor_directory,role='New legal sensor inputs on unchanged original666pilot windows. All exact cell keys/88features/target labels verified. Future targets attach only after input extraction.',roots=31,windows=666,seconds=time.perf_counter()-tick)
    save(pointdest/'manifest.json',dict(common,schema='legal_public_features88_evidence16_sensor256_1',records=pointrecords));save(mapdest/'manifest.json',dict(common,schema='legal_range_bev14_sensor256_1',scenes=len({r['scene_id']for r in maprecords}),maps=sum(len(r['groups'])for r in maprecords),records=maprecords))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--name',default='sensor256');p.add_argument('--sensor-directory',default='range_branches_256');run(**vars(p.parse_args()))
