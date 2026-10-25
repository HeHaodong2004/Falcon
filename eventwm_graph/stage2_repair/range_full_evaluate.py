"""Stream the ORIGINAL full development universe through the legal public API.

The environment's sensor producer is a separate process. Only ancestor sensor
arrays and already observed tracks enter the predictor. Future observations
are read solely for the unchanged target and matching domain/count checks.
Run from a frozen source worktree while continuing implementation elsewhere.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .diagnose import OUT,read,save,sha,source_rows,member
from .range_radix_api import RadixExactSpatialPredictor
from .range_history import ancestor_frames
from .full_evaluate import BASELINE,codes,labels,summarize
from eventwm_graph.stage2_segment.data import replay
from eventwm_graph.stage2_segment.evaluate import groups
from eventwm_graph.stage2_segment.prepared import content_key


def gzread(path):return json.loads(gzip.decompress(path.read_bytes()))


def evaluate_root(model,row,dest,checkpoint_sha,source_sha):
    rid=row['root_id'];path=dest/(rid+'.json.gz')
    sensor_path=OUT/'full_range_branches'/(rid+'.npz')
    sensor_receipt=read(sensor_path.with_suffix('.json'))
    if sha(sensor_path)!=sensor_receipt['sha256']:raise ValueError('sensor arrays changed')
    if sensor_receipt['identity']['files']!=row['files']:raise ValueError('sensor source Root changed')
    if path.exists():
        prior=gzread(path)
        if prior['checkpoint_sha256']!=checkpoint_sha or prior['sensor_sha256']!=sensor_receipt['sha256'] or prior['evaluator_sha256']!=source_sha:raise ValueError('evaluation receipt identity changed')
        return prior
    legal=member(row,'legal_root.json.gz');nodes=member(row,'future_observations.json.gz')
    track_rows=[o['tracks']for o in legal['history']]+[n['observation']['tracks']for n in nodes]
    with np.load(sensor_path)as z:sensors={k:z[k]for k in z.files}
    frozen=gzread(BASELINE/(rid+'.json.gz'))
    old={(e['start_node'],e['end_node']):e for e in frozen['records']}
    with np.load(BASELINE/(rid+'.probabilities.npz'))as z:old_y=z['y'];offsets=z['offsets']
    slices={(e['start_node'],e['end_node']):slice(int(offsets[i]),int(offsets[i+1]))for i,e in enumerate(frozen['records'])}
    previous=gzread(OUT/'full_evaluation/selected_global_point'/(rid+'.json.gz'))
    windows=defaultdict(list);audit=None;tick=time.perf_counter()
    for ex in replay(row,'mixed'):
        if 'audit'in ex:audit=ex['audit'];continue
        windows[ex['start_node']].append(ex)
    records=[]
    for sid,examples in windows.items():
        graph=examples[0]['graph'];before=content_key(graph)
        take=ancestor_frames(sensors['parents'],sensors['history_count'],sid)
        packet=dict(ranges=sensors['depth'][take],origins=sensors['origins'][take],rotations=sensors['rotations'][take],timestamps=sensors['timestamps'][take],sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1')
        results=model.predict_segments(partial_graph=graph,trajectories=[e['trajectory']for e in examples],range_observations=packet,track_history=[track_rows[j]for j in take])
        for ex,delta in zip(examples,results):
            keys=delta['candidate_cells'];hard=delta['states'];y,th,missing=labels(keys,ex['target'])
            prev=old[ex['start_node'],ex['end_node']]
            assert hashlib.sha256(keys.astype('<i4').tobytes()).hexdigest()==prev['domain_sha256']
            assert len(keys)==prev['candidates']and np.count_nonzero(y)+len(missing)==prev['true_counts'][0]
            np.testing.assert_array_equal(y,old_y[slices[ex['start_node'],ex['end_node']]])
            cm=np.bincount(3*y.astype(np.int64)+hard,minlength=9).reshape(3,3)
            for state,height in missing:cm[state,0]+=1
            future=delta['future_partial_graph'];columns=future.cell_columns
            prepared=model.prepare_graph(graph);base=prepared.future_base.table
            fc=codes(columns['cell']);ki=np.searchsorted(fc,codes(base.keys))
            np.testing.assert_array_equal(columns['cell'][ki],base.keys)
            for name,value in [('state',base.states),('height',base.heights),('first_seen',base.first),('last_seen',base.last)]:np.testing.assert_array_equal(columns[name][ki],value)
            chosen=np.flatnonzero(hard);ni=np.searchsorted(fc,codes(keys[chosen]))
            np.testing.assert_array_equal(columns['cell'][ni],keys[chosen]);np.testing.assert_array_equal(columns['state'][ni],hard[chosen]);np.testing.assert_array_equal(columns['height'][ni],delta['heights'][chosen])
            assert len(columns['cell'])==len(base.keys)+len(chosen)and future['hypothetical']is True
            correct=(y>0)&(hard==y);height=[float(abs(delta['heights'][correct]-th[correct]).sum()),int(correct.sum())]
            records.append(dict(start_node=sid,end_node=ex['end_node'],cm=cm.tolist(),baseline_cm=prev['confusion'][6],groups=groups(ex),new_cells=len(ex['target']),poses=len(ex['trajectory']),empty_prediction=not bool(len(chosen)),candidates=len(keys),out_of_domain=len(missing),known_terminal_differences=ex['known_terminal_differences'],height_error=height,future_verified=True))
        assert content_key(graph)==before
    assert audit is not None and len(records)==len(old)
    value=dict(root_id=rid,scene_id=row['scene_id'],human_count=previous['human_count'],checkpoint_sha256=checkpoint_sha,sensor_sha256=sensor_receipt['sha256'],evaluator_sha256=source_sha,records=records,audit=audit,seconds=time.perf_counter()-tick,privileged_runtime_inputs=False)
    temp=path.with_suffix('.tmp');temp.write_bytes(gzip.compress(json.dumps(value,separators=(',',':')).encode()));temp.replace(path)
    model.clear_cache();return value


def run(checkpoint,name,device='cuda:1',shard=0,shards=1,wait=False,limit=0):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if not 0<=shard<shards:raise ValueError('invalid evaluation shard')
    rows,manifest=source_rows('dev');all_rows=rows
    if limit:rows=rows[:limit]
    selected=rows[shard::shards];dest=OUT/'full_evaluation'/name;dest.mkdir(parents=True,exist_ok=True)
    checkpoint_sha=sha(checkpoint);source_sha=sha(__file__)
    model=RadixExactSpatialPredictor(checkpoint,device=device)
    identity=dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,source_manifest_sha256=sha(manifest),evaluator_sha256=source_sha,model_schema=model.predictor_schema,rule=model.decision_rule,threshold=model.threshold,role='LEGAL SENSOR EXTENSION; ORIGINAL full development. No confirmation data. Complete future graph checked at every window.',shards=shards,smoke_limit=limit)
    ip=dest/'identity.json'
    if ip.exists()and read(ip)!=identity:raise ValueError('use a new evaluation destination')
    save(ip,identity);remaining=list(selected);done=[];tick=time.perf_counter();last=0.
    while remaining:
        ready=[r for r in remaining if (OUT/'full_range_branches'/(r['root_id']+'.json')).exists()]
        if not ready:
            if not wait:break
            if time.perf_counter()-last>30:
                print('WAITING',shard,'completed',len(done),'remaining',len(remaining),flush=True);last=time.perf_counter()
            time.sleep(5);continue
        for row in ready:
            done.append(evaluate_root(model,row,dest,checkpoint_sha,source_sha));remaining.remove(row)
            if len(done)%8==0 or not remaining:
                progress=dict(shard=shard,completed=len(done),total=len(selected),pending=len(remaining),seconds=time.perf_counter()-tick)
                save(dest/('progress_shard%d.json'%shard),progress);print(progress,flush=True)
    save(dest/('shard%d.json'%shard),dict(completed=len(done),requested=len(selected),pending=len(remaining)))
    available=[dest/(r['root_id']+'.json.gz')for r in rows]
    if all(p.exists()for p in available):
        values=[gzread(p)for p in available];result=summarize(values)
        assert result['roots']==len(rows)
        if not limit:assert result['roots']==2110 and result['windows']==47128
        result.update(identity=identity,seconds=time.perf_counter()-tick,acceptance_universe_complete=not bool(limit))
        save(dest/'summary.json',result);print('SUMMARY',result['hierarchical'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--wait',action='store_true');p.add_argument('--limit',type=int,default=0);run(**vars(p.parse_args()))
