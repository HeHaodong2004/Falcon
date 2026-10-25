"""Matched exact geometry-cache pilot on recorded consecutive legal maps."""
import argparse
import gc
import gzip
import json
import time
import numpy as np
from eventwm_graph.stage2.data import source_rows
from eventwm_graph.stage2_quality.update import FastPartialGraph
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.temporal_cache_experiment import TemporalReusePredictor
from eventwm_graph.stage2_segment.data import load
from eventwm_graph.stage2_segment.common import OUT, dump, sha, snapshot
from eventwm_graph.stage2.benchmark import quantiles


def build(dest):
    protocol=json.loads((OUT/'protocol.json').read_text())
    wanted=set(protocol['pilot_fit'][:64]);rows,manifest=source_rows('train')
    selected=[r for r in rows if r['root_id'] in wanted];assert len(selected)==64
    groups=[];tick=time.perf_counter()
    for row in selected:
        legal,queries,outcomes,nodes,_=load(row)
        outcome=max(outcomes,key=lambda o:len(o['node_ids']))
        ids=outcome['node_ids'];graph=FastPartialGraph.restore(legal['partial_graph'])
        maps=[graph.snapshot()];poses=[[graph.timestamp,*legal['history'][-1]['robot_pose']]]
        for identity in ids:
            node=nodes[identity];ob=node['observation']
            assert graph.update(ob)==node['graph_delta']
            maps.append(graph.snapshot());poses.append([ob['timestamp'],*ob['robot_pose']])
        cases=[];times=np.asarray(poses)[:,0]
        for start in range(len(ids)):
            end=start+1+int(np.argmin(abs(times[start+1:]-times[start]-.5)))
            cases.append(dict(graph=maps[start],trajectory=poses[start:end+1],start_step=start,end_step=end))
        groups.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],query_id=outcome['query_id'],cases=cases))
    path=dest/'workload.json.gz'
    with gzip.open(path,'wt') as f:json.dump(groups,f)
    dump(dest/'workload_manifest.json',dict(path=str(path),sha256=sha(path),source_manifest_sha256=sha(manifest),
        roots=[r['root_id'] for r in selected],cases=sum(len(g['cases']) for g in groups),extraction_seconds=time.perf_counter()-tick,
        rule='First64 frozen fitting pilotRoots, longest saved Query (first tie), each saved start to nearest future saved0.5s boundary; no interpolation'))
    return groups


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='temporal_cache_pilot_v1');parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();dest=OUT/args.name;dest.mkdir(exist_ok=False)
    checkpoint=OUT/'training/final/last.pt'
    dump(dest/'protocol.json',dict(source=snapshot(),checkpoint_sha256=sha(checkpoint),gc_enabled=gc.isenabled(),repeats=args.repeats,
        accuracy='Exact candidates/probabilities/states/heights/unknowncells and full future snapshots; raw model unchanged',
        timing='Full per-call current graph read through complete future graph and prepared output; sequential saved intermediate maps; both CUDA1 one CPU thread'))
    groups=build(dest)
    kwargs=dict(execution='optimized',future_format='native',device='cuda:1',feature_threads=1)
    models=dict(production=SegmentPredictor(checkpoint,**kwargs),temporal=TemporalReusePredictor(checkpoint,**kwargs))
    first=groups[0]['cases'][0]
    for model in models.values():model.predict(partial_graph=first['graph'],trajectory=first['trajectory']);model.clear_cache()
    calls=[];proofs=[];holders={}
    for repeat in range(args.repeats):
        for gindex,group in enumerate(groups):
            for index,case in enumerate(group['cases']):
                outputs={}
                for name in (('production','temporal') if (index+repeat)%2==0 else ('temporal','production')):
                    model=models[name];cpu=time.process_time();tick=time.perf_counter()
                    holders[name]=model.predict(partial_graph=case['graph'],trajectory=case['trajectory'])
                    seconds=time.perf_counter()-tick;outputs[name]=holders[name]
                    calls.append(dict(model=name,repeat=repeat,root_id=group['root_id'],start_step=case['start_step'],end_step=case['end_step'],
                        total_seconds=seconds,cpu_seconds=time.process_time()-cpu,profile=dict(model.last_profile)))
                left,right=outputs.values()
                for field in ('candidate_cells','probabilities','states','heights','unknown_cells'):
                    if not np.array_equal(left[field],right[field]):raise AssertionError((group['root_id'],index,field))
                if left['static_updates']!=right['static_updates'] or left['future_partial_graph'].to_snapshot()!=right['future_partial_graph'].to_snapshot():
                    raise AssertionError('complete graph mismatch')
            proofs.append(dict(repeat=repeat,root_id=group['root_id'],cases=len(group['cases']),exact=True))
            if gindex%8==0:print(json.dumps(dict(repeat=repeat,roots=gindex+1,calls=len(calls),geometry_hits=models['temporal'].geometry_hits)),flush=True)
    dump(dest/'calls.json',calls);dump(dest/'equivalence.json',proofs)
    dump(dest/'metrics.json',dict(models={name:dict(**quantiles([r['total_seconds'] for r in calls if r['model']==name]),
        calls=sum(r['model']==name for r in calls),mean_prepare_ms=1000*np.mean([r['profile']['prepare_seconds'] for r in calls if r['model']==name])) for name in models},
        geometry_hits=models['temporal'].geometry_hits,geometry_misses=models['temporal'].geometry_misses,
        gc_enabled=gc.isenabled(),all_scientific_outputs_exact=True,scope='Fitting-only bounded engineering experiment, not a replacement acceptance benchmark'))


if __name__=='__main__':main()
