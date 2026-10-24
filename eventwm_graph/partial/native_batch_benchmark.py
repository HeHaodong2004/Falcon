"""Bounded actual-scene exact-ray differential and component benchmark.

Example::
    python -m eventwm_graph.partial.native_batch_benchmark ROOT ROOT --output REPORT

Read-only source roots must belong to train. Each observation is made by the
accepted scalar observer; its exact native Ray payload is recorded, then cast
again by both bindings at the same world state. No aggregate-only comparison.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path
import resource
import time

import numpy as np


def _read(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def _one(root,steps,repeats):
    import habitat_sim
    from habitat.tasks.nav.nav import NavigationEpisode,NavigationGoal
    from ..benchmark.map import load_certified_map
    from ..stage1.runtime import NativeEpisode
    from ..stage1.storage import scene_split
    from .observer import GeometryObserver
    from .native_batch import NativeRayBatch
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text())
    if manifest['split']!='train' or scene_split(manifest['scene_id'])!='train':
        raise ValueError('native_batch_benchmark_requires_train_roots')
    private=_read(root/'privileged_initialization.json.gz');ep=private['episode']
    episode=NavigationEpisode(episode_id=ep['episode_id'],scene_id=ep['scene_id'],
        start_position=ep['start_position'],start_rotation=ep['start_rotation'],
        goals=[NavigationGoal(position=g['position']) for g in ep['goals']],info=copy.deepcopy(ep['info']))
    hmap=load_certified_map(private['source_map_manifest']);rows=[]
    with NativeEpisode(hmap,episode,manifest['seed']) as rt:
        batch=NativeRayBatch(rt.sim);identity=batch.identity
        observer=GeometryObserver(rt,hmap.robot_config.collision_reference/6)
        original_cast=rt.sim.cast_ray
        for step in range(steps):
            recorded=[]
            def capture(ray,max_distance=100.):
                recorded.append((np.array(ray.origin,dtype=np.float32),
                                 np.array(ray.direction,dtype=np.float32),float(max_distance)))
                return original_cast(ray,max_distance=max_distance)
            rt.sim.cast_ray=capture
            try:
                if step==0:observer.observe()
                else:
                    rt.observation=observer.observe
                    rt.step('left' if step%2 else 'forward')
            finally:rt.sim.cast_ray=original_cast
            if not recorded:raise ValueError('no_captured_rays')
            origin=recorded[0][0]
            if not all(np.array_equal(origin,x[0]) for x in recorded):
                raise ValueError('observation_ray_origins_disagree')
            directions=np.asarray([x[1] for x in recorded],dtype=np.float32)
            distances=np.asarray([x[2] for x in recorded],dtype=np.float64)
            expected=[];expected_offsets=[0]
            for direction,distance in zip(directions,distances):
                hits=original_cast(habitat_sim.geo.Ray(origin,direction),max_distance=float(distance)).hits
                expected.extend(hits);expected_offsets.append(len(expected))
            actual=batch.cast_rays(origin,directions,distances)
            checks={'offsets':np.array_equal(actual['offsets'],expected_offsets),
                    'object_ids':np.array_equal(actual['object_ids'],[h.object_id for h in expected]),
                    'distances':np.array_equal(actual['distances'],[h.ray_distance for h in expected]),
                    'points':np.array_equal(actual['points'],np.asarray([h.point for h in expected],dtype=np.float32).reshape(-1,3)),
                    'normals':np.array_equal(actual['normals'],np.asarray([h.normal for h in expected],dtype=np.float32).reshape(-1,3))}
            ignore=observer.robot_ids|observer.inactive_ids
            def scalar_clear(ds=distances):
                return np.asarray([not any(h.object_id not in ignore and h.ray_distance<float(distance)-.015
                    for h in original_cast(habitat_sim.geo.Ray(origin,direction),max_distance=float(distance)).hits)
                    for direction,distance in zip(directions,ds)],dtype=bool)
            expected_clear=scalar_clear();actual_clear=batch.clear_rays(origin,directions,distances,ignore)
            checks['clear']=np.array_equal(expected_clear,actual_clear)
            nearest=batch.nearest_hits(origin,directions,distances,ignore)
            for i in range(len(directions)):
                hits=[h for h in expected[expected_offsets[i]:expected_offsets[i+1]] if h.object_id not in ignore]
                if bool(nearest['valid'][i])!=bool(hits):checks['nearest']=False;break
                if hits and (nearest['object_ids'][i]!=hits[0].object_id or
                             nearest['distances'][i]!=hits[0].ray_distance or
                             not np.array_equal(nearest['points'][i],np.asarray(hits[0].point))):
                    checks['nearest']=False;break
            else:checks['nearest']=True
            offsets=np.unique(np.r_[np.arange(0,len(directions),63),len(directions)]).astype(np.int64)
            groups=batch.clear_groups(origin,directions,distances,ignore,offsets)
            group_expected=np.asarray([expected_clear[a:b].all() for a,b in zip(offsets[:-1],offsets[1:])])
            checks['clear_groups']=np.array_equal(groups['clear'],group_expected)
            group_cast_count=sum(next((j+1 for j,value in enumerate(expected_clear[a:b]) if not value),b-a)
                                 for a,b in zip(offsets[:-1],offsets[1:]))
            checks['clear_group_short_circuit']=bool(groups['rays']==group_cast_count)
            # Exercise the strict endpoint comparison close to native hits.
            boundary_count=0
            for i in range(min(len(directions),512)):
                eligible=[h for h in expected[expected_offsets[i]:expected_offsets[i+1]] if h.object_id not in ignore]
                if not eligible:continue
                for delta in (-1e-7,0.,1e-7):
                    end=float(eligible[0].ray_distance+.015+delta)
                    reference=not any(h.object_id not in ignore and h.ray_distance<end-.015
                        for h in original_cast(habitat_sim.geo.Ray(origin,directions[i]),max_distance=end).hits)
                    candidate=bool(batch.clear_rays(origin,directions[i:i+1],np.array([end]),ignore)[0])
                    if reference!=candidate:checks['endpoint_boundary']=False
                    boundary_count+=1
                if boundary_count>=96:break
            checks.setdefault('endpoint_boundary',True)
            timings={'scalar_clear':[],'native_clear':[],'native_grouped':[]}
            for repeat in range(repeats):
                for name,function in (
                    ('scalar_clear',scalar_clear),
                    ('native_clear',lambda:batch.clear_rays(origin,directions,distances,ignore)),
                    ('native_grouped',lambda:batch.clear_groups(origin,directions,distances,ignore,offsets))):
                    began=time.perf_counter();function();timings[name].append(time.perf_counter()-began)
            rows.append({'step':step,'world_time':float(rt.sim.get_world_time()),'rays':len(directions),
                         'hit_count':len(expected),'boundary_cases':boundary_count,'checks':checks,
                         'timings_seconds':timings,
                         'median_seconds':{k:float(np.median(v)) for k,v in timings.items()},
                         'grouped_actual_rays':groups['rays']})
    return {'source_root':str(root),'scene_id':manifest['scene_id'],'root_id':manifest['root_id'],
            'human_count':ep['info']['human_num'],'seed':manifest['seed'],'identity':identity,
            'observations':rows,'pass':all(all(r['checks'].values()) for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots',nargs='+',type=Path);parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    began=time.perf_counter();rows=[_one(root,args.steps,args.repeats) for root in args.roots]
    report={'schema':'exact_native_batch_differential_1','pass':all(r['pass'] for r in rows),
            'roots':rows,'elapsed_seconds':time.perf_counter()-began,
            'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'measurement_scope':'ray component only; not full pipeline throughput'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'pass':report['pass'],'output':str(args.output),'elapsed_seconds':report['elapsed_seconds']}))
    raise SystemExit(0 if report['pass'] else 1)


if __name__=='__main__':main()
