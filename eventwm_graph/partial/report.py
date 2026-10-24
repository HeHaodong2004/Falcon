"""Offline distribution/storage/compiler profiling; no rendering in collection."""
from collections import Counter
from pathlib import Path
import time
import numpy as np
from ..config import read_json,write_json
from ..stage1.storage import read_gzip
from .map import PartialGraph
from .query import compile_query


def category(provenance):return provenance['source']+':'+provenance.get('intended_band','')


def distribution(folder):
    folder=Path(folder);manifest=read_json(folder/'manifest.json');diag=manifest['sampling']['query_diagnostics']
    proposals=Counter(category(c['provenance']) for c in diag['spatial_candidates'])
    for r in diag['rejected']:
        if 'final_yaw' not in r:proposals[category(r['provenance'])]+=1
    queries=read_gzip(folder/'queries.json.gz');outcomes=read_gzip(folder/'outcomes.json.gz')
    selected=Counter(category(q['provenance'][0]) for q in queries);actual=Counter();cross={}
    for q,o in zip(queries,outcomes):
        s=o['summary'];key=category(q['provenance'][0]);row=cross.setdefault(key,Counter())
        for counter in (actual,row):
            counter['executed_queries']+=1
            counter['human_collision']+=int(s['human_collision'])
            counter['static_collision']+=int(s['static_obstacle_collision'])
            counter['psc_violation']+=int(s['any_psc_violation'])
            counter['intrusive_noncollision']+=int(s['any_psc_violation'] and not s['human_collision'])
            counter['query_complete']+=int(s['query_complete'])
            counter['timeout']+=int(s['termination']=='query_timeout')
    return {'spatial_proposal_attempts':dict(proposals),'geometrically_accepted_orientations':diag['accepted_by_category'],
            'selected_and_executed_queries':dict(selected),'actual_outcomes':dict(actual),
            'actual_outcomes_by_proposal_intention':{k:dict(v) for k,v in cross.items()},
            'raw_proposal_scope':'Explicit spatial-anchor proposal calls; frontier components with no feasible viewpoint are not positions'}


def profile_compiler(folder):
    folder=Path(folder);legal=read_gzip(folder/'legal_root.json.gz');queries=read_gzip(folder/'queries.json.gz')
    graph=PartialGraph.restore(legal['partial_graph'])
    graph_started=time.perf_counter();graph.rebuild_global(legal['history'][-1]['robot_pose'][:3])
    global_graph_seconds=time.perf_counter()-graph_started
    route=graph.route;astar=0.;calls=0
    def timed_route(*args,**kwargs):
        nonlocal astar,calls
        started=time.perf_counter()
        try:return route(*args,**kwargs)
        finally:astar+=time.perf_counter()-started;calls+=1
    graph.route=timed_route;started=time.perf_counter()
    compiled=[compile_query(graph,q,legal['history'][-1]) for q in queries]
    return {'queries':len(queries),'compile_seconds':time.perf_counter()-started,'Astar_route_seconds':astar,
            'global_graph_rebuild_seconds':global_graph_seconds,
            'Astar_calls':calls,'all_schedules_identical':all(a==q['compiled'] for a,q in zip(compiled,queries)),
            'scope':'Offline cold graph, shared exact geometry cache within the query batch; no simulator'}


def summarize(datasets,output):
    roots=[];totals={k:Counter() for k in ('spatial_proposal_attempts','geometrically_accepted_orientations',
                                        'selected_and_executed_queries','actual_outcomes')}
    timings={};outcomes=0;actual_bytes=0
    for dataset in datasets:
        for path in sorted(Path(dataset).glob('roots/*/*/manifest.json')):
            if path.parent.name.startswith('.'):continue
            m=read_json(path)
            if m['split']!='train':continue
            d=distribution(path.parent)
            for key,counter in totals.items():counter.update(d[key])
            size=sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file());actual_bytes+=size;outcomes+=m['queries']
            for key,value in m['performance'].items():
                if key.endswith('_seconds') and isinstance(value,(int,float)):timings.setdefault(key,[]).append(value)
            for key in ('serialization_seconds','validation_seconds'):timings.setdefault(key,[]).append(m.get(key,0.))
            roots.append({'root_id':m['root_id'],'scene':m['scene_id'],'outcomes':m['queries'],'actual_bytes':size,
                          'distribution':d,'compiler':profile_compiler(path.parent),'performance':m['performance']})
    report={'schema':'partial_pipeline_performance_distribution_1','split_scope':'train only','roots':roots,
            'totals':{k:dict(v) for k,v in totals.items()},'outcomes':outcomes,'actual_bytes':actual_bytes,
            'projected_one_million_outcomes_GiB':actual_bytes/max(1,outcomes)*1e6/1024**3,
            'component_timings':{k:{'sum_s':sum(v),'median_s':float(np.median(v)),'max_s':max(v)} for k,v in timings.items()},
            'throughput_note':'Component sums and isolated pilot times are not steady-state aggregate throughput; use campaign status/journal wall time.'}
    write_json(output,report);return report
