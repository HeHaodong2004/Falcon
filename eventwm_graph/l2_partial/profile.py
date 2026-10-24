"""Wall-clock persistent full-call profiling, never just neural forward time."""
import argparse,contextlib,cProfile,gzip,json,pathlib,time,resource
import numpy as np
from .model import GraphBeliefWorldModel

def load_json(path):
    with gzip.open(path,'rt') as f:return json.load(f)

def profile(snapshot,output,limit=30,repeats=8,checkpoint=None,population=None,event_checkpoint=None):
    snapshot=pathlib.Path(snapshot);manifest=json.loads((snapshot/'manifest.json').read_text());rows=[r for r in manifest['rows'] if r['split']=='dev'][:limit]
    model=GraphBeliefWorldModel(population_prior=json.loads(pathlib.Path(population).read_text()) if population else None)
    if checkpoint:
        from .neural import LearnedDynamics
        model.dynamics=LearnedDynamics.load(checkpoint)
    if event_checkpoint:
        from .queries import NativeEventModel
        model.event_model=NativeEventModel(json.loads(pathlib.Path(event_checkpoint).read_text())['coefficients'])
    records=[];initial=[]
    for row in rows:
        folder=snapshot/'roots'/row['root_id'];legal=load_json(folder/'legal_root.json.gz');queries=load_json(folder/'queries.json.gz')
        start=time.perf_counter();b=model.initialize(legal['history'],legal['partial_graph']);initial.append(1000*(time.perf_counter()-start))
        # Reconstruct the preceding prefix exclusively from arrived rows.
        if len(legal['history'])>1:
            from ..partial.map import PartialGraph
            g=PartialGraph(legal['partial_graph']['resolution'],legal['partial_graph']['height'])
            for obs in legal['history'][:-1]:g.update(obs)
            previous=model.initialize(legal['history'][:-1],g.snapshot());actual=legal['history'][-1]
        else:previous=b;actual=legal['history'][-1]
        workload=(queries*max(1,(24+len(queries)-1)//len(queries)))[:24]
        for rep in range(repeats+1):
            invalid=rep==0
            if invalid:model.graph_cache.clear()
            _,batch=model.online_call(previous,actual,workload,seed=27)
            records.append({'root_id':row['root_id'],'humans':max((h.n for h in b.hypotheses),default=0),'graph_cells':len(b.graph.materialize()),
                            'actual_candidates':len(queries),'bench_candidates':len(workload),'worlds':16,'refined':4,
                            'mode':'cold_support_cache' if invalid else 'warm','timings_ms':batch.timings_ms})
    out={'records':records,'initialization_ms':initial,'rss_max_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'threads':__import__('torch').get_num_threads(),'affinity':sorted(__import__('os').sched_getaffinity(0)),
         'workload':'3s/16worlds/24 candidates/4 refined with <=5 branches. Repeated candidates load-test rows are labeled.',
         'checkpoint':checkpoint,'event_checkpoint':event_checkpoint,
         'summary':{}}
    for mode in ('warm','cold_support_cache'):
        values=[r for r in records if r['mode']==mode]
        out['summary'][mode]={k:dict(zip(('p50','p95','p99'),map(float,np.percentile([r['timings_ms'][k] for r in values],[50,95,99])))) for k in values[0]['timings_ms']}
    pathlib.Path(output).write_text(json.dumps(out,indent=2));print(json.dumps(out['summary'],indent=2))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('output');p.add_argument('--limit',type=int,default=30);p.add_argument('--repeats',type=int,default=8);p.add_argument('--checkpoint');p.add_argument('--population');p.add_argument('--event-checkpoint');a=p.parse_args();profile(**vars(a))
