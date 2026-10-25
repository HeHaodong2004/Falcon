"""Matched complete K1/4/6 calls, full hypothetical materialization, normal GC."""
import argparse
from collections import defaultdict
import gc
import hashlib
import os
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import torch
from .diagnose import OUT,HF,CKPT,read,save,sha
from .predictor import make_predictor
from eventwm_graph.stage2_hf.baseline import ThresholdBaseline
from eventwm_graph.stage2_segment.benchmark import inputs


def summary(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['model'],r['cohort'],r['cache'],r['k'])].append(r)
    return [dict(model=k[0],cohort=k[1],cache=k[2],k=k[3],calls=len(v),
        p50_ms=float(np.quantile([r['ms']for r in v],.5)),p95_ms=float(np.quantile([r['ms']for r in v],.95)),
        p99_ms=float(np.quantile([r['ms']for r in v],.99)),max_ms=max(r['ms']for r in v),
        candidate_cells_min=min(sum(r['candidates'])for r in v),candidate_cells_max=max(sum(r['candidates'])for r in v))for k,v in sorted(groups.items())]


def run(checkpoint,device='cuda:1',repetitions=2,name='pilot_cost',increment_format='list',global_features=False):
    torch.set_num_threads(1);gc.enable();tick=time.perf_counter();m=make_predictor(checkpoint,device=device,increment_format=increment_format,global_features=global_features);candidate_loaded=time.perf_counter()-tick
    tick=time.perf_counter();baseline=ThresholdBaseline(CKPT,threshold=.2,execution='optimized',future_format='native',device=device,feature_threads=1);baseline_loaded=time.perf_counter()-tick
    models=dict(baseline=baseline,candidate=m);work=inputs('multi');rows=[];startup=[]
    # Explicit first-call/startup separate from initialized distributions.
    for label,model in models.items():
        ex=work[0];p=[np.array(v['trajectory'])for v in ex['paths'][:6]];before=time.perf_counter();result=model.predict_segments(partial_graph=ex['graph'],trajectories=p);torch.cuda.synchronize(device)
        startup.append(dict(model=label,first_K6_ms=(time.perf_counter()-before)*1000,model_initialization_seconds=candidate_loaded if label=='candidate'else baseline_loaded))
    order=sorted(range(len(work)),key=lambda i:len(work[i]['graph']['cells']));dense=set(order[len(order)//2:]);start=time.perf_counter()
    load=subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,pmem,args','--sort=-pcpu'],text=True).splitlines()[:25]
    gc_events=[];gc_start={}
    def gc_witness(phase,info):
        gen=info['generation']
        if phase=='start':gc_start[gen]=time.perf_counter()
        else:gc_events.append(dict(generation=gen,start=gc_start.pop(gen,time.perf_counter()),end=time.perf_counter(),collected=info['collected']))
    gc.callbacks.append(gc_witness)
    for rep in range(repetitions):
        for i in order[::1 if rep%2 else -1]:
            ex=work[i];g=ex['graph'];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
            assert len(paths)==6 and len({hashlib.sha256(p.tobytes()).hexdigest()for p in paths})==6
            for k in [1,4,6]:
                for label in (['baseline','candidate']if rep%2 else ['candidate','baseline']):
                    model=models[label]
                    for cache in ['cold','warm']:
                        if cache=='cold':model.clear_cache()
                        else:
                            # Establish the same reusable-map state for each call.
                            model.predict_segments(partial_graph=g,trajectories=paths[:k])
                        torch.cuda.synchronize(device);before_gc=len(gc_events);t=time.perf_counter();result=model.predict_segments(partial_graph=g,trajectories=paths[:k])
                        if increment_format=='columns'and label=='baseline':
                            from .increment import SparseIncrement
                            for d in result:
                                ix=np.flatnonzero(d['states']);d['static_updates']=SparseIncrement(d['candidate_cells'][ix],d['states'][ix],d['heights'][ix],d['timestamp'])
                        torch.cuda.synchronize(device);ms=(time.perf_counter()-t)*1000
                        assert len(result)==k and all('future_partial_graph'in d for d in result)
                        rows.append(dict(model=label,cohort='dense'if i in dense else'representative',cache=cache,k=k,ms=ms,group=i,repetition=rep,
                            candidates=[len(d['candidate_cells'])for d in result],updates=[len(d['static_updates'])for d in result],profile=dict(model.last_batch_profile),
                            gc_ms=sum((e['end']-max(e['start'],t))*1000 for e in gc_events[before_gc:]),gc_generations=[e['generation']for e in gc_events[before_gc:]]))
        save(OUT/('benchmark_'+name+'.json'),dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),device=device,hardware=torch.cuda.get_device_name(torch.device(device)),precision='float32 TF32off',
            scope='Complete current graph -> candidates/features/transfers/network/probabilities/hard decode/sparse increment -> normally materialized native hypothetical graphs; sparse serialization EXCLUDED. cold=map cache empty, warm=map retained. Normal GC. Both models on same GPU and exact original workloads.',
            cohort_definition='Original32 multi-map workloads, upper half by starting known cells is dense; six distinct saved real paths per map.',
            increment_format=increment_format,global_features=global_features,baseline_column_adapter_timed=increment_format=='columns',
            concurrent_load=load,loadavg=os.getloadavg(),startup=startup,summary=summary(rows),calls=rows,seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        print(name,'rep',rep+1,'summary',summary(rows),flush=True)
    gc.callbacks.remove(gc_witness)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--repetitions',type=int,default=2);p.add_argument('--name',default='pilot_cost');p.add_argument('--increment-format',choices=['list','columns'],default='list');p.add_argument('--global-features',action='store_true');run(**vars(p.parse_args()))
