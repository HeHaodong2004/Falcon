"""Layer 2 component costs and recurrence; never selects robot actions."""
import argparse
import collections
import copy
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
import numpy as np
from scipy.spatial.distance import cdist

from eventwm_graph.l2_partial.data import read
from eventwm_graph.l2_partial.state import ObservedGraph
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
from eventwm_graph.l2_partial.snapshot import sha
from .data import OUT,SNAP
from .model import load_model
from .evaluate import OPTICS,restore
from .observations import predict_observations,condition_observation_event


def rss():
    return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024


def snapshot_prefix(legal):
    g=ObservedGraph('legal-prefix',0,legal['partial_graph']['resolution'],legal['partial_graph']['height'],{})
    for r in legal['history'][:-1]:g=g.update(r.get('static_updates',[]))
    return dict(timestamp=legal['history'][-2]['timestamp'],resolution=g.resolution,height=g.height,revision=g.version,
        cells=[dict(cell=list(k),state=v[0],height=v[1]) for k,v in g.materialize().items()])


def configured(optimized=True):
    wm=load_model(OUT/'training/checkpoint.pt',json.loads((OUT/'population_prior.json').read_text()),OPTICS,json.loads((OUT/'motion.json').read_text()))
    wm.optimized=optimized
    return wm


def quantiles(records):
    result={}
    for kind in sorted({(r['cache'],r['worlds'],r['workload']) for r in records}):
        part=[r for r in records if (r['cache'],r['worlds'],r['workload'])==kind and 'error' not in r]
        result[str(kind)]={key:dict(n=len(v),p50=float(np.quantile(v,.5)),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)))
            for key in ('observe','forecast','observations','successor','combined','propagate','fixed_query_readout')
            for v in [[r[key] for r in part if key in r]] if v}
    return result


def benchmark(limit=60):
    ti=time.perf_counter()
    cold_demo=subprocess.run([sys.executable,'-m','eventwm_graph.l2_120k.api_example','--trained'],capture_output=True,text=True)
    cold_process_wall_ms=(time.perf_counter()-ti)*1000
    (OUT/'api_trained.json').write_text(cold_demo.stdout);(OUT/'api_trained.stderr').write_text(cold_demo.stderr)
    if cold_demo.returncode:raise RuntimeError('trained cold-process API demonstration failed')
    start=time.perf_counter();wm=configured();startup=(time.perf_counter()-start)*1000
    rows=np.load(OUT/'dev.npy',allow_pickle=True).tolist()
    # Fixed metadata-only cohort: first60 plus12 densest and12 longest histories.
    chosen={r['root_id']:r for r in rows[:limit]}
    if limit>=60:
        for r in sorted(rows,key=lambda r:max(h.n for h in r['filtered_hypotheses']),reverse=True)[:12]+sorted(rows,key=lambda r:r['history_length'],reverse=True)[:12]:chosen[r['root_id']]=r
    calls=[];initialization=[];failures=[];recursion=[];started=time.perf_counter()
    def one(prior,row,query,cache,worlds,meta,workload):
        if cache=='dynamic_cold':
            wm.graph_cache.clear();wm.endpoint_template_cache.clear()
            for name in ('support_footprint_cache',):
                obj=getattr(wm,name,None)
                if hasattr(obj,'clear'):obj.clear()
            from eventwm_graph.l2_partial.structural_kernels import clear_kernel_caches
            clear_kernel_caches()
        tick=time.perf_counter();updated=wm.observe(prior,row);t1=time.perf_counter()
        f=wm.predict_shared(updated,3.,worlds,seed=27);t2=time.perf_counter()
        obs=predict_observations(wm,updated,query,f);t3=time.perf_counter()
        if not obs['successor_supported']:raise ValueError('benchmark route unsupported')
        child=condition_observation_event(wm,f,obs,'none');t4=time.perf_counter()
        p=wm.propagate(updated,updated.timestamp+3.);t5=time.perf_counter()
        # Secondary fixed-query heads remain marked unvalidated; their runtime
        # is separate and is not required to claim the primary combined path.
        wm.evaluate_queries(updated,[query],f,budget_ms=1e6,refined_queries=0);t6=time.perf_counter()
        return {**meta,'cache':cache,'worlds':worlds,'workload':workload,
            'observe':(t1-tick)*1000,'forecast':(t2-t1)*1000,'observations':(t3-t2)*1000,
            'successor':(t4-t3)*1000,'combined':(t4-tick)*1000,'propagate':(t5-t4)*1000,
            'fixed_query_readout':(t6-t5)*1000,'tracked':max(h.n for h in updated.hypotheses),
            'expected_hidden':sum(h.weight*h.expected_hidden for h in updated.hypotheses),
            'retained':sum(h.weight for h in updated.hypotheses),'rss_bytes':rss()}
    for i,c in enumerate(chosen.values()):
        meta=dict(root_id=c['root_id'],scene_id=c['scene_id']);folder=SNAP/'roots'/c['root_id']
        legal=read(folder/'legal_root.json.gz');qs=read(folder/'queries.json.gz')
        query=max(qs,key=lambda q:(len(q['compiled']['actions']),q['query_id']))
        try:
            ti=time.perf_counter();prior=wm.initialize(legal['history'][:-1],snapshot_prefix(legal));initialization.append((time.perf_counter()-ti)*1000)
            row=legal['history'][-1]
            one(prior,row,query,'warm',16,meta,'prime')
            for cache,repeats in [('warm',5),('dynamic_cold',2)]:
                for repeat in range(repeats):calls.append(one(prior,row,query,cache,16,meta,'real'))
            if i<12:
                for worlds in (8,32,64):
                    for cache in ('warm','dynamic_cold'):calls.append(one(prior,row,query,cache,worlds,meta,'world_sensitivity'))
            b=restore(wm,c,legal['partial_graph']);ti=time.perf_counter();direct=wm.propagate(b,b.timestamp+3.);td=(time.perf_counter()-ti)*1000
            ti=time.perf_counter();two=wm.propagate(wm.propagate(b,b.timestamp+1.5),b.timestamp+3.);tt=(time.perf_counter()-ti)*1000
            ti=time.perf_counter();three=wm.propagate(wm.propagate(wm.propagate(b,b.timestamp+1.),b.timestamp+2.),b.timestamp+3.);tr=(time.perf_counter()-ti)*1000
            for name,other in [('two',two),('three',three)]:
                discrepancy=max((float(np.max(np.abs(x.covariance-y.covariance))) if x.n else 0. for x,y in zip(direct.hypotheses,other.hypotheses)),default=0.)
                mean=max((float(np.max(np.abs(x.mean-y.mean))) if x.n else 0. for x,y in zip(direct.hypotheses,other.hypotheses)),default=0.)
                recursion.append({**meta,'segments':name,'mean_error':mean,'covariance_error':discrepancy,
                    'count_support_equal':all(np.array_equal(x.hidden_pmf,y.hidden_pmf) for x,y in zip(direct.hypotheses,other.hypotheses)),
                    'weight_error':max(abs(x.weight-y.weight) for x,y in zip(direct.hypotheses,other.hypotheses)),
                    'direct_ms':td,'segmented_ms':tt if name=='two' else tr})
            if i<12:
                fa=wm.predict_shared(b,3.,256,seed=211,times=[0.,3.])
                mid=wm.propagate(b,b.timestamp+1.5);fb=wm.predict_shared(mid,1.5,256,seed=212,times=[0.,1.5])
                def features(f):
                    x=f.positions[:,-1];valid=np.isfinite(x[...,0]);x=np.nan_to_num(x)
                    return np.stack([valid.sum(1),(np.cos(.5*x[...,0])*valid).sum(1),(np.cos(.5*x[...,1])*valid).sum(1)],axis=1)
                xa,xb=features(fa),features(fb);delta=xa.mean(0)-xb.mean(0);se=np.sqrt(xa.var(0,ddof=1)/256+xb.var(0,ddof=1)/256)
                recursion.append({**meta,'segments':'sample_distribution_direct_vs_projected_1p5',
                    'feature_mean_difference':delta.tolist(),'monte_carlo_standard_error':se.tolist(),
                    'feature_z':(delta/np.maximum(se,1e-15)).tolist(),'worlds_per_arm':256})
        except Exception as e:
            import traceback
            failures.append({**meta,'error':repr(e),'traceback':traceback.format_exc()})
        if (i+1)%10==0:print('systems',i+1,len(calls),'failures',len(failures),'seconds',round(time.perf_counter()-started,1),flush=True)
    # Existing dense fixtures retain tracked AND hidden alternatives. No current
    # campaign certificates are invented; the certificate stress is separate.
    sys.path.insert(0,str(Path('scripts').resolve()))
    from l2_partial_repair_systems_stress import fixture
    for n in (6,12):
        for certificate in (False,True):
            try:
                prior,row,qs=fixture(wm,n,certificate)
                for cache in ('warm','dynamic_cold'):
                    for repeat in range(5):calls.append(one(prior,row,qs[0],cache,16,
                        dict(root_id='synthetic_'+str(n),scene_id='synthetic'),
                        'dense_certificate' if certificate else 'dense_positive'))
            except Exception as e:failures.append(dict(workload='dense',n=n,certificate=certificate,error=repr(e)))
    sigma=wm.reset_std;p=-math.expm1(-wm.maneuver_rate*.048);grid=np.arange(0.,3.,.048)
    a=-np.expm1(-.7*(3-grid))/.7
    cumulants={}
    for name,cut in [('direct',0.),('after_1p5',math.floor(1.5/.048)*.048),('after_2',math.floor(2./.048)*.048)]:
        retained=grid>=cut-1e-10
        cumulants[name]=float(np.sum(3*p*(1-p)*sigma**4*a[retained]**4))
    report=dict(cold_process_complete_synthetic_demo_ms=cold_process_wall_ms,
        startup_model_and_JIT_ms=startup,initialization_ms=dict(p50=float(np.quantile(initialization,.5)),p95=float(np.quantile(initialization,.95)),p99=float(np.quantile(initialization,.99))),
        calls=calls,quantiles=quantiles(calls),recursion=recursion,failures=failures,
        scalar_jump_fourth_cumulant=cumulants,
        recursion_qualification='analytic moments compose; intermediate Gaussian projection changes higher moments, plus bounded joint-mode projection. Finite carrier preserves its held joint paths.',
        workload='single fixed immutable legal query; observe+forecast3s/16worlds+optical events+supported none successor; no planning packet',
        cold='initialized process/JIT; dynamic caches cleared. Full startup and prefix replay reported separately',
        propagation_component_cache='measured after same-call observe/forecast; already warmed. Independent depth timing also recorded.',
        checkpoint_sha256=sha(OUT/'training/checkpoint.pt'),seconds=time.perf_counter()-started)
    (OUT/'systems.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['quantiles'],indent=2),flush=True)


def persistence():
    sys.path.insert(0,str(Path('scripts').resolve()))
    from l2_partial_repair_systems_stress import fixture
    wm=configured();b,_,qs=fixture(wm,1,False);records=[];failures=[];rss_start=rss();forecast_calls=0
    for step in range(1,501):
        t=step*.048
        row=dict(timestamp=t,robot_pose=[0.,0.,0.,0.],static_updates=[],tracks=[])
        if step%5==0:row['tracks']=[dict(position=[3.-.2*t,0.,1.4],last_seen_time=t,visible_now=True,valid=True)]
        try:b=wm.observe(b,row)
        except Exception as e:failures.append(dict(step=step,error=repr(e)));break
        if step%10==0:
            try:
                f=wm.predict_shared(b,3.,16,seed=step);events=predict_observations(wm,b,qs[0],f)
                if events['successor_supported']:condition_observation_event(wm,f,events,'none')
                forecast_calls+=1
            except Exception as e:failures.append(dict(step=step,operation='forecast_events_successor',error=repr(e)))
        if step%25==0:
            diag=b.evidence['normalized_approximation'];records.append(dict(step=step,rss_bytes=rss(),
                hypotheses=len(b.hypotheses),tracks=max(h.n for h in b.hypotheses),retention=diag['retention_product'],
                max_omission=diag['maximum_local_omission'],weight_sum=sum(h.weight for h in b.hypotheses)))
    (OUT/'persistence.json').write_text(json.dumps(dict(records=records,failures=failures,
        rss_start_bytes=rss_start,forecast_event_successor_calls=forecast_calls,
        scope='synthetic500-arrival legal sensor persistence with forecast/event/successor/cache churn; real histories separately evaluated'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['benchmark','persistence']);p.add_argument('--limit',type=int,default=60);a=p.parse_args()
    if a.command=='benchmark':benchmark(a.limit)
    else:persistence()
