"""Source-bound equivalence checks against the frozen pre-speed runtime."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import numpy as np
from .common import BASE,OUT,OLD,SNAP,read,canonical,seed_for,sha,dump
from .model import IntermittentWorldModel
from .neural import LearnedIntermittent
from .numerics import compare
from .observations import predict_observations,condition_observation_event
from eventwm_graph.l2_120k.evaluate import restore
from eventwm_graph.l2_120k.model import load_model


def reference():
    dyn_name='eventwm_graph.l2_intermittent._speed_dynamics_reference'
    dyn=types.ModuleType(dyn_name);dyn.__file__=str(OUT/'speed_reference_34425709/dynamics.py')
    sys.modules[dyn_name]=dyn;exec(compile(Path(dyn.__file__).read_bytes(),dyn.__file__,'exec'),dyn.__dict__)
    name='eventwm_graph.l2_intermittent._speed_model_reference'
    module=types.ModuleType(name);module.__file__=str(OUT/'speed_reference_34425709/model.py')
    sys.modules[name]=module
    source=OUT/'speed_reference_34425709/model.py'
    text=source.read_text().replace('from .dynamics import','from ._speed_dynamics_reference import')
    exec(compile(text,module.__file__,'exec'),module.__dict__)
    return module.IntermittentWorldModel


def compare_future(left,right):
    """Couple full paths, event clocks, hidden states and the continued RNG."""
    error=0.
    for name in ('times','positions','velocities','hidden_positions','hidden_velocities','weights'):
        a,b=getattr(left,name),getattr(right,name)
        if a is None or b is None:
            if a is not b:raise AssertionError('different optional future '+name)
            continue
        if a.shape!=b.shape or not np.array_equal(np.isnan(a),np.isnan(b)):
            raise AssertionError('different support '+name)
        valid=np.isfinite(a)
        e=float(np.max(abs(a[valid]-b[valid]))) if valid.any() else 0.
        error=max(error,e)
    if error>1e-10:raise AssertionError('coupled trajectory changed: '+str(error))
    for name in ('stopped',):
        if not np.array_equal(getattr(left,name),getattr(right,name)):
            raise AssertionError('latent state changed')
    if left.rng_state!=right.rng_state:raise AssertionError('continued RNG changed')
    if not all(np.array_equal(a,b) for a,b in zip(left.slots,right.slots)):
        raise AssertionError('world ancestry changed')
    for index in (0,len(left.times)//2,len(left.times)-1):
        for a,b in zip(left.endpoints[index],right.endpoints[index]):
            for name in ('slots','stopped','next_switch','counters','hidden_pmf'):
                if not np.array_equal(getattr(a,name),getattr(b,name)):
                    raise AssertionError('endpoint event history changed: '+name)
            err,_=compare(types.SimpleNamespace(components=(a,)),types.SimpleNamespace(components=(b,)))
            error=max(error,err)
    if error>1e-10:raise AssertionError('coupled endpoint changed: '+str(error))
    return error


def run(limit,tag,frames,full=False):
    import torch
    torch.set_num_threads(1)
    target=OUT/(tag+'.json')
    if target.exists():raise ValueError('preserve speed evidence')
    paths=[BASE/'eventwm_graph/l2_intermittent'/name for name in
        ('model.py','dynamics.py','neural.py','fast_kernels.py','sampling_kernel.py','random_kernel.py','speed_checks.py')]
    sources={str(p.relative_to(BASE)):sha(p) for p in paths}
    fast=IntermittentWorldModel.load(OUT/'training/checkpoint.pt')
    frozen=reference()(LearnedIntermittent.load(OUT/'training/checkpoint.pt'),law=dict(fast.law),
        observation_law=copy.deepcopy(fast.observation_law),population=json.loads((OLD/'population_prior.json').read_text()))
    if frozen.model_version!=fast.stochastic_version:raise AssertionError('stochastic stream identity changed')
    rows=list(np.load(OLD/'dev.npy',allow_pickle=True));indices=list(range(limit))
    indices+=sorted(range(len(rows)),key=lambda i:(-max((h.n for h in rows[i]['filtered_hypotheses']),default=0),rows[i]['root_id']))[:8]
    indices=list(dict.fromkeys(indices));records=[];errors=[];start=time.perf_counter();bootstrap=load_model()
    for index in indices:
        row=rows[index];folder=SNAP/'roots'/row['root_id'];legal=read(folder/'legal_root.json.gz');_,observations=canonical(folder)
        initial=restore(bootstrap,row,legal['partial_graph'])
        a=fast.from_legacy(initial,seed_for(row['root_id'],10041));b=frozen.from_legacy(initial,seed_for(row['root_id'],10041))
        for ob in observations[:frames]:
            try:
                av=bv=None;ae=be=None;t=time.perf_counter()
                try:av=fast.observe(a,ob)
                except Exception as exc:ae=repr(exc)
                fast_ms=(time.perf_counter()-t)*1000;t=time.perf_counter()
                try:bv=frozen.observe(b,ob)
                except Exception as exc:be=repr(exc)
                frozen_ms=(time.perf_counter()-t)*1000
                if ae is not None or be is not None:
                    if ae!=be:raise AssertionError('different exact-domain refusals: '+str((ae,be)))
                    records.append(dict(root_id=row['root_id'],timestamp=ob['timestamp'],matched_rejection=ae));break
                error,exact=compare(av,bv)
                exact=bool(exact and av.fork(model_version=frozen.model_version).fingerprint()==bv.fingerprint())
                diagnostics_equal=av.diagnostics==bv.diagnostics
                records.append(dict(root_id=row['root_id'],timestamp=ob['timestamp'],fast_ms=fast_ms,frozen_ms=frozen_ms,
                    max_error=error,bitwise=exact,diagnostics_equal=diagnostics_equal))
                if not exact or not diagnostics_equal:raise AssertionError('posterior arithmetic changed')
                a,b=av,bv
            except Exception as exc:
                errors.append(dict(root_id=row['root_id'],timestamp=ob['timestamp'],error=repr(exc)));break
        if full and a.timestamp==b.timestamp:
            try:
                seed=seed_for(row['root_id'],10711)
                af=fast.predict_shared(a,3.,16,seed);bf=frozen.predict_shared(b,3.,16,seed)
                full_error=compare_future(af,bf)
                # Retain exact worlds inside the tape, then extend its RNG beyond
                # the old horizon. Both calls intentionally use an ignored seed.
                aa=fast.successor_at(af,31);bb=frozen.successor_at(bf,31)
                at=fast.predict_shared(aa,2.4,16,42);bt=frozen.predict_shared(bb,2.4,16,42)
                full_error=max(full_error,compare_future(at,bt))
                regions=np.array([[-100.,-100.,100.,100.],[-2.,-2.,2.,2.],[0.,0.,10.,10.]])
                ap=fast.project_future_distribution(af,regions,af.times[[0,10,-1]])
                bp=frozen.project_future_distribution(bf,regions,bf.times[[0,10,-1]])
                for key in ('at_least_one_probability','expected_count','observed_track_expected_count'):
                    if not np.array_equal(ap[key],bp[key]):raise AssertionError('projection mass changed')
                query=max(read(folder/'queries.json.gz'),key=lambda q:(len(q['compiled']['actions']),q['query_id']))
                # The canonical root query starts at the root pose, while a/b
                # above have advanced. Evaluate query events from matching
                # source-root states, retaining every original input check.
                root_a=fast.from_legacy(initial,seed_for(row['root_id'],10041))
                root_b=frozen.from_legacy(initial,seed_for(row['root_id'],10041))
                aq=fast.predict_shared(root_a,3.,16,seed);bq=frozen.predict_shared(root_b,3.,16,seed)
                full_error=max(full_error,compare_future(aq,bq))
                ae=predict_observations(fast,root_a,query,aq);be=predict_observations(frozen,root_b,query,bq)
                if ae['status']!=be['status']:raise AssertionError('query support changed')
                conditioned=[]
                if ae['successor_supported']:
                    for key in ('any_probability','none_probability','first_detection_mass','endpoint_any_probability','per_world_none'):
                        v,w=np.asarray(ae[key]),np.asarray(be[key]);err=float(np.max(abs(v-w))) if v.size else 0.
                        full_error=max(full_error,err)
                        if err>1e-10:raise AssertionError('optical probability changed')
                    for event in ('none','any'):
                        ac=bc=None;ar=br=None
                        try:ac=condition_observation_event(fast,aq,ae,event)
                        except Exception as exc:ar=repr(exc)
                        try:bc=condition_observation_event(frozen,bq,be,event)
                        except Exception as exc:br=repr(exc)
                        if ar!=br:raise AssertionError('conditional support changed')
                        if ac is not None:
                            err,_=compare(ac,bc);full_error=max(full_error,err)
                            if err>1e-10:raise AssertionError('conditional law changed')
                            conditioned.append(event)
                records.append(dict(root_id=row['root_id'],full_world_check=True,max_error=full_error,
                    query_status=ae['status'],conditioned_events=conditioned,rng_continuation_equal=True,
                    temporal_states_clocks_ancestry_equal=True,projection_mass_equal=True))
            except Exception as exc:
                errors.append(dict(root_id=row['root_id'],full_world_check=True,error=repr(exc)))
    result=dict(records=records,errors=errors,passed=not errors,seconds=time.perf_counter()-start,
        roots=len(indices),source_stable=all(sha(BASE/name)==value for name,value in sources.items()),sources=sources,
        reference_sha256=sha(OUT/'speed_reference_34425709/model.py'),
        semantics='Production backward-compatible stochastic identity versus actual archived source/version; implementation ownership stays distinct. Full actual components and diagnostics must be bitwise equal. Optional full worlds, hidden motion, event conditioning, projection and held-tape extension use coupled numerical tolerance1e-10; clocks/modes/ancestry/RNG must agree exactly. Interleaved wall times are exploratory, not the complete-call gate.')
    dump(target,result)
    numeric=[r for r in records if 'fast_ms' in r]
    print(json.dumps(dict(passed=result['passed'],roots=len(indices),updates=len(numeric),errors=errors,
        fast_update_p95=float(np.quantile([r['fast_ms'] for r in numeric],.95)) if numeric else None,
        frozen_update_p95=float(np.quantile([r['frozen_ms'] for r in numeric],.95)) if numeric else None)),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=8);p.add_argument('--frames',type=int,default=6)
    p.add_argument('--tag',default='speed_checks_001');p.add_argument('--full',action='store_true')
    args=p.parse_args();run(args.limit,args.tag,args.frames,args.full)
