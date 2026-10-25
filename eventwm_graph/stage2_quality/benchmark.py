"""Matched fullAPI benchmark; separate singleQuery and multiQuery decisions."""
import argparse
from collections import defaultdict
from collections.abc import Mapping
import gzip
import gc
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
import torch
from eventwm_graph.stage2.api import Stage2Predictor as FrozenPredictor,apply_graph_delta as frozen_apply
from eventwm_graph.stage2.data import source_rows,branches
from eventwm_graph.stage2.benchmark import wire,quantiles
from .api import Stage2Predictor,apply_graph_delta
from .common import BASE,OUT,ORIGINAL,sha,identity,dump


GC_EVENTS=[]
GC_ACTIVE={}
def gc_observer(phase,info):
    generation=info['generation']
    if phase=='start':GC_ACTIVE[generation]=time.perf_counter()
    else:
        end=time.perf_counter();begin=GC_ACTIVE.pop(generation,end)
        GC_EVENTS.append(dict(generation=generation,start=begin,end=end,seconds=end-begin,collected=info.get('collected'),uncollectable=info.get('uncollectable')))

def gc_during(begin,end,index):return [r for r in GC_EVENTS[index:] if begin<=r['start'] and r['end']<=end]


def summary(rows,field='total_seconds'):
    values=[r[field] for r in rows]
    return dict(**quantiles(values),calls=len(rows),roots=len({r['root_id'] for r in rows}),scenes=len({r['scene_id'] for r in rows}))


def call(model,update,graph,path,holder=None):
    cpu=time.process_time();gc_index=len(GC_EVENTS);tick=time.perf_counter();delta=model.predict(partial_graph=graph,trajectory=path)
    if holder is not None:holder[0]=delta
    predicted=time.perf_counter()
    profile=dict(getattr(model,'last_profile',{}) or {});update(graph,delta);updated=time.perf_counter();transport=wire(delta);end=time.perf_counter()
    return dict(total_seconds=end-tick,predict_seconds=predicted-tick,update_seconds=updated-predicted,serialization_seconds=end-updated,
        process_cpu_seconds=time.process_time()-cpu,cpu_utilization_percent=100*(time.process_time()-cpu)/max(end-tick,1e-9),
        profile=profile,gc_events=gc_during(tick,end,gc_index),candidate_cells=len(delta['states']),new_cells=len(delta['static_updates']),wire_bytes=len(transport)),delta,None


def single(checkpoint):
    dest=OUT/'benchmarks/single';dest.mkdir(parents=True,exist_ok=False);rows,_=source_rows('dev');examples=[]
    for index in np.linspace(0,len(rows)-1,128,dtype=int):
        ex=max(list(branches(rows[index])),key=lambda e:e['executed_nodes']);examples.append(ex)
    workloads=dict(representative=examples[::2],dense=sorted(examples,key=lambda e:(len(e['graph']['cells']),e['executed_nodes']),reverse=True)[:32])
    before=time.perf_counter();candidate=Stage2Predictor(checkpoint);candidate_load=time.perf_counter()-before
    before=time.perf_counter();baseline=FrozenPredictor(ORIGINAL/'output/stage2_graphdelta_20261024/training/final/last.pt');baseline_load=time.perf_counter()-before
    models=dict(baseline=(baseline,frozen_apply),candidate=(candidate,apply_graph_delta));records=[];first=[];baseline_profiles=[];candidate_forward_profiles=[];holders={name:[None] for name in models}
    profile_source=BASE/'docs/l2/stage2_quality_runtime_20261024/reviews/reviewer_b/profile_baseline.py'
    spec=importlib.util.spec_from_file_location('independent_baseline_profile',profile_source);profiler=importlib.util.module_from_spec(spec);spec.loader.exec_module(profiler)
    for label,items in workloads.items():
        for index,ex in enumerate(items):
            graph=ex['graph'];path=ex['trajectory']
            # Both reference and candidate are initialized; cache-cold calls
            # explicitly include the new map's preparation inside timing.
            for name,(model,update) in models.items():
                value,_,_=call(model,update,graph,path)
                first.append(dict(workload=label,model=name,root_id=ex['root_id'],scene_id=ex['scene_id'],**value))
            reference=baseline.predict(partial_graph=graph,trajectory=path)
            witness,components,total=profiler.profiled(baseline,dict(partial_graph=graph,trajectory=path))
            for field in ('candidate_cells','probabilities','states','heights'):np.testing.assert_array_equal(reference[field],witness[field])
            baseline_profiles.append(dict(workload=label,root_id=ex['root_id'],components_ms=components,total_ms=total))
            del reference,witness
            forward=[];active=[None]
            def before_forward(module,args):active[0]=time.perf_counter()
            def after_forward(module,args,value):forward.append(time.perf_counter()-active[0])
            h1=candidate.net.register_forward_pre_hook(before_forward);h2=candidate.net.register_forward_hook(after_forward)
            candidate.clear_cache();detail,_,_=call(candidate,apply_graph_delta,graph,path);h1.remove();h2.remove()
            candidate_forward_profiles.append(dict(workload=label,root_id=ex['root_id'],neural_forward_ms=sum(forward)*1000,
                pack_and_probability_ms=(detail['profile']['neural_seconds']-sum(forward))*1000,total_ms=detail['total_seconds']*1000))
            for condition in ('initialized_map_cold','initialized_map_hot'):
                for repeat in range(5):
                    order=('baseline','candidate') if (index+repeat)%2==0 else ('candidate','baseline')
                    keys=None
                    for name in order:
                        model,update=models[name]
                        if condition=='initialized_map_cold' and name=='candidate':model.clear_cache()
                        value,delta,future=call(model,update,graph,path,holders[name])
                        if keys is None:keys=delta['candidate_cells']
                        else:np.testing.assert_array_equal(keys,delta['candidate_cells'])
                        records.append(dict(workload=label,condition=condition,model=name,root_id=ex['root_id'],scene_id=ex['scene_id'],query_id=ex['query_id'],repeat=repeat,
                            known_cells=len(graph['cells']),poses=len(path),**value))
                        delta=None;future=None
    result=dict(scope='Exact frozen128developmentRoot source-selection rule; matched identical Query inputs;5repeats; alternatingmodelorder',checkpoint_sha256=sha(checkpoint),sources=identity(),
        load_seconds=dict(candidate=candidate_load,baseline=baseline_load),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        models={},first_initialized_by_root=first,overall_integrated_layer2_gate_measured=False)
    for name in models:
        result['models'][name]={}
        for label in workloads:
            result['models'][name][label]={}
            for condition in ('initialized_map_cold','initialized_map_hot'):
                selected=[r for r in records if r['model']==name and r['workload']==label and r['condition']==condition]
                means={field:float(np.mean([r[field] for r in selected])*1000) for field in ('predict_seconds','update_seconds','serialization_seconds','total_seconds')}
                profiles={key:float(np.mean([r['profile'][key] for r in selected])*1000) for key in ('prepare_seconds','spatial_seconds','candidate_seconds','feature_seconds','neural_seconds','delta_seconds','trajectory_seconds','map_seconds')} if name=='candidate' else None
                result['models'][name][label][condition]=dict(**summary(selected),mean_ms=means,predict=summary(selected,'predict_seconds'),
                    component_mean_ms=profiles,component_percent_of_total={k:100*v/means['total_seconds'] for k,v in profiles.items()} if profiles is not None else None)
    result['baseline_separate_component_profile']={label:{field:float(np.mean([r['components_ms'][field] for r in baseline_profiles if r['workload']==label])) for field in baseline_profiles[0]['components_ms']} for label in workloads}
    result['candidate_forward_hook_diagnostic']={label:{key:float(np.mean([r[key] for r in candidate_forward_profiles if r['workload']==label])) for key in ('neural_forward_ms','pack_and_probability_ms','total_ms')} for label in workloads}
    dump(dest/'candidate_forward_calls.json',candidate_forward_profiles)
    result['gc_policy']=dict(enabled=gc.isenabled(),thresholds=gc.get_threshold(),controls_changed=False,events=len(GC_EVENTS))
    result['output_lifetime']='Reference apply result is immediately discarded before update timestamp; prior model delta is replaced inside measured prediction, matching original serial benchmark work accounting.'
    result['profile_notes']=['Baseline profile instrumented independently and verified bitwise; unmeasured last_profile fields are null.','Candidate support and fulltrajectory descriptors are separately timed. feature_seconds contains trajectory_seconds plus map_seconds and small bookkeeping; avoid summing nested counters twice.']
    dump(dest/'baseline_component_calls.json',baseline_profiles)
    dump(dest/'calls.json',records);dump(dest/'metrics.json',result)
    # Retain workload identities for exact independent reruns.
    dump(dest/'workloads.json',{label:[dict(root_id=e['root_id'],query_id=e['query_id'],scene_id=e['scene_id']) for e in items] for label,items in workloads.items()})
    print(json.dumps(result),flush=True)


def multi(checkpoint):
    dest=OUT/'benchmarks/multi';dest.mkdir(parents=True,exist_ok=False);rows,_=source_rows('train');eligible=[r for r in rows if r['queries']>=16]
    selected=[eligible[i] for i in np.linspace(0,len(eligible)-1,32,dtype=int)];sets=[list(branches(row))[:16] for row in selected]
    models=dict(baseline=(FrozenPredictor(ORIGINAL/'output/stage2_graphdelta_20261024/training/final/last.pt'),frozen_apply),candidate=(Stage2Predictor(checkpoint),apply_graph_delta));records=[]
    for kind in ('shared_root','unrelated_roots'):
        for index,examples in enumerate(sets):
            for k in (1,4,8,16):
                chosen=examples[:k] if kind=='shared_root' else [sets[(index+j)%len(sets)][0] for j in range(k)]
                inputs=[dict(partial_graph=e['graph'],trajectory=e['trajectory']) for e in chosen]
                order=('baseline','candidate') if (index+k)%2==0 else ('candidate','baseline')
                for name in order:
                    model,update=models[name]
                    # Warm neural/JIT and then explicitly clear map caches for
                    # the cold-spatial condition inside each new decision.
                    warm_deltas=model.predict_batch(inputs)
                    for item,delta in zip(inputs,warm_deltas):update(item['partial_graph'],delta);wire(delta)
                    del warm_deltas,delta
                    for condition in ('initialized_map_cold','initialized_map_hot'):
                        for repeat in range(3):
                            if name=='candidate' and condition=='initialized_map_cold':model.clear_cache()
                            before_hits=getattr(model,'cache_hits',0);before_misses=getattr(model,'cache_misses',0)
                            cpu=time.process_time();tick=time.perf_counter();deltas=model.predict_batch(inputs);predicted=time.perf_counter()
                            future=[update(i['partial_graph'],d) for i,d in zip(inputs,deltas)];updated=time.perf_counter()
                            for delta in deltas:wire(delta)
                            end=time.perf_counter();elapsed=end-tick
                            records.append(dict(kind=kind,k=k,model=name,condition=condition,repeat=repeat,root_id=examples[0]['root_id'],scene_id=examples[0]['scene_id'],
                                query_ids=[e['query_id'] for e in chosen],input_root_ids=[e['root_id'] for e in chosen],total_seconds=elapsed,predict_seconds=predicted-tick,
                                update_seconds=updated-predicted,serialization_seconds=end-updated,per_query_seconds=elapsed/k,cpu_utilization_percent=100*(time.process_time()-cpu)/elapsed,
                                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cache_hits=getattr(model,'cache_hits',0)-before_hits,cache_misses=getattr(model,'cache_misses',0)-before_misses,batch_reuse=dict(getattr(model,'last_batch_profile',{}) or {})))
    metrics={}
    for kind in ('shared_root','unrelated_roots'):
        metrics[kind]={}
        for name in models:
            metrics[kind][name]={}
            for condition in ('initialized_map_cold','initialized_map_hot'):
                metrics[kind][name][condition]={str(k):dict(**summary([r for r in records if r['kind']==kind and r['model']==name and r['condition']==condition and r['k']==k]),
                    mean_ms_per_query=float(np.mean([r['per_query_seconds'] for r in records if r['kind']==kind and r['model']==name and r['condition']==condition and r['k']==k])*1000)) for k in (1,4,8,16)}
    dump(dest/'calls.json',records);dump(dest/'metrics.json',dict(workloads=metrics,eligible_fitting_roots=len(eligible),source_roots=len(selected),native_queries_per_root=16,
        matched=True,no_repeated_query_padding=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,notes=['Same32highQuerycountfittingRoots andnestednativeQueryprefixes forold/new.','Unrelated-Root batches use distinctRoots; no sharedmap speedup claim for them.','initialized_map_hot means retained/no-flushLRU4policy; unrelatedK8/16mayevictallmaps andrecordactualcachemisses.','Single-call100ms gate is not a16Querydecision gate.']))
    print('multiQuery benchmark complete',flush=True)



class FrozenSandboxAdapter:
    def __init__(self,stage1,stage2):self.stage1=stage1;self.stage2=stage2;self.last_profile=None
    def predict_decision(self,context,observation,*,executed_action,trajectories,arrived_graph=None):
        def thaw(v):
            if isinstance(v,Mapping):return {k:thaw(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [thaw(x) for x in v]
            if isinstance(v,np.ndarray):return v.copy()
            return v
        tick=time.perf_counter();people=self.stage1.predict(context,observation,partial_graph=arrived_graph,executed_action=executed_action,horizon=3.,worlds=40);human=time.perf_counter()
        actual=self.stage1.last_context;graph=thaw(actual.partial_graph);adapted=time.perf_counter()
        deltas=self.stage2.predict_batch([dict(partial_graph=graph,trajectory=p) for p in trajectories]);predicted=time.perf_counter()
        future=[frozen_apply(graph,d) for d in deltas];done=time.perf_counter()
        self.last_profile=dict(stage1_seconds=human-tick,graph_adapter_seconds=adapted-human,stage2_batch_seconds=predicted-adapted,update_seconds=done-predicted,total_seconds=done-tick,stage1_calls=1,queries=len(trajectories),worlds=40,knots=64)
        return dict(human_worlds=people,graph_deltas=deltas,hypothetical_graphs=future,actual_context=actual)

def integrated(checkpoint):
    from .integration import load_stage1,WorldModelSandbox
    stage1,source=load_stage1();OnlineContext=importlib.import_module('eventwm_graph.l2_two_stage.context').OnlineContext
    reader=importlib.import_module('eventwm_graph.l2_two_stage.data');dest=OUT/'benchmarks/integrated';dest.mkdir(parents=True,exist_ok=False)
    stage2=Stage2Predictor(checkpoint);sandbox=WorldModelSandbox(stage1,stage2)
    frozen=FrozenPredictor(ORIGINAL/'output/stage2_graphdelta_20261024/training/final/last.pt');adapters=dict(candidate=sandbox,baseline=FrozenSandboxAdapter(stage1,frozen));rows,_=source_rows('dev');examples=[]
    for index in np.linspace(0,len(rows)-1,128,dtype=int):
        row=rows[index];ex=max(list(branches(row)),key=lambda e:e['executed_nodes']);examples.append((row,ex))
    workloads=dict(representative=examples[::2],dense=sorted(examples,key=lambda pair:(len(pair[1]['graph']['cells']),pair[1]['executed_nodes']),reverse=True)[:32])
    records=[];startup=[];failures=[]
    for label,items in workloads.items():
        for index,(row,ex) in enumerate(items):
            folder=Path(row['folder']);legal=reader.read(folder/'legal_root.json.gz')
            outcomes=reader.read(folder/'outcomes.json.gz');outcome=next(o for o in outcomes if o['query_id']==ex['query_id'])
            nodes=reader.read(folder/'future_observations.json.gz');branch=[nodes[i] for i in outcome['node_ids']]
            begun=time.perf_counter();initial=stage1.predict(legal['history'],legal['history'][-1],legal['partial_graph'],actions=legal['history_actions'],horizon=3.,worlds=40);context=stage1.last_context.with_association();init_seconds=time.perf_counter()-begun
            if not branch:failures.append(dict(root_id=row['root_id'],reason='No next actual observation; no native new-frame timing invented'));continue
            ob=branch[0]['observation'];action=branch[0]['action'];path=np.asarray([[node['observation']['timestamp'],*node['observation']['robot_pose']] for node in branch],np.float64)
            # First initialized observation-ingestion call (including first JIT
            # use) retained separately from the initialized gate distribution.
            for model_name,adapter in adapters.items():
                adapter.predict_decision(context,ob,executed_action=action,trajectories=[path]);startup.append(dict(model=model_name,root_id=row['root_id'],context_initialization_seconds=init_seconds,first_profile=dict(adapter.last_profile)))
            for condition in ('initialized_map_cold','initialized_map_hot'):
                for repeat in range(5):
                    order=('baseline','candidate') if (index+repeat)%2==0 else ('candidate','baseline')
                    for model_name in order:
                        if condition=='initialized_map_cold' and model_name=='candidate':stage2.clear_cache()
                        gc_index=len(GC_EVENTS);tick=time.perf_counter();result=adapters[model_name].predict_decision(context,ob,executed_action=action,trajectories=[path]);native=time.perf_counter()
                        delta=result['graph_deltas'][0];wire(delta);sparse=time.perf_counter();people=result['human_worlds']
                        # A separate transport-inclusive diagnostic serializes all
                        # explicit sampled human arrays plus sparse GraphDelta.
                        json.dumps(dict(humans=dict(timestamp=people.timestamp,times=people.times.tolist(),weights=people.weights.tolist(),positions=people.positions.tolist(),present=people.present.tolist(),track_keys=list(people.track_keys),unseen=people.unseen.tolist()),graph_delta=json.loads(wire(delta))),separators=(',',':'),allow_nan=False)
                        complete=time.perf_counter();actual=result['actual_context']
                        if people.positions.shape[:2]!=(40,64) or float(actual.history[-1]['timestamp'])!=float(path[0,0]) or len(actual._positive_history.timestamps)!=len(context._positive_history.timestamps)+1:raise ValueError('Integrated call omitted required Stage1 ingestion or budgets')
                        records.append(dict(workload=label,condition=condition,model=model_name,root_id=row['root_id'],scene_id=row['scene_id'],repeat=repeat,total_seconds=sparse-tick,
                            gc_events=gc_during(tick,sparse,gc_index),all_transport_gc_events=gc_during(tick,complete,gc_index),native_seconds=native-tick,all_samples_transport_seconds=complete-tick,sparse_wire_seconds=sparse-native,full_wire_seconds=complete-sparse,
                            profile=dict(adapters[model_name].last_profile),stage2_profile=dict(stage2.last_profile) if model_name=='candidate' else None,known_cells=len(actual.partial_graph['cells']),poses=len(path),worlds=40,knots=64,
                            stage1_input_old_timestamp=float(context.history[-1]['timestamp']),actual_arrival_timestamp=float(actual.history[-1]['timestamp'])))
    metrics={}
    for model_name in adapters:
        metrics[model_name]={}
        for label in workloads:
            metrics[model_name][label]={}
            for condition in ('initialized_map_cold','initialized_map_hot'):
                selected=[r for r in records if r['model']==model_name and r['workload']==label and r['condition']==condition]
                metrics[model_name][label][condition]=dict(native=summary(selected,'native_seconds'),with_sparse_delta_wire=summary(selected),with_all_explicit_human_samples_json=summary(selected,'all_samples_transport_seconds'))
    gate=all(metrics['candidate'][label][condition]['with_sparse_delta_wire']['p95_ms']<=100 for label in workloads for condition in ('initialized_map_cold','initialized_map_hot')) and not failures
    dump(dest/'calls.json',records);dump(dest/'metrics.json',dict(workloads=metrics,gate_p95_ms=100,gate_passed=gate,baseline_adapter='Newly measured faithful reference adapter using frozen Stage1, original graph materialization, frozenStage2 and frozenupdater; not ahistoricalintegratedclaim.',gate_boundary='Actual frozenStage1native40worlds/64knots/3seconds andnew-observationingestion + Stage2prediction + deterministicupdate + sparseGraphDeltawire. All-sampleJSONtransport reportedseparately.',
        source_stage1=str(source),checkpoint_stage1_sha256=sha(ORIGINAL/'output/l2_two_stage_20261023/experiments/final_stage1_research_20261024/last.pt'),checkpoint_stage2_sha256=sha(checkpoint),
        startup=startup,failures=failures,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,layer25_compilation_measured=False,
        limitations=['StandaloneStage1/Stage2percentiles are not added. Each integratedcall actuallyrunsboth.','NativecurrentL2.5compiler consumes earlierbeliefAPI; no newcompilerintegrationorpassing16Query100msgateclaimed.','No futureobservationmap/humanstate entersStage2; onlyprovidedcandidateexecutedpathshape is used.']))
    print(json.dumps(dict(gate_passed=gate,workloads=metrics)),flush=True)



def frontier_order(model,graph,paths):
    """Benchmark-only legal shortlist cue; never a calibrated model output."""
    from .features import opportunity
    prepared=model.prepare_graph(graph)
    if not hasattr(prepared,'_benchmark_frontiers'):
        keys=prepared.keys
        if not len(keys):frontier=np.empty((0,2),np.int32)
        else:
            lo=keys.min(0)-1;shape=keys.max(0)-lo+2
            if int(np.prod(shape))>4000000:raise ValueError('frontier raster domain exceeds supported bound')
            image=np.zeros(tuple(shape),np.uint8);ij=keys-lo;image[ij[:,0],ij[:,1]]=prepared.values
            unknown=image==0
            boundary=(image==1)&(np.roll(unknown,1,axis=0)|np.roll(unknown,-1,axis=0)|np.roll(unknown,1,axis=1)|np.roll(unknown,-1,axis=1))
            frontier=np.argwhere(boundary).astype(np.int32)+lo
        prepared._benchmark_frontiers=frontier
    scores=[]
    for path in paths:
        _,values=opportunity(prepared._benchmark_frontiers,np.asarray(path,np.float64),prepared.resolution)
        scores.append(float(values[:,2].sum()))
    return np.argsort(-np.asarray(scores),kind='stable'),scores


def native_multi(checkpoint):
    """Actual Root arrival + native Query sets; full certification stays timed."""
    from .integration import load_stage1,WorldModelSandbox
    from eventwm_graph.partial.map import PartialGraph
    stage1,source=load_stage1();OnlineContext=importlib.import_module('eventwm_graph.l2_two_stage.context').OnlineContext
    read=importlib.import_module('eventwm_graph.l2_two_stage.data').read
    candidate=Stage2Predictor(checkpoint);adapters=dict(candidate=WorldModelSandbox(stage1,candidate),baseline=FrozenSandboxAdapter(stage1,FrozenPredictor(ORIGINAL/'output/stage2_graphdelta_20261024/training/final/last.pt')))
    dest=OUT/'benchmarks/native_multi';dest.mkdir(parents=True,exist_ok=False);rows,_=source_rows('train');eligible=[r for r in rows if r['queries']>=16]
    chosen=[eligible[i] for i in np.linspace(0,len(eligible)-1,32,dtype=int)][:16];records=[];selective=[];initialization=[]
    for index,row in enumerate(chosen):
        examples=list(branches(row))[:16];paths=[e['trajectory'] for e in examples];legal=read(Path(row['folder'])/'legal_root.json.gz');history=legal['history'];actions=legal['history_actions'];graph=legal['partial_graph']
        assert len(history)>=2 and len(actions)==len(history)-1
        start=time.perf_counter();g=PartialGraph(graph['resolution'],graph['height'],graph['clearance'])
        for observation in history[:-1]:g.update(observation)
        context=OnlineContext.from_raw(history[:-1],g.snapshot(),actions[:-1]).with_association()
        initialization.append(dict(root_id=row['root_id'],seconds=time.perf_counter()-start,history_frames=len(history)))
        for k in (1,4,8,16):
            order=('baseline','candidate') if (index+k)%2==0 else ('candidate','baseline')
            for name in order:
                adapter=adapters[name];adapter.predict_decision(context,history[-1],executed_action=actions[-1],arrived_graph=graph,trajectories=paths[:k])
                for repeat in range(3):
                    if name=='candidate':candidate.clear_cache()
                    gi=len(GC_EVENTS);tick=time.perf_counter();result=adapter.predict_decision(context,history[-1],executed_action=actions[-1],arrived_graph=graph,trajectories=paths[:k]);predicted=time.perf_counter()
                    for delta in result['graph_deltas']:wire(delta)
                    end=time.perf_counter();actual=result['actual_context']
                    if result['human_worlds'].positions.shape[:2]!=(40,64) or float(actual.history[-1]['timestamp'])!=float(paths[0][0,0]) or len(actual._positive_history.timestamps)!=len(context._positive_history.timestamps)+1:raise ValueError('native decision omitted true arrival or Stage1 budgets')
                    records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],model=name,k=k,repeat=repeat,total_seconds=end-tick,native_seconds=predicted-tick,serialization_seconds=end-predicted,
                        profile=dict(adapter.last_profile),stage2_reuse=dict(candidate.last_batch_profile) if name=='candidate' else None,gc_events=gc_during(tick,end,gi),per_query_seconds=(end-tick)/k,
                        query_ids=[e['query_id'] for e in examples[:k]],trajectory_transformations=False))
        # Explicit optional scheduler experiment. Deferred Queries have NO
        # fabricated posterior or delta. All invoked Queries retainfull support.
        actual=np.array([len(e['target']) for e in examples],np.float64)
        for budget in (1,4):
            for repeat in range(3):
                candidate.clear_cache();gi=len(GC_EVENTS);tick=time.perf_counter();rank,scores=frontier_order(candidate,graph,paths);ranked=time.perf_counter();selected=rank[:budget]
                result=adapters['candidate'].predict_decision(context,history[-1],executed_action=actions[-1],arrived_graph=graph,trajectories=[paths[i] for i in selected]);predicted=time.perf_counter()
                for delta in result['graph_deltas']:wire(delta)
                end=time.perf_counter();best=float(actual.max());retained=float(actual[selected].max()/best) if best else 1.
                selective.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],budget=budget,total_queries=16,repeat=repeat,total_seconds=end-tick,selector_seconds=ranked-tick,
                    native_seconds=predicted-ranked,serialization_seconds=end-predicted,selected_indices=selected.tolist(),legal_proxy_scores=scores,
                    best_actual_quantity_retained=retained,first_prefix_best_quantity_retained=float(actual[:budget].max()/best) if best else 1.,
                    profile=dict(adapters['candidate'].last_profile),gc_events=gc_during(tick,end,gi),deferred_queries_have_no_predictions=True))
    metrics={name:{str(k):dict(**summary([r for r in records if r['model']==name and r['k']==k]),mean_ms_per_query=float(np.mean([r['per_query_seconds'] for r in records if r['model']==name and r['k']==k])*1000)) for k in (1,4,8,16)} for name in adapters}
    selected_metrics={str(b):dict(**summary([r for r in selective if r['budget']==b]),mean_best_quantity_retained=float(np.mean([r['best_actual_quantity_retained'] for r in selective if r['budget']==b])),mean_first_prefix_retention=float(np.mean([r['first_prefix_best_quantity_retained'] for r in selective if r['budget']==b]))) for b in (1,4)}
    dump(dest/'calls.json',records);dump(dest/'selective_calls.json',selective);dump(dest/'metrics.json',dict(models=metrics,selective_stage2=selected_metrics,initialization=initialization,
        checkpoint_sha256=sha(checkpoint),scope='16high-QuerycountfittingRoots fromsame32RootStage2batchcohort; native distinct Queryprefixes; trueRootlastframeingestion with explicitarrivedgraph; fullfrozenStage1graph/historycertification remains inside timing.',
        stage1_once_per_decision=True,trajectories_transformed=False,hard_single_gate_replaced=False,layer25_compilation_measured=False,
        limitations=['Explicit-arrived-graph certification is a different supportedStage1entry boundary from incrementalnextframegate; both are reported.','Selective cue uses only legalFREE/UNKNOWNfrontiers and suppliedpath headings/range; nofuturelabels orhiddengeometry forselection.','Retention uses actualoutcomes offline and measures observationquantity only, notnavigationutility.','DeferredQueries have no fabricatedprobabilities. These timings do not establish a completeL2.5/plannerdecision budget.'],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))

def cold_child(checkpoint,input_path,baseline=False):
    start=time.perf_counter();model=FrozenPredictor(checkpoint) if baseline else Stage2Predictor(checkpoint);loaded=time.perf_counter()
    item=json.loads(gzip.decompress(Path(input_path).read_bytes()));ready=time.perf_counter()
    value,delta,future=call(model,frozen_apply if baseline else apply_graph_delta,item['partial_graph'],item['trajectory'])
    def rss():return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024
    initialized_rss=rss()
    del delta,future
    import gc;gc.collect()
    second,delta,future=call(model,frozen_apply if baseline else apply_graph_delta,item['partial_graph'],item['trajectory'])
    return dict(model_load_seconds=loaded-start,input_read_seconds=ready-loaded,first_complete=value,initialized_second_complete=second,initialized_rss_bytes=initialized_rss,rss_after_second_bytes=rss(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,checkpoint_sha256=sha(checkpoint),jit_cache=os.environ.get('NUMBA_CACHE_DIR'))


def cold(checkpoint):
    dest=OUT/'benchmarks/cold';dest.mkdir(parents=True,exist_ok=False);rows,_=source_rows('dev');byid={r['root_id']:r for r in rows}
    workloads=json.loads((OUT/'benchmarks/single/workloads.json').read_text());records=[]
    for label in ('representative','dense'):
        selected=workloads[label][0];ex=next(e for e in branches(byid[selected['root_id']]) if e['query_id']==selected['query_id'])
        path=dest/(label+'_input.json.gz');path.write_bytes(gzip.compress(json.dumps(dict(partial_graph=ex['graph'],trajectory=ex['trajectory'].tolist())).encode(),mtime=0))
        for name in ('baseline','candidate'):
            weight=ORIGINAL/'output/stage2_graphdelta_20261024/training/final/last.pt' if name=='baseline' else checkpoint
            for repeat in range(3):
                jit=dest/('empty_jit_'+label+'_'+name+'_'+str(repeat));assert not jit.exists()
                env=dict(os.environ,NUMBA_CACHE_DIR=str(jit));command=[sys.executable,str(BASE/'scripts/stage2_quality_run.py'),'eventwm_graph.stage2_quality.benchmark','--kind','cold_child','--checkpoint',str(weight),'--cold-input',str(path)]
                if name=='baseline':command+=['--baseline']
                begun=time.perf_counter();result=subprocess.run(command,cwd=BASE,env=env,text=True,capture_output=True,check=True);full=time.perf_counter()-begun
                row=json.loads(result.stdout.strip().splitlines()[-1]);records.append(dict(workload=label,model=name,repeat=repeat,root_id=selected['root_id'],full_process_seconds=full,empty_jit_at_launch=True,**row))
    dump(dest/'metrics.json',dict(calls=records,scope='Fresh processes and empty candidate JIT caches. Includes interpreter/library imports in parent full_process time;3observations percohort are descriptive, notP95certification.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--kind',choices=['single','multi','integrated','native_multi','cold','cold_child'],required=True);p.add_argument('--cold-input',type=Path);p.add_argument('--baseline',action='store_true');a=p.parse_args()
    gc.callbacks.append(gc_observer)
    if a.kind=='cold_child':print(json.dumps(cold_child(a.checkpoint,a.cold_input,a.baseline)))
    else:globals()[a.kind](a.checkpoint)
