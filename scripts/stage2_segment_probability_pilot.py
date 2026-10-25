"""Bounded post-soak GPU-softmax pilot on unchanged weights and saved inputs."""
import gc
import hashlib
import json
import pickle
import time
import traceback
import numpy as np
import torch
from eventwm_graph.stage2_segment.common import OUT,dump,sha,snapshot


def digest(graph,path):
    value=hashlib.sha256(pickle.dumps(graph,protocol=5));value.update(np.asarray(path,dtype='<f8').tobytes(order='C'));return value.hexdigest()


def compare(left,right,exact_graph=True):
    for delta in (left,right):
        p=delta['probabilities'];assert p.ndim==2 and p.shape[1]==3 and np.isfinite(p).all()
        assert np.all((p>=0)&(p<=1)) and np.all(abs(p.sum(1,dtype=np.float64)-1)<=1e-6)
        assert np.isfinite(delta['heights']).all()
    for name in ('candidate_cells','states','unknown_cells'):np.testing.assert_array_equal(left[name],right[name])
    for name in ('probabilities','heights'):np.testing.assert_allclose(left[name],right[name],atol=2e-6,rtol=0)
    a=left['future_partial_graph'];b=right['future_partial_graph'];a=a.to_snapshot() if hasattr(a,'to_snapshot') else a;b=b.to_snapshot() if hasattr(b,'to_snapshot') else b
    if exact_graph:np.testing.assert_array_equal(left['heights'],right['heights']);assert a==b
    else:
        assert {k:v for k,v in a.items() if k!='cells'}=={k:v for k,v in b.items() if k!='cells'}
        assert len(a['cells'])==len(b['cells'])
        for ac,bc in zip(a['cells'],b['cells']):
            assert {k:v for k,v in ac.items() if k!='height'}=={k:v for k,v in bc.items() if k!='height'}
            assert abs(ac['height']-bc['height'])<=2e-6
    return dict(hard_states_exact=True,future_graph_pass=True,probability_bounds_and_sums_pass=True,
        map_comparison='released_full_map_exact' if exact_graph else 'CPU_states_metadata_exact_heights_within_2e-6',
        maximum_probability_error=float(np.max(abs(left['probabilities']-right['probabilities']))) if len(left['states']) else 0.,
        maximum_height_error=float(np.max(abs(left['heights']-right['heights']))) if len(left['states']) else 0.)


def group_digest(group):
    value=hashlib.sha256(pickle.dumps(group['graph'],protocol=5))
    for item in group['paths'][:8]:
        path=np.asarray(item['trajectory'],dtype='<f8');value.update(str(path.shape).encode());value.update(path.tobytes(order='C'))
    return value.hexdigest()


def synthetic(dest,old_class,new_class):
    from eventwm_graph.partial.map import PartialGraph
    from eventwm_graph.stage2_quality.model import GraphDeltaNet
    from eventwm_graph.stage2_quality.common import SCHEMA
    path=[[1.,0.,0.,0.,0.],[1.5,.25,0.,0.,.2]];records=[]
    biases=([0.,0.,-1.],[0.,.000009,-1.],[0.,.00009,-1.],[0.,.00011,-1.],[1000.,1000.,999.],[-2.,2.,-1.],[-2.,-1.,2.],[-1.,0.,0.],[-1.,0.,.00009],[-1.,0.,.00011])
    cases=[(1.,bias) for bias in biases]+[(.075,[0.,0.,-1.])]
    for index,(resolution,bias) in enumerate(cases):
        graph=PartialGraph(resolution,0.);graph.update(dict(timestamp=1.,robot_pose=[0.,0.,0.,0.],static_updates=[dict(cell=[0,0],state=1,height=0.,timestamp=1.,source='synthetic_test_only')]))
        graph=graph.snapshot()
        net=GraphDeltaNet(input_features=104)
        with torch.no_grad():
            for parameter in net.parameters():parameter.zero_()
            net.net[-1].bias.copy_(torch.tensor([*bias,.125]))
        checkpoint=dest/f'synthetic_{index}.pt';torch.save(dict(schema=SCHEMA,mode='both',model_config=dict(width=48,input_features=104),model=net.state_dict(),config=dict(synthetic_test_only=True)),checkpoint)
        old=old_class(checkpoint,execution='optimized',future_format='native',device='cuda:1');new=new_class(checkpoint,execution='optimized',future_format='native',device='cuda:1');cpu=old_class(checkpoint)
        a=old.predict(partial_graph=graph,trajectory=path);b=new.predict(partial_graph=graph,trajectory=path);c=cpu.predict(partial_graph=graph,trajectory=path)
        proof=compare(a,b);compare(c,b,False)
        if resolution==.075:
            assert len(b['states'])>65536
            assert new.last_profile['near_tie_reference_blocks']==(len(b['states'])+8191)//8192
        records.append(dict(bias=bias,resolution=resolution,candidates=len(b['states']),checkpoint_sha256=sha(checkpoint),**proof,reference_blocks=new.last_profile['near_tie_reference_blocks']))
    return records


def main():
    if not (OUT/'stability_release_dede_v1/COMPLETE.json').exists():raise RuntimeError('Wait for fixed service completion')
    from eventwm_graph.stage2_segment.api import SegmentPredictor
    from eventwm_graph.stage2_segment.probability_experiment import GpuProbabilityPredictor
    from eventwm_graph.stage2_segment.benchmark import inputs,host_state
    from eventwm_graph.stage2.benchmark import quantiles
    from scripts.stage2_segment_supplement_tail import accepted_identity,no_primary_benchmark
    dest=OUT/'probability_execution_pilot_v1';dest.mkdir(exist_ok=False)
    try:
        state=host_state();assert gc.isenabled() and not state['competing_research'];no_primary_benchmark();accepted=accepted_identity();checkpoint=OUT/'training/final/last.pt'
        dump(dest/'protocol.json',dict(source=snapshot(),plan_sha256=sha(OUT/'probability_execution_protocol_v2.json'),accepted=accepted,host_state=state,
            checkpoint_sha256=sha(checkpoint),gc_enabled=True,primary_definitions_unchanged=True,scope='Execution-only same-weight pilot; no labels or new fit'))
        workload=inputs('single');groups=inputs('multi');kwargs=dict(execution='optimized',future_format='native',device='cuda:1',feature_threads=1)
        group_ids={group['root_id']:group_digest(group) for group in groups};assert len(group_ids)==len(groups)==32
        dump(dest/'multi_input_identities.json',group_ids)
        models=dict(released=SegmentPredictor(checkpoint,**kwargs),gpu_probability=GpuProbabilityPredictor(checkpoint,**kwargs));reference=SegmentPredictor(checkpoint)
        unique={}
        for cohort in workload.values():
            for case in cohort:unique.setdefault(digest(case['graph'],case['trajectory']),case)
        proofs=[]
        for identity,case in unique.items():
            args=dict(partial_graph=case['graph'],trajectory=case['trajectory']);old=models['released'].predict(**args);new=models['gpu_probability'].predict(**args);cpu=reference.predict(**args)
            proofs.append(dict(root_id=case['root_id'],input_sha256=identity,released=compare(old,new),cpu=compare(cpu,new,False)))
            assert digest(case['graph'],case['trajectory'])==identity
        del reference
        batch_proofs=[]
        for group in groups:
            for k in (1,2,4,8):
                paths=[p['trajectory'] for p in group['paths'][:k]];a=models['released'].predict_segments(partial_graph=group['graph'],trajectories=paths);b=models['gpu_probability'].predict_segments(partial_graph=group['graph'],trajectories=paths)
                assert len(a)==len(b)==k
                for left,right in zip(a,b):compare(left,right)
                batch_proofs.append(dict(root_id=group['root_id'],k=k,complete_graphs_equal=True))
            assert group_digest(group)==group_ids[group['root_id']]
        synth=synthetic(dest,SegmentPredictor,GpuProbabilityPredictor)
        dump(dest/'equivalence.json',dict(single=proofs,batch=batch_proofs,synthetic=synth));records=[];holder={}
        for cohort,cases in workload.items():
            for index,case in enumerate(cases):
                for model in models.values():model.predict(partial_graph=case['graph'],trajectory=case['trajectory'])
                for condition in ('map_cold','retained_cache'):
                    for repeat in range(5):
                        for name in (('released','gpu_probability') if (index+repeat)%2==0 else ('gpu_probability','released')):
                            model=models[name]
                            if condition=='map_cold':model.clear_cache()
                            tick=time.perf_counter();holder[name]=model.predict(partial_graph=case['graph'],trajectory=case['trajectory']);elapsed=time.perf_counter()-tick
                            records.append(dict(model=name,cohort=cohort,condition=condition,root_id=case['root_id'],repeat=repeat,total_seconds=elapsed,profile=dict(model.last_profile)))
        multi=[]
        for index,group in enumerate(groups):
            assert group_digest(group)==group_ids[group['root_id']]
            for k in (1,2,4,8):
                paths=[p['trajectory'] for p in group['paths'][:k]]
                for condition in ('map_cold','retained_cache'):
                    for repeat in range(5):
                        for name in (('released','gpu_probability') if (index+repeat)%2==0 else ('gpu_probability','released')):
                            model=models[name]
                            if condition=='map_cold':model.clear_cache()
                            tick=time.perf_counter();holder[name]=model.predict_segments(partial_graph=group['graph'],trajectories=paths);elapsed=time.perf_counter()-tick
                            assert len(holder[name])==k
                            multi.append(dict(model=name,k=k,condition=condition,root_id=group['root_id'],repeat=repeat,total_seconds=elapsed,profile=dict(model.last_batch_profile)))
        assert accepted_identity()==accepted
        for identity,case in unique.items():assert digest(case['graph'],case['trajectory'])==identity
        for group in groups:assert group_digest(group)==group_ids[group['root_id']]
        dump(dest/'single_calls.json',records);dump(dest/'multi_calls.json',multi)
        single_summary={name:{cohort:{condition:quantiles([r['total_seconds'] for r in records if r['model']==name and r['cohort']==cohort and r['condition']==condition])
            for condition in ('map_cold','retained_cache')} for cohort in workload} for name in models}
        multi_summary={name:{condition:{str(k):quantiles([r['total_seconds'] for r in multi if r['model']==name and r['condition']==condition and r['k']==k])
            for k in (1,2,4,8)} for condition in ('map_cold','retained_cache')} for name in models}
        k4_gain=all(multi_summary['gpu_probability'][condition]['4'][q]<=.95*multi_summary['released'][condition]['4'][q] for condition in ('map_cold','retained_cache') for q in ('p50_ms','p95_ms'))
        no_single_regression=all(single_summary['gpu_probability'][cohort][condition]['p95_ms']<=1.05*single_summary['released'][cohort][condition]['p95_ms'] for cohort in ('representative','dense') for condition in ('map_cold','retained_cache'))
        dump(dest/'metrics.json',dict(single=single_summary,multi=multi_summary,all_equivalence_proofs_pass=True,
            k4_predeclared_gain_pass=k4_gain,no_single_regression_pass=no_single_regression,
            promotion='LATENCY_CANDIDATE_ONLY_PENDING_ALL_RELEASE_GATES' if k4_gain and no_single_regression else 'NOT_PROMOTED',
            production_promoted=False,memory_and_startup_gate='PENDING_SEPARATE_MATCHED_VERIFICATION',
            full_proper_scores_and_calibration_gate='PENDING_COMPLETE_FROZEN_DEVELOPMENT_REVERIFICATION',
            integrated_and_concurrency_gates='PENDING_MATCHED_RELEASE_BENCHMARKS',
            memory_scope='Joint real/synthetic proof and timing process; no backend-specific peak claim; independent release benchmarks required before promotion'))
    except BaseException:
        dump(dest/'FAILED.json',dict(error=traceback.format_exc(),checkpoint_unchanged=sha(OUT/'training/final/last.pt')));raise


if __name__=='__main__':main()
