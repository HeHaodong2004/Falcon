"""Complete legal-range K1/4/6 latency on original 32 map workloads."""
import argparse
import gc
import os
import subprocess
import time
import numpy as np
import torch
from .diagnose import OUT,CKPT,read,save,sha,source_rows,member
from .range_api import RangeEvidencePredictor
from .benchmark import summary
from eventwm_graph.stage2_segment.benchmark import inputs
from eventwm_graph.stage2_hf.baseline import ThresholdBaseline
from .increment import SparseIncrement


def workloads():
    work=inputs('multi');source,_=source_rows('dev');raw={r['root_id']:r for r in source};out=[]
    for ex in work:
        legal=member(raw[ex['root_id']],'legal_root.json.gz')
        with np.load(OUT/'range_benchmark'/(ex['root_id']+'.npz'))as z:packet=dict(ranges=z['depth'],origins=z['origins'],rotations=z['rotations'],timestamps=z['timestamps'],sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1')
        out.append((ex,packet,[o['tracks']for o in legal['history']]))
    return out


def run(checkpoint,device='cuda:1',repetitions=3,proxy=False,spatial=False,radix=False,ray=False,exact_fast=False,device_canvas=False,batched=False,complete_distance=False,name=None,implementation=None,map_only_warm=False,_workloads=None):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;gc.enable();work=workloads()if _workloads is None else _workloads;rows=[];startup=[]
    from .range_implementations import predictor_class
    cls=predictor_class(spatial,proxy,radix,ray,exact_fast,device_canvas,batched,implementation)
    tick=time.perf_counter();baseline=ThresholdBaseline(CKPT,threshold=.2,execution='optimized',future_format='native',device=device,feature_threads=1);baseline_constructor=time.perf_counter()-tick
    tick=time.perf_counter();candidate=cls(checkpoint,device=device);candidate_constructor=time.perf_counter()-tick
    if complete_distance:candidate.config['observed_obstacle_schema']='capped_2m_complete'
    models=dict(baseline=baseline,range=candidate)
    def call(label,ex,packet,tracks,k,prime_map=False):
        paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:k]]
        if prime_map:paths=[paths[0][:1]]
        kw=dict(partial_graph=ex['graph'],trajectories=paths)
        if label=='range':kw.update(range_observations=packet,track_history=tracks)
        result=models[label].predict_segments(**kw)
        if label=='baseline':
            for d in result:
                at=np.flatnonzero(d['states']);d['static_updates']=SparseIncrement(d['candidate_cells'][at],d['states'][at],d['heights'][at],d['timestamp'])
        assert len(result)==(1 if prime_map else k)and all('future_partial_graph'in d for d in result)
        return result
    for label in models:
        tick=time.perf_counter();call(label,*work[0],6);torch.cuda.synchronize(device);startup.append(dict(model=label,constructor_ms=(baseline_constructor if label=='baseline'else candidate_constructor)*1000,first_complete_K6_ms=(time.perf_counter()-tick)*1000))
    order=sorted(range(len(work)),key=lambda i:len(work[i][0]['graph']['cells']));dense=set(order[len(order)//2:]);load=subprocess.check_output(['ps','-eo','pid,ppid,stat,etime,pcpu,pmem,args','--sort=-pcpu'],text=True).splitlines()[:25]
    for rep in range(repetitions):
        for i in order[::1 if rep%2 else-1]:
            ex,packet,tracks=work[i]
            for k in [1,4,6]:
                for label in ['baseline','range']if rep%2 else['range','baseline']:
                    for cache in ['cold','warm']+(['map_warm_new_queries']if map_only_warm else[]):
                        if cache=='cold':models[label].clear_cache()
                        elif cache=='map_warm_new_queries':models[label].clear_cache();call(label,ex,packet,tracks,k,prime_map=True)
                        else:call(label,ex,packet,tracks,k)
                        torch.cuda.synchronize(device);tick=time.perf_counter();result=call(label,ex,packet,tracks,k);torch.cuda.synchronize(device);ms=(time.perf_counter()-tick)*1000
                        rows.append(dict(model=label,cohort='dense'if i in dense else'representative',cache=cache,k=k,ms=ms,group=i,repetition=rep,candidates=[len(d['candidate_cells'])for d in result],updates=[len(d['static_updates'])for d in result],profile=models[label].last_batch_profile))
        report=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),device=device,hardware=torch.cuda.get_device_name(torch.device(device)),precision={'baseline':'F32 TF32off','range':'AMP spatial encoder/head; frozen F32 prior TF32off; F64 camera/frustum geometry'},input_contract='Range model additionally receives LEGAL measured current/past Euclidean ranges and legal track history; baseline receives original graph/path only. Same original 32 maps/paths and full original cells.',scope='Complete input normalization/content hashing, map/BVH build, graph/candidates/features, transfers/synchronization, network/probability/hard decoding, sparse increment and normal native hypothetical graphs. Cold clears graph AND mesh caches. Warm retains current-map state. Normal GC.',excluded='Input file deserialization, output serialization, environment sensor acquisition; all sensor arrays and input graphs already supplied. Sensor extension is explicit; timing alone does not establish total RL runtime.',startup=startup,concurrent_load=load,load_average=os.getloadavg(),summary=summary(rows),calls=rows)
        report.update(visibility_feature_los_probes=0 if ray else 7 if proxy else 63,spatial_model=spatial or ray,radix_tree=radix or exact_fast or device_canvas or batched or bool(implementation),learned_viewing_lines=ray,implementation=cls.__name__,observed_obstacle_schema=candidate.config.get('observed_obstacle_schema','legacy_8cell_crop'))
        report['map_only_warm']=bool(map_only_warm)
        if map_only_warm:report['map_warm_new_queries_definition']='Clear all caches, prime only the starting-pose query, then time the supplied K complete future trajectories. Warm mode instead repeats identical queries. All timings include normal graph materialization.'
        label=name or 'benchmark_legal_range'+('_'+implementation if implementation else'')+('_spatial'if spatial else'')+('_proxy7'if proxy else'')+('_radix'if radix else'')+('_ray'if ray else'')+('_exactfast'if exact_fast else'')+('_devicecanvas'if device_canvas else'')+('_batched'if batched else'')
        destination=OUT/(label+'.json')
        if destination.exists():
            old=read(destination)
            if old['checkpoint_sha256']!=report['checkpoint_sha256']or old.get('implementation')!=cls.__name__:raise ValueError('use --name for another benchmark identity')
        save(destination,report);print('REPETITION',rep+1,report['summary'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--repetitions',type=int,default=3);p.add_argument('--proxy',action='store_true');p.add_argument('--spatial',action='store_true');p.add_argument('--radix',action='store_true');p.add_argument('--ray',action='store_true');p.add_argument('--complete-distance',action='store_true');p.add_argument('--name');p.add_argument('--map-only-warm',action='store_true');from .range_implementations import add_exact_options;add_exact_options(p);run(**vars(p.parse_args()))
