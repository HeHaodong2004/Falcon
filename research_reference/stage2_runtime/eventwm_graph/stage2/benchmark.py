"""Full Python API, update, wire serialization, cold process and batch timing."""
import argparse
import gzip
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from .common import BASE,OUT,dump,sha,source_identity


def quantiles(values):
    return dict(zip(('p50_ms','p95_ms','p99_ms'),np.quantile(np.asarray(values)*1000,[.5,.95,.99]).tolist()))


def wire(delta):
    """Sparse portable delta; unknown cells represented by no-update semantics."""
    return json.dumps({key:delta[key] for key in ('schema','predicted','base_graph_identity','base_timestamp','timestamp','robot_pose',
        'static_updates','outside_domain','checkpoint_sha256')},separators=(',',':'),allow_nan=False)


def cold(checkpoint,input_path):
    begun=time.perf_counter()
    from .api import Stage2Predictor,apply_graph_delta
    imported=time.perf_counter();predictor=Stage2Predictor(checkpoint);loaded=time.perf_counter()
    item=json.loads(gzip.decompress(Path(input_path).read_bytes()));ready=time.perf_counter()
    delta=predictor.predict(**item);predicted=time.perf_counter();apply_graph_delta(item['partial_graph'],delta);updated=time.perf_counter();wire(delta)
    import eventwm_graph,eventwm_graph.partial,eventwm_graph.partial.map,eventwm_graph.stage2.api
    imports={module.__name__:dict(path=module.__file__,sha256=sha(module.__file__)) for module in
        (eventwm_graph,eventwm_graph.partial,eventwm_graph.partial.map,eventwm_graph.stage2.api)}
    print(json.dumps(dict(import_seconds=imported-begun,load_seconds=loaded-imported,input_read_seconds=ready-loaded,imports=imports,
        cold_predict_seconds=predicted-ready,cold_predict_apply_wire_seconds=time.perf_counter()-ready,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,candidates=len(delta['states']))))


def benchmark(checkpoint,name='final'):
    from .api import Stage2Predictor,apply_graph_delta
    from .data import source_rows,branches
    dest=OUT/'benchmarks'/name;dest.mkdir(parents=True,exist_ok=False)
    before=time.perf_counter();predictor=Stage2Predictor(checkpoint);load=time.perf_counter()-before
    rows,_=source_rows('dev');examples=[]
    for index in np.linspace(0,len(rows)-1,128,dtype=int):
        branch=list(branches(rows[index]));example=max(branch,key=lambda ex:ex['executed_nodes'])
        examples.append(example)
    workloads=dict(representative=examples[::2],dense=sorted(examples,key=lambda ex:(len(ex['graph']['cells']),ex['executed_nodes']),reverse=True)[:32])
    results={};raw=[];rss_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    for label,items in workloads.items():
        times=[];combined=[];withwire=[];forward=[];candidates=[];cells=[];poses=[]
        for example in items:
            item=dict(partial_graph=example['graph'],trajectory=example['trajectory'])
            # Warm each workload through the entire requested API once.
            predictor.predict(**item)
            for repeat in range(3):
                start=time.perf_counter();delta=predictor.predict(**item);predicted=time.perf_counter()
                apply_graph_delta(item['partial_graph'],delta);updated=time.perf_counter();wire(delta);encoded=time.perf_counter()
                times.append(predicted-start);combined.append(updated-start);withwire.append(encoded-start)
                raw.append(dict(workload=label,root_id=example['root_id'],query_id=example['query_id'],repeat=repeat,
                    predict_seconds=predicted-start,predict_apply_seconds=updated-start,predict_apply_wire_seconds=encoded-start,candidates=len(delta['states']),
                    known_cells=len(example['graph']['cells']),trajectory_poses=len(example['trajectory'])))
            candidates.append(len(delta['states']));cells.append(len(example['graph']['cells']));poses.append(len(example['trajectory']))
        results[label]=dict(calls=len(times),independent_roots=len({e['root_id'] for e in items}),scenes=len({e['scene_id'] for e in items}),
            predict=quantiles(times),predict_apply=quantiles(combined),predict_apply_wire=quantiles(withwire),
            candidates_range=[min(candidates),max(candidates)],known_cells_range=[min(cells),max(cells)],trajectory_poses_range=[min(poses),max(poses)])
    batches={}
    for count in (1,4,16):
        inputs=[dict(partial_graph=e['graph'],trajectory=e['trajectory']) for e in examples[:count]];times=[]
        for _ in range(8):
            start=time.perf_counter();predictor.predict_batch(inputs);times.append(time.perf_counter()-start)
        batches[str(count)]=dict(**quantiles(times),median_ms_per_trajectory=float(np.median(times)*1000/count),trajectories_per_second=count/float(np.median(times)))
    cold_results={}
    for label,items in workloads.items():
        example=items[0];path=dest/(label+'_input.json.gz')
        path.write_bytes(gzip.compress(json.dumps(dict(partial_graph=example['graph'],trajectory=example['trajectory'].tolist())).encode(),mtime=0))
        calls=[]
        for _ in range(3):
            start=time.perf_counter();result=subprocess.run([sys.executable,'-m','eventwm_graph.stage2.benchmark','--checkpoint',str(checkpoint),'--cold-input',str(path)],
                cwd='/tmp',env=os.environ,text=True,capture_output=True,check=True)
            row=json.loads(result.stdout.strip().splitlines()[-1]);row['full_process_seconds']=time.perf_counter()-start;calls.append(row)
        cold_results[label]=calls
    dump(dest/'calls.json',raw)
    result=dict(checkpoint_sha256=sha(checkpoint),sources=source_identity(),threads=1,device='CPU',workloads=results,batch=batches,cold=cold_results,
        checkpoint_load_seconds=load,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,rss_before_workloads=rss_before,
        overall_layer2_target_p95_ms=100,integrated_layer2_latency_measured=False,
        notes=['All timings include legal graph parsing/validation, domain/feature construction, neural inference and GraphDelta construction.',
            'predict_apply includes graph copy and existing deterministic update. predict_apply_wire also includes sparse JSON serialization.',
            'Raw probability array serialization excluded from sparse transport; probabilities remain available in the Python API.',
            'Dense is the32 largest observed maps among128 deterministically spaced original dev Roots, not a maximum-scene claim.',
            'Full-process cold times include interpreter/dependency startup; all three observations retained.'])
    dump(dest/'metrics.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--name',default='final');p.add_argument('--cold-input',type=Path);args=p.parse_args()
    if args.cold_input:cold(args.checkpoint,args.cold_input)
    else:benchmark(args.checkpoint,args.name)
