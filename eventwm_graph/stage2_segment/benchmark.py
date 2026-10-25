"""Real saved segments; full graph construction inside every primary timer."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import gc
import gzip
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import types
import numpy as np
import torch
from eventwm_graph.stage2.benchmark import wire, quantiles
from eventwm_graph.stage2.data import source_rows
from .api import SegmentPredictor
from .data import replay, load
from .common import BASE, OUT, QUALITY, CHECKPOINT, CHECKPOINT_SHA, dump, sha, snapshot, stable

# Own original runtime imports: baseline timing uses untouched dfde4322 bytes.
package = 'eventwm_graph.stage2_quality_reference'
if package not in sys.modules:
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(QUALITY/'eventwm_graph/stage2_quality')]
    sys.modules[package] = namespace
reference_api = importlib.import_module(package+'.api')
DeltaPredictor = reference_api.Stage2Predictor
apply_graph_delta = reference_api.apply_graph_delta


GC = []; GC_START = {}
def gc_event(phase, info):
    gen = info['generation']
    if phase == 'start':
        GC_START[gen] = time.perf_counter()
    else:
        end = time.perf_counter(); begin = GC_START.pop(gen, end)
        GC.append(dict(generation=gen, start=begin, end=end, seconds=end-begin, collected=info['collected']))


def reference_identity():
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('original checkpoint changed')
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=QUALITY,text=True).strip()
    if not head.startswith('dfde4322') or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=QUALITY,text=True).strip():
        raise ValueError('original research runtime changed')
    files = {str(p.relative_to(QUALITY)):sha(p) for p in sorted((QUALITY/'eventwm_graph/stage2_quality').glob('*.py'))}
    for rel in ('eventwm_graph/partial/map.py', *[str(p.relative_to(BASE)) for p in sorted((BASE/'eventwm_graph/stage2').glob('*.py'))]):
        if sha(BASE/rel) != sha(QUALITY/rel):
            raise ValueError('original shared dependency changed: '+rel)
        files[rel] = sha(QUALITY/rel)
    return dict(commit=head,source_worktree=str(QUALITY),files=files,checkpoint_sha256=CHECKPOINT_SHA)


def host_state():
    """Record timing conditions and reject overlapping research workloads."""
    modules={'eventwm_graph.stage2_segment.'+x for x in ('data','train','evaluate')}
    competing=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:args=(entry/'cmdline').read_bytes().decode().split('\0')
        except (OSError,UnicodeDecodeError):continue
        matched=sorted(modules.intersection(args)|{a for a in args if a.startswith('scripts.stage2_segment_')})
        if matched:competing.append(dict(pid=int(entry.name),modules=matched))
    cpu=Path('/proc/cpuinfo').read_text();names=[l.split(':',1)[1].strip() for l in cpu.splitlines() if l.startswith('model name')]
    memory={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:','SwapTotal:','SwapFree:'))}
    return dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),cpu_model=names[0] if names else None,
        available_logical_cpus=len(os.sched_getaffinity(0)),available_physical_cores=len(physical_cores()),
        load_average=list(os.getloadavg()),memory=memory,competing_research=competing,torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
        numpy_version=np.__version__,kernel=os.uname().release)


def build_inputs():
    dest = OUT/'benchmark_inputs'; dest.mkdir(parents=True, exist_ok=False)
    rows, source = source_rows('dev'); selected = [rows[int(i)] for i in np.linspace(0, len(rows)-1, 96)]
    specs = []; all_segments = []; groups = []
    # Freeze selection rule before looking at timings or model predictions.
    dump(dest/'protocol.json', dict(source_manifest_sha256=sha(source), roots=[r['root_id'] for r in selected],
        single='96 evenly spaced original development Roots; nearest saved durations0.5/1/2seconds and real>2tail; dense by known-map size among shortsegments',
        multi='up to32sourceRoots with8distinct saved prefixes sharing Root graph; K prefixes1/2/4/8; no repeated trajectory padding',
        accuracy='All models identical saved inputs; K-versus-single scientific arrays and futuregraphs checked outside performance timers'))
    for index, row in enumerate(selected):
        examples = [ex for ex in replay(row, 'mixed') if 'audit' not in ex]
        for ex in examples:
            p = ex['trajectory']; duration = p[-1, 0]-p[0, 0]
            all_segments.append(dict(root_id=row['root_id'], scene_id=row['scene_id'], query_id=ex['query_id'],
                start_node=ex['start_node'], end_node=ex['end_node'], graph=ex['graph'], trajectory=p.tolist(), duration=float(duration)))
        legal, queries, outcomes, nodes, _ = load(row)
        paths = {}; t0 = legal['partial_graph']['timestamp']; pose = legal['history'][-1]['robot_pose']
        for q, o in zip(queries, outcomes):
            ids = o['node_ids']; times = np.array([nodes[i]['observation']['timestamp']-t0 for i in ids])
            if not ids:
                continue
            for duration in (.144, .288, .48, .72, 1.008, 1.44, 2.016, float(times[-1])):
                end = int(np.argmin(abs(times-duration)))+1; identity = tuple(ids[:end])
                if identity not in paths:
                    path = [[t0, *pose]]+[[nodes[i]['observation']['timestamp'], *nodes[i]['observation']['robot_pose']] for i in identity]
                    paths[identity] = dict(query_id=q['query_id'], node_ids=list(identity), trajectory=path)
        if len(paths) >= 8 and len(groups) < 32:
            ordered = sorted(paths.values(), key=lambda v: stable(str(v['node_ids'])))
            groups.append(dict(root_id=row['root_id'], scene_id=row['scene_id'], graph=legal['partial_graph'], paths=ordered[:8]))
        if index % 16 == 0:
            print(json.dumps(dict(input_roots=index+1, segments=len(all_segments), multi_groups=len(groups))), flush=True)
    # One per Root in each duration cohort where a compatible saved window exists.
    cohorts = {}
    for target, tolerance, count, name in ((.5,.12,32,'around0.5s'),(1.,.15,32,'around1s'),(2.,.25,24,'around2s')):
        eligible = [ex for ex in all_segments if abs(ex['duration']-target) <= tolerance]
        chosen = []; seen = set()
        for ex in sorted(eligible, key=lambda ex: (abs(ex['duration']-target), stable(ex['root_id']))):
            if ex['root_id'] not in seen:
                chosen.append(ex); seen.add(ex['root_id'])
            if len(chosen) == count:
                break
        cohorts[name] = chosen
    tails = sorted([e for e in all_segments if e['duration'] > 2.25], key=lambda e: stable(e['root_id']+str(e['end_node'])))
    chosen = []; seen = set()
    for ex in tails:
        if ex['root_id'] not in seen:
            chosen.append(ex); seen.add(ex['root_id'])
        if len(chosen) == 24:
            break
    cohorts['long_tail'] = chosen
    candidates = [e for e in all_segments if .35 <= e['duration'] <= 2.25]
    dense = []; seen = set()
    for ex in sorted(candidates, key=lambda e: (len(e['graph']['cells']), len(e['trajectory'])), reverse=True):
        if ex['root_id'] not in seen:
            dense.append(ex); seen.add(ex['root_id'])
        if len(dense) == 32:
            break
    cohorts['dense'] = dense
    cohorts['representative'] = [cohorts[name][i] for i in range(16) for name in ('around0.5s','around1s','around2s') if i < len(cohorts[name])]
    for name, value in [('single',cohorts), ('multi',groups)]:
        p = dest/(name+'.json.gz')
        with gzip.open(p, 'wt') as f:
            json.dump(value, f)
        specs.append(dict(name=name, path=str(p), sha256=sha(p)))
    dump(dest/'manifest.json', dict(files=specs, single_counts={k:len(v) for k,v in cohorts.items()}, multi_groups=len(groups),
                                   source_manifest_sha256=sha(source), source=snapshot()))


class BaselineComplete(DeltaPredictor):
    """Latest original checkpoint including its deterministic future updater."""
    def predict(self, *, partial_graph, trajectory):
        tick = time.perf_counter(); delta = super().predict(partial_graph=partial_graph, trajectory=trajectory)
        before = time.perf_counter(); delta['future_partial_graph'] = apply_graph_delta(partial_graph, delta)
        self.last_profile.update(update_seconds=time.perf_counter()-before, total_seconds=time.perf_counter()-tick)
        return delta

    def predict_segments(self, *, partial_graph, trajectories):
        tick = time.perf_counter(); deltas = super().predict_batch([dict(partial_graph=partial_graph, trajectory=p) for p in trajectories])
        before = time.perf_counter()
        for d in deltas:
            d['future_partial_graph'] = apply_graph_delta(partial_graph, d)
        self.last_batch_profile.update(update_seconds=time.perf_counter()-before, total_seconds=time.perf_counter()-tick, materialized_future_graphs=len(deltas))
        return deltas


def inputs(name):
    manifest = json.loads((OUT/'benchmark_inputs/manifest.json').read_text())
    spec = next(s for s in manifest['files'] if s['name'] == name)
    assert sha(spec['path']) == spec['sha256']
    with gzip.open(spec['path'], 'rt') as f:
        return json.load(f)


def models(checkpoint):
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    return dict(baseline=BaselineComplete(CHECKPOINT),candidate=SegmentPredictor(checkpoint,execution='optimized',future_format='native',device=DEVICE,feature_threads=FEATURE_THREADS,shared_crops=SHARED_CROPS))


def single(checkpoint, dest):
    workload = inputs('single'); loaded = {}; modeldict = {}
    for name, cls, path in [('baseline',BaselineComplete,CHECKPOINT), ('candidate',SegmentPredictor,checkpoint)]:
        tick=time.perf_counter()
        modeldict[name]=cls(path) if name=='baseline' else cls(path,execution='optimized',future_format='native',device=DEVICE,feature_threads=FEATURE_THREADS,shared_crops=SHARED_CROPS)
        loaded[name]=time.perf_counter()-tick
    records = []; startup = []; holders = {name:None for name in modeldict}
    for label, examples in workload.items():
        for index, ex in enumerate(examples):
            graph = ex['graph']; trajectory = ex['trajectory']
            for name, model in modeldict.items():
                tick = time.perf_counter(); holders[name] = model.predict(partial_graph=graph, trajectory=trajectory)
                startup.append(dict(model=name, workload=label, root_id=ex['root_id'], seconds=time.perf_counter()-tick))
            for condition in ('map_cold', 'retained_cache'):
                for repeat in range(5):
                    for name in (('baseline','candidate') if (index+repeat)%2 == 0 else ('candidate','baseline')):
                        model = modeldict[name]
                        if condition == 'map_cold':
                            model.clear_cache()
                        gc_index = len(GC); cpu = time.process_time(); tick = time.perf_counter()
                        holders[name] = model.predict(partial_graph=graph, trajectory=trajectory)
                        end = time.perf_counter(); process_cpu = time.process_time()-cpu
                        delta = holders[name]; before_wire = time.perf_counter(); encoded = wire(delta); wire_seconds = time.perf_counter()-before_wire
                        records.append(dict(model=name, workload=label, condition=condition, repeat=repeat,
                            root_id=ex['root_id'], scene_id=ex['scene_id'], duration=ex['duration'], poses=len(trajectory),
                            known_cells=len(graph['cells']), candidates=len(delta['candidate_cells']), updates=len(delta['static_updates']),future_format=delta.get('future_format','snapshot'),
                            total_seconds=end-tick, process_cpu_seconds=process_cpu, cpu_utilization_percent=100*process_cpu/(end-tick),
                            wire_seconds=wire_seconds, wire_bytes=len(encoded), profile=dict(model.last_profile),
                            gc_events=[g for g in GC[gc_index:] if g['start']<end and g['end']>tick],
                            wire_gc_events=[g for g in GC[gc_index:] if g['start']>=before_wire],
                            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    summary = {}
    for name in modeldict:
        summary[name] = {}
        for label in workload:
            summary[name][label] = {}
            for condition in ('map_cold','retained_cache'):
                rows = [r for r in records if r['model'] == name and r['workload'] == label and r['condition'] == condition]
                profile = {k: float(np.mean([r['profile'][k] for r in rows if r['profile'].get(k) is not None]))*1000
                           for k in set().union(*(r['profile'] for r in rows)) if k.endswith('_seconds') and any(r['profile'].get(k) is not None for r in rows)}
                summary[name][label][condition] = dict(**quantiles([r['total_seconds'] for r in rows]), calls=len(rows),
                    mean_process_cpu_ms=1000*float(np.mean([r['process_cpu_seconds'] for r in rows])),
                    mean_cpu_utilization_percent=float(np.mean([r['cpu_utilization_percent'] for r in rows])),
                    mean_wire_ms=1000*float(np.mean([r['wire_seconds'] for r in rows])), mean_components_ms=profile,
                    mean_execution_counts={key:float(np.mean([r['profile'].get(key,0) for r in rows])) for key in ('ray_boundary_fallback_rows','opportunity_boundary_fallback_rows','near_tie_reference_blocks')},
                    future_formats={fmt:sum(r['future_format']==fmt for r in rows) for fmt in sorted({r['future_format'] for r in rows})})
    dump(dest/'calls.json', records)
    dump(dest/'metrics.json', dict(workloads=summary, checkpoint_load_seconds=loaded, startup_first_calls=startup,
        gc_enabled=gc.isenabled(), complete_graph_inside_primary_timer=True, sparse_wire_separate=True,
        candidate_targets=dict(representative30ms=all(summary['candidate']['representative'][c]['p95_ms']<=30 for c in ('map_cold','retained_cache')),
                               dense40ms=all(summary['candidate']['dense'][c]['p95_ms']<=40 for c in ('map_cold','retained_cache'))),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,device=DEVICE,feature_threads=FEATURE_THREADS,
        cuda_peak_allocated_bytes=max((r['profile'].get('cuda_peak_allocated_bytes',0) for r in records),default=0),
        cuda_peak_reserved_bytes=max((r['profile'].get('cuda_peak_reserved_bytes',0) for r in records),default=0),
        cuda_feature_kernel_identities=sorted({r['profile']['cuda_feature_kernel_identity'] for r in records if 'cuda_feature_kernel_identity' in r['profile']}),
        memory_scope='joint process lifetime high-water RSS including both models, all decoded workloads, startup and JIT; not attributable per-model'))


def multi(checkpoint, dest):
    groups = inputs('multi'); modeldict = models(checkpoint); records = []; equivalence = []
    holders = {name:None for name in modeldict}
    for i, group in enumerate(groups):
        graph = group['graph']; paths = [p['trajectory'] for p in group['paths']]
        for name, model in modeldict.items():
            singles = [model.predict(partial_graph=graph, trajectory=p) for p in paths]
            for k in (1,2,4,8):
                batch = model.predict_segments(partial_graph=graph, trajectories=paths[:k])
                for a, b in zip(singles, batch):
                    for field in ('candidate_cells','probabilities','states','heights'):
                        np.testing.assert_array_equal(a[field], b[field])
                    assert a['future_partial_graph'] == b['future_partial_graph']
                equivalence.append(dict(model=name, root_id=group['root_id'], k=k, exact=True))
            del singles, batch
        for k in (1,2,4,8):
            for condition in ('map_cold','retained_cache'):
                for repeat in range(3):
                    for name in (('baseline','candidate') if (i+repeat)%2 == 0 else ('candidate','baseline')):
                        model = modeldict[name]
                        if condition == 'map_cold':
                            model.clear_cache()
                        else:
                            model.predict_segments(partial_graph=graph, trajectories=paths[:k])
                        cpu = time.process_time(); tick = time.perf_counter()
                        holders[name] = model.predict_segments(partial_graph=graph, trajectories=paths[:k])
                        end = time.perf_counter(); process_cpu = time.process_time()-cpu
                        records.append(dict(model=name, k=k, condition=condition, repeat=repeat, root_id=group['root_id'], scene_id=group['scene_id'],
                            total_seconds=end-tick, per_segment_seconds=(end-tick)/k, process_cpu_seconds=process_cpu,
                            cpu_utilization_percent=100*process_cpu/(end-tick), batch_profile=dict(model.last_batch_profile),
                            materialized_graphs=len(holders[name]), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    summary = {name:{condition:{str(k):dict(**quantiles([r['total_seconds'] for r in records if r['model']==name and r['condition']==condition and r['k']==k]),
        mean_amortized_ms=1000*float(np.mean([r['per_segment_seconds'] for r in records if r['model']==name and r['condition']==condition and r['k']==k])))
        for k in (1,2,4,8)} for condition in ('map_cold','retained_cache')} for name in modeldict}
    dump(dest/'calls.json', records); dump(dest/'equivalence.json', equivalence)
    dump(dest/'metrics.json', dict(workloads=summary, groups=len(groups), no_padding=True, all_future_graphs_constructed_inside_timer=True,
        exact_single_batch_equivalence=all(e['exact'] for e in equivalence), gc_enabled=gc.isenabled()))


def worker_service(checkpoint,groups,core,queue,start,repetitions,device,condition):
    os.sched_setaffinity(0, {core})
    model=SegmentPredictor(checkpoint,cache_size=8,execution='optimized',future_format='native',device=device,feature_threads=1,shared_crops=SHARED_CROPS)
    jobs = [(g['graph'],[p['trajectory'] for p in g['paths'][:4]]) for g in groups]
    tick = time.perf_counter()
    for graph, paths in jobs:
        model.predict_segments(partial_graph=graph, trajectories=paths)
    init = time.perf_counter()-tick
    queue.put(dict(ready=True, pid=os.getpid(), core=core, startup_seconds=init)); start.wait()
    start_memory=resident_memory();device_memory={}
    if model.cuda is not None:
        free,total=torch.cuda.mem_get_info(model.cuda.device)
        device_memory=dict(global_used_at_service_barrier_bytes=total-free,total_device_bytes=total,
            scope='whole device after every worker warmed; includes CUDA contexts and any unrelated allocations; not a per-worker allocation count')
    cpu = time.process_time(); tick = time.perf_counter(); latencies = []; holder = None
    for _ in range(repetitions):
        for graph, paths in jobs:
            before=time.perf_counter()
            if condition=='map_cold':model.clear_cache()
            holder=model.predict_segments(partial_graph=graph,trajectories=paths);latencies.append(time.perf_counter()-before)
    wall = time.perf_counter()-tick; used = time.process_time()-cpu
    memory = resident_memory()
    rounds = repetitions*len(jobs)
    queue.put(dict(ready=False, pid=os.getpid(), core=core, wall_seconds=wall, cpu_seconds=used,
                   rounds=rounds, segments=rounds*4, graphs=rounds*4, latencies=latencies,
                   workload_root_ids=[g['root_id'] for g in groups],resident_memory=memory,
                   initialized_resident_memory=start_memory,device_memory=device_memory,
                   cuda_peak_allocated_bytes=model.cuda.last_profile.get('cuda_peak_allocated_bytes',0) if model.cuda is not None else 0,
                   cuda_peak_reserved_bytes=model.cuda.last_profile.get('cuda_peak_reserved_bytes',0) if model.cuda is not None else 0,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


def resident_memory():
    result={}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        parts=line.split()
        if len(parts)>=2 and parts[0] in ('Rss:','Pss:','Private_Clean:','Private_Dirty:'):result[parts[0][:-1]]=int(parts[1])*1024
    return result


def physical_cores():
    available = os.sched_getaffinity(0); found = {}
    for cpu in sorted(available):
        top = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
        key = ((top/'physical_package_id').read_text(), (top/'core_id').read_text())
        found.setdefault(key, cpu)
    return list(found.values())


def concurrency(checkpoint, dest):
    groups = inputs('multi')[:24]; records = []; cores = physical_cores()
    if len(groups) != 24:
        raise ValueError('concurrency requires24frozen workload groups')
    if torch.cuda.is_initialized():raise RuntimeError('concurrency requires a fresh parent before worker CUDA initialization')
    for condition,workers in [(c,n) for c in ('retained_cache','map_cold') for n in (4,8,12)]:
        ctx = mp.get_context('fork'); queue = ctx.Queue(); start = ctx.Event()
        jobs=[ctx.Process(target=worker_service,args=(checkpoint,groups[i::workers],cores[i],queue,start,128,DEVICE,condition)) for i in range(workers)]
        for process in jobs:
            process.start()
        startup = [queue.get(timeout=300) for _ in jobs]; assert all(s['ready'] for s in startup)
        coordinator_memory=resident_memory()
        tick = time.perf_counter(); start.set(); rows = [queue.get(timeout=600) for _ in jobs]
        for process in jobs:
            process.join(); assert process.exitcode == 0
        wall = time.perf_counter()-tick; calls = sum(r['rounds'] for r in rows); segments = sum(r['segments'] for r in rows)
        records.append(dict(workers=workers,condition=condition,device=DEVICE,k=4,calls=calls,segments=segments,wall_seconds=wall,
            aggregate_complete_k4_calls_per_second=calls/wall, aggregate_segments_per_second=segments/wall,
            aggregate_cpu_utilization_percent=100*sum(r['cpu_seconds'] for r in rows)/wall,
            sum_worker_peak_rss_bytes=sum(r['peak_rss_bytes'] for r in rows),
            sum_worker_pss_bytes=sum(r['resident_memory']['Pss'] for r in rows),
            sum_worker_private_bytes=sum(r['resident_memory'].get('Private_Clean',0)+r['resident_memory'].get('Private_Dirty',0) for r in rows),
            initialized_service_pss_including_coordinator_bytes=coordinator_memory['Pss']+sum(r['initialized_resident_memory']['Pss'] for r in rows),
            coordinator_memory_at_service_barrier=coordinator_memory,
            global_device_used_at_service_barrier_bytes=max((r['device_memory'].get('global_used_at_service_barrier_bytes',0) for r in rows),default=0),
            sum_cuda_peak_allocated_bytes=sum(r['cuda_peak_allocated_bytes'] for r in rows),
            sum_cuda_peak_reserved_bytes=sum(r['cuda_peak_reserved_bytes'] for r in rows),
            call_latency=quantiles([t for r in rows for t in r['latencies']]), startup=startup, workers_raw=rows))
    dump(dest/'metrics.json', dict(measurements=records, pinned_distinct_physical_cores=True, torch_threads_per_worker=1,
        condition='same24Root workloads,128K4calls perRoot at every concurrency; retained cache_size8 warms every assigned graph; initialized service; IPC included aggregate wall',
        memory_note='summed peak RSS is an upper bound that double-counts fork-shared pages; summed PSS and private resident bytes also reported',
        scope='Stage2 capacity only; no RL decisions/simulator/policy/optimizer throughput measured', gc_enabled=gc.isenabled()))


def cold_call(checkpoint):
    examples = inputs('single')['representative']; ex = examples[0]
    tick=time.perf_counter();model=SegmentPredictor(checkpoint,execution='optimized',future_format='native',device=DEVICE,feature_threads=FEATURE_THREADS,shared_crops=SHARED_CROPS);loaded=time.perf_counter()
    out = model.predict(partial_graph=ex['graph'], trajectory=ex['trajectory']); end = time.perf_counter()
    print(json.dumps(dict(checkpoint_load_seconds=loaded-tick, first_complete_call_seconds=end-loaded,
                         device_properties=model.cuda.device_info if model.cuda is not None else None,
                         future_cells=len(out['future_partial_graph']['cells']), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)))


def cold(checkpoint, dest):
    rows = []
    for i in range(5):
        tick = time.perf_counter()
        result = subprocess.run([sys.executable, str(BASE/'scripts/stage2_segment_run.py'), 'eventwm_graph.stage2_segment.benchmark',
                                 '--kind','cold_call','--checkpoint',str(checkpoint),'--device',DEVICE,'--feature-threads',str(FEATURE_THREADS),*([] if SHARED_CROPS else ['--no-shared-crops'])],capture_output=True,text=True,check=True)
        rows.append(dict(process_wall_seconds=time.perf_counter()-tick, **json.loads(result.stdout.strip().splitlines()[-1])))
    dump(dest/'metrics.json', dict(calls=rows, condition='fresh process with retained on-disk Numba cache; import/startup wall measured externally',
        process_wall=quantiles([r['process_wall_seconds'] for r in rows]), first_complete_call=quantiles([r['first_complete_call_seconds'] for r in rows])))
    cache = dest/'empty_jit_cache'; cache.mkdir()
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache)); tick = time.perf_counter()
    result = subprocess.run([sys.executable, str(BASE/'scripts/stage2_segment_run.py'), 'eventwm_graph.stage2_segment.benchmark',
                             '--kind','cold_call','--checkpoint',str(checkpoint),'--device',DEVICE,'--feature-threads',str(FEATURE_THREADS),*([] if SHARED_CROPS else ['--no-shared-crops'])],env=env,capture_output=True,text=True,check=True)
    dump(dest/'empty_jit_cache.json',dict(process_wall_seconds=time.perf_counter()-tick,**json.loads(result.stdout.strip().splitlines()[-1]),
        condition='fresh process, initially empty isolated on-disk Numba cache; compilation included in first complete call'))


def integrated(checkpoint, dest):
    """Unchanged original gate: actual arrival +40worlds/64knots +full Query.

    Uses the same128 evenly spaced dev Roots, max-length actual Query per Root,
    representative even-index64 and largest-map32 dense cases, five repeats.
    Stage1 source and checkpoint are verified by the frozen loader, not trained.
    """
    from eventwm_graph.stage2_quality.integration import load_stage1
    from eventwm_graph.stage2.data import branches
    stage1, source = load_stage1()
    reader = importlib.import_module('eventwm_graph.l2_two_stage.data')
    rows, manifest = source_rows('dev'); examples = []
    for i in np.linspace(0, len(rows)-1, 128, dtype=int):
        row = rows[int(i)]; ex = max(list(branches(row)), key=lambda e: e['executed_nodes']); examples.append((row, ex))
    workloads = dict(representative=examples[::2], dense=sorted(examples, key=lambda pair:(len(pair[1]['graph']['cells']),pair[1]['executed_nodes']),reverse=True)[:32])
    modeldict = models(checkpoint); records = []; startup = []; failures = []; holders = {n:None for n in modeldict}

    def call(model, context, ob, action, path):
        tick = time.perf_counter()
        people = stage1.predict(context, ob, executed_action=action, horizon=3., worlds=40)
        human_done = time.perf_counter(); actual = stage1.last_context
        deltas = model.predict_segments(partial_graph=actual.partial_graph, trajectories=[path])
        future=[d['future_partial_graph'].to_snapshot() if hasattr(d['future_partial_graph'],'to_snapshot') else d['future_partial_graph'] for d in deltas]
        done = time.perf_counter()
        return dict(human_worlds=people,graph_deltas=deltas,hypothetical_graphs=future,actual_context=actual),dict(stage1_seconds=human_done-tick,
            stage2_complete_seconds=done-human_done, total_seconds=done-tick, stage1_calls=1, worlds=40, knots=64)

    for label, items in workloads.items():
        for index, (row, ex) in enumerate(items):
            folder = Path(row['folder']); legal = reader.read(folder/'legal_root.json.gz')
            outcomes = reader.read(folder/'outcomes.json.gz'); outcome = next(o for o in outcomes if o['query_id']==ex['query_id'])
            nodes = reader.read(folder/'future_observations.json.gz'); branch = [nodes[i] for i in outcome['node_ids']]
            if not branch:
                failures.append(dict(root_id=row['root_id'],reason='no actual arrival observation')); continue
            tick = time.perf_counter()
            stage1.predict(legal['history'],legal['history'][-1],legal['partial_graph'],actions=legal['history_actions'],horizon=3.,worlds=40)
            context = stage1.last_context.with_association(); init = time.perf_counter()-tick
            ob = branch[0]['observation']; action = branch[0]['action']
            path = np.asarray([[n['observation']['timestamp'],*n['observation']['robot_pose']] for n in branch],np.float64)
            for name, model in modeldict.items():
                holders[name], profile = call(model,context,ob,action,path)
                startup.append(dict(model=name,root_id=row['root_id'],context_initialization_seconds=init,first_profile=profile))
            for condition in ('map_cold','retained_cache'):
                for repeat in range(5):
                    for name in (('baseline','candidate') if (index+repeat)%2==0 else ('candidate','baseline')):
                        model = modeldict[name]
                        if condition == 'map_cold':
                            model.clear_cache()
                        gc_index = len(GC); cpu = time.process_time(); tick = time.perf_counter()
                        holders[name], profile = call(model,context,ob,action,path)
                        native = time.perf_counter(); result = holders[name]; wire(result['graph_deltas'][0]); end = time.perf_counter()
                        people = result['human_worlds']; actual = result['actual_context']
                        if people.positions.shape[:2] != (40,64) or float(actual.history[-1]['timestamp']) != float(path[0,0]) or len(actual._positive_history.timestamps) != len(context._positive_history.timestamps)+1:
                            raise ValueError('integrated call omitted required real observation ingestion or Stage1 budget')
                        records.append(dict(model=name,workload=label,condition=condition,repeat=repeat,root_id=row['root_id'],scene_id=row['scene_id'],
                            total_seconds=end-tick,native_seconds=native-tick,process_cpu_seconds=time.process_time()-cpu,
                            sparse_wire_seconds=end-native,profile=profile,stage2_profile=dict(model.last_profile or {}),
                            gc_events=GC[gc_index:],known_cells=len(actual.partial_graph['cells']),poses=len(path),
                            stage1_old_timestamp=float(context.history[-1]['timestamp']),actual_arrival_timestamp=float(actual.history[-1]['timestamp'])))
        print(json.dumps(dict(integrated_workload=label,calls=len(records))),flush=True)
    summary = {name:{label:{condition:dict(native=quantiles([r['native_seconds'] for r in records if r['model']==name and r['workload']==label and r['condition']==condition]),
        with_sparse_delta_wire=quantiles([r['total_seconds'] for r in records if r['model']==name and r['workload']==label and r['condition']==condition]))
        for condition in ('map_cold','retained_cache')} for label in workloads} for name in modeldict}
    gate = not failures and all(summary['candidate'][label][condition]['with_sparse_delta_wire']['p95_ms']<=100 for label in workloads for condition in ('map_cold','retained_cache'))
    dump(dest/'calls.json',records)
    dump(dest/'metrics.json',dict(workloads=summary,gate_passed=gate,gate_p95_ms=100,failures=failures,startup=startup,
        original_workload_preserved=True,source_stage1=str(source),source_manifest_sha256=sha(manifest),gc_enabled=gc.isenabled(),
        gate_boundary='actual frozen Stage1 arrival ingestion40worlds64knots3seconds + supplied fullQuery Stage2 + materialized graph + sparse wire',
        stage1_retrained=False,layer25_compilation_measured=False,rl_training_measured=False,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))


DEVICE='cuda:1'
FEATURE_THREADS=1
SHARED_CROPS=True

def benchmark(kind,checkpoint=None,name=None,device='cuda:1',feature_threads=1,shared_crops=True):
    global DEVICE,FEATURE_THREADS,SHARED_CROPS
    DEVICE=device;FEATURE_THREADS=feature_threads;SHARED_CROPS=shared_crops
    if kind == 'inputs':
        return build_inputs()
    if kind == 'cold_call':
        return cold_call(checkpoint)
    if not gc.isenabled():
        raise ValueError('primary benchmarks require normal GC')
    environment=host_state()
    if environment['competing_research']:raise RuntimeError('acceptance benchmark has competing research jobs: '+repr(environment['competing_research']))
    gc.callbacks.append(gc_event)
    dest = OUT/'benchmarks'/(name or kind); dest.mkdir(parents=True, exist_ok=False)
    dump(dest/'protocol.json', dict(kind=kind, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        source=snapshot(), reference_runtime=reference_identity(), input_manifest_sha256=sha(OUT/'benchmark_inputs/manifest.json'), gc_enabled=gc.isenabled(),
        device=DEVICE,feature_threads=FEATURE_THREADS,torch_threads=torch.get_num_threads(),numba_thread_cap=os.environ.get('NUMBA_NUM_THREADS'),
        shared_crops=SHARED_CROPS,crop_cache_per_graph=4,shared_union_refs_per_graph=int(SHARED_CROPS),
        numba_threading_layer=os.environ.get('NUMBA_THREADING_LAYER'),concurrency_feature_threads=1,
        host=environment,
        timer='in-memory supplied graph through full materialized future graph and prepared output; optional sparse wire separate',
        train_or_evaluation_contention='no competing extraction/fitting/evaluation/profile jobs at benchmark entry'))
    globals()[kind](checkpoint, dest)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--kind', choices=['inputs','single','multi','concurrency','cold','cold_call','integrated'], required=True)
    p.add_argument('--checkpoint');p.add_argument('--name');p.add_argument('--device',default='cuda:1');p.add_argument('--feature-threads',type=int,default=1)
    p.add_argument('--shared-crops',action=argparse.BooleanOptionalAction,default=True)
    benchmark(**vars(p.parse_args()))
