"""Complete Layer 2 (Stage 1 + Stage 2) latency for K = 1/4/6 (and 8, the largest recorded set) and multi-process throughput.

Workloads: the ORIGINAL 32 Stage 2 multi-query benchmark Roots (recorded Query paths, frozen
inputs file, SHA-256 checked); dense = upper half by known cells (the Stage 2 convention). The
decision state is the Root (legal history, legal graph, 256 px range frames); the candidates are
the first K recorded paths, exactly as in the Stage 2 benchmark.

Timed: one complete ``Layer2WorldModel.predict`` call -> candidate validation, Stage 1 legal-state
update + network + 40-world sampling + 64-time output, Stage 2 complete call (graph preparation,
legal voxel map, features, network, decode, sparse increments, complete hypothetical future graphs),
per-candidate joint human-interaction reductions and output assembly. Excluded: reading the
frozen inputs, sensor acquisition, output serialisation.

Modes
  cold    : Stage 2 caches cleared, Stage 1 stream reset (full legal-history replay), no bank reuse.
  online  : Stage 1 stream holds the pre-arrival state (everything but the current frame, restored
            outside the timer), Stage 2 caches cleared: the steady per-decision cost inside an episode.
  same_state_new_candidates : the same decision state was already queried (Stage 2 legal map and the
            human bank reused); K new candidates.
  stage2_only_cold / stage1_only_online : one stage alone under the same load (attribution).
Each mode is measured with Stage 1 overlapped on a worker thread and sequentially.

    python scripts/layer2_run.py eventwm_graph.layer2.benchmark latency NAME [--repetitions 3]
    python scripts/layer2_run.py eventwm_graph.layer2.benchmark throughput NAME --procs 1 2 4 --seconds 60
"""
import argparse
import copy
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'output/layer2/benchmark'
WORKLOADS = Path('/home/haodong/workspace/projects/eventwm-falcon-stage2-segment-20261024/output/stage2_segment_20261024/benchmark_inputs/multi.json.gz')
WORKLOADS_SHA256 = 'a3f7b8d35ed8591bdeab166dab17d0b5d6bf0a56211868029bdc85959fb68869'


def workloads():
    raw = WORKLOADS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != WORKLOADS_SHA256:
        raise ValueError('benchmark workloads changed')
    from .devdata import dev_rows, DevRoot
    rows = {r['root_id']: r for r in dev_rows()}
    out = []
    for ex in json.loads(gzip.decompress(raw)):
        r = DevRoot(rows[ex['root_id']])
        state = r.decision_state()
        if ex['graph'] != state.partial_graph:
            raise ValueError('benchmark graph differs from the legal Root graph')
        paths = [np.asarray(p['trajectory'], np.float64) for p in ex['paths']]
        out.append(dict(root_id=ex['root_id'], state=state, paths=paths, known_cells=len(ex['graph']['cells']),
                        frames=len(state.frames)))
    return out


def stats(x):
    x = np.asarray(x, np.float64)
    return dict(n=int(len(x)), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)), max=float(x.max()), mean=float(x.mean()))


def _sync(device):
    import torch
    torch.cuda.synchronize(device)


def _prime_stream(l2, w):
    """Stage 1 stream with every frame but the current one ingested; returns a restorable copy."""
    s = w['state']
    model = l2.humans.model
    model.streams.pop(s.stream, None)
    if len(s.frames) > 1:
        model.predict(list(s.frames[:-2]), s.frames[-2], None, stream=s.stream)
    return copy.deepcopy(model.streams.get(s.stream))


def latency(name, device='cuda:0', repetitions=3, ks=(1, 4, 6, 8)):
    import torch
    from .model import Layer2WorldModel
    work = workloads()
    order = sorted(range(len(work)), key=lambda i: work[i]['known_cells'])
    dense = set(order[len(order)//2:])
    models = {}
    tick = time.perf_counter()
    models[True] = Layer2WorldModel.from_release(map_device=device, overlap=True)
    construct_ms = (time.perf_counter()-tick)*1000
    # the sequential variant shares the same backends (no second copy of either model)
    models[False] = Layer2WorldModel(models[True].humans, models[True].maps, overlap=False, release=models[True].release)
    l2 = models[True]
    tick = time.perf_counter(); l2.predict(work[0]['state'], work[0]['paths'][:6]); _sync(device)
    startup = dict(construct_ms=construct_ms, first_complete_K6_ms=(time.perf_counter()-tick)*1000)
    for w in work:    # warm every frame-size bucket / kernel once (not timed)
        models[True].reset(); models[True].predict(w['state'], w['paths'][:6])
    load_before = os.getloadavg()
    top = subprocess.check_output(['ps', '-eo', 'pid,stat,etime,pcpu,pmem,args', '--sort=-pcpu'], text=True).splitlines()[:30]
    rows = []

    def record(mode, overlap, i, k, rep, ms, m):
        prof = m.last_profile if m is not None else {}
        rows.append(dict(mode=mode, overlap=overlap, cohort='dense' if i in dense else 'representative', group=i, k=k, repetition=rep, ms=ms,
                         stage1_ms=prof.get('stage1_ms'), stage2_ms=prof.get('stage2_ms'), interaction_ms=prof.get('interaction_ms'),
                         stage2_profile=prof.get('stage2'), stage1_profile=prof.get('stage1')))

    for rep in range(repetitions):
        for i in (order if rep % 2 else order[::-1]):
            w = work[i]; s = w['state']
            saved = _prime_stream(l2, w)
            for k in ks:
                paths = w['paths'][:k]
                for overlap in ((True, False) if rep % 2 else (False, True)):
                    m = models[overlap]
                    # cold
                    m.reset(); gc.collect(); _sync(device)
                    t = time.perf_counter(); out = m.predict(s, paths); _sync(device); ms = (time.perf_counter()-t)*1000
                    assert len(out.candidates) == k and all(c.future_graph is not None for c in out.candidates)
                    record('cold', overlap, i, k, rep, ms, m)
                    # online (pre-arrival Stage 1 stream, new decision state for Stage 2)
                    m.reset(); l2.humans.model.streams[s.stream] = copy.deepcopy(saved); gc.collect(); _sync(device)
                    t = time.perf_counter(); out = m.predict(s, paths); _sync(device); ms = (time.perf_counter()-t)*1000
                    record('online', overlap, i, k, rep, ms, m)
                    # same state, new candidates (Stage 2 legal map + human bank reused)
                    m.reset(); m.predict(s, [paths[0][:1]]); _sync(device)
                    t = time.perf_counter(); out = m.predict(s, paths); _sync(device); ms = (time.perf_counter()-t)*1000
                    record('same_state_new_candidates', overlap, i, k, rep, ms, m)
                # stage 2 alone (cold) and stage 1 alone (online) under the same load
                l2.maps.reset(); _sync(device)
                t = time.perf_counter(); l2.maps.predict(s, paths); _sync(device); ms = (time.perf_counter()-t)*1000
                record('stage2_only_cold', False, i, k, rep, ms, None)
            l2.humans.model.streams[s.stream] = copy.deepcopy(saved)
            t = time.perf_counter(); l2.humans.predict(s); ms = (time.perf_counter()-t)*1000
            record('stage1_only_online', False, i, 0, rep, ms, None)
            l2.humans.model.streams.pop(s.stream, None)
        print('repetition', rep+1, 'done', flush=True)
    summary = []
    for mode in ('cold', 'online', 'same_state_new_candidates', 'stage2_only_cold', 'stage1_only_online'):
        for overlap in (True, False):
            for k in sorted({r['k'] for r in rows if r['mode'] == mode}):
                for cohort in ('representative', 'dense', 'all'):
                    sel = [r for r in rows if r['mode'] == mode and r['overlap'] == overlap and r['k'] == k and (cohort == 'all' or r['cohort'] == cohort)]
                    if not sel:
                        continue
                    row = dict(mode=mode, overlap=overlap, k=k, cohort=cohort, total=stats([r['ms'] for r in sel]))
                    if sel[0]['stage1_ms'] is not None:
                        row.update(stage1=stats([r['stage1_ms'] for r in sel]), stage2=stats([r['stage2_ms'] for r in sel]),
                                   interaction=stats([r['interaction_ms'] for r in sel]))
                    summary.append(row)
    report = dict(name=name, device=device, gpu=torch.cuda.get_device_name(torch.device(device)), cpu_threads=torch.get_num_threads(),
                  stage1_sha256=l2.humans.checkpoint_sha256, stage2_sha256=l2.maps.checkpoint_sha256, lanes=l2.maps.lanes,
                  worlds=l2.humans.worlds, horizon=l2.humans.horizon, startup=startup, workloads=len(work), repetitions=repetitions,
                  load_average_before=load_before, load_average_after=os.getloadavg(), concurrent_processes=top,
                  max_reserved_gb=torch.cuda.max_memory_reserved(torch.device(device))/2**30,
                  measured=[dict(root_id=w['root_id'], known_cells=w['known_cells'], frames=w['frames'],
                                 path_seconds=[float(p[-1, 0]-p[0, 0]) for p in w['paths'][:6]]) for w in work],
                  summary=summary, calls=rows)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/(name+'.json')).write_text(json.dumps(report, indent=1, default=float))
    for s in summary:
        if s['cohort'] != 'all':
            print('%-26s overlap=%-5s K=%d %-14s P50 %7.2f  P95 %7.2f  max %7.2f' % (s['mode'], s['overlap'], s['k'], s['cohort'],
                                                                                    s['total']['p50'], s['total']['p95'], s['total']['max']))
    return report


# ------------------------------------------------------------------ throughput
def throughput_worker(name, device, seconds, k, index):
    """One process: online-mode decisions (K candidates each) over the 32 workloads for ``seconds``."""
    import torch
    from .model import Layer2WorldModel
    work = workloads()
    l2 = Layer2WorldModel.from_release(map_device=device, overlap=True)
    saved = []
    for w in work:
        saved.append(_prime_stream(l2, w))
        l2.reset(); l2.predict(w['state'], w['paths'][:k])     # warm-up, not counted
    print('READY', flush=True)
    sys.stdin.readline()           # start barrier
    done = 0; lat = []; start = time.perf_counter(); i = index
    while time.perf_counter()-start < seconds:
        w = work[i % len(work)]; i += 1
        l2.reset(); l2.humans.model.streams[w['state'].stream] = copy.deepcopy(saved[(i-1) % len(work)])
        t = time.perf_counter(); l2.predict(w['state'], w['paths'][:k]); _sync(device)
        lat.append((time.perf_counter()-t)*1000); done += 1
    wall = time.perf_counter()-start
    print('RESULT '+json.dumps(dict(index=index, decisions=done, seconds=wall, latency=stats(lat),
                                    max_reserved_gb=torch.cuda.max_memory_reserved(torch.device(device))/2**30)), flush=True)


def throughput(name, procs=(1, 2, 4), seconds=60., k=6, device='cuda:0'):
    results = []
    env = dict(os.environ)
    for p in procs:
        workers = [subprocess.Popen([sys.executable, str(ROOT/'scripts/layer2_run.py'), 'eventwm_graph.layer2.benchmark', 'throughput-worker',
                                     name, '--device', device, '--seconds', str(seconds), '--k', str(k), '--index', str(j)],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env) for j in range(p)]
        for w in workers:
            while True:
                line = w.stdout.readline()
                if not line or line.startswith('READY'):
                    break
        load = os.getloadavg()
        for w in workers:
            w.stdin.write('go\n'); w.stdin.flush()
        outs = []
        for w in workers:
            for line in w.stdout:
                if line.startswith('RESULT '):
                    outs.append(json.loads(line[7:]))
            w.wait()
        rate = sum(o['decisions']/o['seconds'] for o in outs)
        results.append(dict(processes=p, k=k, decisions_per_second=rate, decisions_per_day=rate*86400, load_average_start=load,
                            load_average_end=os.getloadavg(), workers=outs))
        print('procs %d: %.1f decisions/s (K=%d), per-call P50 %.1f ms' % (p, rate, k, np.median([o['latency']['p50'] for o in outs])), flush=True)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/(name+'_throughput.json')).write_text(json.dumps(dict(name=name, device=device, seconds=seconds, k=k, mode='online', results=results), indent=1))
    return results


if __name__ == '__main__':
    pa = argparse.ArgumentParser()
    sub = pa.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('latency'); a.add_argument('name'); a.add_argument('--device', default='cuda:0'); a.add_argument('--repetitions', type=int, default=3)
    b = sub.add_parser('throughput'); b.add_argument('name'); b.add_argument('--device', default='cuda:0'); b.add_argument('--procs', type=int, nargs='+', default=[1, 2, 4])
    b.add_argument('--seconds', type=float, default=60.); b.add_argument('--k', type=int, default=6)
    c = sub.add_parser('throughput-worker'); c.add_argument('name'); c.add_argument('--device', default='cuda:0'); c.add_argument('--seconds', type=float)
    c.add_argument('--k', type=int); c.add_argument('--index', type=int)
    x = pa.parse_args()
    if x.cmd == 'latency':
        latency(x.name, x.device, x.repetitions)
    elif x.cmd == 'throughput':
        throughput(x.name, x.procs, x.seconds, x.k, x.device)
    else:
        throughput_worker(x.name, x.device, x.seconds, x.k, x.index)
