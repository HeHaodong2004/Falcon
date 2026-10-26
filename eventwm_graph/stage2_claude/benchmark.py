"""Complete K=1/4/6 latency on the ORIGINAL 32 benchmark map workloads.

Same protocol as stage2_repair/range_benchmark.py: identical workloads and
paths, cold (all graph AND legal-map caches cleared), warm (identical
queries repeated), reusable map / new queries (only the starting pose is
primed), three repetitions with alternating order, dense = upper half by
known cells. Timed: input validation, content identities, graph preparation,
candidate universe, legal voxel map, query features, transfers, network,
probability/hard decode, sparse increment and complete hypothetical graphs.
Excluded: input deserialization, output serialization, sensor acquisition.
"""
import argparse
import gc
import json
import os
import subprocess
import time

import numpy as np
import torch

from eventwm_graph.stage2_repair.diagnose import OUT, save, sha
from eventwm_graph.stage2_repair.range_benchmark import workloads
from eventwm_graph.stage2_repair.benchmark import summary

DEST = OUT.parent/'claude_stage2'/'benchmark'


def run(checkpoint, name, device='cuda:1', repetitions=3, sensor_directory='range_benchmark', threshold=None, lanes=None):
    torch.set_num_threads(1)
    gc.enable()
    from .api import LegalVoxelPredictor
    work = workloads(sensor_directory)
    tick = time.perf_counter(); model = LegalVoxelPredictor(checkpoint, device=device, threshold=threshold, lanes=lanes); constructor = time.perf_counter()-tick

    def call(ex, packet, tracks, k, prime_map=False):
        paths = [np.asarray(p['trajectory'], np.float64) for p in ex['paths'][:k]]
        if prime_map:
            paths = [paths[0][:1]]
        result = model.predict_segments(partial_graph=ex['graph'], trajectories=paths, range_observations=packet, track_history=tracks)
        assert len(result) == (1 if prime_map else k) and all('future_partial_graph' in d for d in result)
        return result

    tick = time.perf_counter(); call(*work[0], 6); torch.cuda.synchronize(device)
    startup = dict(constructor_ms=constructor*1000, first_complete_K6_ms=(time.perf_counter()-tick)*1000)
    order = sorted(range(len(work)), key=lambda i: len(work[i][0]['graph']['cells']))
    dense = set(order[len(order)//2:])
    load = subprocess.check_output(['ps', '-eo', 'pid,ppid,stat,etime,pcpu,pmem,args', '--sort=-pcpu'], text=True).splitlines()[:25]
    load_before = os.getloadavg()
    rows = []
    for rep in range(repetitions):
        for i in order[::1 if rep % 2 else -1]:
            ex, packet, tracks = work[i]
            for k in [1, 4, 6]:
                for cache in ['cold', 'warm', 'map_warm_new_queries']:
                    if cache == 'cold':
                        model.clear_cache()
                    elif cache == 'map_warm_new_queries':
                        model.clear_cache(); call(ex, packet, tracks, k, prime_map=True)
                    else:
                        call(ex, packet, tracks, k)
                    torch.cuda.synchronize(device); tick = time.perf_counter()
                    result = call(ex, packet, tracks, k)
                    torch.cuda.synchronize(device); ms = (time.perf_counter()-tick)*1000
                    rows.append(dict(model='claude_voxel_unet', cohort='dense' if i in dense else 'representative', cache=cache, k=k, ms=ms,
                                     group=i, repetition=rep, candidates=[len(d['candidate_cells']) for d in result],
                                     updates=[len(d['static_updates']) for d in result], profile=dict(model.last_batch_profile)))
        print('REPETITION', rep+1, flush=True)
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), device=device, hardware=torch.cuda.get_device_name(torch.device(device)),
                  sensor_directory=sensor_directory, implementation='LegalVoxelPredictor', lanes=model.lanes, api_sha256=sha(os.path.join(os.path.dirname(__file__), 'api.py')),
                  precision='voxel/visibility kernels F32; U-Net AMP F16; candidate domain exact original F32/F64 semantics',
                  input_contract='LEGAL measured current/past Euclidean ranges + legal track history + current graph + supplied paths; same original 32 maps/paths/full original cells.',
                  scope='Complete input validation/content identity, graph preparation, candidate universe, legal voxel map, query features, transfers/synchronization, network, decode, sparse increment and complete native hypothetical graphs. Cold clears graph AND legal-map caches.',
                  excluded='Input file deserialization, output serialization, environment sensor acquisition.',
                  startup=startup, concurrent_load=load, load_average_before=load_before, load_average_after=os.getloadavg(),
                  summary=summary(rows), calls=rows,
                  measured_workloads=[dict(root_id=e['root_id'], known_cells=len(e['graph']['cells']), range_shape=list(np.shape(p['ranges'])),
                                           path_lengths=[len(q['trajectory']) for q in e['paths'][:6]]) for e, p, t in work])
    DEST.mkdir(parents=True, exist_ok=True)
    save(DEST/(name+'.json'), report)
    for s in report['summary']:
        print('%-22s %-15s %-22s K=%d  P50 %7.3f  P95 %7.3f  max %7.3f' % (s['model'], s['cohort'], s['cache'], s['k'], s['p50_ms'], s['p95_ms'], s['max_ms']))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--sensor-directory', default='range_benchmark')
    p.add_argument('--threshold', type=float)
    p.add_argument('--lanes', type=int)
    run(**vars(p.parse_args()))
