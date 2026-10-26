"""Original development universe through the public legal API.

Mirrors stage2_repair/range_full_evaluate.py exactly: original replay and
windows, legal ancestor sensor frames only, exact candidate-domain digest and
target counts asserted against the frozen original baseline receipts, every
materialized hypothetical graph verified, and the ORIGINAL summarize()
(equal scene/Root/window mean confusion -> rates, plus pooled and bootstrap).
Future observations are read only for targets.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np
import torch

from eventwm_graph.stage2_repair.diagnose import OUT, HF, read, save, sha, source_rows, member
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_repair.full_evaluate import BASELINE, codes, labels, summarize
from eventwm_graph.stage2_segment.data import replay
from eventwm_graph.stage2_segment.evaluate import groups
from eventwm_graph.stage2_segment.prepared import content_key

DEST = OUT.parent/'claude_stage2'/'evaluation'
MODEL = None


def gzread(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def evaluate_root(model, row, dest, checkpoint_sha, source_sha, sensor_directory):
    rid = row['root_id']
    path = dest/(rid+'.json.gz')
    sensor_path = OUT/sensor_directory/(rid+'.npz')
    receipt = read(sensor_path.with_suffix('.json'))
    if receipt['identity']['files'] != row['files']:
        raise ValueError('sensor source Root changed')
    if path.exists():
        prior = gzread(path)
        if prior['checkpoint_sha256'] != checkpoint_sha or prior['sensor_sha256'] != receipt['sha256'] or prior['evaluator_sha256'] != source_sha:
            raise ValueError('evaluation receipt identity changed')
        return prior
    if sha(sensor_path) != receipt['sha256']:
        raise ValueError('sensor arrays changed')
    legal = member(row, 'legal_root.json.gz'); nodes = member(row, 'future_observations.json.gz')
    track_rows = [o['tracks'] for o in legal['history']]+[n['observation']['tracks'] for n in nodes]
    with np.load(sensor_path) as z:
        sensors = {k: z[k] for k in z.files}
    frozen = gzread(BASELINE/(rid+'.json.gz'))
    old = {(e['start_node'], e['end_node']): e for e in frozen['records']}
    with np.load(BASELINE/(rid+'.probabilities.npz')) as z:
        old_y = z['y']; offsets = z['offsets']
    slices = {(e['start_node'], e['end_node']): slice(int(offsets[i]), int(offsets[i+1])) for i, e in enumerate(frozen['records'])}
    previous = gzread(OUT/'full_evaluation/selected_global_point'/(rid+'.json.gz'))
    windows = defaultdict(list); audit = None; tick = time.perf_counter()
    for ex in replay(row, 'mixed'):
        if 'audit' in ex:
            audit = ex['audit']; continue
        windows[ex['start_node']].append(ex)
    records = []
    for sid, examples in windows.items():
        graph = examples[0]['graph']; before = content_key(graph)
        take = ancestor_frames(sensors['parents'], sensors['history_count'], sid)
        packet = dict(ranges=sensors['depth'][take], origins=sensors['origins'][take], rotations=sensors['rotations'][take],
                      timestamps=sensors['timestamps'][take], sensor_protocol='ideal_current_jaw_range_dynamic_occlusion_1')
        results = model.predict_segments(partial_graph=graph, trajectories=[e['trajectory'] for e in examples],
                                         range_observations=packet, track_history=[track_rows[j] for j in take])
        for ex, delta in zip(examples, results):
            keys = delta['candidate_cells']; hard = delta['states']; y, th, missing = labels(keys, ex['target'])
            prev = old[ex['start_node'], ex['end_node']]
            assert hashlib.sha256(keys.astype('<i4').tobytes()).hexdigest() == prev['domain_sha256']
            assert len(keys) == prev['candidates'] and np.count_nonzero(y)+len(missing) == prev['true_counts'][0]
            np.testing.assert_array_equal(y, old_y[slices[ex['start_node'], ex['end_node']]])
            cm = np.bincount(3*y.astype(np.int64)+hard, minlength=9).reshape(3, 3)
            for state, height in missing:
                cm[state, 0] += 1
            future = delta['future_partial_graph']; columns = future.cell_columns
            prepared = model.prepare_graph(graph); base = model._ensure_future_base(prepared, graph).table
            fc = codes(columns['cell']); ki = np.searchsorted(fc, codes(base.keys))
            np.testing.assert_array_equal(columns['cell'][ki], base.keys)
            for name, value in [('state', base.states), ('height', base.heights), ('first_seen', base.first), ('last_seen', base.last)]:
                np.testing.assert_array_equal(columns[name][ki], value)
            chosen = np.flatnonzero(hard); ni = np.searchsorted(fc, codes(keys[chosen]))
            np.testing.assert_array_equal(columns['cell'][ni], keys[chosen]); np.testing.assert_array_equal(columns['state'][ni], hard[chosen])
            np.testing.assert_array_equal(columns['height'][ni], delta['heights'][chosen])
            assert len(columns['cell']) == len(base.keys)+len(chosen) and future['hypothetical'] is True
            correct = (y > 0) & (hard == y); height = [float(abs(delta['heights'][correct]-th[correct]).sum()), int(correct.sum())]
            records.append(dict(start_node=sid, end_node=ex['end_node'], cm=cm.tolist(), baseline_cm=prev['confusion'][6], groups=groups(ex),
                                new_cells=len(ex['target']), poses=len(ex['trajectory']), empty_prediction=not bool(len(chosen)),
                                candidates=len(keys), out_of_domain=len(missing), known_terminal_differences=ex['known_terminal_differences'],
                                height_error=height, future_verified=True))
        assert content_key(graph) == before
    assert audit is not None and len(records) == len(old)
    value = dict(root_id=rid, scene_id=row['scene_id'], human_count=previous['human_count'], checkpoint_sha256=checkpoint_sha,
                 sensor_sha256=receipt['sha256'], evaluator_sha256=source_sha, records=records, audit=audit,
                 seconds=time.perf_counter()-tick, privileged_runtime_inputs=False)
    temp = path.with_suffix('.tmp')
    temp.write_bytes(gzip.compress(json.dumps(value, separators=(',', ':')).encode())); temp.replace(path)
    model.clear_cache()
    return value


def _init(checkpoint, device, threshold):
    global MODEL
    torch.set_num_threads(1)
    from .api import LegalVoxelPredictor
    MODEL = LegalVoxelPredictor(checkpoint, device=device, threshold=threshold)


def _work(args):
    row, dest, checkpoint_sha, source_sha, sensor_directory = args
    return evaluate_root(MODEL, row, Path(dest), checkpoint_sha, source_sha, sensor_directory)


def run(checkpoint, name, device='cuda:1', split='full', workers=4, threshold=None, sensor_directory='full_range_branches', limit=0):
    rows, manifest = source_rows('dev')
    if split == 'pilot':
        ids = set(read(HF/'protocol.json')['pilot_selection_roots'])
        rows = [r for r in rows if r['root_id'] in ids]
    elif split != 'full':
        raise ValueError('split must be pilot or full')
    if limit:
        rows = rows[:limit]
    dest = DEST/name; dest.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha(checkpoint); source_sha = sha(__file__)
    identity = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha, source_manifest_sha256=sha(manifest), evaluator_sha256=source_sha,
                    api_sha256=sha(Path(__file__).with_name('api.py')), split=split, sensor_directory=sensor_directory, threshold=threshold, limit=limit,
                    role='LEGAL SENSOR EXTENSION; ORIGINAL development universe; original windows/domain/labels/summarize.')
    ip = dest/'identity.json'
    if ip.exists() and read(ip) != identity:
        raise ValueError('use a new evaluation destination')
    save(ip, identity)
    tick = time.perf_counter(); done = []
    jobs = [(r, str(dest), checkpoint_sha, source_sha, sensor_directory) for r in rows]
    with get_context('spawn').Pool(workers, initializer=_init, initargs=(str(checkpoint), device, threshold)) as pool:
        for value in pool.imap_unordered(_work, jobs):
            done.append(value)
            if len(done) % 50 == 0 or len(done) == len(rows):
                print(dict(completed=len(done), total=len(rows), seconds=time.perf_counter()-tick), flush=True)
    result = summarize(done)
    if split == 'full' and not limit:
        assert result['roots'] == 2110 and result['windows'] == 47128
    result.update(identity=identity, seconds=time.perf_counter()-tick)
    save(dest/'summary.json', result)
    h = result['hierarchical']['candidate']; p = result['pooled']['candidate']
    print('SUMMARY label-aware IoU %.6f revelation IoU %.6f | pooled %.6f | roots %d windows %d' % (
        h['label_aware_delta_iou'], h['revelation_iou'], p['label_aware_delta_iou'], result['roots'], result['windows']), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--split', default='full')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threshold', type=float)
    p.add_argument('--sensor-directory', default='full_range_branches')
    p.add_argument('--limit', type=int, default=0)
    run(**vars(p.parse_args()))
