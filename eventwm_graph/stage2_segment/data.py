"""Checksummed legal DAG replay, bounded windows and streamed cell sampling.

Each unique action node is replayed once. An undo journal restores its parent
before visiting a sibling; snapshots exist only at requested window starts.
No privileged member or simulator is used. Failures exclude the whole Root.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import orjson
from eventwm_graph.partial.map import PartialGraph
from eventwm_graph.stage2.data import source_rows
from eventwm_graph.stage2_quality.update import FastPartialGraph
from eventwm_graph.stage2_quality.features import Features, PreparedGraph
from .common import OUT, dump, sha, stable, snapshot


def read(row, name):
    raw = (Path(row['folder']) / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['files'][name]:
        raise ValueError('source checksum mismatch: ' + name)
    return orjson.loads(gzip.decompress(raw))


def load(row):
    legal = read(row, 'legal_root.json.gz')
    queries = read(row, 'queries.json.gz')
    outcomes = read(row, 'outcomes.json.gz')
    nodes = read(row, 'future_observations.json.gz')
    initial = legal['history'][-1]
    if initial['timestamp'] != legal['partial_graph']['timestamp']:
        raise ValueError('root clock mismatch')
    if legal['sensor']['max_range_m'] > 10.00001:
        raise ValueError('unsupported sensor range')
    if len(queries) != len(outcomes) or len(queries) != row['queries']:
        raise ValueError('Query cardinality mismatch')
    used = set()
    children = defaultdict(list)
    for i, node in enumerate(nodes):
        if node['node_id'] != i or (node['parent'] is not None and not 0 <= node['parent'] < i):
            raise ValueError('invalid DAG identity or parent')
        children[node['parent']].append(i)
    for q, o in zip(queries, outcomes):
        if q['query_id'] != o['query_id'] or q['decision_timestamp'] != initial['timestamp']:
            raise ValueError('Query identity or initial clock mismatch')
        ids = o['node_ids']; parent = None; t = initial['timestamp']
        actions = []
        for node_id in ids:
            n = nodes[node_id]; ob = n['observation']
            if n['parent'] != parent or not ob['timestamp'] > t:
                raise ValueError('noncausal ancestry')
            if not np.isfinite([ob['timestamp'], *ob['robot_pose']]).all():
                raise ValueError('nonfinite pose or time')
            actions.append(n['action']); used.add(node_id); parent = node_id; t = ob['timestamp']
        if actions != q['compiled']['actions'][:len(actions)] or actions != o['summary']['executed_actions']:
            raise ValueError('executed action mismatch')
        if len(ids) != o['summary']['executed_steps'] or len(ids) > len(q['compiled']['actions']):
            raise ValueError('early termination length mismatch')
        if o['summary']['termination'] == 'query_complete' and len(ids) != len(q['compiled']['actions']):
            raise ValueError('incomplete complete Query')
        pose = nodes[ids[-1]]['observation']['robot_pose'] if ids else initial['robot_pose']
        if o['successor']['timestamp'] != t or o['successor']['robot_pose'] != pose:
            raise ValueError('terminal pose or time mismatch')
    if used != set(range(len(nodes))):
        raise ValueError('orphan frames')
    return legal, queries, outcomes, nodes, children


def windows(legal, queries, outcomes, nodes, strategy):
    """Select only saved boundaries, without looking at future observations.

    mixed: full + 0.5/1/2-second windows at deterministic variable starts.
    partition: nonoverlapping nearest-one-second chunks + full Query.
    Shared windows are deduplicated by (starting node, terminal node).
    """
    found = {}
    for q, o in zip(queries, outcomes):
        ids = o['node_ids']; n = len(ids)
        times = np.array([legal['partial_graph']['timestamp']] +
                         [nodes[i]['observation']['timestamp'] for i in ids])
        chosen = {(0, n)}
        if n and strategy == 'mixed':
            for j, horizon in enumerate((.5, 1., 2.)):
                # Includes early, middle and late starts; never interpolates.
                rng = np.random.default_rng(stable(q['query_id'] + ':' + str(j)))
                start = int(rng.integers(0, n))
                end = min(n, max(start + 1, int(np.argmin(abs(times - (times[start] + horizon))))))
                chosen.add((start, end))
        elif n and strategy == 'partition':
            start = 0
            while start < n:
                end = min(n, max(start + 1, int(np.argmin(abs(times - (times[start] + 1.))))))
                chosen.add((start, end)); start = end
        elif strategy != 'mixed' and strategy != 'partition':
            raise ValueError('unknown segmentation')
        for a, b in sorted(chosen):
            key = (ids[a-1] if a else None, ids[b-1] if b else None)
            if key not in found:
                found[key] = dict(start_node=key[0], end_node=key[1],
                    node_ids=ids[a:b], query_id=q['query_id'], query_aliases=[],
                    start_step=a, end_step=b, termination=o['summary']['termination'])
            found[key]['query_aliases'].append(q['query_id'])
    return list(found.values())


def replay(row, strategy='mixed', verify_history=True):
    """Yield examples while maintaining exactly one branch-specific graph."""
    legal, queries, outcomes, nodes, children = load(row)
    root = legal['partial_graph']
    if verify_history:
        history = PartialGraph(root['resolution'], root['height'], root['clearance'])
        for ob in legal['history']:
            history.update(ob)
        restored = PartialGraph.restore(root)
        if (history.cells != restored.cells or history.timestamp != restored.timestamp or
                history.revision != restored.revision or history.visited != restored.visited or
                history.ego_support.tolist() != restored.ego_support.tolist()):
            raise ValueError('root history replay mismatch')
    selected = windows(legal, queries, outcomes, nodes, strategy)
    starts = {w['start_node'] for w in selected}
    ending = defaultdict(list)
    for w in selected:
        ending[w['end_node']].append(w)
    snapshots = {}
    graph = FastPartialGraph.restore(root)
    path = [[float(root['timestamp']), *legal['history'][-1]['robot_pose']]]
    ledger = Counter(roots=1, queries=len(queries), unique_frames=len(nodes), intermediate_states=len(nodes)+1,
                     candidate_segments=len(selected), query_ancestry_frames=sum(len(o['node_ids']) for o in outcomes))
    terminal_revisions = defaultdict(list)
    for o in outcomes:
        terminal_revisions[o['node_ids'][-1] if o['node_ids'] else None].append(o['successor']['graph_revision'])

    def emit(node_id):
        if any(r != graph.revision for r in terminal_revisions[node_id]):
            raise ValueError('terminal graph revision mismatch')
        if node_id in starts:
            snapshots[node_id] = graph.snapshot()
        for w in ending[node_id]:
            initial = snapshots[w['start_node']]
            known = {tuple(c['cell']): c for c in initial['cells']}
            target = {k: (c['state'], c['height']) for k, c in graph.cells.items() if k not in known}
            correction = sum(c['state'] != graph.cells[k]['state'] or
                             abs(c['height']-graph.cells[k]['height']) > 1e-8 for k, c in known.items())
            ex = dict(w, root_id=row['root_id'], scene_id=row['scene_id'], graph=initial,
                      trajectory=np.asarray(path[len(path)-len(w['node_ids'])-1:], np.float64).copy(),
                      target=target, known_terminal_differences=correction, segments_in_root=len(selected))
            ledger['new_cells_sum'] += len(target)
            ledger['known_terminal_differences_sum'] += correction
            yield ex

    def visit(node_id):
        node = nodes[node_id]; ob = node['observation']; now = ob['timestamp']
        if now <= graph.timestamp:
            raise ValueError('replay clock regressed')
        touched = {tuple(u['cell']) for u in ob['static_updates']}
        old_cells = {k: graph.cells.get(k) for k in touched}
        old = (graph.timestamp, graph.revision, graph.ego_support, len(graph.visited), graph.global_nodes, graph.global_edges)
        for u in ob['static_updates']:
            if u['timestamp'] != now or u['state'] not in (1, 2) or not np.isfinite(u['height']):
                raise ValueError('noncausal or invalid static evidence')
        delta = graph.update(ob)
        if delta != node['graph_delta']:
            raise ValueError('recorded graph_delta replay mismatch at node ' + str(node_id))
        ledger['verified_deltas'] += 1
        changed_known = sum(k in old_cells and old_cells[k] is not None for k in map(tuple, delta['changed_cells']))
        ledger['known_correction_events'] += changed_known
        ledger['frames_with_known_corrections'] += int(changed_known > 0)
        path.append([float(now), *ob['robot_pose']])
        yield from emit(node_id)
        for child in children[node_id]:
            yield from visit(child)
        path.pop()
        for key, value in old_cells.items():
            if value is None:
                graph.cells.pop(key, None)
            else:
                graph.cells[key] = value
        graph.timestamp, graph.revision, graph.ego_support, length, graph.global_nodes, graph.global_edges = old
        del graph.visited[length:]
        graph._fast_grid = None
        if node_id in snapshots:
            del snapshots[node_id]

    yield from emit(None)
    for child in children[None]:
        yield from visit(child)
    # A final sentinel exposes exhaustive audit statistics without a second pass.
    yield dict(audit=dict(ledger))


def sample(ex, rays=False):
    if rays:
        from .features import SegmentFeatures
        context = SegmentFeatures(ex['graph'], ex['trajectory'])
    else:
        context = Features(ex['graph'], ex['trajectory'])
    keys = context.candidates('both'); n = len(keys)
    target = ex['target']; tk = np.asarray(list(target), np.int32).reshape(-1, 2)
    codes = (keys[:, 0]-context.lo[0])*int(context.shape[1])+keys[:, 1]-context.lo[1]
    tc = (tk[:, 0]-context.lo[0])*int(context.shape[1])+tk[:, 1]-context.lo[1]
    at = np.searchsorted(codes, tc)
    valid = (at < n)
    if n:
        valid &= np.all(keys[np.minimum(at, n-1)] == tk, axis=1)
    mask = np.zeros(n, bool); mask[at[valid]] = True
    rng = np.random.default_rng(stable(ex['root_id'] + str(ex['start_node']) + ':' + str(ex['end_node'])))
    chosen = []; weights = []
    for stratum in (np.flatnonzero(mask), np.flatnonzero(~mask)):
        count = min(48, len(stratum))
        if count:
            chosen.extend(rng.choice(stratum, count, replace=False))
            weights.extend([len(stratum)/(count*n*ex['segments_in_root'])]*count)
    x, baseline = context.transform(keys[chosen], 'both')
    y = np.array([target.get(tuple(k), (0, 0.))[0] for k in keys[chosen]], np.uint8)
    height = np.array([target.get(tuple(k), (0, float(h)))[1]-float(h) for k, h in zip(keys[chosen], baseline)], np.float32)
    p = ex['trajectory']; movement = float(np.linalg.norm(np.diff(p[:, [1, 3]], axis=0), axis=1).sum())
    dy = np.diff(p[:, 4]); rotation = float(np.abs(np.arctan2(np.sin(dy), np.cos(dy))).sum())
    meta = {k: ex[k] for k in ('root_id', 'scene_id', 'query_id', 'query_aliases', 'start_node', 'end_node', 'segments_in_root', 'known_terminal_differences')}
    meta.update(candidate_cells=n, out_of_domain_positive=len(tk)-int(valid.sum()),
                new_free=sum(v[0] == 1 for v in target.values()), new_blocked=sum(v[0] == 2 for v in target.values()),
                duration=float(p[-1, 0]-p[0, 0]), poses=len(p), movement=movement, rotation=rotation,
                known_cells=len(ex['graph']['cells']))
    if not np.isfinite(x.astype(np.float16)).all() or not np.isfinite(height.astype(np.float16)).all():
        raise ValueError('feature storage overflow')
    return dict(x=x.astype(np.float16), y=y, height=height.astype(np.float16), weight=np.array(weights, np.float32)), meta


def protocol():
    path = OUT / 'protocol.json'
    if path.exists():
        return json.loads(path.read_text())
    rows, train = source_rows('train'); dev, devpath = source_rows('dev')
    scenes = sorted({r['scene_id'] for r in rows}, key=stable)
    hold_scenes = set(scenes[:64]); fit = sorted([r for r in rows if r['scene_id'] not in hold_scenes], key=lambda r: stable(r['root_id']))[:2048]
    hold = []
    for s in sorted(hold_scenes):
        hold.extend(sorted([r for r in rows if r['scene_id'] == s], key=lambda r: stable(r['root_id']))[:2])
    if set(r['scene_id'] for r in rows) & set(r['scene_id'] for r in dev):
        raise ValueError('training/development scene overlap')
    p = dict(schema='segment_research_protocol_1', seed=2424,
        train_manifest=str(train), train_manifest_sha256=sha(train), dev_manifest=str(devpath), dev_manifest_sha256=sha(devpath),
        training_roots=len(rows), development_roots=len(dev), pilot_fit=[r['root_id'] for r in fit], pilot_hold=[r['root_id'] for r in hold],
        strategies=['partition', 'mixed'], features='retained 76 active legal map/full segment features from latest checkpoint',
        pilots='2048 fitting Roots, 64 scene-disjoint fitting-only holdout scenes (up to2Roots/scene); 3epochs each, warm start latest checkpoint',
        selection='lowest scene-balanced proper log loss among strategies whose Brier <=1.02*pilot baseline; hard IoU/F1 and cost disclosed; no threshold tuning',
        full_fit='one selected strategy, all105838 eligible Roots minus explicit reconstruction failures; fixed3epochs warm start, AdamW lr.0003 wd.0001 batch8192 seed2424',
        weighting='equal Root, then equal unique selected DAG windows; bounded mixed windows per Query; deduplicated exact shared-prefix windows; inverse inclusion48positive/48UNKNOWNcell sampling',
        eval='all2110 original devRoots/allQueries; same deduplicated mixed windows independent of pilot selection; rawargmax; 2000paired scene bootstrap draws seed2425',
        domain='initially UNKNOWN cells within union of11m closed discs around supplied saved poses PLUS all out-of-domain target positives; outside domain deterministicUNKNOWN',
        terminal='compare cell states/heights on the same domain; known-cell terminal corrections audited separately; full graph actually constructed for both models',
        probabilities='per-cell marginal softmax UNKNOWN/FREE/BLOCKED; clipped logloss1e-9; Brier sum3classes; reliability15equalwidthbins; no class weighting',
        cohorts='duration<=.5,.5-1,1-2,>2seconds; motion stationary,rotation,translation,mixed; newcells0,1-99,>=100',
        inference='complete map read/preprocess/fullpath/candidates/features/network/probabilities/sparseupdates/fullfuturegraph; wire separate; GCenabled; cold/hot; K1/2/4/8; workers4/8/12',
        source_at_freeze=snapshot())
    dump(path, p)
    return p


JOBS = None; DEST = None; STRATEGY = None; RAYS = False

def chunk(bounds):
    start, end = bounds; dest = DEST / ('%06d_%06d' % (start, end))
    if dest.with_suffix('.json').exists():
        report = json.loads(dest.with_suffix('.json').read_text())
        assert sha(report['path']) == report['sha256']
        return report
    arrays = defaultdict(list); metas = []; failures = []; counts = Counter(); cursor = 0; tick = time.perf_counter()
    for row in JOBS[start:end]:
        local = []; audit = None
        try:
            for ex in replay(row, STRATEGY):
                if 'audit' in ex:
                    audit = ex['audit']
                else:
                    local.append(sample(ex, RAYS))
        except Exception as error:
            failures.append(dict(root_id=row['root_id'], scene_id=row['scene_id'], error=repr(error), excluded_queries=row['queries']))
            continue
        if audit is None:
            raise ValueError('incomplete Root replay')
        counts.update(audit)
        for value, meta in local:
            meta.update(start=cursor, end=cursor+len(value['y'])); cursor = meta['end']; metas.append(meta)
            for k, v in value.items():
                arrays[k].append(v)
    path = dest.with_suffix('.npz')
    tmp = dest.with_suffix('.tmp.npz')
    np.savez(tmp, **{k: np.concatenate(v) for k, v in arrays.items()}); tmp.replace(path)
    report = dict(path=str(path), sha256=sha(path), counts=dict(counts), segments=metas, failures=failures,
                  seconds=time.perf_counter()-tick, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(dest.with_suffix('.json'), report)
    return report


def prepare(split, strategy, workers=12, limit=None, features='base'):
    global JOBS, DEST, STRATEGY, RAYS
    p = protocol(); rows, source = source_rows('dev' if split == 'dev' else 'train')
    if split in ('pilot_fit', 'pilot_hold'):
        ids = set(p[split]); rows = [r for r in rows if r['root_id'] in ids]
    if limit:
        rows = rows[:limit]
    JOBS = rows; STRATEGY = strategy; RAYS = features == 'ray'
    DEST = OUT / 'prepared' / (split + '_' + strategy + ('_ray' if RAYS else '') + ('_limit' + str(limit) if limit else ''))
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / 'manifest.json').exists():
        raise FileExistsError('completed extraction already exists')
    identity = dict(split=split, strategy=strategy, features=features, limit=limit, source_manifest_sha256=sha(source), sources=snapshot(), protocol_sha256=sha(OUT/'protocol.json'))
    if (DEST / 'identity.json').exists() and json.loads((DEST / 'identity.json').read_text()) != identity:
        raise ValueError('extraction identity changed')
    dump(DEST / 'identity.json', identity)
    tick = time.perf_counter(); reports = []; counts = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for report in pool.map(chunk, [(i, min(i+32, len(rows))) for i in range(0, len(rows), 32)]):
            reports.append(report); counts.update(report['counts'])
            progress = dict(split=split, strategy=strategy, completed_shards=len(reports), total_roots=len(rows), counts=dict(counts), seconds=time.perf_counter()-tick)
            dump(DEST / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    dump(DEST / 'manifest.json', dict(identity=identity, requested_roots=len(rows), roots=counts['roots'],
        scenes=len({m['scene_id'] for r in reports for m in r['segments']}), counts=dict(counts),
        sampled_cells=sum(m['end']-m['start'] for r in reports for m in r['segments']),
        failures=[f for r in reports for f in r['failures']], shards=reports, seconds=time.perf_counter()-tick,
        disk_bytes=sum(Path(r['path']).stat().st_size for r in reports), worker_peak_rss_kib=max(r['peak_rss_kib'] for r in reports)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['pilot_fit', 'pilot_hold', 'full', 'dev'], required=True)
    parser.add_argument('--strategy', choices=['mixed', 'partition'], default='mixed')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--features', choices=['base', 'ray'], default='base')
    prepare(**vars(parser.parse_args()))
