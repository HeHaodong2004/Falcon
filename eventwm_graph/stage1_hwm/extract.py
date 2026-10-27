"""Offline per-anchor extraction: legal inputs plus SEPARATE privileged labels.

Legal features are produced by replaying arrived frames through ``LegalState``
(the same code path as online inference). Privileged simulator positions are
read only to build labels after the legal features of an anchor are frozen.

Anchor schedule: the legacy schedule of ``l2_two_stage.data.one_root`` (root,
nearest frames to +.5/1/2/3 s, first practical gap, first return), plus for
training splits every ``stride``-th branch frame. Legacy tracker-to-person
labels are replicated exactly; an additional training-only label resolves
stale keys last seen before the Root (whose identities the legacy labels leave
unresolved) when the privileged positions make the assignment unambiguous.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from .legal_state import LegalState, positives, camera_yaw, CROP, VIS_POOL, HISTORY

try:
    import orjson
    loads = orjson.loads
except ImportError:  # pragma: no cover
    loads = json.loads

TAU = np.arange(13) * .25
DEV_MANIFEST = Path('/home/haodong/workspace/projects/eventwm-falcon-l2-partial-20261021/output/l2_graph_wm_120k_20261021/snapshot/manifest.json')
DEV_ROOTS = DEV_MANIFEST.parent / 'roots'
TRAIN_MANIFEST = Path('/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023/output/l2_two_stage_20261023/final_collection_20261024/manifest.json')
DEFAULT_DEST = Path('/var/tmp/falcon-stage1-claude/hwm_data')


def read(path):
    with gzip.open(path, 'rb') as f:return loads(f.read())


def selected_branch(folder):
    """Same canonical (longest, then query_id) branch as the legacy pipeline."""
    queries = read(folder / 'queries.json.gz');outcomes = read(folder / 'outcomes.json.gz')
    qi = max(range(len(queries)), key=lambda i:(len(queries[i]['compiled']['actions']), queries[i]['query_id']))
    outcome = outcomes[qi]
    assert outcome['query_id'] == queries[qi]['query_id']
    nodes = read(folder / 'future_observations.json.gz')
    branch = [];parent = None
    for node_id in outcome['node_ids']:
        node = nodes[node_id]
        assert node['node_id'] == node_id and node['parent'] == parent
        branch.append(node);parent = node_id
    return queries[qi], branch


def native_index(times, t, tolerance=1e-7):
    at = int(np.searchsorted(times, t))
    possible = [i for i in (at - 1, at) if 0 <= i < len(times) and abs(times[i] - t) <= tolerance]
    if len(possible) != 1:raise ValueError('timestamp is not uniquely native')
    return possible[0]


def interpolate(times, values, wanted):
    """Linear interpolation of native samples [T,H,2] at wanted times; mask beyond support."""
    wanted = np.asarray(wanted, np.float64)
    ok = (wanted >= times[0] - 1e-9) & (wanted <= times[-1] + 1e-6)
    w = np.clip(wanted, times[0], times[-1])
    hi = np.clip(np.searchsorted(times, w), 1, len(times) - 1);lo = hi - 1
    f = ((w - times[lo]) / (times[hi] - times[lo]))[:, None, None]
    return values[lo] + f * (values[hi] - values[lo]), ok


def resolve_stale(keys, last, last_t, now, resolved_owner, human_now):
    """Training-only label for keys the legacy labels leave unresolved.

    Returns {key_index: human or -1 (absent duplicate)}; ambiguous keys omitted.
    A key is ABSENT when every speed-feasible human is already owned by a more
    recent key. It OWNS a human when exactly one unowned human is feasible under
    a conservative 2.2 m/s(+.2 m) radius, no owned human is 3 m/s-feasible and
    no other unresolved key can reach that human.
    """
    owned = set(resolved_owner.values())
    feasible3 = {};feasible2 = {}
    for i in keys:
        age = now - last_t[i];d = np.linalg.norm(human_now - last[i], axis=-1)
        feasible3[i] = set(np.flatnonzero(d <= 3. * age + .05).tolist())
        feasible2[i] = set(np.flatnonzero(d <= 2.2 * age + .2).tolist())
    result = {}
    for i in keys:
        f3 = feasible3[i];free3 = f3 - owned
        if f3 and not free3:
            result[i] = -1;continue
        free2 = feasible2[i] - owned
        if len(free2) == 1 and not (f3 & owned) and len(free3) == 1:
            h = next(iter(free2))
            if all(h not in feasible3[k] for k in keys if k != i):result[i] = h
    return result


def process_root(job):
    folder, root_id, scene_id, split, stride, wanted = job
    folder = Path(folder)
    legal = read(folder / 'legal_root.json.gz')
    _, branch = selected_branch(folder)
    with np.load(folder / 'privileged_future.npz', allow_pickle=False) as z:
        T = z['timestamps'].astype(np.float64);truth_xyz = z['positions'].astype(np.float64)
    truth = truth_xyz[..., [0, 2]]
    history = legal['history']
    frames = list(history) + [n['observation'] for n in branch]
    R = len(history) - 1
    root_time = float(history[-1]['timestamp'])
    assert abs(root_time - T[0]) < 1e-7
    ft = np.array([float(f['timestamp']) for f in frames[R:]])
    schedule = {0}
    for h in (.5, 1., 2., 3.):
        if ft[-1] >= root_time + h - 1e-7:schedule.add(int(np.abs(ft - root_time - h).argmin()))
    state = LegalState(legal['partial_graph']['resolution'])
    for f in frames[:R]:state.ingest(f)
    # Legacy physical-label replay (frames from the Root onward).
    source_to_person = {};person_last = {};source_last = {};previous_positive = set()
    gap_added = False;return_added = False
    anchors = []
    for j in range(len(frames) - R):
        ob = frames[R + j];state.ingest(ob)
        ti = native_index(T, float(ob['timestamp']));clean = True;mapping = {};current = set()
        for tr in positives(ob):
            zz = np.asarray(tr['position']);close = np.flatnonzero(np.linalg.norm(truth_xyz[ti] - zz, axis=-1) < 2e-4)
            if len(close) != 1 or int(close[0]) in current:
                clean = False;continue
            p = int(close[0]);current.add(p);mapping[str(tr['track_id'])] = p
        if not clean or not ob.get('observation_valid', True) or not ob.get('human_sensor_available', True):
            source_to_person = {};person_last = {};source_last = {};previous_positive = set()
        else:
            returning = [p for p in current if p in person_last and p not in previous_positive]
            if returning and not return_added:schedule.add(j);return_added = True
            for key, p in mapping.items():
                source_to_person[key] = p;source_last[key] = float(ob['timestamp']);person_last[p] = float(ob['timestamp'])
            aged = [p for p, t in person_last.items() if p not in current and .75 - 1e-7 <= ob['timestamp'] - t <= 1.25 + 1e-7]
            if aged and not gap_added:schedule.add(j);gap_added = True
            previous_positive = current
        take = (j in wanted) if wanted is not None else (j in schedule or (stride and j % stride == 0))
        if not take:continue
        now = float(ob['timestamp'])
        mp, vis, corner = state.crops()
        kf = state.key_features();keys = kf['keys'];n = len(keys)
        owners = {}
        for key, p in source_to_person.items():
            if key in keys and (p not in owners or source_last[key] > source_last[owners[p]]):owners[p] = key
        index = {k: i for i, k in enumerate(keys)}
        phys = np.full(n, -1, np.int16);legacy_presence = np.full(n, np.nan, np.float32)
        for i, key in enumerate(keys):
            if key in source_to_person:
                p = source_to_person[key];legacy_presence[i] = float(owners[p] == key)
                if owners[p] == key:phys[i] = p
        resolved_owner = {index[k]: p for p, k in owners.items()}
        unresolved = [i for i, k in enumerate(keys) if k not in source_to_person]
        human_now = truth[ti]
        extra = resolve_stale(unresolved, kf['last'], kf['last_t'], now, resolved_owner, human_now)
        exist = legacy_presence.copy();owner_h = np.full(n, -1, np.int16)
        for i, p in resolved_owner.items():owner_h[i] = p
        for i, h in extra.items():
            exist[i] = float(h >= 0)
            if h >= 0:owner_h[i] = h
        residual_certain = all(not np.isnan(exist[i]) for i in range(n))
        hum, ok = interpolate(T, truth, now + TAU)
        hum = hum.transpose(1, 0, 2)
        owner_key = np.full(truth.shape[1], -1, np.int16)
        for i in range(n):
            if owner_h[i] >= 0:owner_key[owner_h[i]] = i
        anchors.append(dict(j=j, t=now, original=j in schedule,
            robot=np.array([state.robot[-1][1], state.robot[-1][2], state.robot[-1][3], state.robot[-1][4]], np.float32),
            robot_hist=state.robot_history(),
            cam=np.array([ob['camera_origin'][0], ob['camera_origin'][2], camera_yaw(ob['camera_rotation'])], np.float32),
            corner=corner, map=mp, vis=vis, summary=state.summary(), residual_certain=residual_certain, tau_mask=ok,
            keys=keys, hist=kf['hist'], hist_mask=kf['hist_mask'], last=kf['last'], last_t=kf['last_t'],
            velocity=kf['velocity'], velocity_valid=kf['velocity_valid'], count=kf['count'], first=kf['first'],
            phys=phys, legacy_presence=legacy_presence, exist=exist, owner_h=owner_h,
            hum=hum.astype(np.float32), hum_dy=(truth_xyz[ti, :, 1] - state.robot[-1][4]).astype(np.float32), owner_key=owner_key))
    if wanted is not None and {a['j'] for a in anchors} != set(wanted):
        raise ValueError('requested anchors missing for ' + root_id)
    frame_robot = np.array([[f['timestamp'], *f['robot_pose']] for f in frames[R:]], np.float64)
    return dict(root_id=root_id, scene_id=scene_id, anchors=anchors, frame_robot=frame_robot, truth_t=T, truth=truth.astype(np.float32),
                episode_start=float(frames[0]['timestamp']))


def pack(results):
    """Concatenate ragged per-anchor records into flat arrays with offsets."""
    A = [];roots = []
    for r_i, r in enumerate(results):
        roots.append(dict(root_id=r['root_id'], scene_id=r['scene_id'], humans=int(r['truth'].shape[1])))
        for a in r['anchors']:A.append((r_i, a))
    out = {}
    out['anchor_root'] = np.array([r for r, _ in A], np.int32)
    for name, dtype in (('j', np.int16), ('t', np.float64), ('original', bool), ('robot', np.float32), ('robot_hist', np.float32),
                        ('cam', np.float32), ('corner', np.float64), ('map', np.uint8), ('vis', np.uint8), ('summary', np.float32),
                        ('residual_certain', bool), ('tau_mask', bool)):
        out[name] = np.stack([np.asarray(a[name], dtype) for _, a in A]) if A else np.zeros((0,), dtype)
    key_counts = [len(a['keys']) for _, a in A];hum_counts = [len(a['owner_key']) for _, a in A]
    out['key_off'] = np.r_[0, np.cumsum(key_counts)].astype(np.int64)
    out['hum_off'] = np.r_[0, np.cumsum(hum_counts)].astype(np.int64)
    for name, dtype, shape in (('hist', np.float32, (HISTORY, 3)), ('hist_mask', bool, (HISTORY,)), ('last', np.float64, (2,)),
                               ('last_t', np.float64, ()), ('velocity', np.float32, (2,)), ('velocity_valid', bool, ()),
                               ('count', np.int32, ()), ('first', np.float64, ()), ('phys', np.int16, ()),
                               ('legacy_presence', np.float32, ()), ('exist', np.float32, ()), ('owner_h', np.int16, ())):
        parts = [np.asarray(a[name], dtype).reshape((-1,) + shape) for _, a in A]
        out['k_' + name] = np.concatenate(parts) if parts else np.zeros((0,) + shape, dtype)
    for name, dtype, shape in (('hum', np.float32, (13, 2)), ('hum_dy', np.float32, ()), ('owner_key', np.int16, ())):
        parts = [np.asarray(a[name], dtype).reshape((-1,) + shape) for _, a in A]
        out['h_' + name] = np.concatenate(parts) if parts else np.zeros((0,) + shape, dtype)
    out['frame_off'] = np.r_[0, np.cumsum([len(r['frame_robot']) for r in results])].astype(np.int64)
    out['frame_robot'] = np.concatenate([r['frame_robot'] for r in results])
    out['truth_off'] = np.r_[0, np.cumsum([len(r['truth_t']) for r in results])].astype(np.int64)
    out['truth_t'] = np.concatenate([r['truth_t'] for r in results])
    # Native truth is ragged in humans: store [T*H,2] with per-root H.
    out['truth_xz'] = np.concatenate([r['truth'].reshape(-1, 2) for r in results])
    out['truth_flat_off'] = np.r_[0, np.cumsum([r['truth'].size // 2 for r in results])].astype(np.int64)
    out['episode_start'] = np.array([r['episode_start'] for r in results], np.float64)
    keys = [k for _, a in A for k in a['keys']]
    return out, dict(roots=roots, keys=keys)


def jobs(split, stride, limit=0, offset=0):
    if split == 'dev':
        import collections
        m = json.loads(DEV_MANIFEST.read_text())
        rows = [r for r in m['rows'] if r['split'] == 'dev']
        # Exactly the anchors of the legacy prepared development records.
        wanted = collections.defaultdict(set)
        legacy = Path('/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023/output/l2_two_stage_20261023/prepared/dev_native_v2')
        manifest = json.loads((legacy / 'manifest.json').read_text())
        for shard in manifest['shards']:
            for rec in np.load(legacy.parent.parent / shard['path'], allow_pickle=True):
                wanted[rec['root_id']].add(int(rec['anchor_index']))
        out = [(str(DEV_ROOTS / r['root_id']), r['root_id'], r['scene_id'], 'dev', 0, sorted(wanted[r['root_id']])) for r in rows]
    else:
        m = json.loads(TRAIN_MANIFEST.read_text())
        rows = sorted(m['rows'], key=lambda r:r['root_id'])
        out = [(r['folder'], r['root_id'], r['scene_id'], 'train', stride, None) for r in rows]
    out = out[offset:]
    return out[:limit] if limit else out


def run(split, dest, workers, stride, shard_roots, limit, offset):
    dest = Path(dest) / split;dest.mkdir(parents=True, exist_ok=True)
    todo = jobs(split, stride, limit, offset)
    started = time.perf_counter();done = 0
    shards = [todo[i:i + shard_roots] for i in range(0, len(todo), shard_roots)]
    with ProcessPoolExecutor(workers) as pool:
        for s_i, shard in enumerate(shards):
            name = '%s_%07d' % (split, offset + s_i * shard_roots)
            if (dest / (name + '.json')).exists():
                done += len(shard);continue
            results = list(pool.map(process_root, shard, chunksize=4))
            arrays, meta = pack(results)
            tmp = dest / (name + '.tmp.npz');np.savez_compressed(tmp, **arrays);tmp.rename(dest / (name + '.npz'))
            meta.update(split=split, stride=stride, anchors=int(len(arrays['t'])),
                        sha256=hashlib.sha256((dest / (name + '.npz')).read_bytes()).hexdigest())
            (dest / (name + '.json')).write_text(json.dumps(meta))
            done += len(shard)
            print(json.dumps(dict(shard=name, roots=done, total=len(todo), anchors=meta['anchors'],
                                  seconds=round(time.perf_counter() - started, 1))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('split', choices=['dev', 'train'])
    p.add_argument('--dest', default=str(DEFAULT_DEST))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--stride', type=int, default=6)
    p.add_argument('--shard-roots', type=int, default=500)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--offset', type=int, default=0)
    a = p.parse_args()
    run(a.split, a.dest, a.workers, a.stride, a.shard_roots, a.limit, a.offset)
