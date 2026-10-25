"""TRAINING TARGETS ONLY: projected installed-navmesh support completion.

This is an auxiliary traversability raster, not an occupancy map, not the
63-probe observation certificate, and never an input to a WM predictor.
The original initially-UNKNOWN increment supervision is kept unchanged.
"""
import argparse
import fcntl
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import distance_transform_edt
from .diagnose import OUT, HF, read, save, sha, source_rows, member
from .range_bev import SIZE


def crop_targets(keys, heights, lo, floor_y, resolution):
    """Independent world-cell crop; retain only the current projected floor."""
    n = SIZE * 2
    field = np.zeros((n, n), np.float32)
    ij = np.asarray(keys, np.int64) - np.asarray(lo, np.int64)
    valid = np.all((ij >= 0) & (ij < n), 1) & (np.abs(heights-floor_y) <= .18)
    ij = ij[valid]
    field[ij[:, 0], ij[:, 1]] = 1
    signed = (distance_transform_edt(field > 0) - distance_transform_edt(field == 0)) * resolution
    pool = lambda a: a.reshape(SIZE, 2, SIZE, 2).mean((1, 3))
    return np.stack([pool(field), np.clip(pool(signed) / 2., -1., 1.)]).astype(np.float16)


def navigation_targets(navmesh, resolution):
    """Load private training labels in the offline producer, with no renderer."""
    import habitat_sim
    from .geometry_reference import asset_path
    from .geometry_scene import projected_cells
    navmesh = asset_path(navmesh)
    identity = dict(schema='training_aux_projected_nav_support_1',
                    navmesh_sha256=sha(navmesh), resolution=float(resolution),
                    producer_sha256=sha(__file__),
                    raster_source_sha256=sha(Path(__file__).with_name('geometry_scene.py')))
    name = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = OUT/'support_teacher/scenes'/name
    directory.mkdir(parents=True, exist_ok=True)
    receipt, path = directory/'identity.json', directory/'labels.npz'
    with (directory/'lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not receipt.exists():
            pf = habitat_sim.PathFinder()
            if not pf.load_nav_mesh(str(navmesh)):
                raise ValueError('cannot load training teacher navmesh')
            vertices = np.asarray(pf.build_navmesh_vertices(), np.float64)
            indices = np.asarray(pf.build_navmesh_vertex_indices(), np.int64)
            x, z, y = projected_cells(vertices[indices.reshape(-1, 3)], resolution)
            keys = np.column_stack([x, z]).astype(np.int32)
            take = np.unique(np.column_stack([keys, np.rint(y/.05).astype(np.int64)]), axis=0, return_index=True)[1]
            np.savez_compressed(path, keys=keys[take], heights=y[take].astype(np.float32))
            save(receipt, dict(identity=identity, sha256=sha(path), cells=len(take), training_target_only=True))
        record = read(receipt)
        if record['identity'] != identity or record['sha256'] != sha(path):
            raise ValueError('stale private training target raster')
    with np.load(path) as z:
        return z['keys'], z['heights'], record


def one(task):
    row, source = task
    rid = row['root_id']
    private = member(source, 'privileged_initialization.json.gz')
    episode = private['episode']
    legal = member(source, 'legal_root.json.gz')
    resolution = legal['partial_graph']['resolution']
    keys, heights, scene_receipt = navigation_targets(episode['info']['graph_navmesh'], resolution)
    with np.load(row['map_path']) as z:
        maps, low = z['maps'], z['lo']
    # Reconstruct the legal base floor per starting graph from the packed data;
    # neither future certificate labels nor future camera poses are consulted.
    from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
    packed = read(HF/'packed/pilot_fit'/(rid+'.json'))
    arrays = load_arrays(packed)
    lookup = {(e['start_node'], e['end_node']): i for i, e in enumerate(packed['segments'])}
    labels = np.zeros((len(maps), 2, SIZE, SIZE), np.float16)
    for group in row['groups']:
        mi = group['map_index']
        e = group['segments'][0]
        trajectory = get_segment(arrays, lookup[group['start_node'], e['end_node']])[3]
        labels[mi] = crop_targets(keys, heights, low[mi], trajectory[0, 2], resolution)
    destination = OUT/'support_teacher/roots'
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/(rid+'.npz')
    np.savez_compressed(path, support_targets=labels)
    result = dict(root_id=rid, scene_id=row['scene_id'], path=str(path), sha256=sha(path),
                  source_map_sha256=row['map_sha256'],
                  navmesh_label_sha256=scene_receipt['sha256'], maps=len(maps),
                  target_only=True, semantics='Projected installed-navmesh support; auxiliary only, NOT observation certificates or BLOCKED labels')
    save(path.with_suffix('.json'), result)
    return result


def prepare(manifest, workers=4):
    tick = time.perf_counter()
    manifest = Path(manifest)
    m = read(manifest)
    train, _ = source_rows('train')
    dev, _ = source_rows('dev')
    source = {r['root_id']: r for r in train}
    assert set(r['root_id'] for r in m['records']) <= source.keys()
    assert not ({r['scene_id'] for r in m['records']} & {r['scene_id'] for r in dev})
    assert not (set(read(HF/'protocol.json')['opened_protected_roots']) & {r['root_id'] for r in m['records']})
    tasks = [(row, source[row['root_id']]) for row in m['records']]
    records = []
    with get_context('spawn').Pool(workers) as pool:
        for record in pool.imap_unordered(one, tasks):
            records.append(record)
            print('ROOT', len(records), record['root_id'], 'seconds', time.perf_counter()-tick, flush=True)
    save(OUT/'support_teacher'/('manifest_roots%d.json' % len(records)),
         dict(role='PRIVATE TRAINING AUXILIARY TARGET ONLY; never runtime inputs',
              source_manifest=str(manifest), source_manifest_sha256=sha(manifest),
              roots=len(records), records=records, seconds=time.perf_counter()-tick))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--workers', type=int, default=4)
    prepare(**vars(parser.parse_args()))
