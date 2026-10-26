"""Information-budget diagnostic on TRAIN roots (no dev selection).

For each revealed target cell, test whether legal past range measurements
already contain the geometry that produces it:
  BLOCKED: a measured wall-band hit within .25 m (xz) of the cell center.
  FREE: a measured floor hit inside the cell.
"""
import numpy as np, json, sys
from collections import defaultdict
from eventwm_graph.stage2_repair.diagnose import OUT, HF, read, source_rows
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment


def local_dirs(size):
    xx, yy = np.meshgrid((np.arange(size)+.5)*2/size-1, 1-(np.arange(size)+.5)*2/size, indexing='xy')
    v = np.stack([xx, yy, -np.ones_like(xx)], -1).reshape(-1, 3)
    return v/np.linalg.norm(v, axis=1)[:, None]


def points(depth, origins, rotations):
    d = local_dirs(depth.shape[-1])
    out = []
    for r, o, R in zip(depth, origins, rotations):
        r = r.reshape(-1); ok = np.isfinite(r)
        out.append(o+(d[ok]@R.T)*r[ok, None])
    return np.concatenate(out)


def dilate_keys(pts, res, radius=.25):
    # cells whose center lies within radius (xz) of any point
    q = np.unique(np.round(pts[:, [0, 2]]/0.02).astype(np.int64), axis=0)*0.02
    span = int(np.ceil(radius/res))+1
    base = np.floor(q/res).astype(np.int64)
    off = np.array([(a, b) for a in range(-span, span+1) for b in range(-span, span+1)])
    keys = (base[:, None, :]+off[None]).reshape(-1, 2)
    cen = (keys+.5)*res
    src = np.repeat(q, len(off), 0)
    ok = np.linalg.norm(cen-src, axis=1) <= radius+1e-9
    return set(map(tuple, np.unique(keys[ok], axis=0)))


def main(n=40, sensor='expanded_train_range_branches'):
    rows, _ = source_rows('train')
    rng = np.random.default_rng(7)
    pick = rng.choice(16384, n, replace=False)
    tot = defaultdict(float)
    for idx in pick:
        row = rows[int(idx)]
        packed = read(HF/'packed/full'/(row['root_id']+'.json'))
        a = load_arrays(packed)
        z = np.load(OUT/sensor/(row['root_id']+'.npz')); s = {k: z[k] for k in z.files}
        res = float(a['geometry'][0])
        cache = {}
        for i, e in enumerate(packed['segments']):
            gk, gs, gh, path, tk, ts, th, geo = get_segment(a, i)
            sid = e['start_node']
            if sid not in cache:
                take = ancestor_frames(s['parents'], s['history_count'], sid)
                P = points(s['depth'][take], s['origins'][take], s['rotations'][take])
                h0 = path[0, 2]
                wall = P[(P[:, 1] > h0+.2) & (P[:, 1] < h0+1.4)]
                floor = P[np.abs(P[:, 1]-h0) < .06]
                cache[sid] = (dilate_keys(wall, res), set(map(tuple, np.floor(floor[:, [0, 2]]/res).astype(np.int64))), len(take))
            W, F, nf = cache[sid]
            tot['frames'] += nf; tot['windows'] += 1
            for k, st in zip(map(tuple, tk), ts):
                if st == 2:
                    tot['blocked'] += 1; tot['blocked_wall_seen'] += k in W
                else:
                    tot['free'] += 1; tot['free_floor_seen'] += k in F
            tot['poses'] += len(path); tot['duration'] += path[-1, 0]-path[0, 0]
    print(json.dumps({k: v for k, v in tot.items()}, indent=1))
    print('blocked wall seen frac', tot['blocked_wall_seen']/tot['blocked'], 'free floor seen frac', tot['free_floor_seen']/tot['free'])
    print('mean frames', tot['frames']/tot['windows'], 'mean poses', tot['poses']/tot['windows'], 'mean dur', tot['duration']/tot['windows'])


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
