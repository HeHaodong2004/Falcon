"""DIAGNOSTIC ONLY: reconstruct BLOCKED targets from actual future wall-ray
pixels (label side) and from the legal voxel simulation; compare per cell."""
import numpy as np
from collections import Counter
from eventwm_graph.stage2_repair.diagnose import OUT, HF, read, source_rows
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from .geometry import VoxelMap, pack_cams, camera_extrinsic, proxy_cameras, wall_blocked
from .sim_eval import crop_box, keyset_mask

rows, _ = source_rows('train')
pick = np.random.default_rng(7).choice(16384, 24, replace=False)
tot = Counter()
for idx in pick[:12]:
    row = rows[int(idx)]
    packed = read(HF/'packed/full'/(row['root_id']+'.json')); a = load_arrays(packed)
    z = np.load(OUT/'expanded_train_range_branches'/(row['root_id']+'.npz')); s = {k: z[k] for k in z.files}
    res = float(a['geometry'][0]); S = s['depth'].shape[-1]
    xs = np.linspace(-.95, .95, 35)
    c = Counter()
    for i, e in enumerate(packed['segments']):
        seg = get_segment(a, i); gk, gs, path, tk, ts = seg[0], seg[1], seg[3], seg[4], seg[5]
        if len(path) < 2 or e['end_node'] is None: continue
        take = ancestor_frames(s['parents'], s['history_count'], e['start_node'])
        fut = ancestor_frames(s['parents'], s['history_count'], e['end_node'])[len(take):]
        tb = tk[ts == 2]
        # actual rays (label side): fake 'rays' array with actual distances
        act = np.full((len(fut), 105, 6), 1e9, np.float32)
        o, R = proxy_cameras(path[1:], camera_extrinsic(path[0], s['origins'][take[-1]], s['rotations'][take[-1]]))
        for p, f in enumerate(fut):
            for k in range(105):
                x = xs[k//3]; yv = (-.5, 0., .5)[k % 3]
                d = np.array([x, yv, -1.]); d /= np.linalg.norm(d); wd = R[p]@d
                j = int(np.clip(np.floor((x+1)*.5*S), 0, S-1)); ii = int(np.clip(np.floor((1-yv)*.5*S), 0, S-1))
                ra = s['depth'][f][ii, j]
                act[p, k, 2:5] = wd; act[p, k, 5] = 0
                if np.isfinite(ra): act[p, k, 0] = ra-.03
        import torch
        cams = pack_cams(o, R)
        ab = wall_blocked(torch.from_numpy(act), cams.astype(np.float64), path[1:], res, True)
        # drop already-known cells
        if len(ab): ab = ab[~keyset_mask(ab, gk)]
        lo, hi = crop_box(path, res)
        vm = VoxelMap(s['depth'][take], s['origins'][take], s['rotations'][take], lo, hi-lo, res, path[0, 2], 'cuda:0', splat=False)
        sb = wall_blocked(vm.wall_rays(cams), cams.astype(np.float64), path[1:], res, True)
        if len(sb): sb = sb[~keyset_mask(sb, gk)]
        vm.hit_sums(s['depth'][take], s['origins'][take], s['rotations'][take])
        rb = wall_blocked(vm.wall_rays_refined(cams), cams.astype(np.float64), path[1:], res, True)
        if len(rb): rb = rb[~keyset_mask(rb, gk)]
        c['refined_cells'] += len(rb)
        if len(tb) and len(rb): c['targets_in_refined'] += keyset_mask(tb, rb).sum()
        if len(rb) and len(tb): c['refined_in_targets'] += keyset_mask(rb, tb).sum()
        if len(sb) and len(tb): c['sim_in_targets'] += keyset_mask(sb, tb).sum()
        c['targets'] += len(tb); c['actual_cells'] += len(ab); c['sim_cells'] += len(sb)
        if len(tb):
            c['targets_in_actual'] += keyset_mask(tb, ab).sum() if len(ab) else 0
            c['targets_in_sim'] += keyset_mask(tb, sb).sum() if len(sb) else 0
        if len(ab): c['actual_in_targets'] += keyset_mask(ab, tb).sum() if len(tb) else 0
    tot.update(c)
    print(row['root_id'][:8], dict(c), flush=True)
print('TOTAL', dict(tot))
print('target recall: actual-pixel %.3f sim %.3f refined %.3f ; precision sim %.3f refined %.3f' % (tot['targets_in_actual']/tot['targets'], tot['targets_in_sim']/tot['targets'],
      tot['targets_in_refined']/tot['targets'], tot['sim_in_targets']/max(tot['sim_cells'], 1), tot['refined_in_targets']/max(tot['refined_cells'], 1)))
