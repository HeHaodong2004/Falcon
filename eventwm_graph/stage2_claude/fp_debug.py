"""DIAGNOSTIC ONLY (uses future frames as labels): why are known-visible
cells not certified?"""
import numpy as np
from collections import Counter
from scipy.ndimage import distance_transform_edt
from eventwm_graph.stage2_repair.diagnose import OUT, HF, read, source_rows
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from .geometry import VoxelMap, pack_cams, camera_extrinsic, proxy_cameras, DY, Y_BELOW, local_dirs
from .sim_eval import crop_box, candidates, lookup

rows, _ = source_rows('train')
pick = np.random.default_rng(7).choice(16384, 24, replace=False)
best = None
for idx in pick[:10]:
    row = rows[int(idx)]
    packed = read(HF/'packed/full'/(row['root_id']+'.json')); a = load_arrays(packed)
    z = np.load(OUT/'expanded_train_range_branches'/(row['root_id']+'.npz')); s = {k: z[k] for k in z.files}
    res = float(a['geometry'][0])
    for i, e in enumerate(packed['segments']):
        seg = get_segment(a, i); gk, gs, path, tk, ts = seg[0], seg[1], seg[3], seg[4], seg[5]
        if len(path) < 3 or e['end_node'] is None: continue
        take = ancestor_frames(s['parents'], s['history_count'], e['start_node'])
        take_e = ancestor_frames(s['parents'], s['history_count'], e['end_node']); fut = take_e[len(take):]
        lo, hi = crop_box(path, res)
        vm = VoxelMap(s['depth'][take], s['origins'][take], s['rotations'][take], lo, hi-lo, res, path[0, 2], 'cuda:0')
        ext = camera_extrinsic(path[0], s['origins'][take[-1]], s['rotations'][take[-1]])
        o, R = proxy_cameras(path[1:], ext); cams = pack_cams(o, R)
        imgs = vm.render(cams, 96)
        keys = candidates(gk, path, res)
        y = lookup(keys, tk, ts.astype(np.uint8))
        pr = vm.probes(keys, np.full(len(keys), path[0, 2], np.float32), cams, imgs, np.arange(len(cams))).cpu().numpy()
        fp = np.flatnonzero((pr[:, 0] > 0) & (y == 0)); tp = np.flatnonzero((pr[:, 0] > 0) & (y == 1))
        if best is None or len(fp) > best[0]:
            best = (len(fp), row, i, len(tp), fp, keys, path, s, take, fut, cams, vm, res)
    print(row['root_id'][:8], 'best so far', best[0], best[3], flush=True)

nfp, row, i, ntp, fp, keys, path, s, take, fut, cams, vm, res = best
print('window', row['root_id'][:8], i, 'fp', nfp, 'tp', ntp, 'poses', len(path)-1)
# For each FP cell, check against ACTUAL future depth frames: per pose, all 63 probes in frustum and actual range > probe distance - .015?
S = s['depth'].shape[-1]
reasons = Counter()
halo = .25+res/np.sqrt(2)
offs = [(0, 0), (-.49, -.49), (-.49, .49), (.49, -.49), (.49, .49)]
for c in fp[:300]:
    cx, cz = (keys[c]+.5)*res; h = path[0, 2]
    pts = [(cx+dx*res, cz+dz*res) for dx, dz in offs]+[(cx+np.cos(t)*halo, cz+np.sin(t)*halo) for t in np.linspace(0, 2*np.pi, 16, endpoint=False)]
    P = np.array([[x, h+lv, zz] for lv in (.025, .7, 1.4) for x, zz in pts])
    ok_any = False; why = []
    for p_i, f in enumerate(fut):
        O, Rm = s['origins'][f], s['rotations'][f]
        d = P-O; loc = d@Rm; fw = -loc[:, 2]; dist = np.linalg.norm(d, axis=1)
        inside = (fw > 1e-6) & (np.abs(loc[:, 0]) <= fw) & (np.abs(loc[:, 1]) <= fw) & (dist <= 10)
        if not inside.all(): continue
        u = loc[:, 0]/fw; w = loc[:, 1]/fw
        j = np.clip(np.floor((u+1)*.5*S).astype(int), 0, S-1); ii = np.clip(np.floor((1-w)*.5*S).astype(int), 0, S-1)
        r = s['depth'][f][ii, j]; r = np.where(np.isnan(r), 99, r)
        blocked = r < dist-.05
        if not blocked.any(): ok_any = True
        why.append(int(blocked.sum()))
    if ok_any: reasons['actual frame says visible (support/human?)'] += 1
    elif why: reasons['actual depth blocks min %d probes' % min(why) if min(why) < 3 else 'actual blocks >=3'] += 1
    else: reasons['no frustum pose'] += 1
print(reasons)
# Where are the FP cells: distance from start, voxel state of their column
g = vm.grid().cpu().numpy(); lo = vm.lo_key
ij = keys[fp].astype(np.int64)-lo
col = g[ij[:, 0], ij[:, 1]]
print('FP column state counts per height bin (unknown/free/occ):')
for b in range(0, 24, 2):
    print(round(-Y_BELOW+b*DY, 2), np.bincount(col[:, b], minlength=3))
dist = np.linalg.norm((keys[fp]+.5)*res-path[0, [1, 3]], axis=1)
print('FP dist from start pct', np.percentile(dist, [5, 25, 50, 75, 95]))
