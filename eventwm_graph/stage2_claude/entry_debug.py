"""DIAGNOSTIC ONLY (future frames as labels): for real wall-band hits whose
simulated ray enters UNKNOWN first, how far past the unknown entry is the hit?"""
import numpy as np
from eventwm_graph.stage2_repair.diagnose import OUT, HF, read
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from .data import split_roots
from .geometry import VoxelMap, pack_cams, camera_extrinsic, proxy_cameras
from .sim_eval import crop_box

train, _ = split_roots()
rng = np.random.default_rng(3)
gaps = []; free_before = []
for idx in rng.choice(len(train), 30, replace=False):
    row = train[int(idx)]
    packed = read(HF/'packed/full'/(row['root_id']+'.json')); a = load_arrays(packed)
    z = np.load(OUT/'expanded_train_range_branches'/(row['root_id']+'.npz')); s = {k: z[k] for k in z.files}
    res = float(a['geometry'][0]); S = s['depth'].shape[-1]; xs = np.linspace(-.95, .95, 35)
    for i, e in enumerate(packed['segments'][::3]):
        seg = get_segment(a, i*3); path = seg[3]
        e = packed['segments'][i*3]
        if len(path) < 2 or e['end_node'] is None: continue
        take = ancestor_frames(s['parents'], s['history_count'], e['start_node'])
        fut = ancestor_frames(s['parents'], s['history_count'], e['end_node'])[len(take):]
        lo, hi = crop_box(path, res)
        vm = VoxelMap(s['depth'][take], s['origins'][take], s['rotations'][take], lo, hi-lo, res, path[0, 2], 'cuda:1')
        o, R = proxy_cameras(path[1:], camera_extrinsic(path[0], s['origins'][take[-1]], s['rotations'][take[-1]]))
        rays = vm.wall_rays(pack_cams(o, R)).cpu().numpy()
        for p, f in enumerate(fut):
            for k in range(105):
                x = xs[k//3]; yv = (-.5, 0., .5)[k % 3]
                j = int(np.clip(np.floor((x+1)*.5*S), 0, S-1)); ii = int(np.clip(np.floor((1-yv)*.5*S), 0, S-1))
                ra = s['depth'][f][ii, j]
                if not np.isfinite(ra): continue
                d = np.array([x, yv, -1.]); d /= np.linalg.norm(d); wd = R[p]@d
                hy = o[p][1]+wd[1]*ra
                if not (path[p+1, 2]+.2 < hy < path[p+1, 2]+1.4): continue
                to, tu, ul = rays[p, k, 0], rays[p, k, 1], rays[p, k, 5]
                if ul >= .3 and tu < ra:   # ray reaches unknown before the real hit
                    gaps.append(ra-tu); free_before.append(tu)
g = np.array(gaps)
print('rays entering unknown before real hit:', len(g))
print('hit distance past unknown entry (m) percentiles 10/25/50/75/90:', np.round(np.percentile(g, [10, 25, 50, 75, 90]), 2))
print('fraction hit within .3 m of entry %.3f, within .6 m %.3f, within 1 m %.3f' % ((g < .3).mean(), (g < .6).mean(), (g < 1).mean()))
print('entry distance from camera percentiles:', np.round(np.percentile(free_before, [10, 50, 90]), 2))
