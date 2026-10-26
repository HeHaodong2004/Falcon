"""DIAGNOSTIC ONLY (future frames as labels): per wall ray, compare voxel-map
simulation with the actually measured future range along that ray."""
import numpy as np
from collections import Counter
from eventwm_graph.stage2_repair.diagnose import OUT, HF, read, source_rows
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from .geometry import VoxelMap, pack_cams, camera_extrinsic, proxy_cameras
from .sim_eval import crop_box

import sys
from eventwm_graph.stage2_repair.diagnose import member
from .features import human_arrays
SPLAT = sys.argv[1] == '1'; HUMAN = sys.argv[2] == '1'; SUB = int(sys.argv[3]) if len(sys.argv) > 3 else 1; NEAR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.
rows, _ = source_rows('train')
pick = np.random.default_rng(7).choice(16384, 12, replace=False)
total = Counter()
for idx in pick:
    row = rows[int(idx)]
    packed = read(HF/'packed/full'/(row['root_id']+'.json')); a = load_arrays(packed)
    z = np.load(OUT/'expanded_train_range_branches'/(row['root_id']+'.npz')); s = {k: z[k] for k in z.files}
    res = float(a['geometry'][0]); S = s['depth'].shape[-1]
    legal = member(row, 'legal_root.json.gz'); nodes = member(row, 'future_observations.json.gz')
    track_rows = [o['tracks'] for o in legal['history']]+[n['observation']['tracks'] for n in nodes]
    c = Counter()
    for i, e in enumerate(packed['segments']):
        seg = get_segment(a, i); path = seg[3]
        if len(path) < 2 or e['end_node'] is None: continue
        take = ancestor_frames(s['parents'], s['history_count'], e['start_node'])
        fut = ancestor_frames(s['parents'], s['history_count'], e['end_node'])[len(take):]
        lo, hi = crop_box(path, res)
        vm = VoxelMap(s['depth'][take], s['origins'][take], s['rotations'][take], lo, hi-lo, res, path[0, 2], 'cuda:0', splat=SPLAT,
                      humans=human_arrays([track_rows[j] for j in take]) if HUMAN else None, sub=SUB, near=NEAR)
        if HUMAN:
            from .features import current_humans
            vm.stamp_humans(current_humans([track_rows[j] for j in take]))
        ext = camera_extrinsic(path[0], s['origins'][take[-1]], s['rotations'][take[-1]])
        o, R = proxy_cameras(path[1:], ext); cams = pack_cams(o, R)
        rays = vm.wall_rays(cams).cpu().numpy()
        xs = np.linspace(-.95, .95, 35)
        for p, f in enumerate(fut):
            for k in range(105):
                x = xs[k//3]; yv = (-.5, 0., .5)[k % 3]
                d = np.array([x, yv, -1.]); d /= np.linalg.norm(d)
                # nearest pixel to this exact direction
                j = int(np.clip(np.floor((x+1)*.5*S), 0, S-1)); ii = int(np.clip(np.floor((1-yv)*.5*S), 0, S-1))
                ra = s['depth'][f][ii, j]
                wd = R[p]@d; hit_y = o[p][1]+wd[1]*ra if np.isfinite(ra) else np.nan
                in_band = np.isfinite(ra) and (path[p+1, 2]+.2 < hit_y < path[p+1, 2]+1.4)
                if not in_band: c['actual_no_band_hit'] += 1; continue
                to, ul = rays[p, k, 0], rays[p, k, 5]
                if to < 10 and abs(to+.03/SUB-ra) < .2:
                    c['match_known' if ul < .3 else 'match_but_unknown_gap'] += 1
                elif to < ra-.2:
                    c['sim_hits_earlier' + ('_known' if ul < .3 else '_unk')] += 1
                elif ul >= .3:
                    c['sim_unknown_then_later_or_none'] += 1
                else:
                    c['sim_passes_through_known_free'] += 1
    total.update(c)
    n = sum(v for k, v in c.items() if k != 'actual_no_band_hit')
n = sum(v for k, v in total.items() if k != 'actual_no_band_hit')
print('TOTAL', {k: round(v/n, 3) for k, v in sorted(total.items()) if k != 'actual_no_band_hit'}, 'band rays', n)
