"""DIAGNOSTIC ONLY: proxy future cameras vs recorded future cameras."""
import numpy as np
from eventwm_graph.stage2_repair.diagnose import OUT, HF, read, source_rows
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from .geometry import camera_extrinsic, proxy_cameras

rows, _ = source_rows('train')
pick = np.random.default_rng(7).choice(16384, 6, replace=False)
for idx in pick:
    row = rows[int(idx)]
    packed = read(HF/'packed/full'/(row['root_id']+'.json'))
    a = load_arrays(packed)
    z = np.load(OUT/'expanded_train_range_branches'/(row['root_id']+'.npz')); s = {k: z[k] for k in z.files}
    for i, e in enumerate(packed['segments'][:40]):
        seg = get_segment(a, i); path = seg[3]
        if e['end_node'] is None or len(path) < 3: continue
        take_s = ancestor_frames(s['parents'], s['history_count'], e['start_node'])
        take_e = ancestor_frames(s['parents'], s['history_count'], e['end_node'])
        fut = take_e[len(take_s):]
        if len(fut) != len(path)-1: print('len mismatch', len(fut), len(path)); continue
        ext = camera_extrinsic(path[0], s['origins'][take_s[-1]], s['rotations'][take_s[-1]])
        o, R = proxy_cameras(path[1:], ext)
        do = np.linalg.norm(o-s['origins'][fut], axis=1)
        dr = np.abs(R-s['rotations'][fut]).max(axis=(1, 2))
        dt = np.abs(s['timestamps'][fut]-path[1:, 0]).max()
        print(row['root_id'][:8], 'seg', i, 'poses', len(path)-1, 'origin err max %.4f' % do.max(), 'rot err max %.4f' % dr.max(), 'time err %.2e' % dt)
        break
