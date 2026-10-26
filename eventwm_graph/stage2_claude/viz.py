"""Coordinate-aligned window visualization (diagnostic)."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eventwm_graph.stage2_repair.diagnose import OUT, HF, read, source_rows
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from .geometry import VoxelMap, pack_cams, camera_extrinsic, proxy_cameras, wall_blocked, Y_BELOW, DY
from .sim_eval import crop_box, keyset_mask

rows, _ = source_rows('train')
pick = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 3).choice(16384, 40, replace=False)
dest = sys.argv[1]
made = 0
for idx in pick:
    row = rows[int(idx)]
    packed = read(HF/'packed/full'/(row['root_id']+'.json')); a = load_arrays(packed)
    z = np.load(OUT/'expanded_train_range_branches'/(row['root_id']+'.npz')); s = {k: z[k] for k in z.files}
    res = float(a['geometry'][0])
    order = sorted(range(len(packed['segments'])), key=lambda i: -len(get_segment(a, i)[4]))
    i = order[len(order)//4]
    seg = get_segment(a, i); gk, gs, path, tk, ts = seg[0], seg[1], seg[3], seg[4], seg[5]
    if len(tk) < 50: continue
    e = packed['segments'][i]
    take = ancestor_frames(s['parents'], s['history_count'], e['start_node'])
    lo, hi = crop_box(path, res)
    vm = VoxelMap(s['depth'][take], s['origins'][take], s['rotations'][take], lo, hi-lo, res, path[0, 2], 'cuda:0')
    g = vm.grid().cpu().numpy()
    band = g[:, :, int((Y_BELOW+.2)/DY):int((Y_BELOW+1.4)/DY)]
    occ = (band == 2).any(2); free = (band == 1).all(2); unk = ~occ & ~free
    floor = (g[:, :, 1:4] == 2).any(2)
    img = np.ones(occ.shape+(3,))*.15
    img[unk] = [.15, .15, .15]
    img[free] = [.55, .55, .55]
    img[free & floor] = [.75, .75, .75]
    img[occ] = [.35, .2, .1]
    ij = gk.astype(np.int64)-lo; ok = np.all((ij >= 0) & (ij < hi-lo), 1)
    gimg = img.copy()
    gimg[ij[ok & (gs == 1), 0], ij[ok & (gs == 1), 1]] = [.3, .6, 1.]
    gimg[ij[ok & (gs == 2), 0], ij[ok & (gs == 2), 1]] = [1., .3, .3]
    timg = img.copy()
    tij = tk.astype(np.int64)-lo; tok = np.all((tij >= 0) & (tij < hi-lo), 1)
    timg[tij[tok & (ts == 1), 0], tij[tok & (ts == 1), 1]] = [0., 1., 0.]
    timg[tij[tok & (ts == 2), 0], tij[tok & (ts == 2), 1]] = [1., 0., 1.]
    ext = camera_extrinsic(path[0], s['origins'][take[-1]], s['rotations'][take[-1]])
    o, R = proxy_cameras(path[1:], ext); cams = pack_cams(o, R)
    sb = wall_blocked(vm.wall_rays(cams), cams.astype(np.float64), path[1:], res, False)
    simg = img.copy()
    if len(sb):
        sij = sb-lo; sok = np.all((sij >= 0) & (sij < hi-lo), 1); simg[sij[sok, 0], sij[sok, 1]] = [1., 1., 0.]
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    pij = path[:, [1, 3]]/res-lo
    for k, (im, title) in enumerate([(gimg, 'graph FREE(blue)/BLOCKED(red) over legal voxels'), (timg, 'targets FREE(green)/BLOCKED(magenta)'), (simg, 'sim wall-ray BLOCKED (yellow, optimistic)')]):
        ax[k].imshow(np.transpose(im, (1, 0, 2)), origin='lower', interpolation='nearest')
        ax[k].plot(pij[:, 0], pij[:, 1], 'c-', lw=2); ax[k].plot(pij[0, 0], pij[0, 1], 'co')
        for p in range(0, len(path), max(1, len(path)//6)):
            ax[k].arrow(pij[p, 0], pij[p, 1], 8*np.cos(path[p, 4]+np.pi/2)*0, 0, color='c')
        ax[k].set_title(title)
    # camera forward arrows
    for p in range(len(cams)):
        f = R[p]@np.array([0, 0, -1.]); ax[1].arrow(o[p, 0]/res-lo[0], o[p, 2]/res-lo[1], 15*f[0], 15*f[2], color='y', width=.3)
    fig.suptitle('%s seg %d: %d frames, %d poses, %.2fs, targets F%d B%d' % (row['root_id'][:8], i, len(take), len(path), path[-1, 0]-path[0, 0], (ts == 1).sum(), (ts == 2).sum()))
    plt.tight_layout(); plt.savefig('%s/win_%02d.png' % (dest, made), dpi=60); plt.close()
    made += 1
    if made >= 6: break
print('made', made)
