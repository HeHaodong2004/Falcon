"""Self-consistency: simulate wall rays at RECORDED past camera poses on the
legal voxel map and compare with the graph's known BLOCKED certificates."""
import numpy as np
from collections import defaultdict
from eventwm_graph.stage2_repair.diagnose import OUT, HF, read, source_rows
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from .geometry import VoxelMap, pack_cams, wall_blocked
from .sim_eval import crop_box, keyset_mask, lookup


def main(n=12, device='cuda:0'):
    rows, _ = source_rows('train')
    pick = np.random.default_rng(7).choice(16384, n, replace=False)
    for idx in pick:
        row = rows[int(idx)]
        packed = read(HF/'packed/full'/(row['root_id']+'.json'))
        a = load_arrays(packed)
        z = np.load(OUT/'expanded_train_range_branches'/(row['root_id']+'.npz')); s = {k: z[k] for k in z.files}
        res = float(a['geometry'][0])
        seg = get_segment(a, 0)
        gk, gs, path = seg[0], seg[1], seg[3]
        sid = packed['segments'][0]['start_node']
        take = ancestor_frames(s['parents'], s['history_count'], sid)
        lo, hi = crop_box(path, res)
        vm = VoxelMap(s['depth'][take], s['origins'][take], s['rotations'][take], lo, hi-lo, res, path[0, 2], device)
        cams = pack_cams(s['origins'][take], s['rotations'][take])
        rays = vm.wall_rays(cams)
        fake_path = np.tile(path[:1], (len(take), 1))
        blocked = wall_blocked(rays, cams.astype(np.float64), fake_path, res, False)
        gb = gk[gs == 2]
        inter = keyset_mask(blocked, gb).sum() if len(blocked) else 0
        st = vm.state.view(-1)
        frac = [(st == v).float().mean().item() for v in range(3)]
        nan = np.isnan(s['depth'][take]).mean()
        r = rays.cpu().numpy()
        print(row['root_id'][:8], 'frames', len(take), 'graph blocked', len(gb), 'sim blocked', len(blocked), 'sim in graph %.3f' % (inter/max(len(blocked), 1)),
              'graph covered %.3f' % (keyset_mask(gb, blocked).mean() if len(gb) else -1), 'vox u/f/o %.3f %.3f %.3f' % tuple(frac), 'nan %.3f' % nan,
              'ray occ<10 %.3f unk-first %.3f' % ((r[..., 0] < 10).mean(), (r[..., 1] < r[..., 0]).mean()), flush=True)


if __name__ == '__main__':
    main()
