"""Past-pose FREE visibility consistency: graph FREE cells should be
known-visible from some recorded past pose; graph UNKNOWN cells mostly not."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from eventwm_graph.stage2_repair.diagnose import OUT, HF, read, source_rows
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from .geometry import VoxelMap, pack_cams, DY, Y_BELOW
from .sim_eval import crop_box, candidates


import sys
import eventwm_graph.stage2_claude.geometry as G


def main(n=16, device='cuda:1'):
    RENDER = int(sys.argv[1]); G.MARGIN = float(sys.argv[2]); TOL = float(sys.argv[3]); SPLAT = sys.argv[4] == '1'; SUB = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    tot = np.zeros(4)
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
        vm = VoxelMap(s['depth'][take], s['origins'][take], s['rotations'][take], lo, hi-lo, res, path[0, 2], device, splat=SPLAT, sub=SUB)
        cams = pack_cams(s['origins'][take], s['rotations'][take])
        imgs = vm.render(cams, RENDER)
        g = vm.grid(); band = slice(int((Y_BELOW+.05)/DY)*SUB, int((Y_BELOW+1.5)/DY)*SUB)
        obst = (g[:, :, band] == 2).any(2).cpu().numpy()
        if SUB > 1: obst = obst.reshape(obst.shape[0]//SUB, SUB, obst.shape[1]//SUB, SUB).any((1, 3))
        clear = distance_transform_edt(~obst)*res
        free = gk[gs == 1]
        unk = candidates(gk, path, res)
        out = []
        for name, keys in [('graphFREE', free), ('UNKNOWN', unk)]:
            pr = vm.probes(keys, np.full(len(keys), path[0, 2], np.float32), cams, imgs, np.arange(len(cams)), tol_occ=TOL).cpu().numpy()
            ij = keys.astype(np.int64)-lo
            sup = clear[ij[:, 0], ij[:, 1]] > .25+res/np.sqrt(2)+.008-res/2
            if name == 'graphFREE': tot[0] += ((pr[:, 0] > 0) & sup).sum(); tot[1] += len(keys)
            else: tot[2] += ((pr[:, 0] > 0) & sup).sum(); tot[3] += len(keys)
            out.append('%s n=%d frustum %.3f known %.3f possible %.3f known&sup %.3f unc>0 %.3f' % (
                name, len(keys), (pr[:, 4] > 0).mean(), (pr[:, 0] > 0).mean(), (pr[:, 1] > 0).mean(), ((pr[:, 0] > 0) & sup).mean(), (pr[:, 3] > 0).mean()))
            if name == 'UNKNOWN':
                kn = (pr[:, 0] > 0) & sup
                out.append('UNKNOWN known&sup count %d' % kn.sum())
    print('sub %d render %d margin %.2f tol %.2f splat %s: certified FREE reproduced %.3f ; uncertified UNKNOWN wrongly known-visible %d of %d' % (
        SUB, RENDER, G.MARGIN, TOL, SPLAT, tot[0]/tot[1], tot[2], tot[3]))


if __name__ == '__main__':
    main()
