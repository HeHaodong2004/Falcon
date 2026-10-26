"""Why do certified FREE cells look not-known-visible from the poses that
certified them? Per failing probe: height level, blocked vs uncertain, and the
state/height of the voxel where the render ray stops. TRAIN roots, past poses.
"""
import sys
from collections import Counter

import numpy as np
import torch

from eventwm_graph.stage2_repair.diagnose import OUT, HF, read
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from .data import split_roots
from .geometry import VoxelMap, pack_cams, DY, Y_BELOW
from .sim_eval import crop_box


def probe_points(key, res, h):
    cx, cz = (key[0]+.5)*res, (key[1]+.5)*res
    halo = .25+res/np.sqrt(2)
    pts = [(cx, cz)]+[(cx+sx*.49*res, cz+sz*.49*res) for sx, sz in ((-1, -1), (-1, 1), (1, -1), (1, 1))]
    pts += [(cx+np.cos(a)*halo, cz+np.sin(a)*halo) for a in np.arange(16)*2*np.pi/16]
    return np.array([[x, h+lv, z] for lv in (.025, .7, 1.4) for x, z in pts])


def main(n=10, sub=2, render=192, device='cuda:0'):
    train, _ = split_roots()
    rng = np.random.default_rng(11)
    reasons = Counter(); levels = Counter(); stop_state = Counter(); stop_rel_h = []; stop_t = []; stop_cov = Counter()
    cells = 0; failing = 0
    for idx in rng.choice(len(train), n, replace=False):
        row = train[int(idx)]
        packed = read(HF/'packed/full'/(row['root_id']+'.json')); a = load_arrays(packed)
        z = np.load(OUT/'expanded_train_range_branches'/(row['root_id']+'.npz')); s = {k: z[k] for k in z.files}
        res = float(a['geometry'][0])
        seg = get_segment(a, 0); gk, gs, path = seg[0], seg[1], seg[3]
        take = ancestor_frames(s['parents'], s['history_count'], packed['segments'][0]['start_node'])
        lo, hi = crop_box(path, res)
        vm = VoxelMap(s['depth'][take], s['origins'][take], s['rotations'][take], lo, hi-lo, res, path[0, 2], device, splat=False, sub=sub, near=.5)
        cams = pack_cams(s['origins'][take], s['rotations'][take])
        s_depth = s['depth'][take]
        imgs = vm.render(cams, render).cpu().numpy()
        free = gk[gs == 1]
        pr = vm.probes(free, np.full(len(free), path[0, 2], np.float32), cams, torch.as_tensor(imgs).to(device), np.arange(len(cams)),
                       tol_occ=.09, tol_unk=.06).cpu().numpy()
        grid = vm.grid().cpu().numpy()
        for c in np.flatnonzero(pr[:, 0] == 0):
            cells += 1
            P = probe_points(free[c], res, path[0, 2])
            best = None
            for f in range(len(cams)):
                o = cams[f, :3].astype(np.float64); R = cams[f, 3:].reshape(3, 3).astype(np.float64)
                d = P-o; loc = d@R; fw = -loc[:, 2]; dist = np.linalg.norm(d, axis=1)
                inside = (fw > 1e-6) & (np.abs(loc[:, 0]) <= fw) & (np.abs(loc[:, 1]) <= fw) & (dist <= 10)
                if not inside.all():
                    continue
                j = np.clip(np.floor((loc[:, 0]/fw+1)*.5*render).astype(int), 0, render-1)
                i = np.clip(np.floor((1-loc[:, 1]/fw)*.5*render).astype(int), 0, render-1)
                r = imgs[f, i, j]
                blk = r[:, 0] < dist-.09; unc = (~blk) & (r[:, 1] < dist-.06)
                bad = blk | unc
                if best is None or bad.sum() < best[0]:
                    best = (bad.sum(), f, blk, unc, r, dist, o, R, i, j)
            if best is None:
                reasons['no frustum pose'] += 1; continue
            failing += 1
            nb, f, blk, unc, r, dist, o, R, ii, jj = best
            reasons['min bad probes %s' % ('1' if nb == 1 else '2-3' if nb <= 3 else '4-10' if nb <= 10 else '>10')] += 1
            for k in np.flatnonzero(blk | unc):
                levels[('blocked' if blk[k] else 'uncertain', (.025, .7, 1.4)[k//21])] += 1
                # where does the render ray stop?
                lx = (jj[k]+.5)*2/render-1; ly = 1-(ii[k]+.5)*2/render
                dvec = R@np.array([lx, ly, -1.]); dvec /= np.linalg.norm(dvec)
                t = r[k, 0] if blk[k] else r[k, 1]
                pstop = o+dvec*(t+.01)
                stop_rel_h.append(pstop[1]-path[0, 2])
                stop_t.append((t, dist[k]))
                # Was the stop point inside ANY frame's frustum (could carving ever see it)?
                vis = False; deeper = False
                for g2 in range(len(cams)):
                    o2 = cams[g2, :3].astype(np.float64); R2 = cams[g2, 3:].reshape(3, 3).astype(np.float64)
                    l2 = (pstop-o2)@R2; f2 = -l2[2]
                    if f2 > 1e-6 and abs(l2[0]) < f2 and abs(l2[1]) < f2:
                        vis = True
                        S = s_depth.shape[-1]
                        jj2 = int(np.clip(np.floor((l2[0]/f2+1)*.5*S), 0, S-1)); ii2 = int(np.clip(np.floor((1-l2[1]/f2)*.5*S), 0, S-1))
                        rr = s_depth[g2, ii2, jj2]; rr = 99. if not np.isfinite(rr) else rr
                        if np.linalg.norm(pstop-o2) < rr-.045: deeper = True
                stop_cov[('in some frustum' if vis else 'never in frustum', 'some depth passes it' if deeper else 'no depth passes it')] += 1
    print('certified FREE cells not reproduced:', cells, 'with a frustum pose:', failing)
    print(dict(reasons))
    print('failing probes by (type, level):', dict(levels))
    h = np.array(stop_rel_h)
    if len(h):
        print('height (rel. base) where the render ray stops, percentiles 5/25/50/75/95:', np.round(np.percentile(h, [5, 25, 50, 75, 95]), 3))
        print('fraction stopping below base-0.05 (floor band): %.3f' % (h < -.05).mean())
        st = np.array(stop_t)
        print('stop distance from camera percentiles 5/25/50/75/95:', np.round(np.percentile(st[:, 0], [5, 25, 50, 75, 95]), 2))
        print('remaining distance to probe percentiles:', np.round(np.percentile(st[:, 1]-st[:, 0], [5, 25, 50, 75, 95]), 2))
        print('stop voxel coverage:', dict(stop_cov))


if __name__ == '__main__':
    main(*(int(v) for v in sys.argv[1:3])) if len(sys.argv) > 2 else main()
