"""Diagnostic: how far does pure legal-geometry certificate simulation go?

TRAIN roots only (packed windows + 128 measured range history). Reports
label-aware IoU of deterministic simulation variants on all candidate cells.
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from eventwm_graph.stage2_repair.diagnose import OUT, HF, read, source_rows
from eventwm_graph.stage2_repair.range_history import ancestor_frames
from eventwm_graph.stage2_hf.packed import load_arrays, get_segment
from eventwm_graph.stage2_quality.features import candidate_mask
from eventwm_graph.stage2_segment.evaluate import rates
from .geometry import VoxelMap, camera_extrinsic, proxy_cameras, pack_cams, wall_blocked, NY, DY, Y_BELOW

RADIUS = 11.


def crop_box(path, res):
    xy = path[:, [1, 3]].astype(np.float32)
    lo = np.floor((xy.min(0)-RADIUS)/res).astype(np.int64)-4
    hi = np.floor((xy.max(0)+RADIUS)/res).astype(np.int64)+5
    return lo, hi


def candidates(gkeys, path, res):
    lo, hi = crop_box(path, res)
    shape = hi-lo
    known = np.zeros(tuple(shape), bool)
    ij = gkeys.astype(np.int64)-lo
    ok = np.all((ij >= 0) & (ij < shape), axis=1)
    known[ij[ok, 0], ij[ok, 1]] = True
    keys = (np.argwhere(~known)+lo).astype(np.int32)
    xy = np.unique(path[:, [1, 3]].astype(np.float32), axis=0)
    return keys[candidate_mask(keys, xy, res)]


def lookup(keys, table_keys, table_values, default=0):
    out = np.full(len(keys), default, table_values.dtype)
    if not len(table_keys) or not len(keys):
        return out
    code = lambda k: (k[:, 0].astype(np.int64) << 32) + (k[:, 1].astype(np.int64)+(1 << 31))
    tc = code(table_keys); order = np.argsort(tc); tc = tc[order]
    kc = code(keys); at = np.searchsorted(tc, kc); at = np.minimum(at, len(tc)-1)
    hit = tc[at] == kc
    out[hit] = table_values[order][at[hit]]
    return out


def keyset_mask(keys, blocked):
    if not len(blocked):
        return np.zeros(len(keys), bool)
    return lookup(keys, blocked, np.ones(len(blocked), np.uint8)).astype(bool)


def run(n=24, device='cuda:0', sensor='expanded_train_range_branches', render=96, seed=7, offset=.03):
    rows, _ = source_rows('train')
    pick = np.random.default_rng(seed).choice(16384, n, replace=False)
    variants = ['known', 'optimistic', 'known_blocked_only', 'opt_blocked_only', 'known_free_nosupport', 'oracle_target_free_only']
    root_cm = {v: [] for v in variants}
    pooled = {v: np.zeros((3, 3), np.int64) for v in variants}
    tick = time.time()
    stats = defaultdict(float)
    for count, idx in enumerate(pick):
        row = rows[int(idx)]
        packed = read(HF/'packed/full'/(row['root_id']+'.json'))
        a = load_arrays(packed)
        z = np.load(OUT/sensor/(row['root_id']+'.npz')); s = {k: z[k] for k in z.files}
        res = float(a['geometry'][0])
        groups = defaultdict(list)
        for i, e in enumerate(packed['segments']):
            groups[e['start_node']].append(i)
        cms = {v: [] for v in variants}
        for sid, ids in groups.items():
            segs = [get_segment(a, i) for i in ids]
            gk, gs = segs[0][0], segs[0][1]
            take = ancestor_frames(s['parents'], s['history_count'], sid)
            ranges, origins, rotations = s['depth'][take], s['origins'][take], s['rotations'][take]
            p0 = segs[0][3][0]
            lo = np.min([crop_box(v[3], res)[0] for v in segs], 0)
            hi = np.max([crop_box(v[3], res)[1] for v in segs], 0)
            vm = VoxelMap(ranges, origins, rotations, lo, hi-lo, res, p0[2], device)
            ext = camera_extrinsic(p0, origins[-1], rotations[-1])
            # Support proxy: distance to occupied voxels in the obstacle band.
            g = vm.grid()
            band = slice(int((Y_BELOW+.05)/DY), int((Y_BELOW+1.5)/DY))
            obst = (g[:, :, band] == 2).any(2).cpu().numpy()
            clear = distance_transform_edt(~obst)*res if obst.any() else np.full(obst.shape, 99.)
            fl = g[:, :, 1:4]
            floor_occ = (fl == 2).any(2).cpu().numpy()
            void = ((fl == 1).all(2)).cpu().numpy()
            void_d = distance_transform_edt(~void)*res if void.any() else np.full(void.shape, 99.)
            for v in segs:
                path, tk, ts = v[3], v[4], v[5]
                keys = candidates(gk, path, res)
                y = lookup(keys, tk, ts.astype(np.uint8))
                missing = len(tk)-int((y > 0).sum())
                if len(path) < 2:
                    pred = {k: np.zeros(len(keys), np.uint8) for k in variants}
                else:
                    o, R = proxy_cameras(path[1:], ext)
                    cams = pack_cams(o, R)
                    rays = vm.wall_rays(cams)
                    bk = keyset_mask(keys, wall_blocked(rays, cams.astype(np.float64), path[1:], res, True, offset))
                    bo = keyset_mask(keys, wall_blocked(rays, cams.astype(np.float64), path[1:], res, False, offset))
                    imgs = vm.render(cams, render)
                    pr = vm.probes(keys, np.full(len(keys), path[0, 2], np.float32), cams, imgs, np.arange(len(cams))).cpu().numpy()
                    ij = keys.astype(np.int64)-lo
                    thr = .25+res/np.sqrt(2)+.008-res/2
                    sup = (clear[ij[:, 0], ij[:, 1]] > thr) & (void_d[ij[:, 0], ij[:, 1]] > thr)
                    sup_known = sup & floor_occ[ij[:, 0], ij[:, 1]]
                    known_free = (pr[:, 0] > 0) & sup_known
                    opt_free = (pr[:, 1] > 0) & sup
                    pred = {}
                    pred['known'] = np.where(bk, 2, np.where(known_free, 1, 0)).astype(np.uint8)
                    pred['optimistic'] = np.where(bo, 2, np.where(opt_free, 1, 0)).astype(np.uint8)
                    pred['known_blocked_only'] = np.where(bk, 2, 0).astype(np.uint8)
                    pred['opt_blocked_only'] = np.where(bo, 2, 0).astype(np.uint8)
                    pred['known_free_nosupport'] = np.where(bk, 2, np.where(pr[:, 0] > 0, 1, 0)).astype(np.uint8)
                    pred['oracle_target_free_only'] = np.where(y == 1, 1, 0).astype(np.uint8)
                    # coverage stats of true targets
                    f = y == 1; b = y == 2
                    stats['free'] += f.sum(); stats['blocked'] += b.sum()
                    stats['free_known_vis'] += (pr[f, 0] > 0).sum(); stats['free_possible'] += (pr[f, 1] > 0).sum()
                    stats['free_sup'] += sup[f].sum(); stats['free_frustum'] += (pr[f, 4] > 0).sum()
                    stats['blocked_known'] += bk[b].sum(); stats['blocked_opt'] += bo[b].sum()
                    stats['neg_known_vis'] += ((pr[:, 0] > 0) & sup_known)[y == 0].sum(); stats['free_known_sup'] += ((pr[:, 0] > 0) & sup_known)[f].sum(); stats['neg_opt_sup'] += ((pr[:, 1] > 0) & sup)[y == 0].sum(); stats['free_opt_sup'] += ((pr[:, 1] > 0) & sup)[f].sum(); stats['neg_possible'] += (pr[y == 0, 1] > 0).sum()
                for k in variants:
                    cm = np.bincount(3*y.astype(np.int64)+pred[k], minlength=9).reshape(3, 3)
                    cm[1, 0] += 0; cm[0, 0] += 0
                    if missing:
                        cm[1:, 0] += 0
                    cms[k].append(cm); pooled[k] += cm
        for k in variants:
            root_cm[k].append(np.mean(cms[k], 0))
        print(count, row['root_id'][:8], 'elapsed', round(time.time()-tick, 1), flush=True)
    out = {}
    for k in variants:
        out[k] = dict(root_mean=rates(np.mean(root_cm[k], 0)), pooled=rates(pooled[k]))
        print(k, 'root-mean IoU %.4f rev %.4f | pooled IoU %.4f rev %.4f' % (
            out[k]['root_mean']['label_aware_delta_iou'], out[k]['root_mean']['revelation_iou'],
            out[k]['pooled']['label_aware_delta_iou'], out[k]['pooled']['revelation_iou']))
        print('   pooled cm', pooled[k].tolist())
    print(json.dumps({k: float(v) for k, v in stats.items()}))
    st = stats
    print('FREE targets: known-visible %.3f possible %.3f support %.3f frustum %.3f' % (
        st['free_known_vis']/st['free'], st['free_possible']/st['free'], st['free_sup']/st['free'], st['free_frustum']/st['free']))
    print('BLOCKED targets: known %.3f optimistic %.3f' % (st['blocked_known']/st['blocked'], st['blocked_opt']/st['blocked']))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=24)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--render', type=int, default=96)
    p.add_argument('--offset', type=float, default=.03)
    run(**vars(p.parse_args()))
