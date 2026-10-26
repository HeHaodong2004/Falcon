"""Operating-point selection on scene-disjoint TRAIN-split validation roots.

Records, per window, a histogram over (revelation prob bin, BLOCKED prob bin,
true label). The equal-scene/Root/window hierarchical label-aware IoU for any
(tau_r, tau_b) is then exact up to bin edges. Development data is never read.
"""
import argparse
import json
import time

import numpy as np
import torch

from eventwm_graph.stage2_segment.evaluate import rates
from .data import split_roots, load_root
from .features import channel_index, build_inputs, crop_box
from .model import GraphRunner, build_model
from .train import build_context, frame_labels, CIN, DEST

RB, BB = 100, 20


@torch.no_grad()
def collect(checkpoint, device='cuda:0', roots=400, offset=0):
    saved = torch.load(checkpoint, map_location='cpu')
    cidx = channel_index(saved['config']); cin = len(saved['config']['map_channels'])+len(saved['config']['query_channels'])
    model = build_model(cin, saved['config']); model.load_state_dict(saved['model'])
    runner = GraphRunner(model.to(device), device, batches=(1,), sizes=(240, 256, 272, 288, 304, 320, 352, 384), channels=cin)
    _, val = split_roots()
    val = sorted(val, key=lambda r: r['root_id'])[offset:offset+roots]
    records = []
    tick = time.time()
    for row in val:
        root = []
        for g in load_root(row, tracks=bool(saved['config'].get('human_filter'))):
            ctx = build_context(g, device, config=saved['config'])
            windows = g['windows']
            boxes = [crop_box(w['path'], ctx.res) for w in windows]
            x, mask, fr = build_inputs(ctx, [w['path'] for w in windows], [(lo, hi-lo) for lo, hi in boxes])
            xin = x[:, cidx] if cidx is not None else x
            y = frame_labels(mask, fr, windows, ctx.lo)
            logits = runner(xin, frames=fr[:, 2:])
            r = torch.sigmoid(logits[:, 0].float()); b = torch.sigmoid(logits[:, 1].float())
            for i, w in enumerate(windows):
                valid = y[i] >= 0
                yy = y[i][valid].long()
                ri = torch.clamp((r[i][valid]*RB).long(), max=RB-1)
                bi = torch.clamp((b[i][valid]*BB).long(), max=BB-1)
                if len(w['path']) < 2:
                    ri = torch.zeros_like(ri)
                h = torch.bincount((yy*RB+ri)*BB+bi, minlength=3*RB*BB).reshape(3, RB, BB).cpu().numpy()
                missing = len(w['tkeys'])-int((yy > 0).sum())
                root.append((h.astype(np.int32), missing))
        records.append((row['scene_id'], root))
    print('collected', len(records), 'roots in %.1fs' % (time.time()-tick), flush=True)
    return records


def confusion(h, missing, tr, tb):
    """3x3 confusion for revelation threshold index tr and blocked index tb."""
    cm = np.zeros((3, 3))
    cm[:, 0] = h[:, :tr, :].sum((1, 2))
    cm[:, 1] = h[:, tr:, :tb].sum((1, 2))
    cm[:, 2] = h[:, tr:, tb:].sum((1, 2))
    cm[1, 0] += missing
    return cm


def class_map(rule, a, b):
    """Decision per (r bin, b bin): 'reveal' (tau_r, tau_b) or 'class' (tau_F, tau_B)."""
    r = (np.arange(RB)+.5)/RB; bb = (np.arange(BB)+.5)/BB
    R, B = np.meshgrid(r, bb, indexing='ij')
    if rule == 'reveal':
        return np.where(R > a, np.where(B > b, 2, 1), 0)
    pf, pb = R*(1-B), R*B
    return np.where((pf > a) | (pb > b), np.where(pb-b > pf-a, 2, 1), 0)


def score_map(records, cmap):
    onehot = np.stack([cmap == k for k in range(3)], -1).astype(np.float64)
    scenes = {}
    pooled = np.zeros((3, 3))
    for scene, root in records:
        cms = []
        for h, m in root:
            cm = np.einsum('lrb,rbk->lk', h, onehot); cm[1, 0] += m; cms.append(cm)
        if not cms:
            continue
        pooled += np.sum(cms, 0)
        scenes.setdefault(scene, []).append(np.mean(cms, 0))
    hier = np.mean([np.mean(v, 0) for v in scenes.values()], 0)
    return rates(hier), rates(pooled)


def score(records, tr, tb):
    scenes = {}
    pooled = np.zeros((3, 3))
    for scene, root in records:
        cms = [confusion(h, m, tr, tb) for h, m in root]
        if not cms:
            continue
        pooled += np.sum(cms, 0)
        scenes.setdefault(scene, []).append(np.mean(cms, 0))
    hier = np.mean([np.mean(v, 0) for v in scenes.values()], 0)
    return rates(hier), rates(pooled)


def freeze(checkpoint, best, dest):
    """Deployable checkpoint with the frozen operating point embedded."""
    saved = torch.load(checkpoint, map_location='cpu')
    if best['rule'] == 'reveal':
        rule = dict(rule='reveal', threshold=float(best['tau_r']), blocked_threshold=float(best['tau_b']))
    else:
        rule = dict(rule='class', free_threshold=float(best['tau_free']), blocked_class_threshold=float(best['tau_blocked']))
    saved['decision'] = dict(**rule,
                             selected_on='TRAIN-split scene-disjoint validation (select.py)', validation=best)
    torch.save(saved, dest)
    return dest


def run(checkpoint, device='cuda:0', roots=400, out=None, freeze_to=None):
    records = collect(checkpoint, device, roots)
    grid = []
    for tr in range(20, 95, 2):
        for tb in (6, 8, 10, 12, 14):
            h, p = score(records, tr, tb)
            grid.append(dict(rule='reveal', tau_r=tr/RB, tau_b=tb/BB, hier=h['label_aware_delta_iou'], pooled=p['label_aware_delta_iou'], rev=h['revelation_iou']))
    for tf in np.arange(.2, .7, .025):
        for tbb in np.arange(.2, .7, .025):
            h, p = score_map(records, class_map('class', tf, tbb))
            grid.append(dict(rule='class', tau_free=float(tf), tau_blocked=float(tbb), hier=h['label_aware_delta_iou'],
                             pooled=p['label_aware_delta_iou'], rev=h['revelation_iou']))
    best = max(grid, key=lambda g: g['hier'])
    print('BEST', json.dumps(best))
    for g in sorted(grid, key=lambda g: -g['hier'])[:8]:
        print(json.dumps(g))
    result = dict(checkpoint=str(checkpoint), roots=roots, best=best, grid=grid,
                  role='TRAIN-split scene-disjoint validation; operating point frozen before any development evaluation')
    if out:
        (DEST/'selection').mkdir(parents=True, exist_ok=True)
        (DEST/'selection'/(out+'.json')).write_text(json.dumps(result, indent=1))
    if freeze_to:
        freeze(checkpoint, best, freeze_to)
        print('FROZEN', freeze_to)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--roots', type=int, default=400)
    p.add_argument('--out')
    p.add_argument('--freeze-to')
    run(**vars(p.parse_args()))
