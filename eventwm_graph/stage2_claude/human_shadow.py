"""Diagnostic (TRAIN validation): how much of the error sits in cells whose
simulated certificate evidence changes when the currently tracked humans are
stamped as static occluders ("human-shadowed" cells)?

Builds every window's legal inputs twice (with / without the current-human
stamp), runs the frozen model on the normal inputs, and reports the pooled
label-aware IoU and target share of shadowed vs other cells, together with how
often shadowed cells are revealed in truth (the humans move during the window).
"""
import argparse

import numpy as np
import torch

from .data import split_roots, load_root
from .features import channel_index, build_inputs, crop_box, MAP_CHANNELS, QUERY_CHANNELS
from .model import GraphRunner, build_model
from .train import build_context, frame_labels

EVIDENCE = [MAP_CHANNELS.index(n) for n in MAP_CHANNELS if n.startswith('past_')] + \
           [len(MAP_CHANNELS)+QUERY_CHANNELS.index(n) for n in QUERY_CHANNELS if n.startswith(('probe_', 'wall_', 'center_'))]


def iou(c):
    tp = c[1, 1]+c[2, 2]
    return tp/max(c[1:].sum()+c[:, 1:].sum()-tp, 1)


@torch.no_grad()
def run(checkpoint, device='cuda:0', roots=120):
    saved = torch.load(checkpoint, map_location='cpu'); config = saved['config']
    assert config.get('human_occluders'), 'model was trained without the current-human stamp'
    d = saved['decision']; tf, tb = d['free_threshold'], d['blocked_class_threshold']
    cidx = channel_index(config); cin = len(config['map_channels'])+len(config['query_channels'])
    model = build_model(cin, config); model.load_state_dict(saved['model'])
    runner = GraphRunner(model.to(device), device, batches=(1,), sizes=(240, 256, 272, 288, 304, 320, 352, 384), channels=cin)
    _, val = split_roots()
    val = sorted(val, key=lambda r: r['root_id'])[:roots]
    cm = {k: np.zeros((3, 3)) for k in ('shadowed', 'other_in_human_windows', 'no_human_windows', 'all')}
    for row in val:
        for g in load_root(row, tracks=True):
            windows = [w for w in g['windows'] if len(w['path']) > 1]
            if not windows:
                continue
            boxes = [crop_box(w['path'], g['res']) for w in windows]
            ctx = build_context(g, device, config=config)
            x, mask, fr = build_inputs(ctx, [w['path'] for w in windows], [(lo, hi-lo) for lo, hi in boxes])
            has_humans = g.get('current_humans') is not None and len(g['current_humans']) > 0
            if has_humans:
                ctx0 = build_context(g, device, config=dict(config, human_occluders=False))
                x0, _, _ = build_inputs(ctx0, [w['path'] for w in windows], [(lo, hi-lo) for lo, hi in boxes])
                shadow = (x[:, EVIDENCE].float()-x0[:, EVIDENCE].float()).abs().amax(1) > 1e-3
            xin = x[:, cidx] if cidx is not None else x
            y = frame_labels(mask, fr, windows, ctx.lo)
            logits = runner(xin, frames=fr[:, 2:])
            r = torch.sigmoid(logits[:, 0].float()); b = torch.sigmoid(logits[:, 1].float())
            pf, pb = r*(1-b), r*b
            pred = torch.where((pf > tf) | (pb > tb), torch.where(pb-tb > pf-tf, 2, 1), 0).long()
            for i in range(len(windows)):
                valid = y[i] >= 0
                c = lambda m: torch.bincount(3*y[i][m].long()+pred[i][m], minlength=9).reshape(3, 3).cpu().numpy()
                cm['all'] += c(valid)
                if has_humans:
                    cm['shadowed'] += c(valid & shadow[i])
                    cm['other_in_human_windows'] += c(valid & ~shadow[i])
                else:
                    cm['no_human_windows'] += c(valid)
    tot = cm['all'][1:].sum()
    for k, c in cm.items():
        cells = c.sum()
        print('%-24s cells %9d targets %8d (%.3f of all targets) revealed rate %.3f pooled IoU %.4f' %
              (k, cells, c[1:].sum(), c[1:].sum()/tot, c[1:].sum()/max(cells, 1), iou(c)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--roots', type=int, default=120)
    run(**vars(p.parse_args()))
