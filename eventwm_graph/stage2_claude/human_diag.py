"""Diagnostic (TRAIN validation): accuracy of windows with / without a legally
tracked human near the supplied path, and the share of targets they hold."""
import argparse
from collections import defaultdict

import numpy as np
import torch

from .data import split_roots, load_root
from .features import channel_index, build_inputs, crop_box
from .model import GraphRunner, build_model
from .train import build_context, frame_labels


@torch.no_grad()
def run(checkpoint, device='cuda:0', roots=120, tf=.425, tb=.525):
    saved = torch.load(checkpoint, map_location='cpu')
    cidx = channel_index(saved['config']); cin = len(saved['config']['map_channels'])+len(saved['config']['query_channels'])
    model = build_model(cin, saved['config']); model.load_state_dict(saved['model'])
    runner = GraphRunner(model.to(device), device, batches=(1,), sizes=(240, 256, 272, 288, 304, 320, 352, 384), channels=cin)
    _, val = split_roots()
    val = sorted(val, key=lambda r: r['root_id'])[:roots]
    cm = defaultdict(lambda: np.zeros((3, 3)))
    for row in val:
        for g in load_root(row, tracks=True):
            ctx = build_context(g, device, config=saved['config'])
            windows = [w for w in g['windows'] if len(w['path']) > 1]
            if not windows:
                continue
            boxes = [crop_box(w['path'], ctx.res) for w in windows]
            x, mask, fr = build_inputs(ctx, [w['path'] for w in windows], [(lo, hi-lo) for lo, hi in boxes])
            xin = x[:, cidx] if cidx is not None else x
            y = frame_labels(mask, fr, windows, ctx.lo)
            logits = runner(xin, frames=fr[:, 2:])
            r = torch.sigmoid(logits[:, 0].float()); b = torch.sigmoid(logits[:, 1].float())
            pf, pb = r*(1-b), r*b
            hum = g['current_humans'] if g['current_humans'] is not None else np.zeros((0, 3))
            for i, w in enumerate(windows):
                valid = y[i] >= 0
                yy = y[i][valid].long()
                f, bb = pf[i][valid], pb[i][valid]
                pred = torch.where((f > tf) | (bb > tb), torch.where(bb-tb > f-tf, 2, 1), 0).long()
                c = torch.bincount(3*yy+pred, minlength=9).reshape(3, 3).cpu().numpy()
                xy = w['path'][:, [1, 3]]
                if len(hum):
                    d = np.sqrt(((hum[None, :, [0, 2]]-xy[:, None, :])**2).sum(-1)).min()
                else:
                    d = 1e9
                key = 'human<2m' if d < 2 else 'human2-4m' if d < 4 else 'human4-8m' if d < 8 else 'none<8m'
                cm[key] += c
                cm['all'] += c
    def iou(c):
        tp = c[1, 1]+c[2, 2]
        return tp/max(c[1:].sum()+c[:, 1:].sum()-tp, 1)
    tot = cm['all'][1:].sum()
    for k in sorted(cm):
        print('%-10s targets %8d (%.3f) pooled IoU %.4f' % (k, cm[k][1:].sum(), cm[k][1:].sum()/tot, iou(cm[k])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--roots', type=int, default=120)
    run(**vars(p.parse_args()))
