"""How well does the auxiliary head complete UNSEEN geometry? (validation)

Scores the wall-band / floor predictions only in columns that are UNKNOWN in
the legal input but observed somewhere in the Root's recorded frames.
"""
import argparse

import numpy as np
import torch

from .data import split_roots, load_root
from .features import channel_index, build_inputs, crop_box, MAP_CHANNELS
from .model import build_model
from .train import build_context, frame_labels, aux_targets, CIN


@torch.no_grad()
def run(checkpoint, device='cuda:0', roots=60):
    saved = torch.load(checkpoint, map_location='cpu')
    cidx = channel_index(saved['config']); cin = len(saved['config']['map_channels'])+len(saved['config']['query_channels'])
    model = build_model(cin, saved['config']); model.load_state_dict(saved['model']); model = model.to(device).eval()
    _, val = split_roots()
    val = sorted(val, key=lambda r: r['root_id'])[:roots]
    unk_idx = [MAP_CHANNELS.index(n) for n in ('mid_unk', 'high_unk')]
    stats = {k: np.zeros(4) for k in ('wall', 'floor')}   # tp, fp, fn, n
    dist_stats = {}
    for row in val:
        for g in load_root(row, tracks=bool(saved['config'].get('human_filter')), aux=True):
            frames = g.pop('aux_frames', None)
            if frames is None:
                continue
            ctx = build_context(g, device, config=saved['config'])
            windows = [w for w in g['windows'] if len(w['path']) > 1][:1]
            if not windows:
                continue
            boxes = [crop_box(w['path'], ctx.res) for w in windows]
            x, mask, fr = build_inputs(ctx, [w['path'] for w in windows], [(lo, hi-lo) for lo, hi in boxes])
            xin = x[:, cidx] if cidx is not None else x
            A = aux_targets(ctx, frames, fr, x.shape[2], x.shape[3], device)
            with torch.cuda.amp.autocast():
                logits = model(xin)
            pw = torch.sigmoid(logits[0, 2].float()) > .5
            pf = torch.sigmoid(logits[0, 3].float()) > .5
            unknown = (x[0, unk_idx[0]] > .99) & (x[0, unk_idx[1]] > .99) & (x[0, -1] > 0)
            obs = (A[0, 2] > 0) & unknown
            for name, p, t in (('wall', pw, A[0, 0] > 0), ('floor', pf, A[0, 1] > 0)):
                s = stats[name]
                s += np.array([(p & t & obs).sum().item(), (p & ~t & obs).sum().item(), (~p & t & obs).sum().item(), obs.sum().item()])
    for name, (tp, fp, fn, n) in stats.items():
        print('%s in unseen-but-later-observed columns: n=%d positive rate %.3f precision %.3f recall %.3f IoU %.3f' % (
            name, n, (tp+fn)/max(n, 1), tp/max(tp+fp, 1), tp/max(tp+fn, 1), tp/max(tp+fp+fn, 1)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--roots', type=int, default=60)
    run(**vars(p.parse_args()))
