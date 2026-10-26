"""Coordinate-aligned prediction/error figures on TRAIN-split validation roots."""
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from .data import split_roots, load_root
from .features import channel_index, build_inputs, crop_box, MAP_CHANNELS
from .model import GraphRunner, build_model
from .train import build_context, frame_labels, CIN


@torch.no_grad()
def run(checkpoint, dest, device='cuda:0', roots=6, seed=0, tau=.6, taub=.5, short=False, long=False):
    saved = torch.load(checkpoint, map_location='cpu')
    cidx = channel_index(saved['config']); cin = len(saved['config']['map_channels'])+len(saved['config']['query_channels'])
    model = build_model(cin, saved['config']); model.load_state_dict(saved['model'])
    runner = GraphRunner(model.to(device), device, batches=(1,), sizes=(), channels=cin, capture=False)
    _, val = split_roots()
    rng = np.random.default_rng(seed)
    made = 0
    for idx in rng.choice(len(val), roots, replace=False):
        row = val[int(idx)]
        groups = load_root(row, tracks=bool(saved['config'].get('human_filter')))
        if long:
            cands = [(g, i) for g in groups for i, w in enumerate(g['windows']) if len(w['tkeys']) > 300 and w['path'][-1, 0]-w['path'][0, 0] > 1.5]
        elif short:
            cands = [(g, i) for g in groups for i, w in enumerate(g['windows']) if 5 < len(w['tkeys']) < 100 and 1 < len(w['path']) and w['path'][-1, 0]-w['path'][0, 0] <= .5]
        else:
            cands = [(g, i) for g in groups for i, w in enumerate(g['windows']) if len(w['tkeys']) > 80 and len(w['path']) > 1]
        if not cands:
            continue
        g, i = cands[int(rng.integers(len(cands)))]
        ctx = build_context(g, device, config=saved['config'])
        w = g['windows'][i]
        lo, hi = crop_box(w['path'], ctx.res)
        x, mask, fr = build_inputs(ctx, [w['path']], [(lo, hi-lo)])
        xin = x[:, cidx] if cidx is not None else x
        y = frame_labels(mask, fr, [w], ctx.lo)[0].cpu().numpy()
        logits = runner(xin, frames=fr[:, 2:])[0].float()
        r = torch.sigmoid(logits[0]).cpu().numpy(); b = torch.sigmoid(logits[1]).cpu().numpy()
        pf, pb = r*(1-b), r*b
        pred = np.where((pf > .4) | (pb > .5), np.where(pb-.5 > pf-.4, 2, 1), 0)
        h, wd = int(fr[0, 2]), int(fr[0, 3])
        xm = x[0].float().cpu().numpy()
        base = np.zeros((h, wd, 3))
        nm = len(MAP_CHANNELS)
        occ = xm[MAP_CHANNELS.index('mid_occ'), :h, :wd]+xm[MAP_CHANNELS.index('high_occ'), :h, :wd]
        floor = xm[MAP_CHANNELS.index('floor_occ'), :h, :wd]
        base[..., :] = .12
        base[floor > 0] = .35
        base[occ > 0] = [.45, .3, .15]
        base[xm[0, :h, :wd] > 0] = [.25, .45, .8]
        base[xm[1, :h, :wd] > 0] = [.8, .25, .25]
        yy, pp = y[:h, :wd], pred[:h, :wd]
        tgt = base.copy(); tgt[yy == 1] = [0, 1, 0]; tgt[yy == 2] = [1, 0, 1]
        prd = base.copy(); prd[(pp == 1) & (yy >= 0)] = [0, 1, 0]; prd[(pp == 2) & (yy >= 0)] = [1, 0, 1]
        err = base.copy()
        err[(yy > 0) & (pp == 0)] = [1, .6, 0]      # missed
        err[(yy == 0) & (pp > 0)] = [0, .8, 1]      # false revelation
        err[(yy > 0) & (pp > 0) & (yy != pp)] = [1, 1, 0]  # wrong type
        err[(yy > 0) & (pp == yy)] = [1, 1, 1]      # correct
        fig, ax = plt.subplots(1, 4, figsize=(28, 7))
        pij = np.floor(w['path'][:, [1, 3]]/ctx.res)-lo
        for k, (im, title) in enumerate([(tgt, 'target FREE green / BLOCKED magenta'), (prd, 'prediction'), (err, 'white ok, orange missed, cyan false, yellow wrong type'),
                                         (r[:h, :wd], 'revelation probability')]):
            ax[k].imshow(np.transpose(im, (1, 0, 2)) if im.ndim == 3 else im.T, origin='lower', interpolation='nearest')
            ax[k].plot(pij[:, 0], pij[:, 1], 'c-', lw=2); ax[k].set_title(title)
        pp = np.where(yy >= 0, pp, 0)
        nt = (yy > 0).sum(); tp = ((yy > 0) & (pp == yy)).sum(); fp = ((yy == 0) & (pp > 0)).sum(); wt = ((yy > 0) & (pp > 0) & (yy != pp)).sum()
        fig.suptitle('%s: %d poses %.2fs | targets %d correct %d false %d wrongtype %d | window IoU %.3f' % (
            row['root_id'][:8], len(w['path']), w['path'][-1, 0]-w['path'][0, 0], nt, tp, fp, wt, tp/max(nt+(pp > 0).sum()-tp, 1)))
        plt.tight_layout(); plt.savefig('%s/pred_%02d.png' % (dest, made), dpi=55); plt.close()
        made += 1
    print('made', made)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dest', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--roots', type=int, default=6)
    p.add_argument('--tau', type=float, default=.6)
    p.add_argument('--short', action='store_true')
    p.add_argument('--long', action='store_true')
    run(**vars(p.parse_args()))
