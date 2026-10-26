"""Diagnostic figure: current humans vs the constant-velocity future-sweep channel."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .data import split_roots, load_root
from .features import build_inputs, crop_box, MAP_CHANNELS, QUERY_CHANNELS
from . import train as T

dest = sys.argv[1]
T.GEOMETRY.update(splat=False, sub=2, render=192, occluders=True, near=.5)
_, val = split_roots()
made = 0
for row in sorted(val, key=lambda r: r['root_id'])[:60]:
    for g in load_root(row, tracks=True):
        hm = g['human_motion']
        if hm is None or not len(hm) or np.abs(hm[:, [3, 5]]).max() < .3:
            continue
        w = max(g['windows'], key=lambda w: w['path'][-1, 0]-w['path'][0, 0])
        if len(w['path']) < 2:
            continue
        ctx = T.build_context(g, 'cuda:0')
        lo, hi = crop_box(w['path'], ctx.res)
        x, mask, fr = build_inputs(ctx, [w['path']], [(lo, hi-lo)])
        nm = len(MAP_CHANNELS)
        h_, w_ = int(fr[0, 2]), int(fr[0, 3])
        now = x[0, MAP_CHANNELS.index('human_now'), :h_, :w_].float().cpu().numpy()
        fut = x[0, nm+QUERY_CHANNELS.index('human_future'), :h_, :w_].float().cpu().numpy()
        occ = x[0, MAP_CHANNELS.index('mid_occ'), :h_, :w_].float().cpu().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        pij = np.floor(w['path'][:, [1, 3]]/ctx.res)-lo
        ax[0].imshow((occ > 0).T*.4+now.T, origin='lower'); ax[0].set_title('mid occupied + human_now')
        ax[1].imshow((occ > 0).T*.3+fut.T, origin='lower'); ax[1].set_title('human_future (dur %.2fs) v=%s' % (w['path'][-1, 0]-w['path'][0, 0], np.round(hm[:, [3, 5]], 2).tolist()))
        for a in ax:
            a.plot(pij[:, 0], pij[:, 1], 'c-')
        plt.tight_layout(); plt.savefig('%s/humans_%02d.png' % (dest, made), dpi=60); plt.close()
        made += 1
        break
    if made >= 2:
        break
print('made', made)
