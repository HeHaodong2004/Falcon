"""Diagnostic figure: expected unknown-space wall-ray hit density vs BLOCKED targets."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import split_roots, load_root
from .features import build_inputs, crop_box, MAP_CHANNELS, QUERY_CHANNELS
from . import train as T

dest = sys.argv[1]
T.GEOMETRY.update(splat=False, sub=2, render=192, occluders=True, near=.5)
_, val = split_roots()
made = 0
rng = np.random.default_rng(3)
for idx in rng.choice(len(val), 20, replace=False):
    row = val[int(idx)]
    groups = load_root(row, tracks=True)
    cands = [(g, w) for g in groups for w in g['windows'] if len(w['tkeys']) > 400 and len(w['path']) > 1]
    if not cands:
        continue
    g, w = cands[0]
    ctx = T.build_context(g, 'cuda:0')
    lo, hi = crop_box(w['path'], ctx.res)
    x, mask, fr = build_inputs(ctx, [w['path']], [(lo, hi-lo)])
    y = T.frame_labels(mask, fr, [w], ctx.lo)[0].cpu().numpy()
    nm = len(MAP_CHANNELS); h_, w_ = int(fr[0, 2]), int(fr[0, 3])
    hit = x[0, nm+QUERY_CHANNELS.index('wall_unknown_hit'), :h_, :w_].float().cpu().numpy()
    trav = x[0, nm+QUERY_CHANNELS.index('wall_unknown_traversal'), :h_, :w_].float().cpu().numpy()
    occ = x[0, MAP_CHANNELS.index('mid_occ'), :h_, :w_].float().cpu().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    pij = np.floor(w['path'][:, [1, 3]]/ctx.res)-lo
    ax[0].imshow(((occ > 0)*.3+(y[:h_, :w_] == 2)*1.).T, origin='lower'); ax[0].set_title('known mid occupied (dim) + BLOCKED targets')
    ax[1].imshow(trav.T, origin='lower'); ax[1].set_title('wall_unknown_traversal')
    ax[2].imshow(hit.T, origin='lower'); ax[2].set_title('wall_unknown_hit (log1p expected hits)')
    for a in ax:
        a.plot(pij[:, 0], pij[:, 1], 'c-')
    plt.tight_layout(); plt.savefig('%s/hit_%02d.png' % (dest, made), dpi=55); plt.close()
    made += 1
    if made >= 2:
        break
print('made', made, 'max hit', float(hit.max()))
