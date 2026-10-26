"""Informativeness of the past-visibility channels (validation roots)."""
import numpy as np, torch
from collections import defaultdict
from .data import split_roots, load_root
from .features import build_inputs, crop_box, MAP_CHANNELS, QUERY_CHANNELS
from . import train as T
T.GEOMETRY.update(splat=False, sub=2, render=192, occluders=True)
_, val = split_roots()
val = sorted(val, key=lambda r: r['root_id'])[:40]
NM = len(MAP_CHANNELS)
pk = MAP_CHANNELS.index('past_probe_known'); pp = MAP_CHANNELS.index('past_probe_possible')
qk = NM+QUERY_CHANNELS.index('probe_known'); qp = NM+QUERY_CHANNELS.index('probe_possible')
stats = defaultdict(lambda: np.zeros(3))
for row in val:
    for g in load_root(row, tracks=True):
        ctx = T.build_context(g, 'cuda:1')
        windows = [w for w in g['windows'] if len(w['path']) > 1]
        if not windows: continue
        boxes = [crop_box(w['path'], ctx.res) for w in windows]
        x, mask, fr = build_inputs(ctx, [w['path'] for w in windows], [(lo, hi-lo) for lo, hi in boxes])
        y = T.frame_labels(mask, fr, windows, ctx.lo)
        for i, w in enumerate(windows):
            v = y[i] >= 0
            yy = y[i][v]
            short = 'short' if w['path'][-1, 0]-w['path'][0, 0] <= .5 else 'long'
            for name, m in (('future_known&past_known', (x[i, qk] > 0) & (x[i, pk] > 0)), ('future_known&~past_known', (x[i, qk] > 0) & (x[i, pk] == 0)),
                            ('future_possible&past_possible', (x[i, qp] > 0) & (x[i, pp] > 0)), ('future_possible&~past_possible', (x[i, qp] > 0) & (x[i, pp] == 0))):
                mm = m[v]
                stats[(short, name)] += np.array([mm.sum().item(), (mm & (yy == 1)).sum().item(), (mm & (yy == 2)).sum().item()])
for k, (n, f, b) in sorted(stats.items()):
    print('%-6s %-34s cells %9d  FREE rate %.3f  BLOCKED rate %.3f' % (k[0], k[1], n, f/max(n, 1), b/max(n, 1)))
