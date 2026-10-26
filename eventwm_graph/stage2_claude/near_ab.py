"""Sensitivity of a trained model to the near-camera setting (validation)."""
import sys, torch
from .data import split_roots
from .model import build_model
from . import train as T
ckpt = sys.argv[1]
saved = torch.load(ckpt, map_location='cpu')
from .features import channel_index
cidx = channel_index(saved['config'])
m = build_model(len(saved['config']['map_channels'])+len(saved['config']['query_channels']), saved['config'])
m.load_state_dict(saved['model']); m = m.to('cuda:1')
_, val = split_roots(); val = sorted(val, key=lambda r: r['root_id'])[:60]
for near in (0., .5):
    T.GEOMETRY.update(splat=False, sub=2, render=192, occluders=True, near=near)
    v = T.validate(m, val, 'cuda:1', 60, taus=(.5, .6, .7), tracks=True, cidx=cidx)
    print('near', near, {t: round(x['hier'], 4) for t, x in v.items()}, flush=True)
