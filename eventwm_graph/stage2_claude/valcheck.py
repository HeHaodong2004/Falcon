"""TRAIN-validation check of any checkpoint (incl. ensembles) with the training-time
validate() on the first N validation Roots; prints the best single-threshold IoU."""
import json
import sys
import torch
from .data import split_roots
from .features import channel_index
from .model import build_model
from . import train as T

ckpt, dev = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 160
saved = torch.load(ckpt, map_location='cpu'); c = saved['config']
T.GEOMETRY.update(splat=bool(c.get('splat', True)), sub=int(c.get('sub', 1)), render=int(c.get('render', 96)),
                  occluders=bool(c.get('human_occluders', False)), near=float(c.get('near', 0.)))
cidx = channel_index(c)
m = build_model(len(c['map_channels'])+len(c['query_channels']), c); m.load_state_dict(saved['model']); m = m.to(dev)
_, val = split_roots()
val = sorted(val, key=lambda r: r['root_id'])
v = T.validate(m, val, dev, n, tracks=bool(c.get('human_filter')), cidx=cidx)
best = max(v.items(), key=lambda kv: kv[1]['hier'])
print('VALCHECK', ckpt, 'best tau', best[0], 'hier %.4f' % best[1]['hier'], json.dumps({k: round(x['hier'], 4) for k, x in v.items()}), flush=True)
