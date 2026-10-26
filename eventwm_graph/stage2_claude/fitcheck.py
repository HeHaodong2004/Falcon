"""Train-vs-validation gap at one checkpoint (same metric code, the checkpoint's own geometry/channels)."""
import sys
import numpy as np
import torch
from .data import split_roots
from .features import channel_index
from .model import build_model
from . import train as T

ckpt, dev = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 120
saved = torch.load(ckpt, map_location='cpu')
c = saved['config']
T.GEOMETRY.update(splat=bool(c.get('splat', True)), sub=int(c.get('sub', 1)), render=int(c.get('render', 96)),
                  occluders=bool(c.get('human_occluders', False)), near=float(c.get('near', 0.)))
cidx = channel_index(c)
cin = len(c['map_channels'])+len(c['query_channels']) if 'map_channels' in c else T.CIN
m = build_model(cin, c); m.load_state_dict(saved['model']); m = m.to(dev)
train, val = split_roots()
rng = np.random.default_rng(0)
tr = [train[i] for i in rng.choice(len(train), n, replace=False)]
va = sorted(val, key=lambda r: r['root_id'])[:n]
for name, rows in (('train', tr), ('val', va)):
    v = T.validate(m, rows, dev, len(rows), taus=(.5, .6, .7), tracks=bool(c.get('human_filter')), cidx=cidx)
    print(name, {t: round(x['hier'], 4) for t, x in v.items()}, flush=True)
