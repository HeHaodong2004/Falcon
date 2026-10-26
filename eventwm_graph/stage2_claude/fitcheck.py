"""Train-vs-validation gap at one checkpoint (same metric code)."""
import sys, json, torch
import numpy as np
from .data import split_roots
from .model import build_model
from .train import validate, CIN
ckpt, dev = sys.argv[1], sys.argv[2]
saved = torch.load(ckpt, map_location='cpu')
m = build_model(CIN, saved['config']); m.load_state_dict(saved['model']); m = m.to(dev)
train, val = split_roots()
rng = np.random.default_rng(0)
tr = [train[i] for i in rng.choice(len(train), 120, replace=False)]
va = sorted(val, key=lambda r: r['root_id'])[:120]
for name, rows in (('train', tr), ('val', va)):
    v = validate(m, rows, dev, len(rows), taus=(.5, .6, .7), tracks=bool(saved['config'].get('human_filter')))
    print(name, {t: round(x['hier'], 4) for t, x in v.items()}, flush=True)
