"""Weight average ("model soup") of checkpoints with identical architecture and
channels (e.g. a warm-restart lineage): float tensors averaged, the rest copied
from the first. Usage: soup.py <dest> <ckpt> <ckpt> [...]"""
import sys
import torch

dest, paths = sys.argv[1], sys.argv[2:]
saved = [torch.load(p, map_location='cpu') for p in paths]
c0 = saved[0]['config']
DEFAULTS = dict(side=None, sub=1, render=96, splat=True, near=0., human_filter=False, human_occluders=False)
for s in saved[1:]:
    c = s['config']
    for k in ('arch', 'widths', 'blocks', 'side', 'map_channels', 'query_channels', 'valid_channel', 'sub', 'render', 'splat', 'near',
              'human_filter', 'human_occluders'):
        a, b = c.get(k, DEFAULTS.get(k)), c0.get(k, DEFAULTS.get(k))
        if (list(a) if isinstance(a, (list, tuple)) else a) != (list(b) if isinstance(b, (list, tuple)) else b):
            raise ValueError('soup members differ in %s: %s vs %s' % (k, a, b))
state = {}
for k, v in saved[0]['model'].items():
    state[k] = sum(s['model'][k].float() for s in saved)/len(saved) if v.dtype.is_floating_point else v
config = dict(c0, name='soup(%s)' % ','.join(paths), soup=list(paths))
torch.save(dict(model=state, config=config), dest)
print('soup of', len(paths), '->', dest)
