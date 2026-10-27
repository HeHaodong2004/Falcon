"""Function-preserving widening of a trained MaskedUNet (Net2Wider).

Every level's channel count is multiplied by `factor`. A fixed random map per level assigns each new channel to an
old one: layers that PRODUCE a level's activations duplicate their output filters (and BatchNorm statistics),
layers that CONSUME them split each old input channel's weights over its copies with random ratios that sum to
one. The widened network computes exactly the same function at step 0, while the copies receive different
gradients and can diverge in training. Side-branch and head internal widths are kept; the context MLP's
hidden width follows the coarsest level.

usage: python -m ... widen <checkpoint> <dest> [factor] [seed]
"""
import sys

import torch

from .model import build_model


def level_maps(widths, new_widths, gen):
    maps = []
    for w, nw in zip(widths, new_widths):
        extra = torch.randint(0, w, (nw-w,), generator=gen)
        maps.append(torch.cat([torch.arange(w), extra]))
    return maps


def split_ratios(gmap, rows, gen):
    """Per (output row, new input channel) ratios; for every old channel the ratios of its copies sum to 1."""
    w = int(gmap.max())+1
    r = torch.rand(rows, len(gmap), generator=gen)*.5+.25
    sums = torch.zeros(rows, w).index_add_(1, gmap, r)
    return r/sums[:, gmap]


def layer_levels(name, nlev):
    """(input level, output level) of a parameterized layer; None = width unchanged; ('cat', i, j) = concat input."""
    p = name.split('.')
    if p[0] == 'stem':
        return None, 0
    if p[0] == 'stem_blocks':
        return 0, 0
    if p[0] == 'down':
        d = int(p[1])
        return (d, d+1) if p[2] == '0' else (d+1, d+1)
    if p[0] in ('side', 'dside'):
        lev = int(p[1])
        if p[2] == 'inp':
            return lev, None
        if p[2] == 'out':
            return None, lev
        return None, None
    if p[0] == 'context':
        # hidden width = coarsest width: the hidden units use the coarsest level's map (SiLU is elementwise)
        return nlev-1, nlev-1
    if p[0] == 'up':
        i = nlev-1-int(p[1])
        return (('cat', i, i-1), i-1) if p[2] == '0' else (i-1, i-1)
    if p[0] == 'head_a':
        return 0, None
    return None, None


def widen(saved, factor=1.5, seed=0):
    config = dict(saved['config'])
    assert config.get('arch', 'masked_unet') == 'masked_unet' and not config.get('attn'), 'masked U-Net without attention only'
    widths = tuple(config['widths'])
    new_widths = tuple(int(round(w*factor)) for w in widths)
    gen = torch.Generator().manual_seed(seed)
    maps = level_maps(widths, new_widths, gen)
    cin = len(config['map_channels'])+len(config['query_channels'])
    new_config = dict(config, widths=new_widths, widened_from=dict(widths=widths, factor=factor, seed=seed))
    old_sd = saved['model']
    model = build_model(cin, new_config)
    new_sd = model.state_dict()
    nlev = len(widths)
    out = {}
    for key, ref in new_sd.items():
        old = old_sd[key]
        module, leaf = key.rsplit('.', 1)
        lin, lout = layer_levels(_layer_name(module), nlev)
        t = old.clone()
        if t.dim() == 0:
            out[key] = t
            continue
        # output dimension (dim 0 of every weight/bias/BN buffer) for producers
        if lout is not None:
            t = t.index_select(0, maps[lout])
        # input dimension (dim 1 of conv / linear weights) for consumers
        if leaf == 'weight' and t.dim() >= 2 and lin is not None:
            if isinstance(lin, tuple):
                _, i, j = lin
                gi, gj = maps[i], maps[j]
                g = torch.cat([gi, gj+widths[i]])
                ratios = torch.cat([split_ratios(gi, t.shape[0], gen), split_ratios(gj, t.shape[0], gen)], 1)
            else:
                g = maps[lin]
                ratios = split_ratios(g, t.shape[0], gen)
            t = t.index_select(1, g)
            t = t*ratios.view(t.shape[0], t.shape[1], *([1]*(t.dim()-2)))
        assert t.shape == ref.shape, (key, tuple(t.shape), tuple(ref.shape))
        out[key] = t.to(ref.dtype)
    model.load_state_dict(out)
    result = dict(saved)
    result['model'] = out
    result['config'] = new_config
    result.pop('decision', None)
    result.pop('val', None)
    return result


def _layer_name(module):
    """Parameterized layer path (strip the conv / bn / linear leaf containers)."""
    for suffix in ('.conv', '.bn'):
        if module.endswith(suffix):
            return module[:-len(suffix)]
    return module


if __name__ == '__main__':
    src, dest = sys.argv[1], sys.argv[2]
    factor = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    saved = torch.load(src, map_location='cpu')
    wide = widen(saved, factor, seed)
    torch.save(wide, dest)
    print('widened', saved['config']['widths'], '->', wide['config']['widths'], '->', dest)
