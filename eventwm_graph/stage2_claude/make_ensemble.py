"""Combine trained checkpoints into one ensemble checkpoint (logit average)."""
import sys
import torch
from .features import MAP_CHANNELS, QUERY_CHANNELS, channel_index


def main(dest, *paths):
    members = []; state = {}
    geometry = None
    for i, p in enumerate(paths):
        saved = torch.load(p, map_location='cpu'); c = dict(saved['config'])
        idx = channel_index(c)
        c['channel_index'] = list(range(len(MAP_CHANNELS)+len(QUERY_CHANNELS))) if idx is None else list(idx)
        members.append(c)
        for k, v in saved['model'].items():
            state['nets.%d.%s' % (i, k)] = v
        g = (c.get('splat', True), c.get('sub', 1), c.get('render', 96), c.get('human_occluders', False), c.get('human_filter', False))
        if geometry is None:
            geometry = (g, c)
        elif g != geometry[0]:
            raise ValueError('ensemble members must share legal feature geometry: %s vs %s' % (g, geometry[0]))
    base = geometry[1]
    config = dict(arch='ensemble', members=members, map_channels=MAP_CHANNELS, query_channels=QUERY_CHANNELS,
                  splat=base.get('splat', True), sub=base.get('sub', 1), render=base.get('render', 96),
                  human_occluders=base.get('human_occluders', False), human_filter=base.get('human_filter', False),
                  near=max(float(m.get('near', 0.)) for m in members), valid_channel=len(MAP_CHANNELS)+len(QUERY_CHANNELS)-1,
                  sources=list(paths))
    torch.save(dict(model=state, config=config), dest)
    print('ensemble of', len(paths), '->', dest)


if __name__ == '__main__':
    main(sys.argv[1], *sys.argv[2:])
