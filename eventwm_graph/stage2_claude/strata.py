"""Error strata on TRAIN-split validation: where do errors come from?

Each candidate cell is assigned to one simulation-evidence stratum (from the
legal input channels). Reports pooled confusion per stratum and the share of
total missed / false / wrong-type errors.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from .data import split_roots, load_root
from .features import channel_index, build_inputs, crop_box, MAP_CHANNELS, QUERY_CHANNELS
from .model import GraphRunner, build_model
from .train import build_context, frame_labels, CIN

NM = len(MAP_CHANNELS)
Q = {n: NM+i for i, n in enumerate(QUERY_CHANNELS)}
STRATA = ['wall_known', 'wall_after_unknown', 'probe_known', 'probe_possible_mostly_known', 'probe_possible_uncertain',
          'frustum_blocked', 'unk_traversal_only', 'no_evidence']


def stratum(x):
    s = torch.full(x.shape[1:], len(STRATA)-1, dtype=torch.long, device=x.device)
    order = [('unk_traversal_only', x[Q['wall_unknown_traversal']] > 0),
             ('frustum_blocked', x[Q['probe_frustum_poses']] > 0),
             ('probe_possible_uncertain', x[Q['probe_possible']] > 0),
             ('probe_possible_mostly_known', (x[Q['probe_possible']] > 0) & (x[Q['probe_best_uncertain']] < .25)),
             ('probe_known', x[Q['probe_known']] > 0),
             ('wall_after_unknown', x[Q['wall_after_unknown']] > 0),
             ('wall_known', x[Q['wall_known']] > 0)]
    for name, m in order:
        s[m] = STRATA.index(name)
    return s


@torch.no_grad()
def run(checkpoint, device='cuda:0', roots=120, tau=.5, taub=.5):
    saved = torch.load(checkpoint, map_location='cpu')
    cidx = channel_index(saved['config']); cin = len(saved['config']['map_channels'])+len(saved['config']['query_channels'])
    model = build_model(cin, saved['config']); model.load_state_dict(saved['model'])
    runner = GraphRunner(model.to(device), device, batches=(1,), sizes=(240, 256, 272, 288, 304, 320, 352, 384), channels=cin)
    _, val = split_roots()
    val = sorted(val, key=lambda r: r['root_id'])[:roots]
    cm = np.zeros((len(STRATA), 3, 3))
    by_dur = defaultdict(lambda: np.zeros((3, 3)))
    for row in val:
        for g in load_root(row, tracks=bool(saved['config'].get('human_filter'))):
            ctx = build_context(g, device, config=saved['config'])
            windows = [w for w in g['windows'] if len(w['path']) > 1]
            if not windows:
                continue
            boxes = [crop_box(w['path'], ctx.res) for w in windows]
            x, mask, fr = build_inputs(ctx, [w['path'] for w in windows], [(lo, hi-lo) for lo, hi in boxes])
            xin = x[:, cidx] if cidx is not None else x
            y = frame_labels(mask, fr, windows, ctx.lo)
            logits = runner(xin, frames=fr[:, 2:])
            r = torch.sigmoid(logits[:, 0].float()); b = torch.sigmoid(logits[:, 1].float())
            for i, w in enumerate(windows):
                valid = y[i] >= 0
                st = stratum(x[i].float())[valid]
                yy = y[i][valid].long()
                pred = torch.where(r[i][valid] > tau, torch.where(b[i][valid] > taub, 2, 1), 0).long()
                c = torch.bincount((st*3+yy)*3+pred, minlength=len(STRATA)*9).reshape(len(STRATA), 3, 3).cpu().numpy()
                cm += c
                d = w['path'][-1, 0]-w['path'][0, 0]
                by_dur['<=0.5' if d <= .5 else '0.5-1' if d <= 1 else '1-2' if d <= 2 else '>2'] += c.sum(0)
    tot = cm.sum(0)
    def iou(c):
        tp = c[1, 1]+c[2, 2]
        return tp/max(c[1:].sum()+c[:, 1:].sum()-tp, 1)
    print('overall pooled IoU %.4f' % iou(tot), 'targets', int(tot[1:].sum()))
    miss_t = tot[1:, 0].sum(); fp_t = tot[0, 1:].sum(); wt_t = tot[1, 2]+tot[2, 1]
    print('%-30s %9s %9s %9s %7s %7s %7s %7s' % ('stratum', 'cells', 'targets', 'pred', 'IoU', 'miss%', 'false%', 'wtype%'))
    for k, name in enumerate(STRATA):
        c = cm[k]
        print('%-30s %9d %9d %9d %7.3f %7.3f %7.3f %7.3f' % (name, c.sum(), c[1:].sum(), c[:, 1:].sum(), iou(c),
              c[1:, 0].sum()/max(miss_t, 1), c[0, 1:].sum()/max(fp_t, 1), (c[1, 2]+c[2, 1])/max(wt_t, 1)))
    print('per stratum confusion rows=true(U,F,B) cols=pred(U,F,B):')
    for k, name in enumerate(STRATA):
        print('  %-28s %s' % (name, cm[k].astype(np.int64).tolist()))
    for d, c in sorted(by_dur.items()):
        print('duration %-6s targets %8d pooled IoU %.4f' % (d, c[1:].sum(), iou(c)))
    print('targets by class FREE %d BLOCKED %d; missed FREE %d BLOCKED %d; false %d; wrong type %d' % (
        tot[1].sum(), tot[2].sum(), tot[1, 0], tot[2, 0], fp_t, wt_t))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--roots', type=int, default=120)
    p.add_argument('--tau', type=float, default=.5)
    run(**vars(p.parse_args()))
