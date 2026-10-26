"""128 vs 256 px legal sensing for a frozen model, on TRAIN-validation Roots
that have both packet resolutions (selection without development data).

Uses the operating point frozen in the checkpoint (select.py --freeze-to)."""
import json
import sys

import torch

from eventwm_graph.stage2_repair.diagnose import OUT
from .data import split_roots
from .select import collect, score_map, class_map

ckpt, dev = sys.argv[1], sys.argv[2]
decision = torch.load(ckpt, map_location='cpu')['decision']
if decision['rule'] == 'class':
    cmap = class_map('class', decision['free_threshold'], decision['blocked_class_threshold'])
else:
    cmap = class_map('reveal', decision['threshold'], decision['blocked_threshold'])
_, val = split_roots()
rows = sorted([r for r in val if (OUT/'expanded_train_range_branches_256'/(r['root_id']+'.json')).exists()], key=lambda r: r['root_id'])
print('validation Roots with both resolutions:', len(rows), 'scenes', len({r['scene_id'] for r in rows}), flush=True)
result = {}
for sensor in ('expanded_train_range_branches', 'expanded_train_range_branches_256'):
    rec = collect(ckpt, dev, rows=rows, sensor=sensor)
    h, p = score_map(rec, cmap)
    result[sensor] = dict(hier=h['label_aware_delta_iou'], pooled=p['label_aware_delta_iou'])
    print(sensor, 'frozen rule %s hier %.4f pooled %.4f' % (json.dumps({k: v for k, v in decision.items() if k != 'validation'}),
          h['label_aware_delta_iou'], p['label_aware_delta_iou']), flush=True)
best = max(result, key=lambda k: result[k]['hier'])
print('SENSOR', '256' if best.endswith('_256') else '128', json.dumps(result), flush=True)
