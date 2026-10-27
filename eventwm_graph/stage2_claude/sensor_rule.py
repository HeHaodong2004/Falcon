"""Best class-rule operating point per sensor resolution on the 199 TRAIN-validation Roots with both packets (diagnostic)."""
import json, sys
import numpy as np
from eventwm_graph.stage2_repair.diagnose import OUT
from .data import split_roots
from .select import collect, score_map, class_map
ckpt = sys.argv[1]
_, val = split_roots()
rows = sorted([r for r in val if (OUT/'expanded_train_range_branches_256'/(r['root_id']+'.json')).exists()], key=lambda r: r['root_id'])
grid = [(tf, tb) for tf in np.arange(.35, .65, .025) for tb in np.arange(.4, .7, .025)]
best = {}
recs = {}
for sensor in ('expanded_train_range_branches', 'expanded_train_range_branches_256'):
    recs[sensor] = collect(ckpt, 'cuda:0', rows=rows, sensor=sensor)
    scores = {g: score_map(recs[sensor], class_map('class', *g))[0]['label_aware_delta_iou'] for g in grid}
    g = max(scores, key=scores.get); best[sensor] = (g, scores)
    print(sensor, 'best (%.3f, %.3f) %.4f' % (g[0], g[1], scores[g]), flush=True)
a, b = best['expanded_train_range_branches'], best['expanded_train_range_branches_256']
print('256 px scored at the 128-selected point %.4f vs its own best %.4f' % (b[1][a[0]], b[1][b[0]]))
print('frozen v15 point (0.500, 0.575) at 256 px: %.4f' % min(b[1].items(), key=lambda kv: abs(kv[0][0]-.5)+abs(kv[0][1]-.575))[1])
