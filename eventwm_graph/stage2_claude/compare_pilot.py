"""Window-level comparison: previous model A (receipts) vs a new pilot evaluation."""
import gzip, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from eventwm_graph.stage2_repair.diagnose import OUT
from eventwm_graph.stage2_segment.evaluate import rates

new = Path(sys.argv[1])
old = OUT/'range_runtime_online16384_epoch01_columns'
agg = defaultdict(lambda: np.zeros((2, 3, 3)))
wins = defaultdict(int)
for f in sorted(new.glob('*.json.gz')):
    n = json.loads(gzip.decompress(f.read_bytes()))
    o = json.loads((old/(n['root_id']+'.json')).read_text())
    om = {(r['start_node'], r['end_node']): r for r in o['records']}
    for r in n['records']:
        a = np.array(om[(r['start_node'], r['end_node'])]['cm']); b = np.array(r['cm'])
        assert a[1:].sum() == b[1:].sum()
        for g in r['groups']+['all']:
            agg[g] += np.stack([a, b]); wins[g] += 1
for g in sorted(agg):
    ra, rb = rates(agg[g][0]), rates(agg[g][1])
    print('%-22s windows %4d targets %8d | A pooled IoU %.3f (miss %.3f false %.3f) | new %.3f (miss %.3f false %.3f)' % (
        g, wins[g], agg[g][0][1:].sum(), ra['label_aware_delta_iou'], agg[g][0][1:, 0].sum()/agg[g][0][1:].sum(), agg[g][0][0, 1:].sum()/agg[g][0][1:].sum(),
        rb['label_aware_delta_iou'], agg[g][1][1:, 0].sum()/agg[g][1][1:].sum(), agg[g][1][0, 1:].sum()/agg[g][1][1:].sum()))
