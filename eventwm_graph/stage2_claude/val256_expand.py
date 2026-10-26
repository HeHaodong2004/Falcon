"""256 px legal range packets for TRAIN-split VALIDATION Roots (the unchanged
environment replay of stage2_repair.range_train_expand, restricted to chosen
Roots), so the 128 vs 256 px choice is made on more validation Roots.

Usage: val256_expand.py <part> <parts> <count>"""
import sys
from eventwm_graph.stage2_repair import range_train_expand as E
from eventwm_graph.stage2_repair.diagnose import OUT, source_rows
from .data import split_roots

part, parts, count = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
_, val = split_roots()
val = sorted(val, key=lambda r: r['root_id'])[:400]
dest = OUT/'expanded_train_range_branches_256'
todo = [r['root_id'] for r in val if not (dest/(r['root_id']+'.json')).exists()][:count]
mine = set(todo[part::parts])
rows, manifest = source_rows('train')
selected = [r for r in rows if r['root_id'] in mine]
print('part', part, 'of', parts, 'roots', len(selected), flush=True)
E.source_rows = lambda split: (selected, manifest)
E.run(roots=len(selected), offset=0, split='train', sensor_size=256, destination='expanded_train_range_branches_256')
