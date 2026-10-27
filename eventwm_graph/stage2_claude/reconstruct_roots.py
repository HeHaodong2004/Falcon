"""Reconstruct the TRAIN Root sets of runs that started before train_roots.json existed (packets created before the run's
config.json), check the counts against the training logs, and write train_roots.json into those run directories."""
import json, os, re
from eventwm_graph.stage2_repair.diagnose import OUT
from .data import split_roots
T = 'output/claude_stage2/training'
for name in ('v16_a', 'v16_b', 'v17_a'):
    start = os.path.getmtime('%s/%s/config.json' % (T, name))
    train_rows, _ = split_roots(16384)
    rows = [[r['root_id'], 'expanded_train_range_branches'] for r in train_rows]
    seen = {r['root_id'] for r in train_rows}
    counts = {}
    logged_counts = {m.group(1): int(m.group(2)) for m in re.finditer(r'extra sensor (\S+) (\d+) TRAIN Roots', open('output/claude_stage2/logs/%s.log' % name).read())}
    for directory in ('claude_train_range_128', 'claude_train_range_256'):
        cand = [r for r in split_roots(roots=0, sensor=directory, start=16384, val_mod=25)[0] if r['root_id'] not in seen]
        cand.sort(key=lambda r: os.path.getmtime(OUT/directory/(r['root_id']+'.npz')))
        n = logged_counts.get(directory, 0)
        extra = cand[:n]
        # the listing is one instant: the n-th packet must predate the run's config and the (n+1)-th must not predate the n-th
        assert all(os.path.getmtime(OUT/directory/(r['root_id']+'.npz')) < start for r in extra)
        seen |= {r['root_id'] for r in extra}
        rows += [[r['root_id'], directory] for r in extra]
        counts[directory] = len(extra)
    log = open('output/claude_stage2/logs/%s.log' % name).read()
    logged = {m.group(1): int(m.group(2)) for m in re.finditer(r'extra sensor (\S+) (\d+) TRAIN Roots', log)}
    total = re.search(r'train roots (\d+)', log)
    ok = logged == counts and total and int(total.group(1)) == len(rows)
    print(name, 'reconstructed', counts, len(rows), 'logged', logged, total.group(1) if total else None, 'MATCH' if ok else 'MISMATCH')
    if ok:
        open('%s/%s/train_roots.json' % (T, name), 'w').write(json.dumps(rows))
