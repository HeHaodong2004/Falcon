"""Cache contract of the legal-map cache: the provenance range identity is the full-content identity,
repeated packets hit, and packets that share poses/times but differ in content miss (run: test_cache)."""
import copy
import numpy as np, torch
from eventwm_graph.stage2_repair.range_benchmark import workloads
from eventwm_graph.stage2_repair.range_identity import identity as range_identity
from eventwm_graph.stage2_claude.api import LegalVoxelPredictor
work = workloads('range_benchmark_256')
model = LegalVoxelPredictor('output/claude_stage2/final/v15.pt', device='cuda:0', lanes=3)
ok = True
for ex, packet, tracks in work[:6]:
    paths = [np.asarray(p['trajectory'], np.float64) for p in ex['paths'][:6]]
    model.clear_cache()
    a = model.predict_segments(partial_graph=ex['graph'], trajectories=paths, range_observations=packet, track_history=tracks)
    want = range_identity(np.asarray(packet['ranges'], np.float32), np.asarray(packet['origins'], np.float64), np.asarray(packet['rotations'], np.float64),
                          np.asarray(packet['timestamps'], np.float64), tracks, model.prepare_graph(ex['graph']).resolution, 1)
    ok &= all(d['provenance']['range_history_identity'] == want for d in a)
    ctx1 = next(iter(model.map_cache.values()))
    b = model.predict_segments(partial_graph=ex['graph'], trajectories=paths[:3], range_observations=packet, track_history=tracks)
    hit = next(reversed(model.map_cache.values())) is ctx1 and len(model.map_cache) == 1
    ok &= hit and all(np.array_equal(x['probabilities'], y['probabilities']) for x, y in zip(a[:3], b))
    # Same poses and times, one range value changed: the cheap key matches, the full identity must not.
    other = dict(packet); r = np.array(packet['ranges'], np.float32, copy=True); r.flat[12345] = r.flat[12345]+0.5 if np.isfinite(r.flat[12345]) else 3.
    other['ranges'] = r
    model.predict_segments(partial_graph=ex['graph'], trajectories=paths[:1], range_observations=other, track_history=tracks)
    ok &= len(model.map_cache) == 2
print('PASS' if ok else 'FAIL', 'provenance identity, warm hit, quick-key collision miss on 6 workloads')
