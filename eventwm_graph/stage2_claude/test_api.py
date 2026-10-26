"""API contract tests: single vs batched equivalence, CUDA graph vs eager,
cache-state independence, exact domain, zero-horizon rule, no input mutation."""
import copy
import sys
import numpy as np
import torch
from eventwm_graph.stage2_repair.range_benchmark import workloads
from eventwm_graph.stage2_quality.features import candidate_mask
from .api import LegalVoxelPredictor
from .features import crop_box
from eventwm_graph.stage2_segment.prepared import content_key

ckpt = sys.argv[1]
dev = sys.argv[2] if len(sys.argv) > 2 else 'cuda:0'
work = workloads('range_benchmark')
g = LegalVoxelPredictor(ckpt, device=dev)
e = LegalVoxelPredictor(ckpt, device=dev, cuda_graphs=False)
worst = {'eager': 0., 'single': 0.}; sdiff = {'eager': 0, 'single': 0}; total = 0
for ex, packet, tracks in work[:12]:
    paths = [np.asarray(p['trajectory'], np.float64) for p in ex['paths'][:6]]
    graph_before = content_key(ex['graph']); ranges_before = packet['ranges'].copy()
    batched = g.predict_segments(partial_graph=ex['graph'], trajectories=paths, range_observations=packet, track_history=tracks)
    eager = e.predict_segments(partial_graph=ex['graph'], trajectories=paths, range_observations=packet, track_history=tracks)
    g.clear_cache()
    for qi, p in enumerate(paths):
        single = g.predict_segments(partial_graph=ex['graph'], trajectories=[p], range_observations=packet, track_history=tracks)[0]
        total += len(single['states'])
        for name, other in (('eager', eager[qi]), ('single', single)):
            np.testing.assert_array_equal(batched[qi]['candidate_cells'], other['candidate_cells'])
            worst[name] = max(worst[name], float(np.abs(batched[qi]['probabilities']-other['probabilities']).max()))
            sdiff[name] += int((batched[qi]['states'] != other['states']).sum())
        # exact original domain reference (CPU numba)
        res = float(ex['graph']['resolution']); lo, hi = crop_box(p, res)
        known = {tuple(c['cell']) for c in ex['graph']['cells']}
        grid = np.array([(x, z) for x in range(lo[0], hi[0]) for z in range(lo[1], hi[1]) if (x, z) not in known], np.int32)
        ref = grid[candidate_mask(grid, np.unique(p[:, [1, 3]].astype(np.float32), axis=0), res)]
        np.testing.assert_array_equal(ref, batched[qi]['candidate_cells'])
    # zero horizon: starting pose only -> nothing revealed
    z = g.predict_segments(partial_graph=ex['graph'], trajectories=[paths[0][:1]], range_observations=packet, track_history=tracks)[0]
    assert not len(z['static_updates']) and not z['states'].any()
    assert content_key(ex['graph']) == graph_before and np.array_equal(packet['ranges'], ranges_before, equal_nan=True)
print('PASS exact domain (CPU reference), zero-horizon, no input mutation')
print('batched vs single (CUDA graphs): max probability difference %.3g, differing hard states %d of %d' % (worst['single'], sdiff['single'], total))
print('batched graphs vs eager (different canvas, numerics only): max probability difference %.3g, differing hard states %d of %d' % (worst['eager'], sdiff['eager'], total))

# ---- edge cases: many queries, very long path (map rebuild + eager fallback), mixed horizons
ex, packet, tracks = work[0]
base = [np.asarray(p['trajectory'], np.float64) for p in ex['paths'][:6]]
p0 = base[np.argmax([len(p) for p in base])]
long = p0.copy()
steps = 120
ext = np.repeat(long[-1:], steps, 0)
ext[:, 0] = long[-1, 0]+np.arange(1, steps+1)*.05
ext[:, 1] = long[-1, 1]+np.arange(1, steps+1)*.12*np.cos(long[-1, 4]+np.pi/2)
ext[:, 3] = long[-1, 3]+np.arange(1, steps+1)*.12*-np.sin(long[-1, 4]+np.pi/2)
long = np.concatenate([long, ext])
many = base+[b[:max(1, len(b)//2)] for b in base]+[long, base[0][:1]]
g.clear_cache()
res_many = g.predict_segments(partial_graph=ex['graph'], trajectories=many, range_observations=packet, track_history=tracks)
assert len(res_many) == len(many)
for p, r in zip(many, res_many):
    single = g.predict_segments(partial_graph=ex['graph'], trajectories=[p], range_observations=packet, track_history=tracks)[0]
    np.testing.assert_array_equal(single['candidate_cells'], r['candidate_cells'])
    np.testing.assert_array_equal(single['states'], r['states'])
    res = float(ex['graph']['resolution']); lo, hi = crop_box(p, res)
    known = {tuple(c['cell']) for c in ex['graph']['cells']}
    grid = np.array([(x, z) for x in range(lo[0], hi[0]) for z in range(lo[1], hi[1]) if (x, z) not in known], np.int32)
    ref = grid[candidate_mask(grid, np.unique(p[:, [1, 3]].astype(np.float32), axis=0), res)]
    np.testing.assert_array_equal(ref, r['candidate_cells'])
assert not res_many[-1]['states'].any()
print('PASS edge cases: %d queries incl. a %.1f m path (frame %d cells) and a zero-horizon query' % (
    len(many), float(np.linalg.norm(long[-1, [1, 3]]-long[0, [1, 3]])), int(max(crop_box(long, float(ex['graph']['resolution']))[1]-crop_box(long, float(ex['graph']['resolution']))[0]))))

# ---- overlapped lanes (per-query graphs on several streams) must equal one lane exactly
lanes = LegalVoxelPredictor(ckpt, device=dev, lanes=3)
one = LegalVoxelPredictor(ckpt, device=dev, lanes=1)
checked = 0
for ex, packet, tracks in work[:6]:
    paths = [np.asarray(p['trajectory'], np.float64) for p in ex['paths'][:6]]
    a = lanes.predict_segments(partial_graph=ex['graph'], trajectories=paths, range_observations=packet, track_history=tracks)
    b = one.predict_segments(partial_graph=ex['graph'], trajectories=paths, range_observations=packet, track_history=tracks)
    for ra, rb in zip(a, b):
        np.testing.assert_array_equal(ra['probabilities'], rb['probabilities'])
        np.testing.assert_array_equal(ra['states'], rb['states'])
        checked += len(ra['states'])
print('PASS lanes=3 vs lanes=1: bit-identical probabilities and states on %d cells' % checked)
