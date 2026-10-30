"""Stage 2 Sandbox map-stream fine-tune (l3/s2ft.py, l3/s2data.py): label windows, the class rule, the map metrics
against the probe's ``map_stats``, the default-off checkpoint option (synthetic maps; no simulator, no GPU).

    python scripts/l3_run.py tests/l3/test_s2ft.py
"""
import os

import numpy as np

from eventwm_graph.querygen.legal import GridMap, FREE, BLOCKED, UNKNOWN
from eventwm_graph.l3 import s2ft
from eventwm_graph.l3.perceive import predicted_truth_union, s2_states
from eventwm_graph.l3.s2data import new_cells
from eventwm_graph.l3.shortprobe import map_stats


def _keys_of(g, mask):
    ii, jj = np.nonzero(mask)
    return np.stack([ii+g.ox, jj+g.oz], 1).astype(np.int64)


def test_new_cells_and_windows():
    rng = np.random.default_rng(0)
    g = GridMap(center=(0., 0.))
    g.state[150:160, 150:160] = FREE
    fin = g.copy()
    fin.state[140:170, 140:170] = np.where(rng.random((30, 30)) < 0.5, FREE, BLOCKED)
    fin.state[150:160, 150:160] = FREE
    fin.first_seen[140:170, 140:170] = rng.uniform(0.05, 3.1, (30, 30))
    fin.first_seen[150:160, 150:160] = np.nan
    keys, states, first = new_cells(g, fin)
    assert len(keys) == 900-100
    assert set(np.unique(states).tolist()) <= {FREE, BLOCKED}
    c = dict(keys=keys.astype(np.int32), states=states, first=first.astype(np.float32),
             real_t=np.arange(1, 65)*0.0485)
    k32, s32 = s2ft.label_cells(c, 32)
    assert len(k32) == int((first <= c['real_t'][31]+1e-4).sum()) and len(k32) < len(keys)
    k64, _ = s2ft.label_cells(c, 64)
    assert len(k64) == int((first <= c['real_t'][63]+1e-4).sum())
    # a terminated branch: the window ends at its last step
    c2 = dict(c, real_t=c['real_t'][:10])
    assert len(s2ft.label_cells(c2, 32)[0]) == int((first <= c['real_t'][9]+1e-4).sum())


def test_class_rule_matches_perceive():
    rng = np.random.default_rng(1)
    r, b = rng.random(500), rng.random(500)
    p = np.stack([1-r, r*(1-b), r*b], 1).astype(np.float32)
    d = dict(candidate_cells=np.zeros((500, 2), np.int32), probabilities=p, states=np.zeros(500, np.uint8),
             provenance=dict(revelation_threshold=0.475, blocked_threshold=0.575))
    for tb, tf in ((0.3, 0.475), (0.575, 0.475), (0.2, 1.01)):
        assert np.array_equal(s2ft.class_states(p[:, 1], p[:, 2], np.float32(tf), np.float32(tb)),
                              s2_states(d, tb, tf))


def test_union_counts_match_map_stats():
    """The scorer's union map and counts equal ``perceive.predicted_truth_union`` + ``shortprobe.map_stats``."""
    rng = np.random.default_rng(2)
    g = GridMap(center=(0., 0.))
    g.state[155:165, 155:165] = FREE
    deltas = []
    for _ in range(4):
        keys = np.unique(np.stack([rng.integers(-25, 25, 300), rng.integers(-25, 25, 300)], 1), axis=0)
        unknown = np.asarray([g.state[k[0]-g.ox, k[1]-g.oz] == UNKNOWN for k in keys])
        keys = keys[unknown]
        r, b = rng.random(len(keys)), rng.random(len(keys))
        p = np.stack([1-r, r*(1-b), r*b], 1).astype(np.float32)
        st = s2ft.class_states(p[:, 1], p[:, 2], np.float32(0.475), np.float32(0.4))
        deltas.append(dict(candidate_cells=keys.astype(np.int32), probabilities=p, states=st, provenance={}))
    truth = predicted_truth_union(g, deltas)
    fin = g.copy()
    fin.state[135:185, 135:185] = np.where(rng.random((50, 50)) < 0.3, FREE, np.where(rng.random((50, 50)) < 0.5, BLOCKED, UNKNOWN))
    fin.state[155:165, 155:165] = FREE
    fin.first_seen[fin.state != UNKNOWN] = 1.0
    fin.first_seen[155:165, 155:165] = np.nan
    want = map_stats(truth, g, fin)
    # scorer: union of the increments' states (BLOCKED wins), counted against the real window's new cells
    k64 = np.concatenate([s2ft._k64(d['candidate_cells']) for d in deltas])
    uk, inv = np.unique(k64, return_inverse=True)
    us = s2ft.union_states(inv, len(uk), [d['states'] for d in deltas])
    ref = np.zeros(len(uk), np.uint8)
    np.maximum.at(ref, inv, np.concatenate([d['states'] for d in deltas]))
    assert np.array_equal(us, ref)
    rk, rs, _ = new_cells(g, fin)
    got = s2ft.map_counts(uk, us, s2ft._k64(rk), rs, sorted_keys=True)
    assert got == want, (got, want)
    # the scorer's shared-lookup union counts (several real windows) equal the summed per-window map_counts
    wins = [(s2ft._k64(rk), rs)]
    for frac in (0.3, 0.7):
        m = rng.random(len(rk)) < frac
        wins.append((s2ft._k64(rk[m]), rs[m]))
    wins.append((np.zeros(0, np.int64), np.zeros(0, np.uint8)))
    total = dict(s2ft.EMPTY)
    for k64r, r in wins:
        for kk, v in s2ft.map_counts(uk, us, k64r, r, sorted_keys=True).items():
            total[kk] += v
    assert s2ft.union_counts(us, s2ft.union_prep(uk, wins)) == total


def test_checkpoint_option_default_off():
    assert 'L3_S2_CKPT' not in os.environ
    import inspect
    from eventwm_graph.layer2 import model
    src = inspect.getsource(model.Layer2WorldModel.from_release)
    assert "os.environ.get('L3_S2_CKPT')" in src


if __name__ == '__main__':
    import sys
    want = sys.argv[1:]
    names = [k for k in sorted(globals()) if k.startswith('test_') and (not want or any(w in k for w in want))]
    for name in names:
        globals()[name]()
        print('ok', name)
