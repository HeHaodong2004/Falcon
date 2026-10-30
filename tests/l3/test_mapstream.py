"""Stage 2 map-stream options of the imagination (perceive.py): probability thresholds, union / own maps, the legal
observer's certification (synthetic maps; no simulator, no GPU).

    python scripts/l3_run.py tests/l3/test_mapstream.py
"""
import math
import os

import numpy as np

from eventwm_graph.querygen.legal import GridMap, FREE, BLOCKED, UNKNOWN
from eventwm_graph.l3 import perceive
from eventwm_graph.l3.perceive import (Discovery, ObserverReveal, make_reveal, predicted_truth, predicted_truth_union,
                                       s2_states, s2_thresholds)


def _delta(keys, probs):
    """A Stage 2 increment as the release predictor builds it (class rule 0.475 / 0.575, float32 probabilities)."""
    p = np.asarray(probs, np.float32).reshape(-1, 3)
    pf, pb = p[:, 1], p[:, 2]
    tf, tb = np.float32(0.4749999999999999), np.float32(0.575)
    st = np.where((pf > tf) | (pb > tb), np.where(pb-tb > pf-tf, 2, 1), 0).astype(np.uint8)
    return dict(candidate_cells=np.asarray(keys, np.int32).reshape(-1, 2), probabilities=p, states=st,
                provenance=dict(revelation_threshold=0.4749999999999999, blocked_threshold=0.575))


def _old_truth(legal_map, delta):
    truth = legal_map.copy()
    st = np.asarray(delta['states'])
    m = st > 0
    keys = np.asarray(delta['candidate_cells'], np.int64)[m]
    truth._ensure(keys)
    i, j = keys[:, 0]-truth.ox, keys[:, 1]-truth.oz
    unknown = truth.state[i, j] == 0
    truth.state[i[unknown], j[unknown]] = st[m][unknown].astype(np.uint8)
    return truth


def _random_delta(rng, n=400):
    keys = np.stack([rng.integers(-30, 30, n), rng.integers(-30, 30, n)], 1)
    keys = np.unique(keys, axis=0)
    r = rng.random(len(keys)); b = rng.random(len(keys))
    return _delta(keys, np.stack([1-r, r*(1-b), r*b], 1))


def test_thresholds_default_and_operating_point():
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = _random_delta(rng)
        assert s2_states(d) is d['states']                                     # default: Stage 2's own states
        assert np.array_equal(s2_states(d, 0.575, 0.475), d['states'])          # the operating point reproduces them
        assert np.array_equal(s2_states(d, 0.575, None), d['states'])
        lo, hi = s2_states(d, 0.3, 1.01), s2_states(d, 0.8, 1.01)
        assert not (lo == 1).any() and not (hi == 1).any()                      # pf > 1: no predicted FREE cell
        assert ((hi == 2) <= (lo == 2)).all()                                   # a higher pb reveals a subset
        p = d['probabilities']
        assert np.array_equal(lo == 2, p[:, 2] > np.float32(0.3))


def test_predicted_maps_default_unchanged():
    rng = np.random.default_rng(1)
    g = GridMap(center=(0., 0.))
    g.state[150:170, 150:170] = FREE
    g.state[160, 150:170] = BLOCKED
    ds = [_random_delta(rng) for _ in range(4)]
    for d in ds:
        a, b = predicted_truth(g, d), _old_truth(g, d)
        assert np.array_equal(a.state, b.state) and (a.ox, a.oz) == (b.ox, b.oz)
    u = predicted_truth_union(g, ds)
    u2 = predicted_truth_union(g, ds, None, None)
    assert np.array_equal(u.state, u2.state)
    # a threshold changes the predicted cells only, never the legal map's known cells
    t = predicted_truth_union(g, ds, 0.3, 1.01)
    known = g.state != UNKNOWN
    assert np.array_equal(t.state[:g.state.shape[0], :g.state.shape[1]][known], g.state[known])


def test_env_options_default_off():
    for k in ('L3_S2_PB', 'L3_S2_PF', 'L3_S2_REVEAL', 'L3_S2_OBS'):
        assert k not in os.environ, k
    assert s2_thresholds() == (None, None)
    g = GridMap(center=(0., 0.))
    assert type(make_reveal(g.copy(), g)) is Discovery


def _wall_world():
    """Robot at the origin facing +x; a wall band (x in [4.0, 4.5] m) across z in [-3, 3] m, FREE floor before it, a FREE
    room behind it; everything UNKNOWN in the map at t except the robot's own cell."""
    truth = GridMap(center=(0., 0.))
    r = truth.res
    xs = (np.arange(truth.state.shape[0])+truth.ox+.5)*r
    zs = (np.arange(truth.state.shape[1])+truth.oz+.5)*r
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    inside = (np.abs(Z) <= 3.) & (X >= -1.) & (X <= 8.)
    truth.state[inside & (X < 4.0)] = FREE
    truth.state[inside & (X >= 4.0) & (X <= 4.5)] = BLOCKED
    truth.state[inside & (X > 4.5)] = FREE
    work = GridMap(center=(0., 0.))
    return truth, work, X, Z


def test_observer_reveal_geometry():
    truth, work, X, Z = _wall_world()
    rv = ObserverReveal(work, truth)
    pose = np.array([0., 0., 0., 0.])                   # yaw 0: forward = +x
    rv.reveal(pose, 1.0)
    cam_x = perceive.CAM_FWD
    st = work.state
    # wall band in front: the side facing the robot is certified, as are cells up to the band depth
    front = (np.abs(Z) <= 1.) & (np.abs(X-4.05) < 0.05)
    assert (st[front] == BLOCKED).all()
    # nothing behind the wall (FREE behind it: no line of sight; band beyond the hit radius)
    assert not (st[(X > 4.9)] != UNKNOWN).any()
    # FREE floor: certified between free_min and the wall, inside the field of view minus the halo, never next to
    # the camera
    d = np.hypot(X-cam_x, Z)
    near = (truth.state == FREE) & (d < rv.free_min)
    assert not (st[near] != UNKNOWN).any()
    mid = (truth.state == FREE) & (X > 2.) & (X < 3.5) & (np.abs(Z) < 0.5)
    assert (st[mid] == FREE).all()
    side = (truth.state == FREE) & (np.abs(np.arctan2(Z, X-cam_x)) > math.radians(46.)) & (d > 1.5)
    assert not (st[side] != UNKNOWN).any()
    # the map at t comes back on reset; Discovery reveals inside 5 m only, wall cells anywhere in the frustum
    n = rv.revealed
    assert n > 0
    rv.reset()
    assert not (work.state != UNKNOWN).any() and rv.revealed == 0
    work2 = GridMap(center=(0., 0.))
    ds = Discovery(work2, truth)
    ds.reveal(pose, 1.0)
    assert (work2.state[(truth.state == FREE) & (d > 5.2)] == UNKNOWN).all()


def test_observer_wall_range():
    truth, work, X, Z = _wall_world()
    rv = ObserverReveal(work, truth, wall_max=3.0)
    rv.reveal(np.array([0., 0., 0., 0.]), 1.0)
    assert not (work.state == BLOCKED).any()             # the wall is 3.5 m from the camera: beyond wall_max
    rv2 = ObserverReveal(GridMap(center=(0., 0.)), truth, wall_max=3.0)
    rv2.reveal(np.array([1.0, 0., 0., 0.]), 1.0)          # one metre closer: in range
    assert (rv2.work.state == BLOCKED).any()


class _Ro:
    def __init__(self, n, x0=0.):
        self.poses = np.stack([np.linspace(x0, x0+1., n), np.zeros(n), np.zeros(n), np.zeros(n)], 1)
        self.cont = None


class _Humans:
    def fork(self, discovery, blocked):
        h = _Humans()
        h.discovery, h.blocked = discovery, blocked
        return h


def _selector_maps(env, deltas_by_round):
    """Truth maps the selector's map loop (heuristics._discovery_rollouts) hands to every candidate's reveal, with the
    imagination and Layer 2 mocked; deltas_by_round[r][k]: candidate k's Stage 2 increment of round r."""
    from types import SimpleNamespace
    import eventwm_graph.sandbox.imagine as im
    from eventwm_graph.l3 import heuristics
    K = len(deltas_by_round[0])
    calls = []
    seen = []

    def predict(state, trajs, **kw):
        assert len(trajs) == K and all(t.shape[1] == 5 for t in trajs)
        r = len(calls)
        calls.append(r)
        return SimpleNamespace(candidates=[SimpleNamespace(map=x) for x in deltas_by_round[r]])

    def fake_imagine(legal, ex, dyn, hk, steps, **kw):
        seen.append((type(hk.discovery).__name__, hk.discovery.truth.state.copy()))
        return _Ro(steps+1)
    g = GridMap(center=(0., 0.))
    g.state[150:170, 150:170] = FREE
    legal = SimpleNamespace(map=g, time=1.0)
    sb = SimpleNamespace(imagine_all=lambda legal, cands, humans: [_Ro(17) for _ in cands],
                         _execution=lambda c: (object(), 'native'), l2=SimpleNamespace(predict=predict),
                         config=SimpleNamespace(steps_for=lambda c: 16, worlds=None, seed=None, route=False))
    d = SimpleNamespace(candidates=[SimpleNamespace(level=0, planner='orca') for _ in range(K)])
    old = {k: os.environ.get(k) for k in env}
    orig = im.imagine
    try:
        os.environ.update(env)
        im.imagine = fake_imagine
        heuristics._discovery_rollouts(sb, legal, d, _Humans(), state=object(), cont=16)
    finally:
        im.imagine = orig
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return g, seen, calls


def test_selector_map_loop():
    rng = np.random.default_rng(3)
    r1 = [_random_delta(rng) for _ in range(3)]
    r2 = [_random_delta(rng) for _ in range(3)]
    # default (own maps, one round): each candidate gets the old predicted_truth of its own increment, Discovery
    g, seen, calls = _selector_maps({'L3_MW_DISCOVER': '1'}, [r1])
    assert calls == [0] and len(seen) == 3
    for (kind, st), dlt in zip(seen, r1):
        assert kind == 'Discovery' and np.array_equal(st, _old_truth(g, dlt).state)
    # union, thresholds, observer reveal, two rounds: two Stage 2 calls, the last pass on the union of both rounds
    g, seen, calls = _selector_maps({'L3_MW_DISCOVER': 'union', 'L3_S2_ROUNDS': '2', 'L3_S2_REVEAL': 'observer',
                                     'L3_S2_PB': '0.4', 'L3_S2_PF': '1.01'}, [r1, r2])
    assert calls == [0, 1] and len(seen) == 6
    want = predicted_truth_union(g, r1+r2, 0.4, 1.01).state
    assert all(k == 'ObserverReveal' for k, _ in seen) and all(np.array_equal(st, want) for _, st in seen[3:])
    assert not (want[:g.state.shape[0], :g.state.shape[1]][g.state == UNKNOWN] == FREE).any()


if __name__ == '__main__':
    import sys
    want = sys.argv[1:]
    names = [k for k in sorted(globals()) if k.startswith('test_') and (not want or any(w in k for w in want))]
    for name in names:
        globals()[name]()
        print('ok', name)
