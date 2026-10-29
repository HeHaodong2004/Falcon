"""Replay ring: n-step cumulant sums, bootstrap pointers, Monte Carlo returns, eviction of whole episodes.

    python scripts/l3_run.py tests/l3/test_ring.py
"""
import numpy as np

from eventwm_graph.l3.batch import Ring, FIELDS, KMAX


def fake_packed(T):
    out = []
    for t in range(T):
        p = {k: np.zeros(shape, dt) for k, (shape, dt) in FIELDS.items()}
        p['cand_mask'][:3] = True
        p['ego'][0] = t
        out.append(p)
    return out


def test_nstep_and_mc():
    r = Ring(100, nstep=3)
    T = 5
    cum = np.zeros((T, 5), np.float32)
    cum[:, 2] = 0.16                      # 16 steps per decision (/100)
    cum[-1, 0] = 1.                       # success at the last transition
    idx = r.add_episode(fake_packed(T), [0]*T, [1.]*T, cum)
    assert list(r.boot[idx]) == [idx[3], idx[4], -1, -1, -1]
    assert np.allclose(r.ret[idx[0]], [0, 0, 0.48, 0, 0])
    assert np.allclose(r.ret[idx[2]], [1, 0, 0.48, 0, 0])          # terminal inside the window
    assert np.allclose(r.mc[idx[0]], [1, 0, 0.8, 0, 0])
    assert np.allclose(r.mc[idx[4]], [1, 0, 0.16, 0, 0])


def test_eviction():
    r = Ring(10, nstep=2)
    a = r.add_episode(fake_packed(6), [0]*6, [1.]*6, np.zeros((6, 5)))
    b = r.add_episode(fake_packed(6), [0]*6, [1.]*6, np.zeros((6, 5)))   # wraps over the first episode
    assert r.valid[b].all()
    first = np.isin(np.arange(10), a) & ~np.isin(np.arange(10), b)
    assert not r.valid[first].any(), 'a partially overwritten episode must be evicted whole'
    s = r.sample(50, np.random.default_rng(0))
    assert np.isin(s, b).all()


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name, flush=True)
