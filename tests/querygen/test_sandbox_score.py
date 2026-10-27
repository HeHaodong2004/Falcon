"""score_vec (vectorised) == score (reference loop) on random trajectories/futures; timing."""
import time
import numpy as np
from eventwm_graph.querygen.sandbox import score, score_vec

rng = np.random.default_rng(0)
worst = 0.
t_ref = t_vec = 0.
for trial in range(50):
    T, M, N = rng.integers(5, 60), rng.integers(1, 32), rng.integers(1, 6)
    traj = np.zeros((T, 4)); traj[:, 0] = np.cumsum(rng.normal(0, .1, T)); traj[:, 2] = np.cumsum(rng.normal(0, .1, T))
    traj[:, 3] = rng.uniform(-np.pi, np.pi, T)
    F = rng.normal(0, 1.5, (M, T+int(rng.integers(0, 5)), N, 3))
    a = time.perf_counter(); A = score(traj, F); t_ref += time.perf_counter()-a
    a = time.perf_counter(); B = score_vec(traj, F); t_vec += time.perf_counter()-a
    assert np.allclose(A, B, atol=1e-9), (A, B)
    worst = max(worst, float(np.abs(A-B).max()))
print(f'score_vec == score on 50 random cases (max abs diff {worst:.2e}); time ref {t_ref*1e3:.1f} ms vs vec {t_vec*1e3:.1f} ms')
