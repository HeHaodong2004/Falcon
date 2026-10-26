"""Shared-bank candidate evaluation through the L2.5 joint reduction on real sibling Queries.

Run: python scripts/stage1_hwm_run.py tests/stage1_hwm/test_sandbox_candidates.py CHECKPOINT
"""
import pickle
import sys
import time
import numpy as np

from eventwm_graph.stage1_hwm.extract import DEV_ROOTS, read
from eventwm_graph.stage1_hwm.api import HumanWorldModel


def main(checkpoint):
    queries = pickle.load(open('/var/tmp/falcon-stage1-claude/hwm_data/dev_queries.pkl', 'rb'))
    model = HumanWorldModel(checkpoint);checked = 0;timing = []
    for root, qs in list(queries.items())[:40]:
        legal = read(DEV_ROOTS / root / 'legal_root.json.gz');h = legal['history']
        bank = model.predict(h[:-1], h[-1], legal['partial_graph'], worlds=40, horizon=3.)
        T = min(62, min(len(q['poses']) - 1 for q in qs))
        if T < 1:continue
        robots = np.stack([q['poses'][1:T + 1, 1:5] for q in qs])     # x,y,z,yaw at native steps
        tick = time.perf_counter();out = bank.evaluate_candidates(robots);timing.append((time.perf_counter() - tick) * 1000)
        assert out['p_collision'].shape == (len(qs),) and np.all((0 <= out['p_social']) & (out['p_social'] <= 1 + 1e-12))
        # cross-check P(social) against a direct centre-distance computation on the same bank
        rel = np.arange(1, T + 1) * .048;pos = bank.sandbox_positions(rel)
        d = np.linalg.norm(pos[None] - robots[:, None, :, None, :][..., [0, 2]], axis=-1).min(-1)
        assert np.allclose(out['p_social'], (d < 1.).any(-1).mean(-1))
        checked += 1
    print('shared-bank candidate evaluation: %d Roots, median %.3f ms per K-candidate reduction PASS' % (checked, np.median(timing)))


if __name__ == '__main__':
    main(sys.argv[1])
