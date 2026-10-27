"""The HWM API accepts a legacy OnlineContext exactly like raw frame lists (identical banks).

Run: python scripts/stage1_hwm_run.py tests/stage1_hwm/test_legacy_context_input.py CHECKPOINT
"""
import json
import sys
import numpy as np

from eventwm_graph.stage1_hwm.extract import DEV_MANIFEST, DEV_ROOTS, read, selected_branch
from eventwm_graph.stage1_hwm.api import HumanWorldModel
from eventwm_graph.l2_two_stage.context import OnlineContext


def main(checkpoint):
    rows = [r for r in json.loads(DEV_MANIFEST.read_text())['rows'] if r['split'] == 'dev'][:12]
    a = HumanWorldModel(checkpoint);b = HumanWorldModel(checkpoint);n = 0
    for r in rows:
        legal = read(DEV_ROOTS / r['root_id'] / 'legal_root.json.gz');_, br = selected_branch(DEV_ROOTS / r['root_id'])
        h = legal['history'];ctx = OnlineContext.from_raw(h, legal['partial_graph'], legal['history_actions'])
        x = a.predict(h[:-1], h[-1], legal['partial_graph'], seed=7, stream=r['root_id'])
        y = b.predict(ctx, h[-1], seed=7, stream=r['root_id'])
        assert x.track_keys == y.track_keys and np.array_equal(x.positions, y.positions) and np.array_equal(x.present, y.present)
        if br:   # next native frame through legacy ingest vs raw list
            ob = br[0]['observation'];ctx2 = ctx.ingest(ob, br[0]['action'])
            x = a.predict(list(h), ob, None, seed=9, stream=r['root_id']);y = b.predict(ctx2, ob, seed=9, stream=r['root_id'])
            assert np.array_equal(x.positions, y.positions) and np.array_equal(x.present, y.present)
        n += 1
    print('legacy OnlineContext input == raw frame input on %d Roots PASS' % n)


if __name__ == '__main__':
    main(sys.argv[1])
