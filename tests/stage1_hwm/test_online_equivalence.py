"""Online API features equal offline extraction features on real development Roots.

Run: python scripts/stage1_hwm_run.py tests/stage1_hwm/test_online_equivalence.py [checkpoint]
"""
import sys
from pathlib import Path
import numpy as np
import torch

from eventwm_graph.stage1_hwm.data import Store, collate
from eventwm_graph.stage1_hwm.extract import DEV_ROOTS, read, selected_branch
from eventwm_graph.stage1_hwm.api import HumanWorldModel, _LiveStore

FIELDS = ('map', 'vis', 'corner', 'global', 'key_feat', 'key_mask', 'key_hist', 'key_last', 'key_vel', 'key_age', 'origin', 'theta')


def replay(model, root_id, wanted):
    folder = DEV_ROOTS / root_id
    legal = read(folder / 'legal_root.json.gz');_, branch = selected_branch(folder)
    history = list(legal['history']);frames = history + [n['observation'] for n in branch]
    R = len(history) - 1;out = {}
    for j in range(len(frames) - R):
        if j not in wanted:continue
        prefix = frames[:R + j + 1]
        model.predict(prefix[:-1], prefix[-1], partial_graph=dict(resolution=legal['partial_graph']['resolution'], timestamp=prefix[-1]['timestamp']), worlds=4)
        st = model.streams['default'];live = _LiveStore(st.state, st.episode_start)
        out[j] = collate(live, [0])
    return out


def main(checkpoint):
    store = Store('dev', limit_shards=1)
    model = HumanWorldModel(checkpoint)
    roots = sorted(set(store.a['anchor_root'][:200].tolist()))[:25]
    checked = 0
    for r in roots:
        idx = np.flatnonzero(store.a['anchor_root'] == r)
        wanted = {int(store.a['j'][i]): i for i in idx}
        online = replay(model, store.root_ids[r], set(wanted))
        for j, i in wanted.items():
            offline = collate(store, [i])
            for f in FIELDS:
                a, b = np.asarray(offline[f], np.float64), np.asarray(online[j][f], np.float64)
                assert a.shape == b.shape and np.allclose(a, b, atol=1e-5, rtol=0), (store.root_ids[r], j, f, np.abs(a - b).max())
            checked += 1
    print('online/offline feature equivalence: %d anchors over %d Roots PASS' % (checked, len(roots)))


if __name__ == '__main__':
    main(sys.argv[1])
