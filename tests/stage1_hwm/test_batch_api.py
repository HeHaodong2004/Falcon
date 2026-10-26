"""Batched multi-stream API: per-stream features equal single-stream features; outputs are valid banks.

Run: python scripts/stage1_hwm_run.py tests/stage1_hwm/test_batch_api.py CHECKPOINT
"""
import sys
import numpy as np

from eventwm_graph.stage1_hwm.data import collate
from eventwm_graph.stage1_hwm.extract import DEV_MANIFEST, DEV_ROOTS, read, selected_branch
from eventwm_graph.stage1_hwm.api import HumanWorldModel, _LiveStore, _MultiStore
import json

FIELDS = ('map', 'vis', 'corner', 'global', 'key_feat', 'key_mask', 'key_hist', 'key_last', 'key_vel', 'key_age', 'origin', 'theta', 'min_count')


def main(checkpoint):
    rows = [r for r in json.loads(DEV_MANIFEST.read_text())['rows'] if r['split'] == 'dev'][:40]
    single = HumanWorldModel(checkpoint);batched = HumanWorldModel(checkpoint)
    requests = []
    for r in rows:
        legal = read(DEV_ROOTS / r['root_id'] / 'legal_root.json.gz');_, branch = selected_branch(DEV_ROOTS / r['root_id'])
        frames = list(legal['history']) + [n['observation'] for n in branch[:3]]
        requests.append(dict(history=frames[:-1], obs=frames[-1], partial_graph=dict(resolution=legal['partial_graph']['resolution'], timestamp=frames[-1]['timestamp']), stream=r['root_id']))
    outs = batched.predict_batch(requests, worlds=40)
    lives = [_LiveStore(batched.streams[q['stream']].state, batched.streams[q['stream']].episode_start) for q in requests]
    multi = collate(_MultiStore(lives), np.arange(len(lives)))
    for j, q in enumerate(requests):
        one = single.predict(q['history'], q['obs'], q['partial_graph'], worlds=40, stream=q['stream'])
        st = single.streams[q['stream']];ref = collate(_LiveStore(st.state, st.episode_start), [0])
        for f in FIELDS:
            assert np.allclose(np.asarray(ref[f], np.float64)[0], np.asarray(multi[f], np.float64)[j], atol=1e-6), (j, f)
        for o in (outs[j], one):   # batched torch sampler and single-call CPU fast path
            assert o.positions.shape[:2] == (40, 64) and (o.present.sum(1) >= o.minimum_count).all()
            assert abs(np.logaddexp.reduce(o.count_log_probability)) < 1e-9
            step = np.linalg.norm(np.diff(o.knots, axis=2), axis=-1) / .25
            assert (step[o.present] <= 3. * (1 + 1e-5)).all(), step[o.present].max()
        o = outs[j]
        assert o.track_keys[:len([k for k in one.track_keys if k.startswith('track:')])] == tuple(k for k in one.track_keys if k.startswith('track:'))
        assert o.positions.shape[:2] == (40, 64) and (o.present.sum(1) >= o.minimum_count).all()
        assert abs(np.logaddexp.reduce(o.count_log_probability) - 0) < 1e-9
        step = np.linalg.norm(np.diff(o.knots, axis=2), axis=-1) / .25
        assert (step[o.present] <= 3. * (1 + 1e-5)).all(), step[o.present].max()
    print('batched API: %d streams, features equal single-stream, banks valid PASS' % len(requests))


if __name__ == '__main__':
    main(sys.argv[1])
