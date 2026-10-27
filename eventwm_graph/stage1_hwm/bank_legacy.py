"""Dump legacy HistoryConditionedWorldModel world banks on development anchors.

Run with the FROZEN legacy inference source first on PYTHONPATH (see
docs/stage1_hwm). Calls ``predict_features`` exactly as the legacy evaluator
(same records, seeds and 40 worlds) but requests every native truth time up
to 3.028 s so downstream-risk metrics can be scored. The legacy sampled worlds
do not depend on the requested time grid (only the path basis is evaluated).
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint');p.add_argument('dest');p.add_argument('--worlds', type=int, default=40)
    p.add_argument('--shard', type=int, default=1000)
    p.add_argument('--truth-times', default='/var/tmp/falcon-stage1-claude/hwm_data/dev_truth_times.pkl')
    a = p.parse_args()
    from eventwm_graph.l2_two_stage.data import load_records
    from eventwm_graph.l2_two_stage.common import seed_for
    from eventwm_graph.l2_two_stage.conditioned import HistoryConditionedWorldModel
    from eventwm_graph.l2_two_stage.positive_history import attach_histories
    from eventwm_graph.l2_two_stage.last_event_metadata import attach_metadata
    import pickle
    truth_times = pickle.load(open(a.truth_times, 'rb'))
    records = load_records('dev_native_v2')
    attach_histories(records, 'dev');attach_metadata(records, 'dev_native_v2')
    model = HistoryConditionedWorldModel(a.checkpoint)
    dest = Path(a.dest);dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter();rows = [];shard = 0
    for idx, r in enumerate(records):
        native = truth_times[r['root_id']] - float(r['timestamp'])
        native = native[(native > 1e-9) & (native <= 3.028001)]
        requested = np.unique(np.r_[0, r['times'][r['times'] > 0], native])
        features = {k: r[k] for k in ('keys', 'x', 'graph', 'global_graph', 'origin', 'velocity', 'age', 'global_features', 'robotxy', 'timestamp')}
        features['minimum_count'] = r['minimum_count']
        features['positive_history'] = r['positive_history'];features['source_last_event_indices'] = r['source_last_event_indices']
        for stream in (41, 107):
            seed = seed_for(r['root_id'], stream, r['anchor_index'])
            t0 = time.perf_counter()
            result = model.predict_features(features, worlds=a.worlds, seed=seed, times=requested)
            ms = (time.perf_counter() - t0) * 1000
            ownership = np.asarray(result.association.event_person)[:, np.asarray(result.source_event_indices)]
            rows.append(dict(root_id=r['root_id'], anchor=int(r['anchor_index']), stream=stream, times=requested,
                             positions=np.asarray(result.positions, np.float32), present=np.asarray(result.present),
                             key_cols=ownership.astype(np.int16), ms=ms))
        if len(rows) >= a.shard or idx == len(records) - 1:
            np.save(dest / ('bank_%03d.npy' % shard), np.asarray(rows, dtype=object), allow_pickle=True)
            shard += 1;rows = []
            print(json.dumps(dict(records=idx + 1, seconds=round(time.perf_counter() - started, 1))), flush=True)


if __name__ == '__main__':
    main()
