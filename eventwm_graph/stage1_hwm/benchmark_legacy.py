"""Legacy HistoryConditionedWorldModel complete-call latency with the SAME protocol as ``benchmark``.

Run with the frozen legacy source first on PYTHONPATH (as ``bank_legacy``). Same cohorts
(first 200 development Roots; top 30 by legacy positive episodes / static updates / cells),
startup = full raw-history public call, then ``--warm`` repeats of the legacy public call
``predict(pre_arrival_context, next_native_frame, executed_action=...)`` (40 worlds, 64
times, 3 s), exactly as the legacy runtime review timed it.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path
import numpy as np
import torch


def stats(x):
    x = np.asarray(x)
    return dict(n=len(x), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)), max=float(x.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint');p.add_argument('out');p.add_argument('--warm', type=int, default=5);p.add_argument('--worlds', type=int, default=40)
    a = p.parse_args()
    torch.set_num_threads(1)
    from eventwm_graph.l2_two_stage.data import read, selected_branch
    from eventwm_graph.l2_two_stage.features import positives
    from eventwm_graph.l2_two_stage.conditioned import HistoryConditionedWorldModel
    manifest = json.loads(Path('/home/haodong/workspace/projects/eventwm-falcon-l2-partial-20261021/output/l2_graph_wm_120k_20261021/snapshot/manifest.json').read_text())
    roots = Path('/home/haodong/workspace/projects/eventwm-falcon-l2-partial-20261021/output/l2_graph_wm_120k_20261021/snapshot/roots')
    rows = [r for r in manifest['rows'] if r['split'] == 'dev'];meta = []
    for i, r in enumerate(rows):
        legal = read(roots / r['root_id'] / 'legal_root.json.gz');h = legal['history']
        meta.append(dict(root_id=r['root_id'], positive_episodes=len({str(t['track_id']) for ob in h for t in positives(ob)}), frames=len(h),
                         static_updates=sum(len(ob['static_updates']) for ob in h), cells=len(legal['partial_graph']['cells'])))
    cohorts = [('representative', meta[:200]), ('dense', sorted(meta, key=lambda r:(r['positive_episodes'], r['static_updates'], r['cells']), reverse=True)[:30]),
               ('longest', sorted(meta, key=lambda r:(r['frames'], r['static_updates']), reverse=True)[:30])]
    model = HistoryConditionedWorldModel(a.checkpoint)
    online = [];startup = []
    for cohort, roots_ in cohorts:
        for m in roots_:
            folder = roots / m['root_id'];legal = read(folder / 'legal_root.json.gz')
            tick = time.perf_counter()
            model.predict(legal['history'], legal['history'][-1], legal['partial_graph'], actions=legal['history_actions'], horizon=3., worlds=a.worlds)
            startup.append(dict(cohort=cohort, ms=(time.perf_counter() - tick) * 1000))
            context = model.last_context.with_association()
            branch = selected_branch(folder)[2]
            if not branch:continue
            ob = branch[0]['observation'];action = branch[0]['action']
            for rep in range(a.warm):
                tick = time.perf_counter()
                out = model.predict(context, ob, executed_action=action, horizon=3., worlds=a.worlds)
                online.append(dict(cohort=cohort, ms=(time.perf_counter() - tick) * 1000, people=int(out.positions.shape[2])))
            gc.collect()
    summary = dict(model='legacy HistoryConditionedWorldModel', checkpoint=a.checkpoint, worlds=a.worlds, threads=torch.get_num_threads(), load_average=os.getloadavg(),
                   online={c: stats([r['ms'] for r in online if r['cohort'] == c]) for c in ('representative', 'dense', 'longest')},
                   startup={c: stats([r['ms'] for r in startup if r['cohort'] == c]) for c in ('representative', 'dense', 'longest')})
    out = Path(a.out);out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
