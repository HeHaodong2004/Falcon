"""Multi-environment throughput: one ``predict_batch`` call serves B concurrent robot streams.

Each timed call ingests one new actual frame per stream (pre-arrival states restored outside
the timer), builds features, runs one batched network call, samples 40 worlds per stream and
returns 64-time banks. Reports per-call latency and per-stream cost for B in {1,8,32,64}.
Also times the shared-bank risk reduction for K candidate trajectories per stream.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import torch

from .extract import DEV_MANIFEST, DEV_ROOTS, read, selected_branch
from .api import HumanWorldModel
from .benchmark import risk_features, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint');p.add_argument('name');p.add_argument('--device', default='cuda:0')
    p.add_argument('--batches', type=int, nargs='*', default=[1, 8, 32, 64]);p.add_argument('--repeats', type=int, default=20)
    a = p.parse_args()
    torch.set_num_threads(1)
    model = HumanWorldModel(a.checkpoint, a.device)
    rows = [r for r in json.loads(DEV_MANIFEST.read_text())['rows'] if r['split'] == 'dev'][:max(a.batches) + 40]
    cases = []
    for r in rows:
        legal = read(DEV_ROOTS / r['root_id'] / 'legal_root.json.gz');_, br = selected_branch(DEV_ROOTS / r['root_id'])
        if not br:continue
        h = list(legal['history']);g = legal['partial_graph']
        model.predict(h[:-1], h[-1], dict(resolution=g['resolution'], timestamp=h[-1]['timestamp']), stream=r['root_id'])
        ob = br[0]['observation']
        cases.append(dict(stream=r['root_id'], history=h, obs=ob, partial_graph=dict(resolution=g['resolution'], timestamp=ob['timestamp']),
                          saved=copy.deepcopy(model.streams[r['root_id']])))
        if len(cases) >= max(a.batches):break
    out = {}
    for B in a.batches:
        reqs = cases[:B];ts = [];rk = []
        for rep in range(a.repeats + 3):
            for c in reqs:model.streams[c['stream']] = copy.deepcopy(c['saved'])
            if model.device.type == 'cuda':torch.cuda.synchronize(model.device)
            tick = time.perf_counter()
            banks = model.predict_batch([{k: c[k] for k in ('stream', 'history', 'obs', 'partial_graph')} for c in reqs], worlds=40)
            if model.device.type == 'cuda':torch.cuda.synchronize(model.device)
            el = (time.perf_counter() - tick) * 1000
            t2 = time.perf_counter()
            for bk, c in zip(banks, reqs):   # K=6 candidate trajectories per stream from one shared bank
                x0 = np.asarray(c['obs']['robot_pose'])[[0, 2]];yaw = float(c['obs']['robot_pose'][3])
                ang = yaw + np.linspace(-1., 1., 6)
                robots = x0 + bk.times[None, :, None] * 1.04 * np.stack([np.sin(ang), np.cos(ang)], -1)[:, None]
                risk_features(bk.positions, bk.present, robots)
            if rep >= 3:ts.append(el);rk.append((time.perf_counter() - t2) * 1000)
        out['B%d' % B] = dict(call=stats(ts), per_stream_ms=float(np.median(ts)) / B, risk_K6_all_streams=stats(rk))
        print(json.dumps({('B%d' % B): out['B%d' % B]['call']['p50'], 'per_stream_ms': out['B%d' % B]['per_stream_ms']}), flush=True)
    summary = dict(checkpoint=a.checkpoint, device=a.device, worlds=40, times=64, load_average=os.getloadavg(), results=out,
                   note='GPU shared with other training jobs unless stated otherwise')
    dest = Path(__file__).resolve().parents[2] / 'output/stage1_hwm/benchmark' / a.name;dest.mkdir(parents=True, exist_ok=True)
    (dest / 'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
