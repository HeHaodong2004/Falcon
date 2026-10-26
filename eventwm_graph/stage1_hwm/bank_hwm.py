"""Dump human-world-model banks on development anchors in the legacy bank format.

Same anchors, streams (41,107), 40 worlds and requested native times as
``bank_legacy``. Adds the exact count law ``count_logp``.
"""
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch

from .data import Store, collate, to_torch
from .model import HWM
from .sampler import sample, interpolate, to_world


def load(checkpoint, device):
    state = torch.load(checkpoint, map_location='cpu')
    cfg = state['config'].get('model', {})
    net = HWM(**cfg);net.load_state_dict(state['model']);return net.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint');p.add_argument('dest');p.add_argument('--worlds', type=int, default=40)
    p.add_argument('--device', default='cuda:0');p.add_argument('--chunk', type=int, default=64)
    p.add_argument('--truth-times', default='/var/tmp/falcon-stage1-claude/hwm_data/dev_truth_times.pkl')
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    device = torch.device(a.device)
    net = load(a.checkpoint, device)
    store = Store('dev');truth_times = pickle.load(open(a.truth_times, 'rb'))
    from .metrics import legacy_records
    recs = legacy_records()
    anchor_of = {(store.root_ids[store.a['anchor_root'][i]], int(store.a['j'][i])): i for i in range(store.n)}
    order = [(k, anchor_of[k]) for k in recs]
    if a.limit:order = order[:a.limit]
    dest = Path(a.dest);dest.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter();rows = [];shard = 0;timing = []
    for stream in (41, 107):
        g = torch.Generator(device=device);g.manual_seed(1000 + stream)
        for c0 in range(0, len(order), a.chunk):
            part = order[c0:c0 + a.chunk];idx = np.array([i for _, i in part])
            b = to_torch(collate(store, idx), device)
            t0 = time.perf_counter()
            out = sample(net, b, a.worlds, g)
            if device.type == 'cuda':torch.cuda.synchronize(device)
            timing.append((time.perf_counter() - t0) / len(idx))
            knots = out['knots'].cpu().numpy();present = out['present'].cpu().numpy();clp = out['count_logp'].cpu().numpy()
            for j, ((root, anchor), i) in enumerate(part):
                r = recs[(root, anchor)]
                native = truth_times[root] - float(r['timestamp'])
                native = native[(native > 1e-9) & (native <= 3.028001)]
                requested = np.unique(np.r_[0, r['times'][r['times'] > 0], native])
                n = len(r['keys']);col = b['key_column'][j].cpu().numpy()
                key_cols = np.zeros(n, np.int16)
                for slot, c in enumerate(col):
                    if c >= 0:key_cols[c] = slot
                kn = knots[j];pr = present[j]
                pos = interpolate(kn, requested)                                   # [W,N,T,2]
                world = to_world(pos, b['origin'][j].cpu().numpy().astype(np.float64), b['rot'][j].cpu().numpy().astype(np.float64))
                keep = np.flatnonzero(pr.any(0) | (np.arange(pr.shape[1]) < out['n_keys']))
                rows.append(dict(root_id=root, anchor=anchor, stream=stream, times=requested,
                                 positions=world.transpose(0, 2, 1, 3)[:, :, keep].astype(np.float32), present=pr[:, keep],
                                 key_cols=np.broadcast_to(key_cols, (a.worlds, n)).copy(), count_logp=clp[j]))
            if len(rows) >= 1000:
                np.save(dest / ('bank_%03d.npy' % shard), np.asarray(rows, dtype=object), allow_pickle=True);shard += 1;rows = []
                print(json.dumps(dict(stream=stream, done=c0 + len(part), seconds=round(time.perf_counter() - started, 1))), flush=True)
    if rows:np.save(dest / ('bank_%03d.npy' % shard), np.asarray(rows, dtype=object), allow_pickle=True)
    (dest / 'timing.json').write_text(json.dumps(dict(per_anchor_ms=1000 * float(np.mean(timing)), device=str(device), chunk=a.chunk)))
    print(json.dumps(dict(done=True, seconds=time.perf_counter() - started, per_anchor_ms=1000 * float(np.mean(timing)))), flush=True)


if __name__ == '__main__':
    main()
