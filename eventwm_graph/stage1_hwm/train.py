"""Train the human world model by exact joint negative log-likelihood.

Model selection uses a scene-disjoint validation subset of the TRAINING split
(scene hash rule below); the development split is reserved for reporting.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch

from .data import Store, collate, to_torch, rasters
from .model import HWM

OUT = Path(__file__).resolve().parents[2] / 'output/stage1_hwm'


def validation_scene(scene):
    """Deterministic ~6% scene hold-out of the training split."""
    return int(hashlib.sha256(('hwm-val:' + scene).encode()).hexdigest()[:8], 16) % 16 == 0


class Batches(torch.utils.data.Dataset):
    def __init__(self, store, indices, batch, seed, train=True, hard_labels=False):
        self.store = store;self.indices = np.asarray(indices);self.batch = batch;self.train = train;self.hard_labels = hard_labels
        self.rng = np.random.default_rng(seed);self.epoch_order()

    def epoch_order(self):
        order = self.rng.permutation(self.indices) if self.train else self.indices
        self.chunks = [order[i:i + self.batch] for i in range(0, len(order), self.batch)]
        if self.train and len(self.chunks[-1]) < self.batch:self.chunks = self.chunks[:-1]

    def __len__(self):return len(self.chunks)

    def __getitem__(self, i):
        idx = self.chunks[i]
        flip = np.random.default_rng(hash((i, int(idx[0]))) % (2 ** 32)).random(len(idx)) < .5 if self.train else None
        return collate(self.store, idx, flip, self.hard_labels)


def loader(ds, workers):
    return torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers,
                                       persistent_workers=False, pin_memory=True, prefetch_factor=4 if workers else 2)


def evaluate(net, ds, device, workers, max_batches=0):
    net.eval();tot = {};n = 0
    with torch.no_grad():
        for i, b in enumerate(loader(ds, workers)):
            if max_batches and i >= max_batches:break
            b = to_torch(b, device);m, v = rasters(b, *net.raster_spec)
            _, st = net.loss(b, m, v)
            for k, val in st.items():tot[k] = tot.get(k, 0.) + float(val)
            n += 1
    net.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name');p.add_argument('--epochs', type=int, default=6);p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3);p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--device', default='cuda:0');p.add_argument('--workers', type=int, default=6)
    p.add_argument('--limit-shards', type=int, default=0);p.add_argument('--seed', type=int, default=17)
    p.add_argument('--val-every', type=int, default=2000);p.add_argument('--threads', type=int, default=4)
    p.add_argument('--width', type=int, default=128);p.add_argument('--modes', type=int, default=6);p.add_argument('--arch', default='full');p.add_argument('--truncate', action='store_true');p.add_argument('--ablate', nargs='*', default=[]);p.add_argument('--hard-labels', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    dest = OUT / 'runs' / a.name;dest.mkdir(parents=True, exist_ok=False)
    store = Store('train', a.limit_shards)
    scenes = np.array([validation_scene(s) for s in store.scene_ids])
    is_val = scenes[store.a['anchor_root']]
    train_idx = np.flatnonzero(~is_val);val_idx = np.flatnonzero(is_val)
    rng = np.random.default_rng(0);val_sub = np.sort(rng.choice(val_idx, min(len(val_idx), 12000), replace=False))
    print(json.dumps(dict(train_anchors=len(train_idx), val_anchors=len(val_idx), val_used=len(val_sub),
                          train_scenes=len({sc for sc, v in zip(store.scene_ids, scenes) if not v}), val_scenes=len({sc for sc, v in zip(store.scene_ids, scenes) if v}), roots=len(store.root_ids))), flush=True)
    device = torch.device(a.device)
    net = HWM(d=a.width, modes=a.modes, arch=a.arch, truncate=a.truncate, ablate=a.ablate).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    ds = Batches(store, train_idx, a.batch, a.seed, hard_labels=a.hard_labels);vds = Batches(store, val_sub, 256, 0, train=False)
    total = a.epochs * len(ds);step = 0;started = time.time();best = math.inf
    config = dict(vars(a), train_anchors=len(train_idx), val_anchors=len(val_idx), model=dict(d=a.width, modes=a.modes, arch=a.arch, truncate=a.truncate, ablate=a.ablate))
    (dest / 'config.json').write_text(json.dumps(config, indent=1))
    log = open(dest / 'log.jsonl', 'a')
    for epoch in range(a.epochs):
        ds.epoch_order();run = {};nrun = 0
        for b in loader(ds, a.workers):
            lr = a.lr * min(1., (step + 1) / 500) * .5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:g['lr'] = lr
            b = to_torch(b, device);m, v = rasters(b, *net.raster_spec)
            loss, st = net.loss(b, m, v)
            opt.zero_grad(set_to_none=True);loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            if torch.isfinite(gn):opt.step()
            for k, val in st.items():run[k] = run.get(k, 0.) + float(val)
            nrun += 1;step += 1
            if step % 200 == 0:
                row = dict(step=step, epoch=epoch, lr=lr, seconds=round(time.time() - started, 1), grad=float(gn),
                           **{k: v / nrun for k, v in run.items()})
                print(json.dumps(row), flush=True);log.write(json.dumps(row) + '\n');log.flush();run = {};nrun = 0
            if step % a.val_every == 0 or step == total:
                val = evaluate(net, vds, device, 2)
                row = dict(step=step, epoch=epoch, val=val);print(json.dumps(row), flush=True);log.write(json.dumps(row) + '\n');log.flush()
                state = dict(model=net.state_dict(), config=config, step=step, val=val)
                torch.save(state, dest / 'last.pt')
                if val['nll'] < best:
                    best = val['nll'];torch.save(state, dest / 'best.pt')
    print(json.dumps(dict(done=True, best=best, seconds=time.time() - started)), flush=True)


if __name__ == '__main__':
    main()
