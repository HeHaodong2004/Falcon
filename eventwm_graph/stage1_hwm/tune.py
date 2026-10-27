"""Post-hoc sampling recalibration tuned on the scene-held-out VALIDATION split only.

Scores, per setting (mode-logit temperature, within-mode spread scale; keys and residual
people separately):
  * key-path Energy Score at the 0.25 s knots (resolved keys, observed future), normalised
    by sqrt(#knots), and the self-referenced Coverage90 gap (obs - self, 39 worlds);
  * downstream risk Brier: P(any human within 1 m of the robot within H in {1,2,3} s) for
    the executed robot path and a stationary robot, using every human (keys + residual).
The development split is never touched here.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from .data import Store, collate, to_torch, FIXED_PRESENT
from .bank_hwm import load
from .sampler import sample
from .train import validation_scene
from .model import STEP

OUT = Path(__file__).resolve().parents[2] / 'output/stage1_hwm/tune'


def es_cov(x, y):
    """x [W,D] samples, y [D] -> (energy score, obs39 coverage, self39 coverage)."""
    W = len(x);d1 = np.linalg.norm(x - y, axis=1).mean()
    dd = np.linalg.norm(x[:, None] - x[None], axis=-1).sum() / (2 * W * (W - 1))
    obs = [];ref = []
    for k in range(8):
        rest = np.delete(x, k, 0);c = rest.mean(0);q = np.quantile(np.linalg.norm(rest - c, axis=-1), .9)
        obs.append(np.linalg.norm(y - c) <= q);ref.append(np.linalg.norm(x[k] - c) <= q)
    return d1 - dd, float(np.mean(obs)), float(np.mean(ref))


def evaluate(net, store, idx, calib, device, worlds=40, seed=5):
    g = torch.Generator(device=device);g.manual_seed(seed)
    es = [];obs = [];ref = [];risk = {}
    for c0 in range(0, len(idx), 128):
        part = idx[c0:c0 + 128];b = to_torch(collate(store, part), device)
        out = sample(net, b, worlds, g, calib=calib)
        knots = out['knots'].cpu().numpy();present = out['present'].cpu().numpy()
        hum = b['hum'].cpu().numpy();hmask = b['hum_mask'].cpu().numpy();tm = b['tau_mask'].cpu().numpy()
        kt = b['key_type'].cpu().numpy();ko = b['key_owner'].cpu().numpy();K = kt.shape[1]
        rot = b['rot'].cpu().numpy().astype(np.float64);origin = b['origin'].cpu().numpy().astype(np.float64)
        for j, i in enumerate(part):
            fut = np.flatnonzero(tm[j][1:]) + 1
            if len(fut):
                for k in range(K):
                    if kt[j, k] != FIXED_PRESENT:continue
                    x = knots[j, :, k][:, fut].reshape(worlds, -1);y = hum[j, ko[j, k]][fut].reshape(-1)
                    e, o, r = es_cov(x, y);es.append(e / np.sqrt(len(fut)));obs.append(o);ref.append(r)
            # downstream risk at the knots (canonical frame), all present humans vs truth
            ri = store.a['anchor_root'][i];t0 = store.a['t'][i];frames = store.frame[ri]
            for probe in ('stationary', 'executed'):
                times = t0 + np.arange(1, 13) * STEP
                if probe == 'executed':
                    if frames[-1, 0] < times[0] - 1e-6:continue
                    sel = times <= frames[-1, 0] + 1e-6
                    rob = np.stack([np.interp(times[sel], frames[:, 0], frames[:, 1]), np.interp(times[sel], frames[:, 0], frames[:, 3])], -1)
                    rob = np.einsum('ij,tj->ti', rot[j], rob - origin[j])
                else:
                    sel = np.ones(12, bool);rob = np.zeros((12, 2))
                sel = sel & tm[j][1:]
                if not sel.any():continue
                steps = np.flatnonzero(sel) + 1
                dm = np.linalg.norm(knots[j][:, :, steps] - rob[:sel.sum()][None, None], axis=-1)       # [W,N,T]
                dm = np.where(present[j][:, :, None], dm, np.inf).min(1)
                dt = np.linalg.norm(hum[j][hmask[j]][:, steps] - rob[:sel.sum()][None], axis=-1).min(0) if hmask[j].any() else np.full(len(steps), np.inf)
                for H in (1., 2., 3.):
                    w = steps * STEP <= H + 1e-9
                    if steps[-1] * STEP < H - 1e-9:continue
                    p = float((dm[:, w] < 1.).any(1).mean());yv = float((dt[w] < 1.).any())
                    risk.setdefault('%s_H%g' % (probe, H), []).append((p - yv) ** 2)
    return dict(key_es=float(np.mean(es)), cov_gap=float(np.mean(obs) - np.mean(ref)), keys=len(es),
                **{'risk_' + k: float(np.mean(v)) for k, v in sorted(risk.items())})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint');p.add_argument('name');p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit-shards', type=int, default=15);p.add_argument('--anchors', type=int, default=4000)
    a = p.parse_args()
    device = torch.device(a.device);net = load(a.checkpoint, device)
    store = Store('train', a.limit_shards)
    val = np.flatnonzero(np.array([validation_scene(s) for s in store.scene_ids])[store.a['anchor_root']])
    idx = np.sort(np.random.default_rng(0).choice(val, min(a.anchors, len(val)), replace=False))
    OUT.mkdir(parents=True, exist_ok=True);rows = []
    grid = [dict(key_temp=t, key_scale=s) for t in (1., .8, .6) for s in (1., .9, .8, .7)]
    for calib in grid:
        r = dict(calib, **evaluate(net, store, idx, calib, device));rows.append(r);print(json.dumps(r), flush=True)
    base = min(rows, key=lambda r:r['key_es'] if abs(r['cov_gap']) <= .03 else 1e9)
    for calib in [dict(base, hidden_temp=t, hidden_scale=s) for t in (1., .8) for s in (1., .85, .7)]:
        calib = {k: calib[k] for k in ('key_temp', 'key_scale', 'hidden_temp', 'hidden_scale')}
        r = dict(calib, **evaluate(net, store, idx, calib, device));rows.append(r);print(json.dumps(r), flush=True)
    (OUT / (a.name + '.json')).write_text(json.dumps(dict(checkpoint=a.checkpoint, validation_anchors=len(idx), rows=rows), indent=1))


if __name__ == '__main__':
    main()
