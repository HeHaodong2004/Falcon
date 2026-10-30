"""Layer 2 v2 people module: short-horizon multimodal forecaster for TRACKED people (legal inputs, privileged labels).

Stage 1 returns 40 equally weighted worlds; for tracked people the right future is usually among them (best world
0.07 m at 0.77 s) but nothing ranks it (average world 0.61 m, mean 0.37 m).  This model predicts, from the legal track
history alone, K weighted modes of a tracked person's displacement at 0.77 / 1.5 / 3 s with per-mode spreads:

* inputs: the last 2 s of the person's detections (``peopledata.track_histories``: 10 points every 0.2 s relative to
  the last detection, validity flags), the tracker's velocity estimate, the age of the last detection -- rotated into
  the frame of the tracker velocity (heading invariance);
* output: a K-component isotropic Gaussian mixture per horizon with shared component weights (a mode is one coherent
  future);
* training: negative log-likelihood of the recorded future (privileged supervision).

At inference the K modes (with their weights) replace Stage 1's worlds for tracked people in the closed loop.

    python scripts/l3_run.py eventwm_graph.l3.peoplefc fit --data output/peopledata/R10 --eval-bank output/statebank/orca_dev \
        --save output/peoplefc/v1
"""
import argparse
import glob
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np

K = 6
NH = 3            # horizons 0.77 / 1.5 / 3 s


def canonical(hist, vel):
    """Rotation into the velocity frame: returns (features [F], rotation matrix R (world->canonical))."""
    sp = float(np.hypot(*vel))
    if sp > 0.1:
        c, s = vel[0]/sp, vel[1]/sp
    else:
        valid = hist[:, 2] > 0
        if valid.sum() >= 2:
            d = hist[valid][-1, :2]-hist[valid][0, :2]
            n = float(np.hypot(*d))
            c, s = (d[0]/n, d[1]/n) if n > 0.05 else (1., 0.)
        else:
            c, s = 1., 0.
    R = np.array([[c, s], [-s, c]])
    h = hist.copy()
    h[:, :2] = h[:, :2] @ R.T
    v = R @ vel
    return np.concatenate([h.reshape(-1), v, [sp]]).astype(np.float32), R


def load_data(dirs):
    X, Y, A = [], [], []
    for d in str(dirs).split(','):
        for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
            z = np.load(f)
            if 't_hist' not in z.files:
                continue
            for h, v, a, fut in zip(z['t_hist'], z['t_vel'], z['t_age'], z['t_fut']):
                if not np.isfinite(fut).all():
                    continue
                x, R = canonical(h, v)
                X.append(np.concatenate([x, [a]]).astype(np.float32))
                Y.append((fut @ R.T).astype(np.float32))
    return np.asarray(X), np.asarray(Y)


class MDN:
    def __init__(self, n_in, hidden=128, seed=0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, K*(1+NH*2+NH)))
        self.mu = None
        self.sd = None

    def heads(self, X):
        import torch
        out = self.net(X)
        logit = out[:, :K]
        mean = out[:, K:K+K*NH*2].reshape(-1, K, NH, 2)
        logs = out[:, K+K*NH*2:].reshape(-1, K, NH).clamp(-4., 2.)
        return logit, mean, logs

    def nll(self, X, Y):
        import torch
        logit, mean, logs = self.heads(X)
        logw = torch.log_softmax(logit, -1)                                      # [B, K]
        d2 = ((Y[:, None]-mean)**2).sum(-1)                                      # [B, K, NH]
        ll = -d2/(2*torch.exp(2*logs))-2*logs-math.log(2*math.pi)               # 2D isotropic per horizon
        return -(torch.logsumexp(logw+ll.sum(-1), -1)).mean()

    def predict(self, X):
        """-> weights [B, K], means [B, K, NH, 2] (canonical frame), sigmas [B, K, NH]."""
        import torch
        with torch.no_grad():
            Xt = torch.as_tensor((X-self.mu)/self.sd, dtype=torch.float32)
            logit, mean, logs = self.heads(Xt)
            return torch.softmax(logit, -1).numpy(), mean.numpy(), torch.exp(logs).numpy()


def fit(X, Y, epochs=60, lr=2e-3, seed=0, batch=512):
    import torch
    m = MDN(X.shape[1], seed=seed)
    m.mu, m.sd = X.mean(0), X.std(0)+1e-6
    Xt = torch.as_tensor((X-m.mu)/m.sd, dtype=torch.float32)
    Yt = torch.as_tensor(Y, dtype=torch.float32)
    opt = torch.optim.Adam(m.net.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        idx = rng.permutation(len(Xt))
        for b in range(0, len(idx), batch):
            j = idx[b:b+batch]
            loss = m.nll(Xt[j], Yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def eval_bank(m, bank_dir):
    """Dev bank tracked people: MDN weighted / top-mode / best-mode errors vs Stage 1 (uniform weighted, mean, best)."""
    from .peopledata import track_histories
    res = {h: dict(mdn_w=[], mdn_top=[], mdn_best=[], s1_w=[], s1_mean=[], s1_best=[], cv=[]) for h in range(NH)}
    hz = (16, 32, 62)
    for p in sorted(glob.glob(os.path.join(bank_dir, '*.pkl'))):
        item = pickle.load(open(p, 'rb'))
        T = np.asarray(item['truth_people'], np.float64)
        b = item['bank']; P = np.asarray(b['positions'], np.float64); keys = list(b['track_keys'])
        W = np.asarray(b['weights'], np.float64); W = W/W.sum()
        for h in track_histories(item):
            x, R = canonical(h['hist'], h['vel'])
            w, mean, _ = m.predict(np.concatenate([x, [h['age']]])[None].astype(np.float32))
            w, mean = w[0], mean[0] @ R                                         # canonical -> world (R orthonormal)
            ref, j, e = h['ref'], h['j'], h['est']
            for i, hh in enumerate(hz):
                tp = T[min(hh, len(T)-1), j]
                if not np.isfinite(tp).all():
                    continue
                err = np.hypot(ref[0]+mean[:, i, 0]-tp[0], ref[1]+mean[:, i, 1]-tp[1])
                res[i]['mdn_w'].append(float((w*err).sum())); res[i]['mdn_top'].append(float(err[int(np.argmax(w))]))
                res[i]['mdn_best'].append(float(err.min()))
                k = 'track:%s' % h['track_id']
                if k in keys:
                    pr = P[:, min(hh, P.shape[1])-1, keys.index(k)]
                    ok = np.isfinite(pr).all(1)
                    if ok.any():
                        e1 = np.hypot(pr[ok, 0]-tp[0], pr[ok, 1]-tp[1])
                        res[i]['s1_w'].append(float((W[ok]*e1).sum()/W[ok].sum())); res[i]['s1_best'].append(float(e1.min()))
                        mm = (W[ok, None]*pr[ok]).sum(0)/W[ok].sum(); res[i]['s1_mean'].append(float(np.hypot(*(mm-tp))))
                x1, y1 = e.at(hh*0.048); res[i]['cv'].append(float(math.hypot(x1-tp[0], -y1-tp[1])))
    for i, hh in enumerate(hz):
        r = res[i]
        q = lambda v: '%.2f/%.2f' % (np.median(v), np.percentile(v, 90)) if v else 'n/a'
        print('%.2f s (n=%d): MDN weighted %s top-mode %s best-of-%d %s | Stage 1 weighted %s mean %s best-of-40 %s | CV %s' % (
            hh*0.048, len(r['mdn_w']), q(r['mdn_w']), q(r['mdn_top']), K, q(r['mdn_best']), q(r['s1_w']), q(r['s1_mean']),
            q(r['s1_best']), q(r['cv'])), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['fit'])
    ap.add_argument('--data', default='output/peopledata/R10')
    ap.add_argument('--eval-bank', default='output/statebank/orca_dev')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--save', default=None)
    a = ap.parse_args()
    X, Y = load_data(a.data)
    print('tracked-person instances %d' % len(X), flush=True)
    n = len(X)
    perm = np.random.default_rng(0).permutation(n)
    va = perm[:max(1, n//10)]; tr = perm[max(1, n//10):]
    m = fit(X[tr], Y[tr], a.epochs)
    import torch
    with torch.no_grad():
        Xv = torch.as_tensor((X[va]-m.mu)/m.sd, dtype=torch.float32)
        print('held-out NLL %.3f' % float(m.nll(Xv, torch.as_tensor(Y[va]))), flush=True)
    if a.eval_bank:
        eval_bank(m, a.eval_bank)
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=m.net.state_dict(), mu=m.mu, sd=m.sd, n_in=X.shape[1]), Path(a.save, 'mdn.pt'))
        Path(a.save, 'meta.json').write_text(json.dumps(dict(K=K, horizons=[16, 32, 62], data=a.data)))


if __name__ == '__main__':
    main()


# ------------------------------------------------------------------------------------------------ closed-loop bank
class Forecaster:
    """A saved MDN (``fit --save``) with the deployment interface."""

    def __init__(self, path):
        import torch
        d = torch.load(Path(path, 'mdn.pt'), map_location='cpu')
        self.m = MDN(int(d['n_in']))
        self.m.net.load_state_dict(d['state'])
        self.m.mu, self.m.sd = d['mu'], d['sd']

    def modes(self, legal, track):
        """(weights [K], world displacements [K, NH, 2] relative to the last detection, last detection (x, z)) or None."""
        from .peopledata import HIST_S, HIST_N
        now = float(legal.time)
        hs = []
        for tm, trs in (legal.track_log or []):
            if now-float(tm) > HIST_S:
                continue
            for t in trs:
                if t.get('visible_now') and str(t['track_id']) == str(track['track_id']):
                    hs.append((float(tm), float(t['position'][0]), float(t['position'][2])))
        if not hs:
            return None
        hs.sort()
        t_last, xl, zl = hs[-1]
        H = np.zeros((HIST_N, 3), np.float32)
        for i, tq in enumerate(np.linspace(now-HIST_S, now, HIST_N)):
            cand = [h for h in hs if abs(h[0]-tq) <= 0.15]
            if cand:
                h = min(cand, key=lambda h: abs(h[0]-tq))
                H[i] = (h[1]-xl, h[2]-zl, 1.)
        v3 = track.get('estimated_velocity') if track.get('velocity_valid') else None
        vel = np.array([v3[0], v3[2]], np.float32) if v3 is not None else np.zeros(2, np.float32)
        x, R = canonical(H, vel)
        w, mean, _ = self.m.predict(np.concatenate([x, [now-t_last]])[None].astype(np.float32))
        return w[0], mean[0] @ R, (xl, zl), vel*(now-t_last)


class PeopleBankV2:
    """Bank interface for the closed loop: tracked people from the forecaster's weighted modes (stratified samples per
    world), everyone else from Stage 1's hidden slots of the corresponding world; equal world weights (the mixture is
    in the sampling)."""

    def __init__(self, legal, stage1, forecaster, worlds=12, seed=0):
        from eventwm_graph.querygen.control import human_estimates
        self.s1 = stage1
        keys = [str(k) for k in stage1.track_keys]
        self.track_keys = tuple(keys)
        self.unseen = np.asarray(getattr(stage1, 'unseen', np.zeros(len(keys))), bool)
        self.times = np.asarray(stage1.times, np.float64)
        self.long_steps = 0
        self.n = int(worlds)
        self.weights = np.full(self.n, 1./self.n)
        self.present = np.ones((self.n, len(keys)), bool)
        self.s1_world = np.arange(self.n) % len(np.asarray(stage1.weights))
        rng = np.random.default_rng(seed)
        self.override = {}                         # slot -> (sample mode per world [W], displacements [K, NH, 2], ref)
        tracks = {str(t['track_id']): t for t in legal.tracks}
        for e in human_estimates(legal.tracks, legal.time):
            k = 'track:%s' % e.track_id
            if k not in keys or str(e.track_id) not in tracks:
                continue
            r = forecaster.modes(legal, tracks[str(e.track_id)])
            if r is None:
                continue
            w, disp, ref, d_now = r
            # stratified sampling of modes by weight (deterministic per seed)
            u = (np.arange(self.n)+rng.random())/self.n
            picks = np.searchsorted(np.cumsum(w), u).clip(0, len(w)-1)
            self.override[keys.index(k)] = (picks, disp, ref, d_now)

    def sandbox_positions(self, rel):
        rel = np.asarray(rel, np.float64)
        base = np.asarray(self.s1.sandbox_positions(rel), np.float64)[self.s1_world]       # [W, T, N, 2]
        out = base.copy()
        knots = np.array([0., 16*0.048, 32*0.048, 62*0.048])
        for slot, (picks, disp, ref, d_now) in self.override.items():
            for wi in range(self.n):
                d = disp[picks[wi]]                                                    # [NH, 2]
                kx = np.concatenate([[d_now[0]], d[:, 0]]); kz = np.concatenate([[d_now[1]], d[:, 1]])
                tt = np.clip(rel, 0., knots[-1])
                x = ref[0]+np.interp(tt, knots, kx)+np.where(rel > knots[-1], (rel-knots[-1])*(kx[-1]-kx[-2])/(knots[-1]-knots[-2]), 0.)
                z = ref[1]+np.interp(tt, knots, kz)+np.where(rel > knots[-1], (rel-knots[-1])*(kz[-1]-kz[-2])/(knots[-1]-knots[-2]), 0.)
                out[wi, :, slot, 0] = x; out[wi, :, slot, 1] = z
        return out
