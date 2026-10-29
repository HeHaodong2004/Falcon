"""Packing compiled graphs into fixed-size arrays, the replay ring buffer and batch transfer.

Every decision is packed once (float16, K <= KMAX candidates, N <= NMAX tracked humans) into preallocated ring
arrays, so sampling a training batch is one fancy-index gather.  Episodes are inserted whole; the n-step returns of
the decomposed value heads are precomputed at insertion (undiscounted, episodic: every episode ends at an official
terminal -- success, human collision, STOP elsewhere or the 500-step limit, and the remaining time is in the state).
"""
import threading

import numpy as np
import torch

from . import compiler as C

KMAX = 13 if C.LAYOUT == 'v3' else 10       # v3: + Layer 3 scan queries (env.scan_queries)
NMAX = 8
CUM = ('succ', 'coll', 'steps', 'path', 'soc')          # cumulants of the value heads (network units, see model.SCALE)
CUM_SCALE = np.array([1., 1., 100., 10., 50.], np.float32)


def pack(g, kmax=KMAX, nmax=NMAX):
    """Compiled graph -> dict of fixed-size arrays (float16 features, bool masks)."""
    K = min(len(g['cand']), kmax)
    N = min(len(g['hum']), nmax)
    out = dict(ego=g['ego'].astype(np.float16),
               cand=np.zeros((kmax, len(C.CAND)), np.float16), cand_mask=np.zeros(kmax, bool),
               traj=np.zeros((kmax, C.P, len(C.TRAJ)), np.float16),
               hum=np.zeros((nmax, len(C.HUM)), np.float16), hum_mask=np.zeros(nmax, bool),
               hfut=np.zeros((nmax, C.P, len(C.HFUT)), np.float16),
               pair=np.zeros((kmax, C.P, nmax, len(C.PAIR)), np.float16))
    out['cand'][:K] = g['cand'][:K]
    out['cand_mask'][:K] = True
    out['traj'][:K] = g['traj'][:K]
    if N:
        # keep the N nearest tracked humans (hum column 'dist')
        order = np.argsort(g['hum'][:, C.HUM.index('dist')])[:N]
        out['hum'][:N] = g['hum'][order]
        out['hum_mask'][:N] = True
        out['hfut'][:N] = g['hfut'][order]
        out['pair'][:K, :, :N] = g['pair'][:K][:, :, order]
    return out


FIELDS = dict(ego=((len(C.EGO),), np.float16), cand=((KMAX, len(C.CAND)), np.float16), cand_mask=((KMAX,), bool),
              traj=((KMAX, C.P, len(C.TRAJ)), np.float16), hum=((NMAX, len(C.HUM)), np.float16),
              hum_mask=((NMAX,), bool), hfut=((NMAX, C.P, len(C.HFUT)), np.float16),
              pair=((KMAX, C.P, NMAX, len(C.PAIR)), np.float16))
GRAPH_KEYS = tuple(FIELDS)


def stack(packed_list):
    return {k: np.stack([p[k] for p in packed_list]) for k in GRAPH_KEYS}


def to_torch(arrs, device, non_blocking=True):
    out = {}
    for k, v in arrs.items():
        t = torch.from_numpy(v)
        if t.dtype == torch.float16:
            t = t.float() if device == 'cpu' else t
        out[k] = t.to(device, non_blocking=non_blocking)
        if out[k].dtype == torch.float16:
            out[k] = out[k].float()
    return out


class Ring:
    """Replay ring of decisions with precomputed n-step cumulant sums and bootstrap pointers."""

    def __init__(self, capacity, nstep=3):
        self.cap = int(capacity)
        self.nstep = int(nstep)
        self.g = {k: np.zeros((self.cap,)+shape, dt) for k, (shape, dt) in FIELDS.items()}
        self.action = np.zeros(self.cap, np.int16)
        self.prob = np.ones(self.cap, np.float32)
        self.ncand = np.zeros(self.cap, np.int16)
        self.ret = np.zeros((self.cap, len(CUM)), np.float32)       # n-step cumulant sums (network units)
        self.boot = np.full(self.cap, -1, np.int64)                  # ring index of s_{t+n}, -1 = terminal inside
        self.mc = np.zeros((self.cap, len(CUM)), np.float32)         # Monte Carlo return to the episode end
        self.cfg = np.zeros(self.cap, np.int16)
        self.version = np.zeros(self.cap, np.int32)
        self.episode = np.full(self.cap, -1, np.int64)
        self.oracle = np.full(self.cap, -1, np.int16)                # Oracle's choice (warm-up data), -1 none
        self.cf = np.zeros((self.cap, KMAX, len(CUM)), np.float32)    # counterfactual labels per candidate
        self.cf_mask = np.zeros((self.cap, KMAX), bool)
        self.source = np.zeros(self.cap, np.int8)                    # 0 online RL, 1 oracle demo, 2 other
        self.valid = np.zeros(self.cap, bool)
        self.head = 0
        self.size = 0
        self.inserted = 0
        self.episodes = 0
        self.lock = threading.Lock()

    def add_episode(self, packed, actions, probs, cumulants, cfg_id=0, version=0, oracle=None, cf=None, cf_mask=None,
                    source=0):
        """packed: list of T packed graphs; cumulants: [T, 5] (network units; succ / coll are 1 at the terminal)."""
        T = len(packed)
        if T == 0 or T > self.cap:
            return None
        cum = np.asarray(cumulants, np.float32).reshape(T, len(CUM))
        with self.lock:
            idx = (self.head+np.arange(T)) % self.cap
            # evict whole episodes overlapping the new slots
            old = np.unique(self.episode[idx][self.episode[idx] >= 0])
            if len(old):
                self.valid[np.isin(self.episode, old)] = False
            for k in GRAPH_KEYS:
                self.g[k][idx] = np.stack([p[k] for p in packed])
            self.action[idx] = np.asarray(actions, np.int16)
            self.prob[idx] = np.asarray(probs, np.float32)
            self.ncand[idx] = np.asarray([int(p['cand_mask'].sum()) for p in packed], np.int16)
            csum = np.concatenate([np.zeros((1, len(CUM)), np.float32), np.cumsum(cum, 0)])
            for t in range(T):
                e = min(t+self.nstep, T)
                self.ret[idx[t]] = csum[e]-csum[t]
                self.boot[idx[t]] = idx[e] if e < T else -1
                self.mc[idx[t]] = csum[T]-csum[t]
            self.cfg[idx] = cfg_id
            self.version[idx] = version
            self.episode[idx] = self.episodes
            self.oracle[idx] = -1 if oracle is None else np.asarray(oracle, np.int16)
            if cf is not None:
                self.cf[idx] = np.asarray(cf, np.float32)
                self.cf_mask[idx] = np.asarray(cf_mask, bool)
            else:
                self.cf_mask[idx] = False
            self.source[idx] = source
            self.valid[idx] = True
            self.head = (self.head+T) % self.cap
            self.size = min(self.size+T, self.cap)
            self.inserted += T
            self.episodes += 1
        return idx

    def sample(self, n, rng, source=None, where=None):
        with self.lock:
            ok = self.valid if source is None else self.valid & np.isin(self.source, np.atleast_1d(source))
            if where is not None:
                ok = ok & where
            cand = np.flatnonzero(ok)
        if not len(cand):
            return None
        return cand[rng.integers(0, len(cand), n)]

    def gather(self, idx):
        """Graph arrays of ring indices (boot indices -1 -> zero graphs, masked by the caller)."""
        safe = np.where(idx >= 0, idx, 0)
        return {k: self.g[k][safe] for k in GRAPH_KEYS}
