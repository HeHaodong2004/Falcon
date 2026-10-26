"""Vectorised training likelihood == brute-force enumeration (and dense Gaussian) reference.

Run: python scripts/stage1_hwm_run.py tests/stage1_hwm/test_likelihood_reference.py CHECKPOINT
"""
import itertools
import math
import sys
import numpy as np
import torch

from eventwm_graph.stage1_hwm.data import Store, collate, to_torch, rasters, FIXED_PRESENT, FIXED_ABSENT, LATENT, P_MAX
from eventwm_graph.stage1_hwm.bank_hwm import load
from eventwm_graph.stage1_hwm.model import mixture_logpdf, log_tail, SEGMENTS


def dense_logpdf(p, d, mask):
    """Explicit mixture of full-covariance Gaussians over the observed segments."""
    M = p['logits'].shape[-1];obs = np.flatnonzero(mask)
    comps = []
    for m in range(M):
        mu = p['mu'][m][obs].reshape(-1).double();sig = p['sigma'][m][obs].double();L = p['L'][m].double();dl = p['delta'][obs].double()
        U = torch.cat([dl[i] * L for i in range(len(obs))], 0)
        cov = torch.diag(torch.repeat_interleave(sig ** 2, 2)) + U @ U.T
        comps.append(torch.distributions.MultivariateNormal(mu, cov).log_prob(d[obs].reshape(-1).double()))
    return torch.logsumexp(torch.stack(comps) + torch.log_softmax(p['logits'].double(), -1), 0)


def main(checkpoint):
    torch.manual_seed(0)
    net = load(checkpoint, 'cpu');store = Store('dev', limit_shards=4)
    lat = [i for i in range(store.n) if np.isnan(store.k['k_legacy_presence'][store.key_off[i]:store.key_off[i + 1]]).any()]
    idx = np.array(lat[:24] + list(range(8)))
    b = to_torch(collate(store, idx), 'cpu')
    m, v = rasters(b, *net.raster_spec)
    with torch.no_grad():
        net.loss(b, m, v);total = net._debug['total'];usable = net._debug['usable']
        enc = net.encode(b, m, v);kp, el = net.key_params(b, enc);lc = net.log_count(enc)
        hum = b['hum'];hm = b['hum_mask'];tm = b['tau_mask'].float()
        hp = net.hidden_params(enc, hum[:, :, 0])
        worst = 0.;dense_worst = 0.;checked = 0
        for j in range(len(idx)):
            if not usable[j]:continue
            K = int(b['key_mask'][j].sum());P = int(hm[j].sum())
            def logq(k, h):
                stale = float(b['key_age'][j, k]) > 1e-6
                d = torch.cat([(hum[j, h, 0] - b['key_last'][j, k])[None], hum[j, h, 1:] - hum[j, h, :-1]], 0)
                mask = torch.cat([torch.tensor([float(stale)]), tm[j, 1:]])
                pk = {kk: vv[j, k] for kk, vv in kp.items()}
                val = mixture_logpdf(pk, d, mask)
                if checked < 3 and k == 0 and h == 0:
                    nonlocal_dense.append(abs(float(val - dense_logpdf(pk, d, mask.numpy() > 0))))
                if not stale and float((hum[j, h, 0] - b['key_last'][j, k]).norm()) >= .05:val = val - 1e4
                return float(val)
            def res(h):
                d = torch.cat([torch.zeros(1, 2), hum[j, h, 1:] - hum[j, h, :-1]], 0)
                mask = torch.cat([torch.zeros(1), tm[j, 1:]])
                g = mixture_logpdf({kk: vv[j, h] for kk, vv in hp.items()}, d, mask)
                return float(net.log_density({k_: (vv[j:j + 1] if torch.is_tensor(vv) else vv) for k_, vv in enc.items()}, hum[j:j + 1, h:h + 1, 0])[0, 0] + g)
            vis = [float(b['key_age'][j, k]) <= 1e-6 for k in range(K)]
            lp = [0. if vis[k] else float(torch.nn.functional.logsigmoid(el[j, k])) for k in range(K)]
            lnp = [-1e4 if vis[k] else float(torch.nn.functional.logsigmoid(-el[j, k])) for k in range(K)]
            types = [int(b['key_type'][j, k]) for k in range(K)];own = [int(b['key_owner'][j, k]) for k in range(K)]
            fixed = sum(lp[k] + logq(k, own[k]) for k in range(K) if types[k] == FIXED_PRESENT) + sum(lnp[k] for k in range(K) if types[k] == FIXED_ABSENT)
            owned = {own[k] for k in range(K) if types[k] == FIXED_PRESENT}
            U = [h for h in range(P) if h not in owned];L = sorted([k for k in range(K) if types[k] == LATENT], key=lambda k:float(b['key_age'][j, k]))
            terms = []
            for assign in itertools.product([None] + U, repeat=len(L)):
                used = [a for a in assign if a is not None]
                if len(used) != len(set(used)):continue
                t = 0.
                for k, a in zip(L, assign):t += lnp[k] if a is None else lp[k] + logq(k, a)
                rest = [h for h in U if h not in used];n = len(rest)
                t += (float(lc[j, n]) if n < lc.shape[1] else -1e4) + math.lgamma(n + 1) + sum(res(h) for h in rest)
                terms.append(t)
            ref = fixed + float(torch.logsumexp(torch.tensor(terms), 0))
            if net.truncate:
                pe = torch.where(b['key_age'][j:j + 1] <= 1e-6, torch.ones_like(el[j:j + 1]), torch.sigmoid(el[j:j + 1])) * b['key_mask'][j:j + 1]
                ref -= float(log_tail(pe, (b['key_age'][j:j + 1] <= 1e-6) & b['key_mask'][j:j + 1], lc[j:j + 1], b['min_count'][j:j + 1]))
            worst = max(worst, abs(ref - float(total[j])));checked += 1
    print('likelihood: %d anchors (%d with unresolved keys), max |vectorised - enumeration| = %.2e; dense-Gaussian check max %.2e'
          % (checked, len(lat[:24]), worst, max(nonlocal_dense) if nonlocal_dense else float('nan')))
    assert worst < 2e-3 and (not nonlocal_dense or max(nonlocal_dense) < 2e-3)   # float32 network


nonlocal_dense = []
if __name__ == '__main__':
    main(sys.argv[1])
