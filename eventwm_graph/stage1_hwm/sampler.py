"""IID joint world sampling from the human world model, plus its exact count law.

Each world independently draws: key existence, one mixture mode and Gaussian
displacements per existing key; a residual count, residual positions from the
spatial law and residual trajectories conditioned on those positions. Paths
are piecewise linear between 0.25 s knots (the model's declared continuous
law), so any requested time in [0, 3.028] is evaluated exactly.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F

from .model import SEGMENTS, STEP, GRID, CELL, HALF, MAX_COUNT

OUTSIDE_RADII = (13., 25.)
MAX_SPEED = 3.0


def draw_mixture(p, worlds, generator, zero_first, temp=1., scale=1.):
    """p: params with leading [B,N]; returns knots [B,W,N,SEGMENTS,2] (knot 0 = segment-0 endpoint offset).

    ``temp``/``scale``: declared post-hoc sampling recalibration (mode-logit temperature and
    within-mode spread scale for sigma and the shared offset); identity by default."""
    mu, sigma, L, logits = p['mu'], p['sigma'] * scale, p['L'] * scale, p['logits'] / temp
    B, N, M = logits.shape;device = logits.device
    mode = torch.multinomial(F.softmax(logits, -1).reshape(-1, M), worlds, replacement=True, generator=generator)
    mode = mode.reshape(B, N, worlds).permute(0, 2, 1)                       # [B,W,N]
    bi = torch.arange(B, device=device)[:, None, None];ni = torch.arange(N, device=device)[None, None, :]
    mu_m = mu[bi, ni, mode];sig_m = sigma[bi, ni, mode];L_m = L[bi, ni, mode]  # [B,W,N,S,2],[B,W,N,S],[B,W,N,2,2]
    z = torch.randn(B, worlds, N, 2, device=device, generator=generator)
    eps = torch.randn(B, worlds, N, SEGMENTS, 2, device=device, generator=generator)
    offset = (L_m @ z[..., None])[..., 0]                                    # [B,W,N,2]
    delta = p['delta'][:, None].expand(-1, worlds, -1, -1)                   # [B,W,N,S]
    d = mu_m + delta[..., None] * offset[:, :, :, None] + sig_m[..., None] * eps
    # Certified horizontal human speed bound (3 m/s): project each constant-velocity segment.
    norm = d.norm(dim=-1, keepdim=True);limit = MAX_SPEED * delta[..., None]
    d = d * torch.clamp(limit / norm.clamp_min(1e-12), max=1.)
    if zero_first is not None:
        d[..., 0, :] = torch.where(zero_first[:, None, :, None], torch.zeros_like(d[..., 0, :]), d[..., 0, :])
    return torch.cumsum(d, -2)


def count_log_law(exist_p, visible, key_mask, log_count, minimum=None, size=24):
    """Exact log P(total count = c), c < size: visible + Poisson-binomial(stale) + residual (float64 numpy)."""
    as_np = lambda x:x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    p = as_np(exist_p).astype(np.float64);vis_m = as_np(visible).astype(bool);km = as_np(key_mask).astype(bool)
    res = np.exp(as_np(log_count).astype(np.float64))
    B, K = p.shape
    stale = np.where(km & ~vis_m, p, 0.)
    pb = np.zeros((B, size));pb[:, 0] = 1.
    for k in range(K):
        q = stale[:, k:k + 1]
        pb = pb * (1 - q) + np.concatenate([np.zeros((B, 1)), pb[:, :-1]], 1) * q
    conv = np.zeros_like(pb)
    for c in range(res.shape[1]):
        conv[:, c:] += pb[:, :size - c] * res[:, c:c + 1]
    vis = (vis_m & km).sum(1)
    j = np.arange(size)[None] - vis[:, None]
    shifted = np.where(j >= 0, np.take_along_axis(conv, np.clip(j, 0, None), 1), 0.)
    if minimum is not None:
        m = np.asarray(minimum)[:, None];shifted = np.where(np.arange(size)[None] >= m, shifted, 0.)
        shifted = shifted / shifted.sum(1, keepdims=True)
    with np.errstate(divide='ignore'):
        return np.log(shifted)


def draw_population(p_exist, key_mask, count_prob, minimum, worlds, generator, max_hidden):
    """Exact sampling of (key existence, residual count) from the law conditioned on total >= minimum.

    Draws the total from the truncated total-count law, then key existences sequentially from
    suffix (Poisson-binomial * residual) distributions; the remainder is the residual count.
    Equivalent to rejection sampling but cannot fail when the legal event is improbable.
    """
    B, K = p_exist.shape;device = p_exist.device;C = count_prob.shape[1];size = K + C
    q = torch.where(key_mask, p_exist, torch.zeros_like(p_exist)).double()
    g = torch.zeros(B, K + 1, size, dtype=torch.float64, device=device);g[:, K, :C] = count_prob.double()
    for k in range(K - 1, -1, -1):
        nxt = g[:, k + 1]
        g[:, k] = (1 - q[:, k:k + 1]) * nxt + q[:, k:k + 1] * F.pad(nxt[:, :-1], (1, 0))
    support = torch.arange(size, device=device)[None] >= minimum[:, None]
    total = g[:, 0] * support
    empty = total.sum(1) <= 0
    if empty.any():   # impossible under the model's finite support: place the legal minimum
        total = torch.where(empty[:, None], (torch.arange(size, device=device)[None] == minimum.clamp(max=size - 1)[:, None]).double(), total)
    t_rem = torch.multinomial(total / total.sum(1, keepdim=True), worlds, replacement=True, generator=generator)   # [B,W]
    present = torch.zeros(B, worlds, K, dtype=torch.bool, device=device)
    for k in range(K):
        num = q[:, k:k + 1] * torch.gather(g[:, k + 1], 1, (t_rem - 1).clamp_min(0)) * (t_rem >= 1)
        den = torch.gather(g[:, k], 1, t_rem)
        e = torch.rand(B, worlds, device=device, generator=generator, dtype=torch.float64) * den < num
        present[:, :, k] = e;t_rem = t_rem - e.long()
    return present & key_mask[:, None], t_rem.clamp(max=max_hidden)


def draw_population_np(p_exist, key_mask, count_prob, minimum, worlds, rng):
    """NumPy twin of ``draw_population`` for one anchor."""
    K = len(p_exist);C = len(count_prob);size = K + C
    q = np.where(key_mask, p_exist, 0.).astype(np.float64)
    g = np.zeros((K + 1, size));g[K, :C] = count_prob
    for k in range(K - 1, -1, -1):
        g[k] = (1 - q[k]) * g[k + 1];g[k, 1:] += q[k] * g[k + 1, :-1]
    total = g[0] * (np.arange(size) >= minimum)
    if total.sum() <= 0:total = (np.arange(size) == min(minimum, size - 1)).astype(np.float64)
    cdf = np.cumsum(total / total.sum())
    t_rem = np.minimum(np.searchsorted(cdf, rng.random(worlds), side='right'), size - 1)
    present = np.zeros((worlds, K), bool)
    for k in range(K):
        num = q[k] * g[k + 1][np.maximum(t_rem - 1, 0)] * (t_rem >= 1);den = g[k][t_rem]
        e = rng.random(worlds) * den < num
        present[:, k] = e;t_rem = t_rem - e
    return present & key_mask[None], t_rem


@torch.no_grad()
def sample(net, b, worlds, generator, max_hidden=MAX_COUNT, calib=None):
    """Canonical worlds for a batch: dict(knots [B,W,N,13,2], present [B,W,N], n_keys, count_logp)."""
    from .data import rasters
    m, v = rasters(b, *net.raster_spec)
    enc = net.encode(b, m, v)
    kp, exist_logit = net.key_params(b, enc)
    B, K = exist_logit.shape;device = exist_logit.device
    visible = b['key_age'] <= 1e-6
    p_exist = torch.where(visible, torch.ones_like(exist_logit), torch.sigmoid(exist_logit))
    calib = calib or {}
    key_knots = draw_mixture(kp, worlds, generator, visible, calib.get('key_temp', 1.), calib.get('key_scale', 1.)) + b['key_last'][:, None, :, None]
    log_count = net.log_count(enc)
    key_present, counts = draw_population(p_exist, b['key_mask'], log_count.exp(), b['min_count'], worlds, generator, max_hidden)
    H = int(counts.max()) if counts.numel() else 0
    if H:
        flat = torch.cat([enc['density'].reshape(B, -1), enc['outside'][:, None]], -1)
        cell = torch.multinomial(F.softmax(flat, -1), worlds * H, replacement=True, generator=generator)  # [B,W*H]
        inside = cell < GRID * GRID
        iu = torch.div(cell, GRID, rounding_mode='floor').float();iv = (cell % GRID).float()
        jit = torch.rand(B, worlds * H, 2, device=device, generator=generator)
        pos_in = torch.stack([(iu + jit[..., 0]) * CELL - HALF, (iv + jit[..., 1]) * CELL - HALF], -1)
        ang = torch.rand(B, worlds * H, device=device, generator=generator) * 2 * math.pi
        rad = OUTSIDE_RADII[0] + torch.rand(B, worlds * H, device=device, generator=generator) * (OUTSIDE_RADII[1] - OUTSIDE_RADII[0])
        pos_out = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], -1)
        pos = torch.where(inside[..., None], pos_in, pos_out)
        hidden_present = torch.arange(H, device=device)[None, None] < counts[..., None]   # [B,W,H]
        # Trajectory heads only for slots that exist in their world.
        flat_present = hidden_present.reshape(B, worlds * H)
        n_max = int(flat_present.sum(1).max())
        order = torch.argsort((~flat_present).to(torch.int8), dim=1)[:, :n_max]
        pos_c = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
        hp = net.hidden_params(enc, pos_c)
        hk_c = draw_mixture(hp, 1, generator, None, calib.get('hidden_temp', 1.), calib.get('hidden_scale', 1.))[:, 0]
        # hidden knot 0 is the sampled position itself (segment 0 has zero duration)
        hk_c = hk_c - hk_c[..., :1, :] + pos_c[..., None, :]
        hk = pos[..., None, :].expand(-1, -1, SEGMENTS, -1).clone()
        hk.scatter_(1, order[..., None, None].expand(-1, -1, SEGMENTS, 2), hk_c)
        hk = hk.reshape(B, worlds, H, SEGMENTS, 2)
        knots = torch.cat([key_knots, hk], 2);present = torch.cat([key_present, hidden_present], 2)
    else:
        knots, present = key_knots, key_present
    count_logp = count_log_law(p_exist, visible, b['key_mask'], log_count, b['min_count'].cpu().numpy())
    expected_residual = (log_count.exp() * torch.arange(log_count.shape[1], device=device)).sum(1).cpu().numpy()
    return dict(knots=knots, present=present, n_keys=K, count_logp=count_logp, p_exist=p_exist, expected_residual=expected_residual)


def interpolate(knots, times):
    """knots [...,13,2] at 0.25 s; times (relative, <=3.028) -> [..., T, 2]."""
    t = np.asarray(times, np.float64)
    s = np.clip(np.floor(t / STEP).astype(int), 0, SEGMENTS - 2)
    f = (t - s * STEP) / STEP
    a = knots[..., s, :];c = knots[..., s + 1, :]
    return a + f[:, None] * (c - a)


def to_world(canon, origin, rot):
    """canonical [...,2] -> world, rot world->canonical (orthogonal)."""
    return np.einsum('ji,...j->...i', rot, canon) + origin
