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


def draw_mixture(p, worlds, generator, zero_first):
    """p: params with leading [B,N]; returns knots [B,W,N,SEGMENTS,2] (knot 0 = segment-0 endpoint offset)."""
    mu, sigma, L, logits = p['mu'], p['sigma'], p['L'], p['logits']
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
    p = exist_p.detach().cpu().numpy().astype(np.float64);vis_m = visible.cpu().numpy();km = key_mask.cpu().numpy()
    res = np.exp(log_count.detach().cpu().numpy().astype(np.float64))
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


def draw_population(p_exist, key_mask, count_prob, minimum, worlds, generator, max_hidden, max_rounds=200):
    """Exact rejection sampling of (key existence, residual count) given total >= minimum."""
    B, K = p_exist.shape;device = p_exist.device
    present = torch.zeros(B, worlds, K, dtype=torch.bool, device=device);counts = torch.zeros(B, worlds, dtype=torch.long, device=device)
    filled = torch.zeros(B, dtype=torch.long, device=device)
    for _ in range(max_rounds):
        need = worlds - filled
        if int(need.max()) == 0:break
        pool = max(worlds, 2 * int(need.max()))
        pr = (torch.rand(B, pool, K, device=device, generator=generator) < p_exist[:, None]) & key_mask[:, None]
        ct = torch.multinomial(count_prob, pool, replacement=True, generator=generator).clamp(max=max_hidden)
        ok = (pr.sum(-1) + ct) >= minimum[:, None]
        rank = torch.cumsum(ok.long(), 1) - 1 + filled[:, None]               # destination slot of each accepted draw
        take = ok & (rank < worlds)
        bi, pi = torch.nonzero(take, as_tuple=True)
        present[bi, rank[bi, pi]] = pr[bi, pi];counts[bi, rank[bi, pi]] = ct[bi, pi]
        filled = filled + take.sum(1)
    if int((worlds - filled).max()) > 0:raise RuntimeError('population rejection sampling exhausted its round budget')
    return present, counts


@torch.no_grad()
def sample(net, b, worlds, generator, max_hidden=MAX_COUNT):
    """Canonical worlds for a batch: dict(knots [B,W,N,13,2], present [B,W,N], n_keys, count_logp)."""
    from .data import rasters
    m, v = rasters(b, *net.raster_spec)
    enc = net.encode(b, m, v)
    kp, exist_logit = net.key_params(b, enc)
    B, K = exist_logit.shape;device = exist_logit.device
    visible = b['key_age'] <= 1e-6
    p_exist = torch.where(visible, torch.ones_like(exist_logit), torch.sigmoid(exist_logit))
    key_knots = draw_mixture(kp, worlds, generator, visible) + b['key_last'][:, None, :, None]
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
        hk_c = draw_mixture(hp, 1, generator, None)[:, 0]
        # hidden knot 0 is the sampled position itself (segment 0 has zero duration)
        hk_c = hk_c - hk_c[..., :1, :] + pos_c[..., None, :]
        hk = pos[..., None, :].expand(-1, -1, SEGMENTS, -1).clone()
        hk.scatter_(1, order[..., None, None].expand(-1, -1, SEGMENTS, 2), hk_c)
        hk = hk.reshape(B, worlds, H, SEGMENTS, 2)
        knots = torch.cat([key_knots, hk], 2);present = torch.cat([key_present, hidden_present], 2)
    else:
        knots, present = key_knots, key_present
    count_logp = count_log_law(p_exist, visible, b['key_mask'], log_count, b['min_count'].cpu().numpy())
    return dict(knots=knots, present=present, n_keys=K, count_logp=count_logp, p_exist=p_exist)


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
