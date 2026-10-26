"""Low-overhead CPU inference engine for the human world model (same law as ``sampler``).

The trained network runs as ONE TorchScript module (canonical raster resampling, BEV,
key/global encoders, transformer and every head) returning raw head outputs. Mixture
post-processing, exact population rejection, residual cell sampling and path assembly are
vectorised NumPy (PCG64), which avoids ~2,000 small tensor dispatches per call. The
sampled law is identical to ``sampler.sample``; random streams differ.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .legal_state import AGE_STEP, CROP, COARSE, VIS_POOL, HISTORY
from .model import SEGMENTS, STEP, GRID, CELL, HALF, MAX_COUNT, fold_batchnorm, sample_field
from .sampler import OUTSIDE_RADII, MAX_SPEED, count_log_law
from .kernels import draw_paths


class Engine(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.bev = net.bev;self.glob = net.glob;self.hist = net.hist;self.key_mlp = net.key_mlp;self.mixer = net.mixer
        self.key_net = net.key_traj.net;self.exist = net.exist;self.count = net.count;self.film = net.film
        self.density = net.density;self.outside = net.outside;self.hidden_embed = net.hidden_embed;self.hidden_net = net.hidden_traj.net
        if net.arch not in ('fast', 'lite'):raise ValueError('scripted engine supports the fast/lite deployment architectures')
        self.field = int(net.field)
        assert VIS_POOL == 2
        self.half = float(HALF);self.age_step = float(AGE_STEP);self.coarse = float(COARSE);self.crop = int(CROP);self.history = int(HISTORY)
        out, cell = net.raster_spec;self.out = int(out)
        c = (torch.arange(out).float() + .5) * cell - out * cell / 2
        U, V = torch.meshgrid(c, c, indexing='ij')
        self.register_buffer('uv', torch.stack([U, V], -1).reshape(-1, 2))
        self.register_buffer('ahead', net.ahead.clone());self.register_buffer('around', net.around.clone())

    def _field(self, field, points):
        g = torch.stack([points[..., 1] / self.half, points[..., 0] / self.half], -1)
        out = F.grid_sample(field, g[:, :, None], mode='bilinear', padding_mode='zeros', align_corners=False)
        return out[..., 0].transpose(1, 2)

    def _sample(self, img, rel, cell: float, size: int):
        gx = rel[:, 1] / (cell * size) * 2 - 1;gy = rel[:, 0] / (cell * size) * 2 - 1
        grid = torch.stack([gx, gy], -1).reshape(1, self.out, self.out, 2)
        return F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    def forward(self, mp, vis, rot, origin, corner, glob, key_feat, key_hist, key_vel, key_last, key_age, key_mask):
        # canonical rasters (exactly data.rasters for batch size 1)
        m8 = mp.long()
        onehot = torch.stack([m8 == 0, m8 == 1, m8 == 2], 1).float()
        age = vis.float();never = (age == 255).float();now = (age == 0).float()
        seconds = torch.where(age == 255, torch.full_like(age, 254.), age) * self.age_step
        vfeat = torch.stack([never, torch.log1p(seconds) / 2.6 * (1 - never), now], 1)
        world = self.uv @ rot[0] + origin
        rel = world - corner
        ones = torch.ones_like(onehot[:, :1])
        m = self._sample(torch.cat([onehot, ones], 1), rel, self.coarse, self.crop)
        m = torch.cat([m[:, :1] + (1 - m[:, 3:4]), m[:, 1:3]], 1)
        v = self._sample(torch.cat([vfeat, ones[:, :, ::2, ::2]], 1), rel, self.coarse * 2, self.crop // 2)
        v = torch.cat([v[:, :1] + (1 - v[:, 3:4]), v[:, 1:3]], 1)
        # encoder (exactly HWM.encode)
        f, pooled = self.bev(m, v)
        K = key_mask.shape[1]
        g = self.glob(torch.cat([pooled, glob], -1))
        hh = self.hist(key_hist.reshape(1, K, self.history * 4))
        speed = torch.sqrt((key_vel * key_vel).sum(-1, keepdim=True))
        heading = torch.where(speed > .2, key_vel / speed.clamp_min(1e-6), torch.tensor([1., 0.]))
        normal = torch.stack([-heading[..., 1], heading[..., 0]], -1)
        center = key_last + key_vel * key_age[..., None].clamp(max=2.)
        pts = center[..., None, :] + self.ahead[:, :1] * heading[..., None, :] + self.ahead[:, 1:] * normal[..., None, :]
        sf = self._field(f, pts.reshape(1, -1, 2)).reshape(1, K, -1)
        keys = self.key_mlp(torch.cat([key_feat, hh, sf], -1))
        tokens = torch.cat([g[:, None], keys], 1)
        mask = torch.cat([torch.zeros(1, 1, dtype=torch.bool), ~key_mask], 1)
        o = self.mixer(tokens, src_key_padding_mask=mask)
        gt = o[:, 0];kt = o[:, 1:]
        film = self.film(gt)
        dens = self.density(f * (1 + film[:, :self.field, None, None]) + film[:, self.field:, None, None])[:, 0]
        flat = torch.cat([dens.reshape(1, -1), self.outside(gt)], -1)
        return f, gt, self.key_net(kt)[0], self.exist(kt)[0, :, 0], F.log_softmax(self.count(gt), -1)[0], F.log_softmax(flat, -1)[0]

    @torch.jit.export
    def hidden(self, f, gt, pos):
        N = pos.shape[1]
        pts = pos[:, :, None, :] + self.around[None, None]
        sf = self._field(f, pts.reshape(1, -1, 2)).reshape(1, N, -1)
        h = self.hidden_embed(torch.cat([sf, pos / 10, gt[:, None].expand(-1, N, -1)], -1))
        return self.hidden_net(h)[0]


def softplus(x):
    return np.logaddexp(0., x)


def mixture(raw, modes, base_velocity, age):
    """NumPy TrajectoryHead post-processing; raw [N, M*(1+3S+3)] -> dict of [N,M,...] arrays."""
    N = raw.shape[0];raw = raw.reshape(N, modes, -1)
    delta = np.full((N, SEGMENTS), STEP);delta[:, 0] = age
    mu = (base_velocity[:, None, None, :] + raw[:, :, 1:1 + 2 * SEGMENTS].reshape(N, modes, SEGMENTS, 2) * 1.5) * delta[:, None, :, None]
    sigma = (softplus(raw[:, :, 1 + 2 * SEGMENTS:1 + 3 * SEGMENTS] - 1.) + .01) * np.sqrt(delta[:, None, :] / STEP + 1e-4)
    lt = raw[:, :, 1 + 3 * SEGMENTS:]
    L = np.zeros((N, modes, 2, 2))
    L[..., 0, 0] = softplus(lt[..., 0]) * .5 + 1e-3;L[..., 1, 1] = softplus(lt[..., 1]) * .5 + 1e-3;L[..., 1, 0] = lt[..., 2] * .3
    logits = raw[:, :, 0]
    p = np.exp(logits - logits.max(-1, keepdims=True));p /= p.sum(-1, keepdims=True)
    return dict(prob=p, mu=mu, sigma=sigma, L=L, delta=delta)


def draw(p, worlds, rng, zero_first=None):
    """Knots [W,N,S,2] (cumulative displacements) for N entities, speed-bounded."""
    N, M = p['prob'].shape
    if N == 0:return np.zeros((worlds, 0, SEGMENTS, 2))
    u = rng.random((worlds, N))
    mode = (u[..., None] > np.cumsum(p['prob'], -1)[None]).sum(-1).clip(max=M - 1)    # [W,N]
    ni = np.arange(N)[None]
    mu = p['mu'][ni, mode];sig = p['sigma'][ni, mode];L = p['L'][ni, mode]
    z = rng.standard_normal((worlds, N, 2));eps = rng.standard_normal((worlds, N, SEGMENTS, 2))
    offset = np.einsum('wnij,wnj->wni', L, z)
    d = mu + p['delta'][None, :, :, None] * offset[:, :, None] + sig[..., None] * eps
    if zero_first is not None:d[:, zero_first, 0] = 0.
    norm = np.linalg.norm(d, axis=-1, keepdims=True);limit = MAX_SPEED * p['delta'][None, :, :, None]
    d *= np.minimum(1., limit / np.maximum(norm, 1e-12))
    return np.cumsum(d, -2)


class FastSampler:
    """Callable used by the API: tensors for one anchor -> canonical worlds (NumPy)."""

    def __init__(self, net):
        net = fold_batchnorm(net.eval())
        self.modes = net.key_traj.modes;self.use_kernels = True
        self.engine = torch.jit.script(Engine(net).eval())

    def _paths(self, raw, base_v, age, zero_first, worlds, rng):
        """Fused numba kernel; consumes the RNG exactly like mixture()+draw()."""
        N = raw.shape[0]
        if N == 0 or not self.use_kernels:
            return draw(mixture(raw, self.modes, base_v, age), worlds, rng, zero_first)
        u = rng.random((worlds, N));z = rng.standard_normal((worlds, N, 2));eps = rng.standard_normal((worlds, N, SEGMENTS, 2))
        zf = np.zeros(N, bool) if zero_first is None else zero_first
        return draw_paths(raw, self.modes, base_v, age, zf, u, z, eps, MAX_SPEED)

    @torch.no_grad()
    def __call__(self, b, worlds, rng, max_hidden=MAX_COUNT):
        f, gt, key_raw, exist_logit, log_count, log_cell = self.engine(
            b['map'], b['vis'], b['rot'], b['origin'], b['corner'], b['global'], b['key_feat'], b['key_hist'], b['key_vel'],
            b['key_last'], b['key_age'], b['key_mask'])
        key_mask = b['key_mask'][0].numpy();age = b['key_age'][0].numpy().astype(np.float64)
        visible = age <= 1e-6;K = len(key_mask)
        p_exist = np.where(visible, 1., 1. / (1. + np.exp(-exist_logit.numpy().astype(np.float64))))
        key_knots = self._paths(key_raw.numpy().astype(np.float64), b['key_vel'][0].numpy().astype(np.float64), age, visible, worlds, rng) \
            + b['key_last'][0].numpy().astype(np.float64)[None, :, None]
        count_p = np.exp(log_count.numpy().astype(np.float64));count_p /= count_p.sum()
        minimum = int(b['min_count'][0])
        # exact rejection on total >= minimum
        present = np.zeros((worlds, K), bool);counts = np.zeros(worlds, int);filled = 0
        ccdf = np.cumsum(count_p)
        for _ in range(200):
            need = worlds - filled
            if need == 0:break
            pool = max(worlds, 2 * need)
            pr = (rng.random((pool, K)) < p_exist[None]) & key_mask[None]
            ct = np.minimum((rng.random(pool)[:, None] > ccdf[None]).sum(-1), max_hidden)
            ok = np.flatnonzero(pr.sum(-1) + ct >= minimum)[:need]
            present[filled:filled + len(ok)] = pr[ok];counts[filled:filled + len(ok)] = ct[ok];filled += len(ok)
        if filled < worlds:raise RuntimeError('population rejection sampling exhausted its round budget')
        H = int(counts.max())
        knots = key_knots;pres = present
        if H:
            total = int(counts.sum())
            cell_p = np.exp(log_cell.numpy().astype(np.float64));cdf = np.cumsum(cell_p);cdf /= cdf[-1]
            cell = np.searchsorted(cdf, rng.random(total), side='right').clip(max=len(cdf) - 1)
            inside = cell < GRID * GRID
            jit = rng.random((total, 2))
            pos = np.stack([(cell // GRID + jit[:, 0]) * CELL - HALF, (cell % GRID + jit[:, 1]) * CELL - HALF], -1)
            ang = rng.random(total) * 2 * math.pi;rad = OUTSIDE_RADII[0] + rng.random(total) * (OUTSIDE_RADII[1] - OUTSIDE_RADII[0])
            pos = np.where(inside[:, None], pos, np.stack([rad * np.cos(ang), rad * np.sin(ang)], -1))
            raw = self.engine.hidden(f, gt, torch.from_numpy(pos.astype(np.float32))[None]).numpy().astype(np.float64)
            hk = self._paths(raw, np.zeros((total, 2)), np.zeros(total), None, 1, rng)[0]
            hk = hk - hk[:, :1] + pos[:, None]
            slots = np.zeros((worlds, H, SEGMENTS, 2))
            world_of = np.repeat(np.arange(worlds), counts);slot_of = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            slots[world_of, slot_of] = hk
            hidden_present = np.arange(H)[None] < counts[:, None]
            knots = np.concatenate([key_knots, slots], 1);pres = np.concatenate([present, hidden_present], 1)
        count_logp = count_log_law(p_exist[None], visible[None], key_mask[None], np.log(count_p)[None], np.array([minimum]))[0]
        return dict(knots=knots, present=pres, n_keys=K, count_logp=count_logp, p_exist=p_exist,
                    expected_residual=np.array([float((count_p * np.arange(len(count_p))).sum())]))
