"""Human world model network and its exact training likelihood.

Joint law of one anchor (canonical camera frame):
  * every tracker key exists with probability p_k (1 for keys visible now);
  * an existing key follows a mixture of M Gaussian displacement sequences:
    segment 0 = last observation -> now (stale keys), segments 1..12 = 0.25 s
    steps. Within a mode d_s = mu_s + Delta_s L z + eps_s, z~N(0,I2) shared
    across segments (persistent velocity offset), eps_s~N(0, sigma_s^2 I2);
  * humans not represented by an existing key form an IID cluster process:
    count ~ Categorical(0..10), current position ~ softmax over a 64x64 0.4 m
    canonical grid (+ one outside bin), future ~ the same mixture family
    conditioned on the sampled position.
Unresolved keys are marginalized over every injective key->human assignment.
"""
import itertools
import math
import torch
from torch import nn
from torch.nn import functional as F

from .data import K_MAX, P_MAX, L_MAX, FIXED_PRESENT, FIXED_ABSENT, LATENT
from .legal_state import HISTORY

SEGMENTS = 13
STEP = .25
GRID = 64
CELL = .4
HALF = GRID * CELL / 2
OUTSIDE_AREA = 50. ** 2 - (2 * HALF) ** 2
MAX_COUNT = 10
LOG2PI = math.log(2 * math.pi)


def conv(i, o, s=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, padding=d, dilation=d, bias=False), nn.GroupNorm(4, o), nn.SiLU())


class Res(nn.Module):
    def __init__(self, c, d=1):
        super().__init__();self.a = conv(c, c, d=d);self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=d, dilation=d, bias=False), nn.GroupNorm(4, c))

    def forward(self, x):return F.silu(x + self.b(self.a(x)))


class BEV(nn.Module):
    """128x128 (0.2 m) canonical rasters -> 64x64 (0.4 m) feature field + pooled context."""

    def __init__(self, c=(24, 48, 64, 96), out=48):
        super().__init__()
        self.stem = conv(5, c[0], s=2)                      # map3 + coords2 -> 64
        self.mix = nn.Sequential(conv(c[0] + 3, c[1]), Res(c[1]))  # + visibility at 64
        self.d1 = nn.Sequential(conv(c[1], c[2], s=2), Res(c[2]))   # 32
        self.d2 = nn.Sequential(conv(c[2], c[3], s=2), Res(c[3], 2), Res(c[3], 4))  # 16
        self.u1 = nn.Sequential(nn.Conv2d(c[3] + c[2], c[2], 1), nn.SiLU(), Res(c[2]))
        self.u2 = nn.Sequential(nn.Conv2d(c[2] + c[1], out, 1), nn.SiLU(), conv(out, out))
        self.pooled = 2 * c[3]

    def forward(self, m, v):
        B, _, H, W = m.shape
        lin = torch.linspace(-1, 1, H, device=m.device)
        coords = torch.stack(torch.meshgrid(lin, lin, indexing='ij'), 0)[None].expand(B, -1, -1, -1)
        x = self.stem(torch.cat([m, coords], 1))
        v64 = F.avg_pool2d(v, 2)
        a = self.mix(torch.cat([x, v64], 1))
        b = self.d1(a);c = self.d2(b)
        pooled = torch.cat([c.mean((2, 3)), c.amax((2, 3))], 1)
        u = self.u1(torch.cat([F.interpolate(c, scale_factor=2, mode='nearest'), b], 1))
        f = self.u2(torch.cat([F.interpolate(u, scale_factor=2, mode='nearest'), a], 1))
        return f, pooled


def cbr(i, o, s=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, padding=d, dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class ResB(nn.Module):
    def __init__(self, c, d=1):
        super().__init__();self.a = cbr(c, c, d=d);self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=d, dilation=d, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):return F.relu(x + self.b(self.a(x)))


class BEVFast(nn.Module):
    """64x64 (0.4 m) canonical rasters -> 64x64 feature field; ~10x fewer MACs than ``BEV``."""

    def __init__(self, c=(16, 32, 48), out=32):
        super().__init__()
        self.stem = cbr(8, c[0])                                    # map3 + vis3 + coords2 at 64
        self.d1 = nn.Sequential(cbr(c[0], c[1], s=2), ResB(c[1]))   # 32
        self.d2 = nn.Sequential(cbr(c[1], c[2], s=2), ResB(c[2]))  # 16
        self.u1 = nn.Sequential(nn.Conv2d(c[2] + c[1], c[1], 1), nn.ReLU(inplace=True), cbr(c[1], c[1]))
        self.u2 = nn.Sequential(nn.Conv2d(c[1] + c[0], out, 1), nn.ReLU(inplace=True))
        self.pooled = 2 * c[2]

    def forward(self, m, v):
        B, _, H, W = m.shape
        lin = torch.linspace(-1, 1, H, device=m.device)
        coords = torch.stack(torch.meshgrid(lin, lin, indexing='ij'), 0)[None].expand(B, -1, -1, -1)
        a = self.stem(torch.cat([m, v, coords], 1))
        b = self.d1(a);c = self.d2(b)
        pooled = torch.cat([c.mean((2, 3)), c.amax((2, 3))], 1)
        u = self.u1(torch.cat([F.interpolate(c, scale_factor=2, mode='nearest'), b], 1))
        f = self.u2(torch.cat([F.interpolate(u, scale_factor=2, mode='nearest'), a], 1))
        return f, pooled


def sample_field(field, points):
    """Bilinear sample [B,C,64,64] canonical field at canonical metres points [B,N,2] -> [B,N,C]."""
    # field indexed [u (H), v (W)]; grid x -> W (v), y -> H (u)
    g = torch.stack([points[..., 1] / HALF, points[..., 0] / HALF], -1)
    out = F.grid_sample(field, g[:, :, None], mode='bilinear', padding_mode='zeros', align_corners=False)
    return out[..., 0].transpose(1, 2)


class TrajectoryHead(nn.Module):
    """Mixture parameters from an entity embedding."""

    def __init__(self, d, modes=6):
        super().__init__();self.modes = modes
        self.net = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, modes * (1 + SEGMENTS * 3 + 3)))

    def forward(self, h, base_velocity, age):
        """h [...,d]; base_velocity [...,2]; age [...] -> dict of mixture parameters."""
        M = self.modes;raw = self.net(h).reshape(*h.shape[:-1], M, -1)
        logits = raw[..., 0]
        delta = torch.full((SEGMENTS,), STEP, device=h.device).expand(*h.shape[:-1], SEGMENTS).clone()
        delta[..., 0] = age
        mu_res = raw[..., 1:1 + 2 * SEGMENTS].reshape(*h.shape[:-1], M, SEGMENTS, 2)
        mu = (base_velocity[..., None, None, :] + mu_res * 1.5) * delta[..., None, :, None]
        log_sigma = raw[..., 1 + 2 * SEGMENTS:1 + 3 * SEGMENTS]
        # per-segment std grows with segment duration (sqrt for stale age)
        sigma = (F.softplus(log_sigma - 1.) + .01) * torch.sqrt(delta[..., None, :] / STEP + 1e-4)
        lt = raw[..., 1 + 3 * SEGMENTS:]
        L = torch.zeros(*lt.shape[:-1], 2, 2, device=h.device)
        L[..., 0, 0] = F.softplus(lt[..., 0]) * .5 + 1e-3
        L[..., 1, 1] = F.softplus(lt[..., 1]) * .5 + 1e-3
        L[..., 1, 0] = lt[..., 2] * .3
        return dict(logits=logits, mu=mu, sigma=sigma, L=L, delta=delta)


def mixture_logpdf(p, d, mask):
    """log density of displacements d [...,S,2] under mixture p (params broadcast over leading dims).

    p tensors have shape [..., M, ...]; d/mask have the same leading dims as p minus M.
    """
    mu, sigma, L, delta = p['mu'], p['sigma'], p['L'], p['delta']
    r = (d[..., None, :, :] - mu) * mask[..., None, :, None]
    w = mask[..., None, :] / sigma.square()                     # [...,M,S]
    quad1 = (w * r.square().sum(-1)).sum(-1)
    dl = delta[..., None, :]                                    # [...,1,S]
    LtL = L.transpose(-1, -2) @ L                               # [...,M,2,2]
    A = torch.eye(2, device=d.device) + (w * dl.square()).sum(-1)[..., None, None] * LtL
    b = (L.transpose(-1, -2) @ (w[..., None] * dl[..., None] * r).sum(-2)[..., None])[..., 0]
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    inv = torch.stack([torch.stack([A[..., 1, 1], -A[..., 0, 1]], -1), torch.stack([-A[..., 1, 0], A[..., 0, 0]], -1)], -2) / det[..., None, None]
    quad = quad1 - (b[..., None, :] @ inv @ b[..., :, None])[..., 0, 0]
    n = mask.sum(-1)[..., None]
    logdet = (mask[..., None, :] * 2 * torch.log(sigma.square())).sum(-1) + torch.log(det)
    comp = -.5 * (quad + logdet + 2 * n * LOG2PI)
    return torch.logsumexp(comp + F.log_softmax(p['logits'], -1), -1)


def log_tail(p_exist, visible, log_count, minimum, size=24):
    """Differentiable log P(total >= minimum); p_exist already masked (0 for padded keys)."""
    B, K = p_exist.shape
    stale = torch.where(visible, torch.zeros_like(p_exist), p_exist)
    pb = torch.zeros(B, size, device=p_exist.device, dtype=p_exist.dtype);pb = torch.cat([torch.ones_like(pb[:, :1]), pb[:, 1:]], 1)
    for k in range(K):
        q = stale[:, k:k + 1]
        pb = pb * (1 - q) + F.pad(pb[:, :-1], (1, 0)) * q
    res = log_count.exp()
    conv = sum(F.pad(pb[:, :size - c], (c, 0)) * res[:, c:c + 1] for c in range(res.shape[1]))
    vis = visible.sum(1)
    j = torch.arange(size, device=p_exist.device)[None] - vis[:, None]
    law = torch.where(j >= 0, torch.gather(conv, 1, j.clamp_min(0)), torch.zeros_like(conv))
    below = (law * (torch.arange(size, device=p_exist.device)[None] < minimum[:, None])).sum(1)
    return torch.log1p(-below.clamp(max=1 - 1e-6))


def injections(nl, nh):
    """All partial injective maps from nl latent keys into nh humans (-1 = unmatched)."""
    rows = []
    for assign in itertools.product(range(-1, nh), repeat=nl):
        used = [a for a in assign if a >= 0]
        if len(used) == len(set(used)):rows.append(assign)
    return torch.tensor(rows, dtype=torch.long)


class HWM(nn.Module):
    def __init__(self, d=128, modes=6, field=48, global_in=43, key_in=12, arch='full', truncate=False, ablate=()):
        super().__init__()
        self.arch = arch;self.truncate = truncate;self.ablate = tuple(ablate)
        if arch == 'fast':
            field = 32;self.bev = BEVFast(out=field);self.raster_spec = (64, .4)
        else:
            self.bev = BEV(out=field);self.raster_spec = (128, .2)
        self.register_buffer('ahead', torch.tensor([[0., 0.], [.6, 0.], [1.2, 0.], [2., 0.], [3., 0.], [1., .8], [1., -.8], [-.8, 0.]]), persistent=False)
        if arch == 'fast':
            self.hist = nn.Sequential(nn.Linear(4 * HISTORY, 64), nn.SiLU(), nn.Linear(64, 64))
        else:
            self.hist = nn.GRU(4, 64, batch_first=True)
        self.key_mlp = nn.Sequential(nn.Linear(key_in + 64 + field * len(self.ahead), d), nn.SiLU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.Linear(self.bev.pooled + global_in, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0., batch_first=True, norm_first=True)
        self.mixer = nn.TransformerEncoder(layer, 2)
        self.key_traj = TrajectoryHead(d, modes)
        self.exist = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.count = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, MAX_COUNT + 1))
        self.film = nn.Linear(d, 2 * field)
        self.density = nn.Sequential(nn.Conv2d(field, field, 1), nn.SiLU(), nn.Conv2d(field, 1, 1))
        self.outside = nn.Linear(d, 1)
        self.hidden_embed = nn.Sequential(nn.Linear(field * len(self.ahead) + 2 + d, d), nn.SiLU(), nn.Linear(d, d))
        self.hidden_traj = TrajectoryHead(d, modes)
        self.d = d;self.field = field
        self.register_buffer('inj', injections(L_MAX, P_MAX), persistent=False)

    # ------------------------------------------------------------------
    def encode(self, b, m, v):
        if 'map' in self.ablate:m = torch.cat([torch.ones_like(m[:, :1]), torch.zeros_like(m[:, 1:])], 1)
        if 'map' in self.ablate or 'vis' in self.ablate:v = torch.cat([torch.ones_like(v[:, :1]), torch.zeros_like(v[:, 1:])], 1)
        f, pooled = self.bev(m, v)
        B = f.shape[0]
        g = self.glob(torch.cat([pooled, b['global']], -1))
        # key tokens
        if self.arch == 'fast':
            hh = self.hist(b['key_hist'].reshape(B, K_MAX, HISTORY * 4))
        else:
            _, hh = self.hist(b['key_hist'].reshape(-1, HISTORY, 4));hh = hh[-1].reshape(B, K_MAX, 64)
        vel = b['key_vel'];speed = vel.norm(dim=-1, keepdim=True)
        heading = torch.where(speed > .2, vel / speed.clamp_min(1e-6), torch.tensor([1., 0.], device=vel.device))
        normal = torch.stack([-heading[..., 1], heading[..., 0]], -1)
        ahead = self.ahead
        center = b['key_last'] + vel * b['key_age'][..., None].clamp(max=2.)
        pts = center[..., None, :] + ahead[:, :1] * heading[..., None, :] + ahead[:, 1:] * normal[..., None, :]
        sf = sample_field(f, pts.reshape(B, -1, 2)).reshape(B, K_MAX, -1)
        keys = self.key_mlp(torch.cat([b['key_feat'], hh, sf], -1))
        tokens = torch.cat([g[:, None], keys], 1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=f.device), ~b['key_mask']], 1)
        out = self.mixer(tokens, src_key_padding_mask=mask)
        gt, kt = out[:, 0], out[:, 1:]
        film = self.film(gt)
        dens_feat = f * (1 + film[:, :self.field, None, None]) + film[:, self.field:, None, None]
        logits = self.density(dens_feat)[:, 0]
        outside = self.outside(gt)[:, 0]
        return dict(field=f, global_token=gt, key_token=kt, density=logits, outside=outside)

    def key_params(self, b, enc):
        p = self.key_traj(enc['key_token'], b['key_vel'], b['key_age'])
        exist_logit = self.exist(enc['key_token'])[..., 0]
        return p, exist_logit

    def hidden_params(self, enc, pos):
        """Mixture for hidden humans at canonical positions pos [B,N,2]."""
        B, N = pos.shape[:2]
        ahead = self.ahead
        pts = pos[..., None, :] + ahead[None, None]
        sf = sample_field(enc['field'], pts.reshape(B, -1, 2)).reshape(B, N, -1)
        h = self.hidden_embed(torch.cat([sf, pos / 10, enc['global_token'][:, None].expand(-1, N, -1)], -1))
        zeros = torch.zeros_like(pos)
        return self.hidden_traj(h, zeros, torch.zeros(B, N, device=pos.device))

    def log_count(self, enc):
        return F.log_softmax(self.count(enc['global_token']), -1)

    def log_density(self, enc, pos):
        """log spatial density (per m^2) of hidden humans at canonical pos [B,N,2]."""
        B = pos.shape[0]
        flat = torch.cat([enc['density'].reshape(B, -1), enc['outside'][:, None]], -1)
        logp = F.log_softmax(flat, -1)
        iu = torch.floor((pos[..., 0] + HALF) / CELL).long();iv = torch.floor((pos[..., 1] + HALF) / CELL).long()
        inside = (iu >= 0) & (iu < GRID) & (iv >= 0) & (iv < GRID)
        cell = torch.where(inside, iu.clamp(0, GRID - 1) * GRID + iv.clamp(0, GRID - 1), torch.full_like(iu, GRID * GRID))
        lp = torch.gather(logp, 1, cell)
        return torch.where(inside, lp - math.log(CELL * CELL), lp - math.log(OUTSIDE_AREA))

    # ------------------------------------------------------------------
    def loss(self, b, m, v):
        enc = self.encode(b, m, v)
        B = m.shape[0]
        kp, exist_logit = self.key_params(b, enc)
        hum, hmask, tmask = b['hum'], b['hum_mask'], b['tau_mask'].float()
        # Displacements of every human relative to every key: [B,K,P,S,2]
        steps = hum[:, :, 1:] - hum[:, :, :-1]                               # [B,P,12,2]
        d0 = hum[:, None, :, 0] - b['key_last'][:, :, None]                  # [B,K,P,2]
        disp = torch.cat([d0[..., None, :], steps[:, None].expand(-1, K_MAX, -1, -1, -1)], -2)
        stale = (b['key_age'] > 1e-6).float()
        seg_mask = torch.cat([stale[..., None, None].expand(-1, -1, P_MAX, 1),
                              tmask[:, None, None, 1:].expand(-1, K_MAX, P_MAX, -1)], -1)
        kp_e = {k: v.unsqueeze(2) for k, v in kp.items()}                    # broadcast over P
        logq = mixture_logpdf(kp_e, disp, seg_mask)                          # [B,K,P]
        visible = ~(b['key_age'] > 1e-6)
        exact = (d0.norm(dim=-1) < .05)
        logq = torch.where(visible[..., None] & ~exact, torch.full_like(logq, -1e4), logq)
        log_p = torch.where(visible, torch.zeros_like(exist_logit), F.logsigmoid(exist_logit))
        log_np = torch.where(visible, torch.full_like(exist_logit, -1e4), F.logsigmoid(-exist_logit))
        # Residual explanation of each human.
        hp = self.hidden_params(enc, hum[:, :, 0])
        fut_mask = torch.cat([torch.zeros_like(tmask[:, :1]), tmask[:, 1:]], -1)[:, None].expand(-1, P_MAX, -1)
        disp_h = torch.cat([torch.zeros_like(steps[:, :, :1]), steps], -2)
        logg = mixture_logpdf(hp, disp_h, fut_mask)                          # [B,P]
        logf = self.log_density(enc, hum[:, :, 0])                           # [B,P]
        res = (logf + logg) * hmask
        log_count = self.log_count(enc)                                      # [B,11]
        # Fixed keys.
        ktype, owner, kmask = b['key_type'], b['key_owner'], b['key_mask']
        fixed_p = (ktype == FIXED_PRESENT) & kmask
        fixed_a = (ktype == FIXED_ABSENT) & kmask
        own = owner.clamp(0, P_MAX - 1)
        fixed_ll = (torch.gather(logq, 2, own[..., None])[..., 0] + log_p) * fixed_p + log_np * fixed_a
        owned = torch.zeros(B, P_MAX + 1, dtype=torch.bool, device=m.device)
        owned.scatter_(1, torch.where(fixed_p, own, torch.full_like(own, P_MAX)), True)
        unowned = hmask & ~owned[:, :P_MAX]
        # Latent keys (at most L_MAX, most recent first) and the assignment marginal.
        latent = (ktype == LATENT) & kmask
        nlat = latent.sum(1)
        order = torch.argsort((~latent).float() * 1e6 + b['key_age'], dim=1)[:, :L_MAX]
        lat_ok = torch.gather(latent, 1, order)                               # [B,L]
        lq = torch.gather(logq, 1, order[..., None].expand(-1, -1, P_MAX))     # [B,L,P]
        lp_ = torch.gather(log_p, 1, order);lnp = torch.gather(log_np, 1, order)
        gain = lq + lp_[..., None] - res[:, None]                             # matching u->h replaces residual h
        gain = torch.where(unowned[:, None] & lat_ok[..., None], gain, torch.full_like(gain, -1e4))
        inj = self.inj                                                       # [J,L]
        matched = inj >= 0
        lidx = torch.arange(L_MAX, device=inj.device)[None].expand(inj.shape[0], -1)
        g = gain[:, lidx, inj.clamp_min(0)]                                  # [B,J,L]
        term = torch.where(matched[None], g, torch.where(lat_ok[:, None], lnp[:, None], torch.zeros_like(g)))
        nu = unowned.sum(1)
        k = matched.sum(1)[None].expand(B, -1)
        resid_n = nu[:, None] - k
        valid = resid_n >= 0
        cnt = torch.gather(log_count, 1, resid_n.clamp(0, MAX_COUNT))
        cnt = torch.where(resid_n > MAX_COUNT, torch.full_like(cnt, -1e4), cnt)
        score = term.sum(-1) + cnt + torch.lgamma(resid_n.clamp_min(0).float() + 1)
        score = torch.where(valid, score, torch.full_like(score, -1e4))
        marginal = torch.logsumexp(score, 1) + (res * unowned).sum(1)
        usable = (nlat <= L_MAX)
        total = fixed_ll.sum(1) + marginal
        if self.truncate:
            # Exact conditioning on the legal event total >= max simultaneous detections.
            total = total - log_tail(log_p.exp() * kmask, visible & kmask, log_count, b['min_count'])
        nll = -(total * usable).sum() / usable.sum().clamp_min(1)
        stats = dict(nll=nll.detach(), key_nll=-(fixed_ll.sum(1) * usable).sum().detach() / usable.sum().clamp_min(1),
                     usable=usable.float().mean().detach())
        return nll, stats
