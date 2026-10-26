"""In-memory anchor store, canonical camera frame and padded batches.

Canonical frame: origin at the robot base (x,z), +x along the current camera
forward direction. Every legal input and every label of one anchor is rotated
by the same rigid transform; rasters are resampled on the device.

Key label types (privileged, offline only):
  FIXED_PRESENT  legacy-resolved key owning a specific human
  FIXED_ABSENT   legacy-resolved stale duplicate of a person with a newer key
  LATENT         unresolved key; its assignment is marginalized in training
"""
import glob
import json
import math
from pathlib import Path
import numpy as np
import torch

from .legal_state import AGE_STEP, CROP, COARSE, VIS_POOL, HISTORY

DATA = Path('/var/tmp/falcon-stage1-claude/hwm_data')
K_MAX = 10          # keys per anchor (most recent kept)
P_MAX = 8           # humans per anchor
L_MAX = 4           # latent keys in the assignment marginal
FIXED_PRESENT, FIXED_ABSENT, LATENT = 0, 1, 2
TAU = np.arange(13) * .25


class Store:
    """Concatenation of extracted shards (all arrays held in RAM)."""

    def __init__(self, split, limit_shards=0, root_filter=None):
        files = sorted(glob.glob(str(DATA / split / '*.npz')))
        files = [f for f in files if not f.endswith('.tmp.npz')]
        if limit_shards:files = files[:limit_shards]
        parts = [];self.root_ids = [];self.scene_ids = [];self.key_names = [];self.humans = []
        root_base = 0
        for f in files:
            z = dict(np.load(f));meta = json.loads(Path(f[:-4] + '.json').read_text())
            z['anchor_root'] = z['anchor_root'] + root_base
            parts.append(z);root_base += len(meta['roots'])
            self.root_ids += [r['root_id'] for r in meta['roots']];self.scene_ids += [r['scene_id'] for r in meta['roots']]
            self.key_names += meta['keys']
        anchor_fields = ('anchor_root', 'j', 't', 'original', 'robot', 'robot_hist', 'cam', 'corner', 'map', 'vis', 'summary',
                         'residual_certain', 'tau_mask')
        self.a = {k: np.concatenate([p[k] for p in parts]) for k in anchor_fields}
        key_fields = [k for k in parts[0] if k.startswith('k_')];hum_fields = [k for k in parts[0] if k.startswith('h_')]
        self.k = {k: np.concatenate([p[k] for p in parts]) for k in key_fields}
        self.h = {k: np.concatenate([p[k] for p in parts]) for k in hum_fields}
        koff = [];hoff = [];kb = 0;hb = 0
        for p in parts:
            koff.append(p['key_off'][:-1] + kb);hoff.append(p['hum_off'][:-1] + hb)
            kb += p['key_off'][-1];hb += p['hum_off'][-1]
        self.key_off = np.r_[np.concatenate(koff), kb];self.hum_off = np.r_[np.concatenate(hoff), hb]
        # Root-level native truth / robot frames for downstream evaluation.
        self.frame = [];self.truth = []
        for p in parts:
            for r in range(len(p['frame_off']) - 1):
                self.frame.append(p['frame_robot'][p['frame_off'][r]:p['frame_off'][r + 1]])
                tt = p['truth_t'][p['truth_off'][r]:p['truth_off'][r + 1]]
                xz = p['truth_xz'][p['truth_flat_off'][r]:p['truth_flat_off'][r + 1]].reshape(len(tt), -1, 2)
                self.truth.append((tt, xz))
        self.episode_start = np.concatenate([p['episode_start'] for p in parts])
        self.n = len(self.a['t'])

    def scene_of_anchor(self, i):return self.scene_ids[self.a['anchor_root'][i]]

    def root_of_anchor(self, i):return self.root_ids[self.a['anchor_root'][i]]


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)  # world->canonical


def canon(points, origin, rot):
    return np.einsum('...ij,...j->...i', rot, points - origin)


def collate(store, idx, flip=None):
    """Padded numpy batch in the canonical frame. ``flip``: per-anchor reflection v->-v."""
    idx = np.asarray(idx);B = len(idx)
    a = store.a
    origin = a['robot'][idx, :2].astype(np.float64);theta = a['cam'][idx, 2].astype(np.float64)
    rot = rotation(theta)
    if flip is None:flip = np.zeros(B, bool)
    sign = np.where(flip, -1., 1.)
    rot = rot * np.stack([np.ones(B), sign], -1)[:, :, None]   # reflect second canonical axis
    batch = dict(index=idx, origin=origin, theta=theta, flip=flip, rot=rot.astype(np.float32))
    batch['map'] = a['map'][idx];batch['vis'] = a['vis'][idx];batch['corner'] = a['corner'][idx]
    # Robot history in the canonical frame (relative to current pose).
    rh = a['robot_hist'][idx].astype(np.float64)
    rpos = canon(rh[:, :, :2], origin[:, None], rot[:, None])
    ryaw = rh[:, :, 2] - a['robot'][idx, 2:3]
    cam_rel = np.angle(np.exp(1j * (a['robot'][idx, 2] - theta)))
    elapsed = a['t'][idx] - store.episode_start[a['anchor_root'][idx]]
    summ = a['summary'][idx]
    glob_feat = np.concatenate([rpos.reshape(B, -1) / 5., np.sin(ryaw), np.cos(ryaw),
        np.stack([np.sin(cam_rel) * sign, np.cos(cam_rel), elapsed / 5., np.log1p(summ[:, 0]) / 5., summ[:, 1] / np.maximum(summ[:, 0], 1),
                  summ[:, 2] / np.maximum(summ[:, 1], 1), summ[:, 3] / 5.], -1)], -1)
    batch['global'] = glob_feat.astype(np.float32)
    # Keys.
    K = K_MAX
    kf = np.zeros((B, K, 12), np.float32);kmask = np.zeros((B, K), bool);khist = np.zeros((B, K, HISTORY, 4), np.float32)
    klast = np.zeros((B, K, 2), np.float32);kvel = np.zeros((B, K, 2), np.float32);kage = np.zeros((B, K), np.float32)
    ktype = np.full((B, K), -1, np.int64);kown = np.full((B, K), -1, np.int64);kcol = np.full((B, K), -1, np.int64)
    P = P_MAX
    hum = np.zeros((B, P, 13, 2), np.float32);hmask = np.zeros((B, P), bool);hfixed = np.zeros((B, P), bool)
    tmask = a['tau_mask'][idx]
    t_now = a['t'][idx]
    for b, i in enumerate(idx):
        k0, k1 = store.key_off[i], store.key_off[i + 1]
        n = k1 - k0
        order = np.arange(n)
        if n > K:order = np.sort(np.argsort(-store.k['k_last_t'][k0:k1], kind='stable')[:K])
        sel = k0 + order
        m = len(sel)
        kmask[b, :m] = True;kcol[b, :m] = order
        last = canon(store.k['k_last'][sel], origin[b], rot[b])
        vel = store.k['k_velocity'][sel].astype(np.float64) @ rot[b].T
        age = t_now[b] - store.k['k_last_t'][sel]
        hh = store.k['k_hist'][sel].astype(np.float64);hm = store.k['k_hist_mask'][sel]
        hp = canon(hh[:, :, 1:], origin[b], rot[b])
        khist[b, :m, :, 0] = hh[:, :, 0] / 3.;khist[b, :m, :, 1:3] = (hp - last[:, None]) / 3.
        khist[b, :m, :, 3] = hm
        khist[b, :m, :, :3] *= hm[..., None]
        klast[b, :m] = last;kvel[b, :m] = vel * store.k['k_velocity_valid'][sel, None];kage[b, :m] = age
        count = store.k['k_count'][sel];first = t_now[b] - store.k['k_first'][sel]
        dist = np.linalg.norm(last, axis=-1)
        kf[b, :m] = np.stack([last[:, 0] / 10, last[:, 1] / 10, vel[:, 0] / 2, vel[:, 1] / 2,
            store.k['k_velocity_valid'][sel].astype(np.float64), np.log1p(age), (age < 1e-6).astype(np.float64),
            np.log1p(count) / 3, np.log1p(first), dist / 10, np.full(m, n / 5.), np.full(m, float(m < n))], -1)
        lp = store.k['k_legacy_presence'][sel]
        ktype[b, :m] = np.where(np.isnan(lp), LATENT, np.where(lp > .5, FIXED_PRESENT, FIXED_ABSENT))
        kown[b, :m] = np.where(lp > .5, store.k['k_phys'][sel], -1)
        h0, h1 = store.hum_off[i], store.hum_off[i + 1]
        nh = min(P, h1 - h0)
        hum[b, :nh] = canon(store.h['h_hum'][h0:h0 + nh].astype(np.float64), origin[b], rot[b])
        hmask[b, :nh] = True
        for kk in range(m):
            if ktype[b, kk] == FIXED_PRESENT and kown[b, kk] < nh:hfixed[b, kown[b, kk]] = True
    batch.update(key_feat=kf, key_mask=kmask, key_hist=khist, key_last=klast, key_vel=kvel, key_age=kage,
                 key_type=ktype, key_owner=kown, key_column=kcol, hum=hum, hum_mask=hmask, hum_fixed=hfixed,
                 tau_mask=tmask)
    return batch


def to_torch(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray) and v.dtype != object:v = torch.from_numpy(np.ascontiguousarray(v))
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.float64:v = v.float()
            out[k] = v.to(device, non_blocking=True)
        else:out[k] = v
    return out


def rasters(batch, out=CROP):
    """Canonical map (3 one-hot channels, 0.2 m) and visibility (3 channels, 0.4 m) on device."""
    import torch.nn.functional as F
    device = batch['map'].device;B = batch['map'].shape[0]
    mp = batch['map'].long()
    onehot = torch.stack([mp == 0, mp == 1, mp == 2], 1).float()
    age = batch['vis'].float()
    never = (age == 255).float();now = (age == 0).float()
    seconds = torch.where(age == 255, torch.full_like(age, 254), age) * AGE_STEP
    vis = torch.stack([never, torch.log1p(seconds) / 2.6 * (1 - never), now], 1)
    # Canonical grid: pixel centres u,v in [-12.8,12.8).
    half = out * COARSE / 2
    c = (torch.arange(out, device=device).float() + .5) * COARSE - half
    U, V = torch.meshgrid(c, c, indexing='ij')
    uv = torch.stack([U, V], -1).reshape(1, -1, 2).expand(B, -1, -1)
    rot = batch['rot']  # world->canonical; canonical->world is transpose
    world = torch.einsum('bji,bpj->bpi', rot, uv) + batch['origin'][:, None].float()
    rel = world - batch['corner'][:, None].float()
    def sample(img, cell, size):
        # img [B,C,Hx,Wz] indexed [x,z]; grid_sample x-> W (z index), y -> H (x index)
        gx = rel[..., 1] / (cell * size) * 2 - 1;gy = rel[..., 0] / (cell * size) * 2 - 1
        grid = torch.stack([gx, gy], -1).reshape(B, out, out, 2)
        return F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    ones = torch.ones_like(onehot[:, :1])
    m = sample(torch.cat([onehot, ones], 1), COARSE, CROP)
    # Outside the extracted crop is UNKNOWN / never seen.
    m = torch.cat([m[:, :1] + (1 - m[:, 3:4]), m[:, 1:3]], 1)
    v = sample(torch.cat([vis, ones[:, :, ::VIS_POOL, ::VIS_POOL]], 1), COARSE * VIS_POOL, CROP // VIS_POOL)
    v = torch.cat([v[:, :1] + (1 - v[:, 3:4]), v[:, 1:3]], 1)
    return m, v
