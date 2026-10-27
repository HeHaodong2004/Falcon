"""Legal tri-state voxel map and certificate simulation (GPU).

Inputs are the measured range history, its recorded camera poses, the
current graph and supplied base trajectories. Future camera poses come from
the current rigid camera/base extrinsic applied to the supplied path.
"""
import ctypes as ct
import math

import numpy as np
import torch

from .kernels import Vol, module

DY = .1          # vertical voxel size (m)
Y_BELOW = .4     # volume starts this far below the current base height
NY = 24          # vertical voxels: covers base-.4 .. base+2.0
MARGIN = .08     # carving margin (m)


def upload(arr, device):
    """Host->device copy that never blocks on queued GPU work: the array is
    staged in (cached) pinned memory and copied asynchronously in stream order.
    Large arrays (range packets) are staged in 8 slices along the first axis, so
    the host copy of one slice overlaps the transfer of the previous one."""
    t = torch.from_numpy(np.ascontiguousarray(arr))
    device = torch.device(device)
    if device.type != 'cuda':
        return t
    if t.dim() == 0 or t.shape[0] < 8 or t.numel()*t.element_size() < (8 << 20):
        return t.pin_memory().to(device, non_blocking=True)
    out = torch.empty(t.shape, dtype=t.dtype, device=device)
    step = -(-t.shape[0]//8)
    for a in range(0, t.shape[0], step):
        out[a:a+step].copy_(t[a:a+step].pin_memory(), non_blocking=True)
    return out


def local_dirs(size):
    xx, yy = np.meshgrid((np.arange(size)+.5)*2/size-1, 1-(np.arange(size)+.5)*2/size, indexing='xy')
    v = np.stack([xx, yy, -np.ones_like(xx)], -1).reshape(-1, 3)
    return v/np.linalg.norm(v, axis=1)[:, None]


def base_rotations(yaw):
    yaw = np.asarray(yaw, np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.zeros((len(yaw), 3, 3), np.float64)
    out[:, 0, 0] = c; out[:, 0, 2] = s; out[:, 1, 1] = 1.; out[:, 2, 0] = -s; out[:, 2, 2] = c
    return out


def camera_extrinsic(start_pose, origin, rotation):
    """Rigid camera-in-base transform from the CURRENT legal camera/base pair."""
    frame = base_rotations(np.asarray([start_pose[4]]))[0]
    return frame.T@(np.asarray(origin, np.float64)-np.asarray(start_pose[1:4], np.float64)), frame.T@np.asarray(rotation, np.float64)


def proxy_cameras(path, extrinsic):
    translation, rotation = extrinsic
    p = np.asarray(path, np.float64)
    base = base_rotations(p[:, 4])
    return p[:, 1:4]+base@translation, base@rotation


def pack_cams(origins, rotations):
    return np.concatenate([np.asarray(origins, np.float64).reshape(-1, 3),
                           np.asarray(rotations, np.float64).reshape(-1, 9)], 1).astype(np.float32)


class VoxelMap:
    """Tri-state voxel volume over an axis-aligned cell-key box."""

    def __init__(self, ranges, origins, rotations, lo_key, shape, res, base_y, device, keep=None, splat=True, humans=None, sub=1, near=0.):
        # sub: voxels per cell edge (1: voxel = cell; 2: half-size voxels, also halving DY)
        self.device = torch.device(device)
        self.cell_res = float(res)
        self.sub = int(sub)
        self.res = self.cell_res/self.sub
        self.lo_key = np.asarray(lo_key, np.int64)
        self.shape = (int(shape[0])*self.sub, int(shape[1])*self.sub)
        self.ny = NY*self.sub
        self.y0 = float(base_y)-Y_BELOW
        self.cell_shape = (int(shape[0]), int(shape[1]))
        self.vol = Vol(float(self.lo_key[0]*self.cell_res), float(self.lo_key[1]*self.cell_res), self.y0, self.res, DY/self.sub,
                       self.shape[0], self.shape[1], self.ny, self.cell_res, self.sub, float(near))
        m = module(self.device)
        ranges = upload(np.asarray(ranges, np.float32), self.device)
        cams = upload(pack_cams(origins, rotations), self.device)
        frames, size = ranges.shape[0], ranges.shape[-1]
        n = self.shape[0]*self.shape[1]*self.ny
        occ = torch.zeros(n, dtype=torch.uint8, device=self.device)
        self.state = torch.empty(n, dtype=torch.uint8, device=self.device)
        if splat:
            hp, hs = None, None
            if humans is not None:
                hp = upload(np.asarray(humans[0], np.float32).reshape(-1), self.device)
                hs = upload(np.asarray(humans[1], np.int32), self.device)
                if not len(hp):
                    hp = torch.zeros(3, dtype=torch.float32, device=self.device)
            m.launch('splat_hits', frames*size*size, [ranges, cams, ct.c_int(frames), ct.c_int(size), self.vol, occ,
                                                      ct.c_float(.04), ct.c_float(.02), ct.c_float(1.), hp, hs, ct.c_float(.4)], 128)
        else:
            m.launch('scatter_hits', frames*size*size, [ranges, cams, ct.c_int(frames), ct.c_int(size), self.vol, occ, keep], 256)
        # Projective carving over only the frames that can reach each 8x8-column block (exact culling).
        nbx, nbz, words = -(-self.shape[0]//8), -(-self.shape[1]//8), max(1, -(-frames//32))
        masks = torch.zeros(nbx*nbz*words, dtype=torch.int32, device=self.device)
        m.launch('cull_frames', nbx*nbz*frames, [cams, ct.c_int(frames), self.vol, ct.c_int(nbx), ct.c_int(nbz), ct.c_int(words), masks], 128)
        m.launch('carve_culled', n, [ranges, cams, ct.c_int(frames), ct.c_int(size), self.vol, occ, self.state,
                                     ct.c_float(MARGIN if self.sub == 1 else .045), ct.c_int(nbz), ct.c_int(words), masks], 128)
        self.module = m

    def hit_sums(self, ranges, origins, rotations, humans=None):
        ranges = torch.as_tensor(np.ascontiguousarray(ranges, np.float32), device=self.device)
        cams = torch.as_tensor(pack_cams(origins, rotations), device=self.device)
        frames, size = ranges.shape[0], ranges.shape[-1]
        sums = torch.zeros((self.shape[0]*self.shape[1]*self.ny, 4), dtype=torch.int64, device=self.device)
        hp, hs = None, None
        if humans is not None:
            hp = torch.as_tensor(np.asarray(humans[0], np.float32).reshape(-1), device=self.device)
            hs = torch.as_tensor(np.asarray(humans[1], np.int32), device=self.device)
            if not len(hp):
                hp = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.module.launch('hit_sums', frames*size*size, [ranges, cams, ct.c_int(frames), ct.c_int(size), self.vol, sums, hp, hs, ct.c_float(.4)], 256)
        self.sums = sums
        return sums

    def wall_rays_refined(self, cams):
        cams_t = torch.as_tensor(cams, device=self.device)
        poses = len(cams)
        out = torch.empty((poses*105, 6), dtype=torch.float32, device=self.device)
        self.module.launch('wall_rays_refined', poses*105, [cams_t, ct.c_int(poses), self.vol, self.state, self.sums, out], 128)
        return out.view(poses, 105, 6)

    def stamp_humans(self, positions, radius=.3):
        """Current humans as dynamic occluders (state 3)."""
        pos = np.asarray(positions, np.float32).reshape(-1, 3)
        if not len(pos):
            return
        bx = int(np.ceil(2*radius/self.res))+2; by = int(np.ceil(2.1/(DY/self.sub)))+2
        t = upload(pos.reshape(-1), self.device)
        self.module.launch('stamp_humans', len(pos)*bx*bx*by, [self.vol, self.state, t, ct.c_int(len(pos)), ct.c_float(radius),
                                                               ct.c_int(bx), ct.c_int(bx), ct.c_int(by)], 256)

    def grid(self):
        return self.state.view(self.shape[0], self.shape[1], self.ny)

    def wall_rays(self, cams):
        cams_t = torch.as_tensor(cams, device=self.device)
        poses = len(cams)
        out = torch.empty((poses*105, 6), dtype=torch.float32, device=self.device)
        self.module.launch('wall_rays', poses*105, [cams_t, ct.c_int(poses), self.vol, self.state, out], 128)
        return out.view(poses, 105, 6)

    def render(self, cams, size=96):
        cams_t = torch.as_tensor(cams, device=self.device)
        poses = len(cams)
        out = torch.empty((poses, size, size, 2), dtype=torch.float32, device=self.device)
        self.module.launch('render', poses*size*size, [cams_t, ct.c_int(poses), ct.c_int(size), self.vol, self.state, out], 128)
        return out

    def probes(self, keys, floor_h, cams, images, pose_index, tol_occ=.15, tol_unk=.1):
        keys_t = torch.as_tensor(np.ascontiguousarray(keys, np.int32), device=self.device)
        fh = torch.as_tensor(np.ascontiguousarray(floor_h, np.float32), device=self.device)
        pi = torch.as_tensor(np.ascontiguousarray(pose_index, np.int32), device=self.device)
        cams_t = torch.as_tensor(cams, device=self.device)
        out = torch.empty((len(keys), 8), dtype=torch.float32, device=self.device)
        self.module.launch('probes', len(keys), [keys_t, fh, ct.c_int(len(keys)), cams_t, pi, ct.c_int(len(pose_index)),
                                                 ct.c_int(images.shape[1]), images, ct.c_float(self.cell_res), ct.c_float(tol_occ),
                                                 ct.c_float(tol_unk), out], 128)
        return out


def wall_blocked(rays, cams, path, res, known_only=True, offset=.03, gap=.3):
    """Cells within .25 m (xz) of simulated wall-band hits (observer rule)."""
    r = rays.detach().cpu().numpy()
    origin = cams[:, None, :3].astype(np.float64)
    t_occ, u_len, d = r[..., 0], r[..., 5], r[..., 2:5].astype(np.float64)
    hit = t_occ < 10.
    if known_only:
        hit &= u_len < gap
    p = origin+d*(t_occ[..., None]+offset)
    py = path[:, 2][:, None]
    hit &= (p[..., 1] > py+.2) & (p[..., 1] < py+1.4)
    pts = p[hit][:, [0, 2]]
    if not len(pts):
        return np.empty((0, 2), np.int64)
    span = int(math.ceil(.25/res))+1
    off = np.array([(a, b) for a in range(-span, span+1) for b in range(-span, span+1)], np.int64)
    key = np.floor(pts/res).astype(np.int64)
    near = key[:, None, :]+off[None]
    cen = (near+.5)*res
    ok = np.linalg.norm(cen-pts[:, None, :], axis=2) <= .25
    return np.unique(near[ok], axis=0)
