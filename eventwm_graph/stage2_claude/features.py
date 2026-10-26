"""Legal BEV canvases for the trajectory-conditioned observation model.

GroupContext holds everything derived from the CURRENT legal state (graph,
measured range history, current camera calibration). query_canvases() adds
the supplied hypothetical trajectories. Nothing here reads future sensing,
scene meshes, navmeshes or simulator state.

Canvas frame: cell keys lo .. lo+shape (x index first, z index second), the
same frame as the voxel volume. Channel order is fixed by MAP_CHANNELS and
QUERY_CHANNELS and recorded in checkpoints.
"""
import ctypes as ct
import math

import numpy as np
import torch

from .geometry import VoxelMap, camera_extrinsic, proxy_cameras, pack_cams, upload, NY, DY, Y_BELOW
from .kernels import module

RADIUS = 11.
# Height bands relative to the current base height (voxel bin b covers
# [base-.4+.1b, base-.3+.1b)).
BANDS = (('below', 0, 1), ('floor', 1, 4), ('low', 4, 6), ('mid', 6, 11), ('high', 11, 18), ('top', 18, NY))
MAP_CHANNELS = (['graph_free', 'graph_blocked'] + ['%s_%s' % (b[0], s) for b in BANDS for s in ('occ', 'free', 'unk')] + ['human_now'] +
                ['past_probe_known', 'past_probe_possible', 'past_probe_best_visible', 'past_frustum', 'past_wall_hit'])
PAST_FRAMES = 16
QUERY_CHANNELS = ['probe_known', 'probe_possible', 'probe_best_visible', 'probe_best_uncertain', 'probe_frustum_poses',
                  'probe_best_blocked', 'probe_min_distance', 'center_frustum_poses',
                  'wall_known', 'wall_after_unknown', 'wall_unknown_traversal', 'wall_free_traversal',
                  'pose_cells', 'start_cell', 'end_cell', 'probe_unknown_depth', 'probe_unknown_depth_max',
                  'wall_unknown_entry', 'probe_possible_poses', 'probe_known_poses',
                  'query_poses', 'query_duration', 'query_length', 'wall_hit_distance', 'human_future', 'frame_valid']
VALID = len(QUERY_CHANNELS)-1


def channel_index(config):
    """Indices of a checkpoint's recorded channels within the current builder."""
    names = MAP_CHANNELS+QUERY_CHANNELS
    wanted = list(config['map_channels'])+list(config['query_channels'])
    missing = [n for n in wanted if n not in names]
    if missing:
        raise ValueError('checkpoint channels unavailable: %s' % missing)
    idx = [names.index(n) for n in wanted]
    return None if idx == list(range(len(names))) else idx
RENDER = 96
WALL_GAP = .3
WALL_OFFSET = .03


def crop_box(path, res):
    """Original candidate-domain crop (stage2_quality.features.Features)."""
    xy = np.asarray(path)[:, [1, 3]].astype(np.float32)
    lo = np.floor((xy.min(0)-RADIUS)/res).astype(np.int64)-4
    hi = np.floor((xy.max(0)+RADIUS)/res).astype(np.int64)+5
    return lo, hi


class GroupContext:
    """Legal state shared by every query from one starting graph."""

    def __init__(self, graph_keys, graph_states, ranges, origins, rotations, start_pose, res, device, lo, hi, humans=None, splat=True,
                 sub=1, render=96, current_humans=None, near=0., human_motion=None):
        self.device = torch.device(device)
        # Current legal tracks [position, estimated velocity] for the future-sweep channel.
        self.human_motion = None if human_motion is None else np.asarray(human_motion, np.float32).reshape(-1, 6)
        self.res = float(res)
        self.lo = np.asarray(lo, np.int64)
        self.shape = tuple(int(v) for v in (np.asarray(hi, np.int64)-self.lo))
        self.start_pose = np.asarray(start_pose, np.float64)
        self.base_y = float(self.start_pose[2])
        self.sub = int(sub)
        self.render = int(render)
        self.vm = VoxelMap(ranges, origins, rotations, self.lo, self.shape, self.res, self.base_y, self.device, humans=humans, splat=splat,
                           sub=self.sub, near=near)
        if current_humans is not None:
            self.vm.stamp_humans(current_humans)
        self.extrinsic = camera_extrinsic(self.start_pose, origins[-1], rotations[-1])
        nx, nz = self.shape
        keys = np.asarray(graph_keys, np.int64).reshape(-1, 2)-self.lo
        states = np.asarray(graph_states, np.uint8)
        ok = np.all((keys >= 0) & (keys < np.asarray(self.shape)), axis=1)
        flat = np.zeros(nx*nz, np.uint8)
        flat[keys[ok, 0]*nz+keys[ok, 1]] = states[ok]
        gstate = upload(flat, self.device)
        bands = upload(np.asarray([(a, b) for _, a, b in BANDS], np.int32).reshape(-1), self.device)
        channels = torch.zeros((len(MAP_CHANNELS), nx, nz), dtype=torch.float32, device=self.device)
        module(self.device).launch('map_channels', nx*nz, [self.vm.vol, self.vm.state, gstate, bands, ct.c_int(len(BANDS)), channels,
                                                            ct.c_int(int(current_humans is not None))], 256)
        self.map_channels = channels
        self.graph_known = (gstate > 0).to(torch.uint8).view(nx, nz).contiguous()
        # Legal PAST camera poses (actual recorded cameras, most recent frames):
        # what was already visible but never certified is unlikely to be.
        n = len(origins)
        take = np.arange(n) if n <= PAST_FRAMES else np.unique(np.concatenate([np.arange(n-PAST_FRAMES//2, n),
                                                                                np.linspace(0, n-PAST_FRAMES//2-1, PAST_FRAMES//2).astype(int)]))
        self.past_cams = pack_cams(np.asarray(origins)[take], np.asarray(rotations)[take])
        past_channels(self, channels[len(MAP_CHANNELS)-5:])


def past_channels(ctx, out):
    """Probe statistics and wall-ray hits from the legal PAST camera poses over
    the whole context frame (written into out: 5 x nx x nz)."""
    nx, nz = ctx.shape
    dev = ctx.device
    m = module(dev)
    P = len(ctx.past_cams)
    Hp, Wp = nx, nz
    fr = np.asarray([[0, 0, nx, nz]], np.int32)
    ints = np.concatenate([fr.reshape(-1), [0], [P], np.arange(P), np.zeros(P)]).astype(np.int32)
    ti = upload(ints, dev)
    t_frames, t_qs, t_qc, t_pl, t_wq = ti[0:4], ti[4:5], ti[5:6], ti[6:6+P], ti[6+P:6+2*P]
    tf = upload(np.concatenate([ctx.past_cams.reshape(-1), np.full(1, ctx.base_y, np.float32), np.full(P, ctx.base_y, np.float32)]).astype(np.float32), dev)
    t_cams, t_floor, t_wy = tf[:12*P], tf[12*P:12*P+1], tf[12*P+1:]
    images = ctx.vm.render(t_cams.view(-1, 12), ctx.render)
    C = 20
    buf = torch.zeros((1, C, Hp, Wp), dtype=torch.float32, device=dev)
    m.launch('probe_frame', Hp*Wp, [t_cams, t_qs, t_qc, t_pl, ct.c_int(1), ct.c_int(ctx.render), images, ctx.vm.vol, ctx.graph_known,
                                     t_frames, t_floor, ct.c_float(.15 if ctx.sub == 1 else .09), ct.c_float(.1 if ctx.sub == 1 else .06),
                                     ct.c_int(Hp), ct.c_int(Wp), ct.c_int(C), ct.c_int(0), buf], 128)
    wall = torch.zeros((1, 4, Hp, Wp), dtype=torch.float32, device=dev)
    trav = torch.zeros((4, Hp, Wp), dtype=torch.int32, device=dev)
    trav[3] = 2**30
    m.launch('wall_frame', P*105, [t_cams, t_wq, t_wy, ct.c_int(P), ctx.vm.vol, ctx.vm.state, t_frames, ct.c_int(Hp), ct.c_int(Wp),
                                   ct.c_int(4), ct.c_int(0), wall, ct.c_float(WALL_GAP), ct.c_float(WALL_OFFSET/ctx.sub), trav, ct.c_int(1)], 128)
    out[0] = buf[0, 0]; out[1] = buf[0, 1]; out[2] = buf[0, 2]; out[3] = buf[0, 4]
    out[4] = torch.clamp(wall[0, 0]+wall[0, 1], max=1.)


def unique_positions(path):
    """Unique float32 supplied positions (original domain semantics)."""
    return np.unique(np.asarray(path)[:, [1, 3]].astype(np.float32), axis=0)


def build_inputs(ctx, paths, frames, pad=16):
    """Network inputs on per-query frames.

    frames: list of (lo_key (2,), shape (2,)) per query in cell keys. Returns
    x (Q, C, Hp, Wp) float16 and the EXACT original candidate-domain mask
    (Q, Hp, Wp) uint8 (initially UNKNOWN, within 11 m of a supplied position,
    inside the frame). Frames must lie inside the context frame.
    """
    q = len(paths)
    nm, nq = len(MAP_CHANNELS), len(QUERY_CHANNELS)
    C = nm+nq
    fr = np.zeros((q, 4), np.int32)
    for i, (lo, shape) in enumerate(frames):
        fr[i, :2] = np.asarray(lo, np.int64)-ctx.lo
        fr[i, 2:] = np.asarray(shape, np.int64)
    Hp = int(fr[:, 2].max()); Wp = int(fr[:, 3].max())
    Hp += (-Hp) % pad; Wp += (-Wp) % pad
    dev = ctx.device
    x = torch.zeros((q, C, Hp, Wp), dtype=torch.float32, device=dev)
    for i in range(q):
        a0, b0, h, w = (int(v) for v in fr[i])
        ga0, gb0 = max(a0, 0), max(b0, 0)
        ga1, gb1 = min(a0+h, ctx.shape[0]), min(b0+w, ctx.shape[1])
        if ga1 > ga0 and gb1 > gb0:
            x[i, :nm, ga0-a0:ga1-a0, gb0-b0:gb1-b0] = ctx.map_channels[:, ga0:ga1, gb0:gb1]
        x[i, nm+VALID, :h, :w] = 1.
        pp = np.asarray(paths[i], np.float64)
        x[i, nm+20, :h, :w] = np.log1p(len(pp)-1)/5.
        x[i, nm+21, :h, :w] = float(pp[-1, 0]-pp[0, 0])/5.
        x[i, nm+22, :h, :w] = float(np.linalg.norm(np.diff(pp[:, [1, 3]], axis=0), axis=1).sum())/5. if len(pp) > 1 else 0.
    # One host->device upload of all small per-call arrays.
    unique = {}
    pose_rows, pose_list, query_start, query_count = [], [], [], []
    positions, pos_start, pos_count = [], [], []
    for p in paths:
        p = np.asarray(p, np.float64)
        query_start.append(len(pose_list)); query_count.append(len(p)-1)
        for row in p[1:]:
            key = row[1:].tobytes()
            if key not in unique:
                unique[key] = len(pose_rows); pose_rows.append(row)
            pose_list.append(unique[key])
        u = unique_positions(p)
        pos_start.append(len(positions)); pos_count.append(len(u)); positions.extend(u)
    m = module(dev)
    ints = np.concatenate([fr.reshape(-1), np.asarray(query_start, np.int32), np.asarray(query_count, np.int32),
                           np.asarray(pos_start, np.int32), np.asarray(pos_count, np.int32), np.asarray(pose_list, np.int32),
                           np.repeat(np.arange(q, dtype=np.int32), query_count)]).astype(np.int32)
    ti = upload(ints, dev)
    o = 0
    t_frames = ti[o:o+4*q]; o += 4*q
    t_qs = ti[o:o+q]; o += q
    t_qc = ti[o:o+q]; o += q
    t_ps = ti[o:o+q]; o += q
    t_pc = ti[o:o+q]; o += q
    t_pl = ti[o:o+len(pose_list)]; o += len(pose_list)
    t_wq = ti[o:o+len(pose_list)]; o += len(pose_list)
    floats = [np.asarray(positions, np.float32).reshape(-1), np.asarray([np.asarray(p)[0, 2] for p in paths], np.float32)]
    if pose_rows:
        rows = np.asarray(pose_rows, np.float64)
        oo, R = proxy_cameras(rows, ctx.extrinsic)
        floats += [pack_cams(oo, R).reshape(-1), rows[np.asarray(pose_list)][:, 2].astype(np.float32)]
    motion = getattr(ctx, 'human_motion', None)
    nh = 0 if motion is None else len(motion)
    if nh:
        floats += [np.asarray([float(np.asarray(p)[-1, 0]-np.asarray(p)[0, 0]) for p in paths], np.float32), motion.reshape(-1)]
    tf = upload(np.concatenate(floats).astype(np.float32), dev)
    o = 0
    t_pos = tf[o:o+2*len(positions)]; o += 2*len(positions)
    t_floor = tf[o:o+q]; o += q
    mask = torch.empty((q, Hp, Wp), dtype=torch.uint8, device=dev)
    m.launch('domain_mask', q*Hp*Wp, [t_frames, ct.c_int(q), ct.c_int(Hp), ct.c_int(Wp), ctx.vm.vol, ctx.graph_known,
                                       t_pos, t_ps, t_pc, ct.c_float(np.float32(ctx.res)), mask], 256)
    if pose_rows:
        t_cams = tf[o:o+12*len(pose_rows)]; o += 12*len(pose_rows)
        t_wy = tf[o:o+len(pose_list)]; o += len(pose_list)
        images = ctx.vm.render(t_cams.view(-1, 12), ctx.render)
        m.launch('probe_frame', q*Hp*Wp, [t_cams, t_qs, t_qc, t_pl, ct.c_int(q), ct.c_int(ctx.render), images, ctx.vm.vol, ctx.graph_known,
                                           t_frames, t_floor, ct.c_float(.15 if ctx.sub == 1 else .09), ct.c_float(.1 if ctx.sub == 1 else .06),
                                           ct.c_int(Hp), ct.c_int(Wp), ct.c_int(C),
                                           ct.c_int(nm), x], 128)
        wcams = t_cams.view(-1, 12)[t_pl.long()].contiguous()
        traversal = torch.zeros((4*q, Hp, Wp), dtype=torch.int32, device=dev)
        traversal[3*q:] = 2**30
        m.launch('wall_frame', len(pose_list)*105, [wcams, t_wq, t_wy, ct.c_int(len(pose_list)), ctx.vm.vol, ctx.vm.state, t_frames,
                                                     ct.c_int(Hp), ct.c_int(Wp), ct.c_int(C), ct.c_int(nm+8), x,
                                                     ct.c_float(WALL_GAP), ct.c_float(WALL_OFFSET/ctx.sub), traversal, ct.c_int(q)], 128)
        x[:, nm+10:nm+12] = torch.clamp(traversal[:2*q].view(q, 2, Hp, Wp).float()*(1./(4096.*105*.5)), max=4.)
        x[:, nm+17] = torch.log1p(traversal[2*q:3*q].float())*.5
        x[:, nm+23] = torch.clamp(traversal[3*q:].float()*1e-4, max=.45)/.45
    if nh:
        t_dur = tf[o:o+q]; o += q
        t_motion = tf[o:o+6*nh]; o += 6*nh
        m.launch('human_sweep', q*Hp*Wp, [t_frames, ct.c_int(q), ct.c_int(Hp), ct.c_int(Wp), ctx.vm.vol, t_motion, ct.c_int(nh),
                                           t_dur, ct.c_float(.35), ct.c_int(C), ct.c_int(nm+QUERY_CHANNELS.index('human_future')), x], 256)
    if not pose_rows:
        x[:, nm+3] = 1.
        x[:, nm+15:nm+17] = 1.
        x[:, nm+23] = 1.
    # Probe "uncertain" defaults to 1 where no frustum pose exists (cells with
    # no probe evaluation keep the kernel's untouched zeros elsewhere).
    # Path raster (all queries, one upload): channel, query, a, b.
    raster = []
    for i, p in enumerate(paths):
        p = np.asarray(p, np.float64)
        ij = np.floor(p[:, [1, 3]]/ctx.res).astype(np.int64)-ctx.lo-fr[i, :2]
        ok = np.all((ij >= 0) & (ij < fr[i, 2:]), axis=1)
        for ch, sel in ((nm+12, ok), (nm+13, np.arange(len(ij)) == 0), (nm+14, np.arange(len(ij)) == len(ij)-1)):
            sel = sel & ok
            if sel.any():
                raster.append(np.column_stack([np.full(sel.sum(), ch), np.full(sel.sum(), i), ij[sel]]))
    if raster:
        r = upload(np.concatenate(raster).astype(np.int64), dev)
        x[r[:, 1], r[:, 0], r[:, 2], r[:, 3]] = 1.
    return x.half(), mask, fr


def current_humans(track_history, max_age=.5, motion=False):
    """Legally tracked humans at the CURRENT frame (positions extrapolated by
    their estimated velocity over the observation age, at most max_age s).
    motion=True returns rows [position, estimated velocity (0 if invalid)]."""
    rows = track_history[-1] if len(track_history) else []
    out = []
    for t in rows or ():
        if not t.get('valid', True):
            continue
        age = float(t.get('observation_age', 0.) or 0.)
        if age > max_age:
            continue
        p = np.asarray(t['position'], np.float64)
        v = np.zeros(3)
        if t.get('velocity_valid') and t.get('estimated_velocity') is not None:
            v = np.asarray(t['estimated_velocity'], np.float64)
            p = p+v*age
        out.append(np.concatenate([p, v]) if motion else p)
    return np.asarray(out, np.float32).reshape(-1, 6 if motion else 3)


def human_arrays(track_history):
    """Per-frame positions of legally tracked humans VISIBLE in that frame."""
    pos, start = [], [0]
    for rows in track_history:
        for t in rows or ():
            if t.get('valid', True) and t.get('visible_now', False):
                pos.append(t['position'])
        start.append(len(pos))
    return np.asarray(pos, np.float32).reshape(-1, 3), np.asarray(start, np.int32)
