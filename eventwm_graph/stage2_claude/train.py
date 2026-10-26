"""Train the BEV certificate-revelation U-Net on TRAIN roots.

Labels: window targets (initially UNKNOWN cells that become FREE/BLOCKED).
Inputs: GroupContext/query canvases (legal only). Validation uses scene-
disjoint TRAIN roots; development data is never read here.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eventwm_graph.stage2_repair.diagnose import OUT
from eventwm_graph.stage2_segment.evaluate import rates
from .data import GroupStream, split_roots, collate, scene_factors, load_root
from .features import GroupContext, crop_box, build_inputs, MAP_CHANNELS, QUERY_CHANNELS, RADIUS
from .geometry import upload
from .model import UNet, build_model

DEST = OUT.parent/'claude_stage2'
CIN = len(MAP_CHANNELS)+len(QUERY_CHANNELS)


def group_frame(windows, res):
    lo = np.min([crop_box(w['path'], res)[0] for w in windows], 0)
    hi = np.max([crop_box(w['path'], res)[1] for w in windows], 0)
    return lo, hi


GEOMETRY = dict(splat=True, sub=1, render=96, occluders=False)   # set from the run/checkpoint configuration


def build_context(g, device, splat=None, config=None):
    lo, hi = group_frame(g['windows'], g['res'])
    geo = dict(GEOMETRY)
    if config is not None:
        geo.update(splat=bool(config.get('splat', True)), sub=int(config.get('sub', 1)), render=int(config.get('render', 96)),
                   occluders=bool(config.get('human_occluders', False)))
    if splat is not None:
        geo['splat'] = splat
    ctx = GroupContext(g['gkeys'], g['gstates'], g['ranges'], g['origins'], g['rotations'], g['windows'][0]['path'][0],
                       g['res'], device, lo, hi, humans=g.get('humans'),
                       current_humans=g.get('current_humans') if geo.pop('occluders', False) else None, **geo)
    return ctx


def domain_mask(ctx, path, lo, shape, device):
    """Cells (in a lo/shape frame) whose float32 centers lie within 11 m of a pose."""
    res = ctx.res
    xy = torch.as_tensor(np.unique(np.asarray(path)[:, [1, 3]].astype(np.float32), axis=0), device=device, dtype=torch.float64)
    ix = (torch.arange(shape[0], device=device, dtype=torch.float64)+float(lo[0])+.5)*res
    iz = (torch.arange(shape[1], device=device, dtype=torch.float64)+float(lo[1])+.5)*res
    dx = (ix[:, None]-xy[None, :, 0])**2
    dz = (iz[:, None]-xy[None, :, 1])**2
    best = (dx[:, None, :]+dz[None, :, :]).amin(2)
    return best <= RADIUS**2


def labels_canvas(ctx, window, device):
    """Group-frame labels: -1 ignore, 0 stays UNKNOWN, 1 FREE, 2 BLOCKED."""
    nx, nz = ctx.shape
    y = torch.full((nx, nz), -1, dtype=torch.int8, device=device)
    dom = domain_mask(ctx, window['path'], ctx.lo, ctx.shape, device) & (ctx.graph_known == 0)
    y[dom] = 0
    tk = window['tkeys']-ctx.lo
    ok = np.all((tk >= 0) & (tk < np.asarray(ctx.shape)), axis=1)
    if ok.any():
        t = torch.as_tensor(tk[ok], device=device)
        y[t[:, 0], t[:, 1]] = torch.as_tensor(window['tstates'][ok].astype(np.int8), device=device)
    return y, int((~ok).sum())


def crop(t, lo, size, fill):
    """Crop the last two axes of t (frame starting at 0) at lo with size, padding with fill."""
    out_shape = t.shape[:-2]+(size, size)
    out = torch.full(out_shape, fill, dtype=t.dtype, device=t.device)
    a0, b0 = max(lo[0], 0), max(lo[1], 0)
    a1, b1 = min(lo[0]+size, t.shape[-2]), min(lo[1]+size, t.shape[-1])
    if a1 > a0 and b1 > b0:
        out[..., a0-lo[0]:a1-lo[0], b0-lo[1]:b1-lo[1]] = t[..., a0:a1, b0:b1]
    return out


def frame_labels(mask, fr, windows, lo_group):
    """Labels on network frames: -1 outside the exact domain, else 0/1/2."""
    y = torch.where(mask > 0, 0, -1).to(torch.int8)
    rows = []
    for i, w in enumerate(windows):
        tk = w['tkeys']-lo_group-fr[i, :2]
        ok = np.all((tk >= 0) & (tk < fr[i, 2:]), axis=1)
        if ok.any():
            rows.append(np.column_stack([np.full(ok.sum(), i), tk[ok], w['tstates'][ok].astype(np.int64)]))
    if rows:
        t = upload(np.concatenate(rows).astype(np.int64), mask.device)
        inside = mask[t[:, 0], t[:, 1], t[:, 2]] > 0
        t = t[inside]
        y[t[:, 0], t[:, 1], t[:, 2]] = t[:, 3].to(torch.int8)
    return y


def window_samples(g, ctx, size, rng, augment, aux_frames=None):
    """Training samples on the EXACT inference frames (original crop boxes)."""
    windows = [w for w in g['windows'] if len(w['path']) >= 2]
    if not windows:
        return []
    frames = []
    for w in windows:
        lo, hi = crop_box(w['path'], ctx.res)
        shape = np.minimum(hi-lo, size)
        if np.any(hi-lo > size):
            # Rare long paths: keep a size x size frame centered on the path.
            xy = w['path'][:, [1, 3]]
            center = np.floor(((xy.min(0)+xy.max(0))/2)/ctx.res).astype(np.int64)
            lo = np.where(hi-lo > size, center-size//2, lo)
        frames.append((lo, shape))
    x, mask, fr = build_inputs(ctx, [w['path'] for w in windows], frames, pad=16)
    y = frame_labels(mask, fr, windows, ctx.lo)
    aux = aux_targets(ctx, aux_frames, fr, x.shape[2], x.shape[3], x.device) if aux_frames is not None else None
    out = []
    for i in range(len(windows)):
        h, w_ = int(fr[i, 2]), int(fr[i, 3])
        xc, yc = x[i, :, :h, :w_], y[i, :h, :w_]
        ac = aux[i, :, :h, :w_] if aux is not None else None
        if augment:
            k = int(rng.integers(8))
            xc = torch.rot90(xc, k % 4, (1, 2)); yc = torch.rot90(yc, k % 4, (0, 1))
            if ac is not None:
                ac = torch.rot90(ac, k % 4, (1, 2))
            if k >= 4:
                xc = torch.flip(xc, (1,)); yc = torch.flip(yc, (0,))
                if ac is not None:
                    ac = torch.flip(ac, (1,))
        # Unpadded frame tensors; batches are padded to their own maximum
        # (the masked network is padding-invariant).
        weight = g['scene_factor']/g['windows_in_root']
        out.append((xc.contiguous(), yc.contiguous(), weight, ac.contiguous() if ac is not None else None))
    return out


def collate_frames(items, device):
    hmax = max(it[0].shape[1] for it in items); wmax = max(it[0].shape[2] for it in items)
    hmax += (-hmax) % 16; wmax += (-wmax) % 16
    C = items[0][0].shape[0]
    x = torch.zeros((len(items), C, hmax, wmax), dtype=items[0][0].dtype, device=device)
    y = torch.full((len(items), hmax, wmax), -1, dtype=torch.int8, device=device)
    A = torch.zeros((len(items), 3, hmax, wmax), dtype=torch.uint8, device=device) if items[0][3] is not None else None
    for i, it in enumerate(items):
        h, w = it[0].shape[1], it[0].shape[2]
        x[i, :, :h, :w] = it[0]; y[i, :h, :w] = it[1]
        if A is not None:
            A[i, :, :h, :w] = it[3]
    return x.contiguous(memory_format=torch.channels_last), y, A


def aux_targets(ctx, frames, fr, Hp, Wp, device):
    """LABEL SIDE: wall-band occupancy / floor presence from ALL Root frames.

    Returns (Q, 3, Hp, Wp) uint8: wall, floor, observed-mask on network frames.
    """
    from .geometry import VoxelMap
    from .features import BANDS
    vm = VoxelMap(frames[0], frames[1], frames[2], ctx.lo, ctx.shape, ctx.res, ctx.base_y, device)
    g = vm.grid()
    wall = (g[:, :, 6:18] == 2).any(2)
    floor = (g[:, :, 1:4] == 2).any(2)
    observed = (g[:, :, 1:18] != 0).any(2)
    full = torch.stack([wall, floor, observed]).to(torch.uint8)
    out = torch.zeros((len(fr), 3, Hp, Wp), dtype=torch.uint8, device=device)
    for i in range(len(fr)):
        a0, b0, h, w = (int(v) for v in fr[i])
        ga0, gb0 = max(a0, 0), max(b0, 0)
        ga1, gb1 = min(a0+h, ctx.shape[0]), min(b0+w, ctx.shape[1])
        if ga1 > ga0 and gb1 > gb0:
            out[i, :, ga0-a0:ga1-a0, gb0-b0:gb1-b0] = full[:, ga0:ga1, gb0:gb1]
    return out


def loss_fn(logits, y, w, pos_weight, blocked_weight, norm, aux=None, aux_weight=0.):
    valid = (y >= 0).float()
    rev = (y > 0).float()
    lr = F.binary_cross_entropy_with_logits(logits[:, 0].float(), rev, reduction='none',
                                            pos_weight=torch.tensor(pos_weight, device=logits.device))
    lb = F.binary_cross_entropy_with_logits(logits[:, 1].float(), (y == 2).float(), reduction='none')
    per = (lr*valid).sum((1, 2))+blocked_weight*(lb*rev).sum((1, 2))
    if aux is not None and aux_weight > 0:
        m = aux[:, 2].float()*(y > -2).float()
        la = F.binary_cross_entropy_with_logits(logits[:, 2:4].float(), aux[:, :2].float(), reduction='none').sum(1)
        per = per+aux_weight*(la*m).sum((1, 2))
    return (per*w).sum()/w.sum()/norm


@torch.no_grad()
def validate(model, rows, device, limit=160, taus=(.3, .4, .5, .6, .7), amp=True, tracks=False):
    model.eval()
    factors = scene_factors(rows)
    per_scene = {}
    cms_total = {t: np.zeros((3, 3)) for t in taus}
    roots = {t: [] for t in taus}
    scenes = {t: {} for t in taus}
    for row in rows[:limit]:
        groups = load_root(row, tracks=tracks)
        root_cms = {t: [] for t in taus}
        for g in groups:
            ctx = build_context(g, device)
            windows = g['windows']
            frames = [crop_box(w['path'], ctx.res) for w in windows]
            x, mask, fr = build_inputs(ctx, [w['path'] for w in windows], [(lo, hi-lo) for lo, hi in frames])
            y = frame_labels(mask, fr, windows, ctx.lo)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
            r = torch.sigmoid(logits[:, 0].float()); b = torch.sigmoid(logits[:, 1].float())
            for i, w in enumerate(windows):
                valid = y[i] >= 0
                yy = y[i][valid].long()
                missing = len(w['tkeys'])-int((yy > 0).sum())
                for t in taus:
                    if len(w['path']) < 2:
                        pred = torch.zeros_like(yy)
                    else:
                        pred = torch.where(r[i][valid] > t, torch.where(b[i][valid] > .5, 2, 1), 0).long()
                    cm = torch.bincount(3*yy+pred, minlength=9).reshape(3, 3).cpu().numpy().astype(np.float64)
                    cm[1, 0] += missing
                    root_cms[t].append(cm)
        for t in taus:
            m = np.mean(root_cms[t], 0)
            scenes[t].setdefault(row['scene_id'], []).append(m)
            cms_total[t] += np.sum(root_cms[t], 0)
    out = {}
    for t in taus:
        hier = np.mean([np.mean(v, 0) for v in scenes[t].values()], 0)
        out[t] = dict(hier=rates(hier)['label_aware_delta_iou'], pooled=rates(cms_total[t])['label_aware_delta_iou'],
                      rev=rates(hier)['revelation_iou'], free_f1=rates(hier)['free']['f1'], blocked_f1=rates(hier)['blocked']['f1'])
    model.train()
    return out


def run(name, device='cuda:0', epochs=1, batch=16, size=256, lr=2e-3, workers=6, pos_weight=3., blocked_weight=2.,
        val_every=4000, val_roots=160, roots=16384, widths='32,64,96,128,192', max_steps=0, seed=0, warm=None,
        arch='masked_unet', blocks='1,2,2,2,2', human_filter=False, extra_sensor=None, aux_weight=0., splat=1, sub=1, render=96,
        human_occluders=False):
    GEOMETRY.update(splat=bool(splat), sub=int(sub), render=int(render), occluders=bool(human_occluders))
    torch.backends.cudnn.benchmark = True
    dest = DEST/'training'/name
    dest.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = split_roots(roots)
    val_rows = sorted(val_rows, key=lambda r: r['root_id'])
    widths = tuple(int(v) for v in widths.split(','))
    arch = dict(arch=arch, widths=widths, blocks=tuple(int(v) for v in blocks.split(',')), valid_channel=CIN-1, aux=2 if aux_weight > 0 else 0)
    model = build_model(CIN, arch).to(device).to(memory_format=torch.channels_last)
    if warm:
        model.load_state_dict(torch.load(warm, map_location='cpu')['model'])
    if extra_sensor:
        extra = split_roots(roots=0, sensor=extra_sensor, start=roots, val_mod=25)
        for r in extra[0]:
            r['sensor_dir'] = extra_sensor
        train_rows = train_rows+extra[0]
    config = dict(name=name, map_channels=MAP_CHANNELS, query_channels=QUERY_CHANNELS, **arch, batch=batch, size=size, human_filter=human_filter,
                  extra_sensor=extra_sensor, aux_weight=aux_weight, splat=bool(splat), sub=int(sub), render=int(render),
                  human_occluders=bool(human_occluders),
                  lr=lr, pos_weight=pos_weight, blocked_weight=blocked_weight, epochs=epochs, train_roots=len(train_rows),
                  val_roots=len(val_rows), seed=seed, warm=warm)
    (dest/'config.json').write_text(json.dumps(config, indent=1))
    print('train roots', len(train_rows), 'val roots', len(val_rows), 'params', sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    est_steps = int(epochs*len(train_rows)*23.5/batch) if not max_steps else max_steps
    sched = lambda s: min(1., (s+1)/500)*.5*(1+math.cos(math.pi*min(1., s/est_steps)))
    rng = np.random.default_rng(seed)
    step = 0; tick = time.time(); pool = []; losses = []; seen = 0
    history = []
    for epoch in range(epochs):
        loader = DataLoader(GroupStream(train_rows, seed*1000+epoch, tracks=human_filter, aux=aux_weight > 0), batch_size=None, num_workers=workers,
                            collate_fn=collate, prefetch_factor=4, persistent_workers=False)
        for item in loader:
          if max_steps and step >= max_steps:
              break
          groups, aux_frames = (item['groups'], item['aux_frames']) if item.get('root') else ([item], None)
          for g in groups:
            ctx = build_context(g, device)
            pool.extend(window_samples(g, ctx, size, rng, True, aux_frames))
            while len(pool) >= 4*batch:
                pick = rng.choice(len(pool), batch, replace=False)
                items = [pool[i] for i in pick]
                for i in sorted(pick, reverse=True):
                    pool.pop(i)
                x, y, A = collate_frames(items, device)
                w = torch.tensor([it[2] for it in items], device=device, dtype=torch.float32)
                for gparam in opt.param_groups:
                    gparam['lr'] = lr*sched(step)
                with torch.cuda.amp.autocast():
                    logits = model(x)
                loss = loss_fn(logits, y, w, pos_weight, blocked_weight, 2000., A, aux_weight)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(opt); scaler.update()
                losses.append(loss.item()); step += 1; seen += batch
                if step % 200 == 0:
                    print(dict(epoch=epoch, step=step, loss=float(np.mean(losses[-200:])), lr=lr*sched(step),
                               windows_per_s=seen/(time.time()-tick), elapsed=time.time()-tick), flush=True)
                if step % val_every == 0:
                    v = validate(model, val_rows, device, val_roots, tracks=human_filter)
                    history.append(dict(step=step, val=v)); print('VAL', step, json.dumps(v), flush=True)
                    torch.save(dict(model=model.state_dict(), config=config, step=step, val=v), dest/('step%06d.pt' % step))
                    (dest/'history.json').write_text(json.dumps(history, indent=1))
                if max_steps and step >= max_steps:
                    break
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
    v = validate(model, val_rows, device, val_roots, tracks=human_filter)
    history.append(dict(step=step, val=v, final=True)); print('VAL FINAL', step, json.dumps(v), flush=True)
    torch.save(dict(model=model.state_dict(), config=config, step=step, val=v), dest/'final.pt')
    (dest/'history.json').write_text(json.dumps(history, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--size', type=int, default=288)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--pos-weight', type=float, default=3.)
    p.add_argument('--blocked-weight', type=float, default=2.)
    p.add_argument('--val-every', type=int, default=4000)
    p.add_argument('--val-roots', type=int, default=160)
    p.add_argument('--roots', type=int, default=16384)
    p.add_argument('--widths', default='32,64,96,128,192')
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--warm')
    p.add_argument('--arch', default='masked_unet')
    p.add_argument('--blocks', default='1,2,2,2,2')
    p.add_argument('--human-filter', action='store_true')
    p.add_argument('--extra-sensor')
    p.add_argument('--aux-weight', type=float, default=0.)
    p.add_argument('--splat', type=int, default=1)
    p.add_argument('--sub', type=int, default=1)
    p.add_argument('--render', type=int, default=96)
    p.add_argument('--human-occluders', action='store_true')
    run(**vars(p.parse_args()))
