"""Stage 2 fine-tuned for the Sandbox's imagined map stream (Layer 2 option; default off).

The Sandbox (``heuristics._discovery_rollouts``) calls Stage 2 on every candidate's frozen-map imagined path and reveals
the union of the predicted cells to the imagined planner.  The released Stage 2 (v17a) was trained on recorded query
executions (Roots) for label-aware IoU.  Here it is fine-tuned on what the Sandbox consumes (``s2data.py``, TRAIN
episodes only): at a floor decision state, the conditioning path is a candidate's IMAGINED path (legal constant-velocity
people or true people, core physics; the Sandbox's first discovery pass) or its REAL path, over 1.5 s (32 steps) or
3.1 s (64 steps), and the target is what the legal camera REALLY certified along the candidate's real branch in that
window (cells UNKNOWN at t; FREE / BLOCKED / stays UNKNOWN).  Wall cells carry extra weight (the imagined planner reacts
to discovered walls; docs/l3/LAYER2_V2_DESIGN.md section 1.2).  Inputs, features, network and the ``predict_segments``
interface are unchanged; the checkpoint stores its own operating point (class rule), chosen on scene-disjoint TRAIN
validation.  Load it in the Sandbox with ``L3_S2_CKPT=<path>`` (layer2/model.py).

    python scripts/l3_run.py eventwm_graph.l3.s2ft train --data output/agents/stage2/s2ft/data_train --out <ckpt>
    python scripts/l3_run.py eventwm_graph.l3.s2ft score --data <dir> --split val|all --ckpts release,<ckpt> --out <json>
    python scripts/l3_run.py eventwm_graph.l3.s2ft select --stats <json> --ckpt <ckpt>      (TRAIN validation only)
    python scripts/l3_run.py eventwm_graph.l3.s2ft table --stats <json> [--out <txt>]
    python scripts/l3_run.py eventwm_graph.l3.s2ft fidelity --probes release=<dir>,ft=<dir> [--out <json>]

Recipe of ``s2ft_v1`` (output/agents/stage2/s2ft/ckpt; results in docs/l3/LAYER2_V2_DESIGN.md section 1.2): data
``s2data --suite train --shards 8`` (first ``--every 3 --max-dec 6``, then ``--every 1 --max-dec 5`` on the remaining
episodes; 76 training + 11 validation episodes), ``train --epochs 4 --lr 3e-4 --wall-weight 3 --batch 16 --micro 8``,
``score --split val --kinds cv --windows 32`` + ``select --kind cv --window 32`` (BLOCKED > 0.725); DEV data
``s2data --suite dev --shards 2 --every 3 --max-dec 6`` (36 episodes), scored once.

Privileged inputs appear only as labels (the real branches) and as one alternative conditioning path family ('true'
imagined paths); the deployed model's inputs are the legal Stage 2 inputs and whatever path the Sandbox supplies.
"""
import argparse
import json
import math
import os
import time
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

RELEASE = 'release/stage2_claude_v17a/stage2_claude_v17a.pt'
WINDOWS = (32, 64)
KINDS = ('cv', 'true', 'real')
PROTOCOL = 'ideal_current_jaw_range_dynamic_occlusion_1'


def is_val(scene):
    """Scene-disjoint TRAIN validation split (about 1 scene in 8)."""
    return zlib.crc32(str(scene).encode()) % 8 == 0


# ------------------------------------------------------------------------------------------------ data
class Episodes:
    """LRU of decompressed ``s2data`` episode files."""

    def __init__(self, size=6):
        self.size, self.cache = int(size), OrderedDict()

    def get(self, path):
        from .s2data import load_episode
        path = str(path)
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        ep = load_episode(path)
        self.cache[path] = ep
        while len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return ep


def index(data_dirs, split):
    """[(file, decision index, scene)] of the s2data files in ``data_dirs`` (comma-separated) for a split: 'train' /
    'val' (scene-disjoint TRAIN validation) / 'all'.  Reads only the file names (scene = the name's third field)."""
    out = []
    for d in str(data_dirs).split(','):
        for f in sorted(Path(d).glob('*.pkl')):
            scene = f.name.split('_')[2]
            if split == 'train' and is_val(scene):
                continue
            if split == 'val' and not is_val(scene):
                continue
            out.append((str(f), scene))
    return out


def decision_inputs(ep, d):
    nf = int(d['frames'])
    fr = ep['frames']
    packet = dict(ranges=fr['ranges'][:nf], origins=fr['origins'][:nf], rotations=fr['rotations'][:nf],
                  timestamps=fr['timestamps'][:nf], sensor_protocol=PROTOCOL)
    return d['graph'], packet, list(fr['tracks'][:nf])


def path_of(d, c, kind, W):
    """Conditioning trajectory N x 5 [t, x, y, z, yaw] (0.048 s steps, the imagination clock) of candidate ``c``: its
    imagined path ('cv' / 'true') or real path ('real'), first ``W`` steps."""
    from eventwm_graph.layer2.state import trajectory_from_poses
    if kind == 'real':
        P = np.concatenate([np.asarray(d['pose'], np.float64)[None], np.asarray(c['real'][:W], np.float64)])
    else:
        P = c['imag'].get(kind)
        if P is None:
            return None
        P = np.asarray(P[:W+1], np.float64)
    return trajectory_from_poses(P, float(d['time']))


def window_end(c, W):
    rt = np.asarray(c['real_t'], np.float64)
    return float(rt[min(W, len(rt))-1]) if len(rt) else 0.


def label_cells(c, W):
    """Cells the real branch certified within ``W`` steps (UNKNOWN at t): keys [n, 2], states [n] (1 FREE, 2 BLOCKED)."""
    m = np.asarray(c['first'], np.float64) <= window_end(c, W)+1e-4
    return np.asarray(c['keys'])[m], np.asarray(c['states'])[m]


# ------------------------------------------------------------------------------------------------ features
class Featurizer:
    """Stage 2's own legal feature pipeline (``LegalVoxelPredictor._context`` + ``features.build_inputs``)."""

    def __init__(self, device='cuda:0', checkpoint=RELEASE):
        from eventwm_graph.stage2_claude.api import LegalVoxelPredictor
        self.p = LegalVoxelPredictor(checkpoint, device=device, cuda_graphs=False, lanes=1)
        self.device = self.p.device

    def __call__(self, graph, packet, tracks, paths):
        import torch
        from eventwm_graph.stage2_claude.features import crop_box, build_inputs
        p = self.p
        with torch.no_grad():
            prepared = p.prepare_graph(graph)
            res = prepared.resolution
            boxes = [crop_box(q, res) for q in paths]
            ctx, _ = p._context(prepared, graph, packet, tracks, paths)
            x, mask, fr = build_inputs(ctx, paths, [(lo, hi-lo) for lo, hi in boxes])
            if p.channels is not None:
                x = x[:, p.channels]
        return x, mask, fr, boxes


def frame_labels(mask, fr, boxes, labels):
    """Labels on the network frames: -1 outside the candidate domain, else 0 (stays UNKNOWN) / 1 FREE / 2 BLOCKED."""
    import torch
    y = torch.where(mask > 0, 0, -1).to(torch.int8)
    rows = []
    for i, (keys, states) in enumerate(labels):
        if not len(keys):
            continue
        ij = np.asarray(keys, np.int64)-np.asarray(boxes[i][0], np.int64)
        ok = (ij >= 0).all(1) & (ij[:, 0] < fr[i, 2]) & (ij[:, 1] < fr[i, 3])
        if ok.any():
            rows.append(np.column_stack([np.full(ok.sum(), i), ij[ok], np.asarray(states, np.int64)[ok]]))
    if rows:
        t = torch.as_tensor(np.concatenate(rows), device=mask.device)
        inside = mask[t[:, 0], t[:, 1], t[:, 2]] > 0
        t = t[inside]
        y[t[:, 0], t[:, 1], t[:, 2]] = t[:, 3].to(torch.int8)
    return y


# ------------------------------------------------------------------------------------------------ training
def loss_fn(logits, y, wall_weight=3., pos_weight=3., blocked_weight=2.):
    """Revelation BCE (positives = certified cells; wall cells weighted ``wall_weight``) + BLOCKED-given-revelation BCE
    on certified cells (walls weighted ``wall_weight``), normalised by the valid cells."""
    import torch
    import torch.nn.functional as F
    valid = (y >= 0).float()
    rev = (y > 0).float()
    wall = (y == 2).float()
    w = 1.+(wall_weight-1.)*wall
    lr = F.binary_cross_entropy_with_logits(logits[:, 0].float(), rev, reduction='none',
                                            pos_weight=torch.tensor(pos_weight, device=logits.device))
    lb = F.binary_cross_entropy_with_logits(logits[:, 1].float(), wall, reduction='none')
    return ((lr*w*valid).sum()+blocked_weight*(lb*w*rev).sum())/valid.sum().clamp(min=1.)


def load_model(checkpoint, device):
    import torch
    from eventwm_graph.stage2_claude.model import build_model
    saved = torch.load(checkpoint, map_location='cpu')
    config = saved['config']
    cin = len(config['map_channels'])+len(config['query_channels'])
    model = build_model(cin, config)
    model.load_state_dict(saved['model'])
    return model.to(device).to(memory_format=torch.channels_last), saved


def freeze_bn(model):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def sample_queries(d, rng, batch, kinds=KINDS, windows=WINDOWS, kind_p=None):
    """Up to ``batch`` (candidate, kind, window) samples of one decision (without replacement)."""
    opts = []
    for k, c in enumerate(d['cands']):
        for kind in kinds:
            if kind != 'real' and kind not in c['imag']:
                continue
            for W in windows:
                opts.append((k, kind, W))
    if not opts:
        return []
    p = None
    if kind_p:
        p = np.asarray([kind_p[o[1]] for o in opts], np.float64)
        p = p/p.sum()
    idx = rng.choice(len(opts), size=min(batch, len(opts)), replace=False, p=p)
    return [opts[i] for i in idx]


def train(a):
    import torch
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    torch.backends.cudnn.benchmark = True
    feats = Featurizer(a.device)
    model, saved = load_model(a.init, dev)
    model.train()
    freeze_bn(model)
    ema = None
    if a.ema > 0:
        import copy
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.wd)
    scaler = torch.cuda.amp.GradScaler()
    rng = np.random.default_rng(a.seed)
    files = index(a.data, 'train')
    eps = Episodes(4)
    items = []
    for f, scene in files:
        ep = eps.get(f)
        items += [(f, i) for i in range(len(ep['decisions']))]
    print(json.dumps(dict(train_files=len(files), train_decisions=len(items))), flush=True)
    kind_p = dict(cv=a.p_cv, true=a.p_true, real=a.p_real)
    total = a.epochs*len(items)
    step = 0
    log = []
    t0 = time.perf_counter()
    for epoch in range(a.epochs):
        # episodes in random order, decisions in random order within each (episode-local cache)
        by = OrderedDict()
        for f, i in items:
            by.setdefault(f, []).append(i)
        order = list(by)
        rng.shuffle(order)
        run = {'loss': 0., 'n': 0}
        for f in order:
            ep = eps.get(f)
            dl = list(by[f])
            rng.shuffle(dl)
            for i in dl:
                d = ep['decisions'][i]
                qs = sample_queries(d, rng, a.batch, kind_p=kind_p)
                paths, labs = [], []
                for k, kind, W in qs:
                    p = path_of(d, d['cands'][k], kind, W)
                    if p is None or len(p) < 2:
                        continue
                    paths.append(p)
                    labs.append(label_cells(d['cands'][k], W))
                if not paths:
                    continue
                graph, packet, tracks = decision_inputs(ep, d)
                x, mask, fr, boxes = feats(graph, packet, tracks, paths)
                y = frame_labels(mask, fr, boxes, labs)
                # pad to multiples of 64 (the masked network is padding-invariant): few distinct shapes for cudnn
                ph, pw = (-x.shape[2]) % 64, (-x.shape[3]) % 64
                if ph or pw:
                    x = torch.nn.functional.pad(x, (0, pw, 0, ph))
                    y = torch.nn.functional.pad(y, (0, pw, 0, ph), value=-1)
                lr = a.lr*0.5*(1+math.cos(math.pi*min(step/max(total, 1), 1.)))
                lr = lr*min(1., (step+1)/max(a.warmup, 1))
                for g in opt.param_groups:
                    g['lr'] = lr
                opt.zero_grad(set_to_none=True)
                # micro-batches (gradient accumulation; each chunk normalised by its own valid cells, chunks averaged)
                chunks = [slice(s, s+a.micro) for s in range(0, len(paths), max(a.micro, 1))]
                loss = 0.
                for sl in chunks:
                    with torch.cuda.amp.autocast():
                        logits = model(x[sl].contiguous(memory_format=torch.channels_last))
                    li = loss_fn(logits, y[sl], a.wall_weight, a.pos_weight, a.blocked_weight)/len(chunks)
                    scaler.scale(li).backward()
                    loss = loss+li.detach()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                if ema is not None:
                    with torch.no_grad():
                        for pe, pm in zip(ema.parameters(), model.parameters()):
                            pe.mul_(a.ema).add_(pm.detach(), alpha=1-a.ema)
                        for be, bm in zip(ema.buffers(), model.buffers()):
                            be.copy_(bm)
                run['loss'] += float(loss); run['n'] += 1
                step += 1
                if step % 50 == 0:
                    rec = dict(step=step, of=total, epoch=epoch, loss=round(run['loss']/max(run['n'], 1), 5), lr=lr,
                               seconds=round(time.perf_counter()-t0, 1))
                    log.append(rec)
                    print(json.dumps(rec), flush=True)
                    run = {'loss': 0., 'n': 0}
        save(ema if ema is not None else model, saved, a, Path(a.out), dict(epochs_done=epoch+1, steps=step, log=log,
                                                                             train_files=len(files),
                                                                             train_decisions=len(items)))
        print(json.dumps(dict(saved=a.out, epoch=epoch+1, steps=step, seconds=round(time.perf_counter()-t0, 1),
                              peak_gb=round(torch.cuda.max_memory_allocated(dev)/2**30, 2))), flush=True)


def save(model, saved, a, out, info):
    """Checkpoint in the release format (config / model / decision): ``LegalVoxelPredictor`` loads it unchanged.  The
    decision keeps the release operating point until ``select`` stores the TRAIN-validation choice."""
    import torch
    config = dict(saved['config'])
    config.update(name=Path(out).stem, warm=str(a.init), sandbox_ft=dict(
        data=str(a.data), windows=list(WINDOWS), kinds=dict(cv=a.p_cv, true=a.p_true, real=a.p_real),
        wall_weight=a.wall_weight, pos_weight=a.pos_weight, blocked_weight=a.blocked_weight, lr=a.lr, epochs=a.epochs,
        batch=a.batch, micro=a.micro, ema=a.ema, seed=a.seed, split='TRAIN only (scene-disjoint validation held out: crc32(scene) % 8 == 0)',
        **{k: v for k, v in info.items() if k != 'log'}))
    state = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out)+'.tmp')
    torch.save(dict(model=state, config=config, decision=dict(saved['decision']), train_log=info.get('log', [])), tmp)
    os.replace(tmp, out)


# ------------------------------------------------------------------------------------------------ evaluation
def _k64(keys):
    keys = np.asarray(keys, np.int64).reshape(-1, 2)
    return (keys[:, 0] << 32) ^ (keys[:, 1] & 0xffffffff)


def union_states(inv, n, states):
    """Union of several increments' states on ``n`` unique keys (``inv``: key index of every concatenated row): BLOCKED
    wins over FREE wins over UNKNOWN (``perceive.predicted_truth_union``), i.e. the maximum state per key."""
    us = np.zeros(n, np.uint8)
    if n:
        sa = np.concatenate(states)
        us[np.bincount(inv, weights=(sa == 1), minlength=n) > 0] = 1
        us[np.bincount(inv, weights=(sa == 2), minlength=n) > 0] = 2
    return us


def class_states(pf, pb, tf, tb):
    """Stage 2's class rule at thresholds (perceive.s2_states): 0 / 1 FREE / 2 BLOCKED."""
    return np.where((pf > tf) | (pb > tb), np.where(pb-tb > pf-tf, 2, 1), 0).astype(np.uint8)


def map_counts(pk, ps, rk, rs, sorted_keys=False):
    """``shortprobe.map_stats`` on key sets: predicted map cells (int64 keys ``pk``, unique, states ``ps``; new at t) vs
    one real window's certified cells (``rk`` / ``rs``)."""
    ps, rs = np.asarray(ps), np.asarray(rs)
    sel = ps > 0
    pk, ps = pk[sel], ps[sel]
    if not sorted_keys and len(pk):
        o = np.argsort(pk)
        pk, ps = pk[o], ps[o]
    pr = np.zeros(len(rk), np.uint8)
    seen = np.zeros(len(pk), bool)
    if len(pk) and len(rk):
        pos = np.minimum(np.searchsorted(pk, rk), len(pk)-1)
        hit = pk[pos] == rk
        pr = np.where(hit, ps[pos], 0)
        seen[pos[hit]] = True
    return dict(rB=int((rs == 2).sum()), rF=int((rs == 1).sum()), pB=int((ps == 2).sum()), pF=int((ps == 1).sum()),
                BB=int(((rs == 2) & (pr == 2)).sum()), FF=int(((rs == 1) & (pr == 1)).sum()),
                FB=int(((rs == 1) & (pr == 2)).sum()), BF=int(((rs == 2) & (pr == 1)).sum()),
                B0=int(((ps == 2) & ~seen).sum()), F0=int(((ps == 1) & ~seen).sum()))


def rates(S):
    g = lambda a, b: float(a)/max(float(b), 1.)
    return dict(B_prec=g(S['BB'], S['pB']-S['B0']), B_rec=g(S['BB'], S['rB']), F_prec=g(S['FF'], S['pF']-S['F0']),
                F_rec=g(S['FF'], S['rF']), false_walls=g(S['FB'], S['rF']), false_openings=g(S['BF'], S['rB']),
                B_pred_unseen=g(S['B0'], S['pB']), B_pred=S['pB'], rB=S['rB'])


TB_GRID = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.575, 0.6, 0.65, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825,
           0.85, 0.9, 0.95)
TF_GRID = (0.3, 0.4, 0.475, 1.01)
EMPTY = dict(rB=0, rF=0, pB=0, pF=0, BB=0, FF=0, FB=0, BF=0, B0=0, F0=0)


def union_prep(uk, windows):
    """Per real window (int64 keys ``rk``, states ``rs``; unique keys): the positions of its keys in the sorted unique
    predicted keys ``uk`` and hit flags -- shared by every threshold setting of the union map."""
    out = []
    for rk, rs in windows:
        if len(uk) and len(rk):
            pos = np.minimum(np.searchsorted(uk, rk), len(uk)-1)
            hit = uk[pos] == rk
        else:
            pos, hit = np.zeros(len(rk), np.int64), np.zeros(len(rk), bool)
        out.append((pos, hit, np.asarray(rs)))
    return out


def union_counts(us, prep):
    """``map_counts`` of one union map (states ``us`` on the sorted unique keys of ``union_prep``) against every real
    window, summed over the windows (equal to summing ``map_counts(uk, us, rk, rs, sorted_keys=True)``)."""
    pB, pF = int((us == 2).sum()), int((us == 1).sum())
    S = dict(EMPTY)
    for pos, hit, rs in prep:
        pr = np.where(hit, us[pos], 0) if len(us) else np.zeros(len(rs), np.uint8)
        b, f, qb, qf = rs == 2, rs == 1, pr == 2, pr == 1
        S['rB'] += int(b.sum()); S['rF'] += int(f.sum()); S['pB'] += pB; S['pF'] += pF
        S['BB'] += int((b & qb).sum()); S['FF'] += int((f & qf).sum())
        S['FB'] += int((f & qb).sum()); S['BF'] += int((b & qf).sum())
        S['B0'] += pB-int(qb.sum()); S['F0'] += pF-int(qf.sum())
    return S


def evaluate(a):
    """Map metrics of each checkpoint on the decisions of a split: per (conditioning kind, window), the union of every
    candidate's predicted cells (as the Sandbox reveals them, ``perceive.predicted_truth_union``: BLOCKED wins) and each
    candidate's own prediction, against every candidate's REAL certified cells in the window (summed over candidates,
    ``shortprobe.map_stats``).  Settings: 'op' -- the checkpoint's own operating point (its ``states``); 'opw' -- walls
    only at that point (``L3_S2_PF=1.01``); union map only: the class rule over ``TB_GRID`` x ``TF_GRID``.  Latency of
    the ``predict_segments`` call (CUDA graphs, 3 lanes, as the Layer 2 backend) is recorded per checkpoint."""
    import torch
    from eventwm_graph.stage2_claude.api import LegalVoxelPredictor
    names = a.ckpts.split(',')
    preds = {}
    for n in names:
        path = RELEASE if n == 'release' else n
        preds[n] = LegalVoxelPredictor(path, device=a.device, lanes=3)
    files = index(a.data, a.split)
    if a.limit:
        files = files[:a.limit]
    eps = Episodes(2)
    kinds = a.kinds.split(',')
    windows = [int(w) for w in a.windows.split(',')]
    grid = [('tb%.3f_tf%.3f' % (tb, tf), tb, tf) for tb in TB_GRID for tf in TF_GRID]
    acc = {}
    lat = {n: [] for n in names}
    ndec = 0

    def add(key, c):
        S = acc.setdefault(key, dict(EMPTY))
        for kk, v in c.items():
            S[kk] += v
    for f, scene in files:
        ep = eps.get(f)
        for d in ep['decisions']:
            graph, packet, tracks = decision_inputs(ep, d)
            ndec += 1
            for kind in kinds:
                for W in windows:
                    ks = [k for k, c in enumerate(d['cands']) if path_of(d, c, kind, W) is not None]
                    paths = [path_of(d, d['cands'][k], kind, W) for k in ks]
                    if not paths:
                        continue
                    real = [label_cells(d['cands'][k], W) for k in ks]
                    rk64 = [(_k64(rk), rs) for rk, rs in real]
                    for n, P in preds.items():
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        out = P.predict_segments(partial_graph=graph, trajectories=paths, range_observations=packet,
                                                 track_history=tracks, materialize=False)
                        lat[n].append(dict(K=len(paths), ms=1000*(time.perf_counter()-t)))
                        keys = [_k64(o['candidate_cells']) for o in out]
                        sizes = [len(x) for x in keys]
                        k64 = np.concatenate(keys) if keys else np.zeros(0, np.int64)
                        prob = (np.concatenate([np.asarray(o['probabilities'], np.float32).reshape(-1, 3) for o in out])
                                if out else np.zeros((0, 3), np.float32))
                        own = np.concatenate([np.asarray(o['states'], np.uint8) for o in out]) if out else np.zeros(0, np.uint8)
                        uk, inv = np.unique(k64, return_inverse=True)
                        prep = union_prep(uk, rk64)
                        tbo = np.float32(P.blocked_threshold)
                        pf, pb = prob[:, 1], prob[:, 2]
                        sets = [('op', own)]
                        if P.rule == 'class':        # walls only at the checkpoint's own BLOCKED threshold (PF=1.01)
                            sets.append(('opw', class_states(pf, pb, np.float32(1.01), tbo)))
                        sets += [(tag, class_states(pf, pb, np.float32(tf), np.float32(tb))) for tag, tb, tf in grid]
                        for tag, st in sets:
                            add((n, kind, W, 'union', tag), union_counts(union_states(inv, len(uk), [st]), prep))
                            if tag in ('op', 'opw'):
                                at = 0
                                for j, (rk, rs) in enumerate(rk64):
                                    add((n, kind, W, 'own', tag), map_counts(keys[j], st[at:at+sizes[j]], rk, rs))
                                    at += sizes[j]
        print(json.dumps(dict(file=Path(f).name, decisions=ndec)), flush=True)
    out = dict(split=a.split, data=a.data, files=len(files), decisions=ndec, ckpts=names, stats=[], latency={},
               operating_points={n: dict(blocked=float(P.blocked_threshold), free=float(P.threshold), rule=P.rule,
                                         sha256=P.checkpoint_sha256) for n, P in preds.items()})
    for (n, kind, W, mode, tag), S in sorted(acc.items()):
        out['stats'].append(dict(ckpt=n, kind=kind, W=W, mode=mode, setting=tag, counts=S, **rates(S)))
    for n, L in lat.items():
        ms = np.asarray([x['ms'] for x in L])
        if len(ms):
            out['latency'][n] = dict(calls=len(ms), K_mean=float(np.mean([x['K'] for x in L])), p50=float(np.median(ms)),
                                     p95=float(np.percentile(ms, 95)), mean=float(ms.mean()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    table(out)


def _curve(rows, ck, kind, W, tf=0.475, mode='union'):
    pts = [s for s in rows if s['ckpt'] == ck and s['kind'] == kind and s['W'] == W and s['mode'] == mode
           and s['setting'].startswith('tb') and s['setting'].endswith('tf%.3f' % tf)]
    return sorted(pts, key=lambda s: float(s['setting'][2:].split('_')[0]))


def _interp(pts, xkey, x, ykey):
    """``ykey`` where the threshold curve's ``xkey`` equals x (linear between grid points; nan outside)."""
    if not pts:
        return float('nan')
    xs, ys = np.asarray([p[xkey] for p in pts]), np.asarray([p[ykey] for p in pts])
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    return float(np.interp(x, xs, ys)) if xs[0] <= x <= xs[-1] else float('nan')


def table(out, file=None):
    """Compact table of a ``score`` result: per map (union / own), conditioning and window, each checkpoint at its
    operating point ('op') and walls only at that point ('opw', ``L3_S2_PF=1.01``); union map: wall recall / precision
    interpolated at the release operating point's false-wall rate (the release's own curve gives the release value)."""
    rows = out['stats']
    lines = ['== %s split, %d files, %d decisions' % (out['split'], out['files'], out['decisions'])]
    fmt = lambda s: 'walls P %.3f R %.3f | free P %.3f R %.3f | false walls %.3f false openings %.3f' % (
        s['B_prec'], s['B_rec'], s['F_prec'], s['F_rec'], s['false_walls'], s['false_openings'])
    for mode in ('union', 'own'):
        for kind in sorted({s['kind'] for s in rows}):
            for W in sorted({s['W'] for s in rows}):
                ref = [s for s in rows if s['ckpt'] == 'release' and s['kind'] == kind and s['W'] == W
                       and s['mode'] == mode and s['setting'] == 'op']
                secs = {32: '1.5 s', 64: '3.1 s'}.get(W, '%.2f s' % (W*0.0485))
                lines.append(' %s map, conditioning %s, %d steps (%s)' % (mode, kind, W, secs))
                for ck in out['ckpts']:
                    for tag in ('op', 'opw'):
                        s = [x for x in rows if x['ckpt'] == ck and x['kind'] == kind and x['W'] == W and x['mode'] == mode
                             and x['setting'] == tag]
                        if not s:
                            continue
                        line = '   %-18s %-3s %s' % (Path(ck).stem[:18], tag, fmt(s[0]))
                        if tag == 'op' and mode == 'union' and ref:
                            pts = _curve(rows, ck, kind, W)
                            fw = ref[0]['false_walls']
                            line += ' || at false walls %.3f: walls R %.3f P %.3f' % (
                                fw, _interp(pts, 'false_walls', fw, 'B_rec'), _interp(pts, 'false_walls', fw, 'B_prec'))
                        lines.append(line)
    for ck, L in out.get('latency', {}).items():
        lines.append(' latency %-18s calls %d, K %.1f: p50 %.1f ms, p95 %.1f ms, mean %.1f ms' % (
            Path(ck).stem[:18], L['calls'], L['K_mean'], L['p50'], L['p95'], L['mean']))
    for ck, o in out.get('operating_points', {}).items():
        lines.append(' operating point %-18s rule %s: BLOCKED > %.3f, FREE > %.3f (sha256 %s)' % (
            Path(ck).stem[:18], o['rule'], o['blocked'], o['free'], o['sha256'][:12]))
    text = '\n'.join(lines)
    print(text)
    if file:
        Path(file).write_text(text+'\n')
    return text


def select(a):
    """Operating point of a fine-tuned checkpoint on TRAIN validation (``score --split val`` statistics): the BLOCKED
    threshold with the highest union wall recall whose union wall precision is at least the release's at its own
    operating point and whose false-wall rate (predicted BLOCKED where the real camera certified FREE) is at most the
    release's (same conditioning / window): more walls found at no more false walls.  The FREE threshold of the class
    rule is kept (0.475).  Stored in the checkpoint's ``decision`` (the predictor's ``states``)."""
    import torch
    st = json.loads(Path(a.stats).read_text())
    if st['split'] != 'val':
        raise SystemExit('operating points are chosen on TRAIN validation only')
    rows = [s for s in st['stats'] if s['kind'] == a.kind and s['W'] == a.window and s['mode'] == 'union']
    ref = [s for s in rows if s['ckpt'] == 'release' and s['setting'] == 'op'][0]
    mine = [s for s in rows if s['ckpt'] != 'release' and Path(s['ckpt']).resolve() == Path(a.ckpt).resolve()
            and s['setting'].startswith('tb') and s['setting'].endswith('tf0.475')]
    if not mine:
        raise SystemExit('no statistics of %s in %s' % (a.ckpt, a.stats))
    ok = [s for s in mine if s['B_prec'] >= ref['B_prec']-a.slack and s['false_walls'] <= ref['false_walls']+a.slack]
    best = max(ok, key=lambda s: s['B_rec']) if ok else max(mine, key=lambda s: s['B_prec'])
    tb = float(best['setting'][2:].split('_')[0])
    keys = ('B_prec', 'B_rec', 'F_prec', 'F_rec', 'false_walls', 'false_openings')
    saved = torch.load(a.ckpt, map_location='cpu')
    dec = dict(saved['decision'])
    dec.update(rule='class', blocked_class_threshold=tb, selected_on='TRAIN scene-disjoint validation (s2ft.select): max '
               'union wall recall with union wall precision >= release and false-wall rate <= release (slack %.3f), '
               'conditioning %s, %d steps; FREE threshold kept' % (a.slack, a.kind, a.window),
               validation=dict(stats=a.stats, decisions=st['decisions'], release={k: ref[k] for k in keys},
                               chosen={k: best[k] for k in keys}, feasible=bool(ok)))
    saved['decision'] = dec
    tmp = Path(a.ckpt+'.tmp')
    torch.save(saved, tmp)
    os.replace(tmp, a.ckpt)
    print(json.dumps(dict(chosen=best['setting'], feasible=bool(ok), **{k: round(best[k], 4) for k in keys},
                          release={k: round(ref[k], 4) for k in keys}), indent=1))


def _probe_load(d):
    out = {}
    for f in sorted(Path(d).glob('*.json')):
        r = json.loads(f.read_text())
        for s in r.get('probe', []):
            for c in s['cands']:
                out[(r['key'], s['decision'], c['label'])] = c
    return out


def fidelity(a):
    """Imagined-vs-real divergence of shortprobe runs that differ only in the Sandbox server's Stage 2 checkpoint
    (``--probes name=dir,...``; the first is the reference): per variant, non-thru candidates' planar divergence at
    16 / 32 / 64 steps (median / p90 (mean)) on the candidates present in every run; the Stage 2 variants (camS2*) per
    run, the others from the first run (checked identical across runs); each camS2 variant's predicted map against the
    real windows (``shortprobe.map_stats`` sums); paired camS2 differences against the first run."""
    runs = [(kv.split('=', 1)[0], _probe_load(kv.split('=', 1)[1])) for kv in a.probes.split(',')]
    keys = sorted(set.intersection(*[set(r) for _, r in runs]))
    ref = runs[0][1]
    res = dict(candidates={n: len(r) for n, r in runs}, common=len(keys), variants={}, paired={}, identical={})
    lines = ['candidates %s, common %d' % (', '.join('%s %d' % (n, len(r)) for n, r in runs), len(keys))]
    have = {n: {v for k in keys for v in R[k]['imag']} for n, R in runs}
    variants = sorted(set.union(*have.values()))
    for n, R in runs[1:]:
        for v in [v for v in variants if not v.startswith('camS2') and v in have[n] and v in have[runs[0][0]]]+['real']:
            same = sum(json.dumps(R[k]['imag'].get(v) if v != 'real' else R[k]['real'])
                       == json.dumps(ref[k]['imag'].get(v) if v != 'real' else ref[k]['real']) for k in keys)
            res['identical']['%s:%s' % (n, v)] = same
            lines.append('  %s: %s identical to %s in %d / %d' % (n, v, runs[0][0], same, len(keys)))

    def div(R, v, lag, thru=False):
        return np.asarray([R[k]['imag'][v][0]['div'][str(lag)] for k in keys if R[k]['imag'].get(v)
                           and R[k]['label'].endswith('_th') == thru and R[k]['imag'][v][0]['div'].get(str(lag)) is not None],
                          np.float64)
    for n, R in runs:
        for v in variants:
            if v not in have[n] or (not v.startswith('camS2') and n != [m for m, _ in runs if v in have[m]][0]):
                continue
            name = v if not v.startswith('camS2') else '%s[%s]' % (v, n)
            o, line = {}, '  %-20s' % name
            for lag in (16, 32, 64):
                x = div(R, v, lag)
                if len(x):
                    o['d%d' % lag] = dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)), mean=float(x.mean()),
                                          n=int(len(x)))
                    line += '  d%d %.2f / %.2f (%.2f) n=%d' % (lag, np.median(x), np.percentile(x, 90), x.mean(), len(x))
            st = [R[k]['imag'][v][0]['s2'] for k in keys if R[k]['imag'].get(v) and R[k]['imag'][v][0].get('s2')]
            if st:
                S = {kk: sum(x[kk] for x in st) for kk in st[0]}
                o['s2'] = dict(S, **rates(dict(dict(EMPTY), **{kk: S.get(kk, 0) for kk in EMPTY})))
                line += '\n%22swalls P %.2f R %.2f | free P %.2f R %.2f | false walls %.3f false openings %.3f' % (
                    '', o['s2']['B_prec'], o['s2']['B_rec'], o['s2']['F_prec'], o['s2']['F_rec'], o['s2']['false_walls'],
                    o['s2']['false_openings'])
            res['variants'][name] = o
            lines.append(line)
    for n, R in runs[1:]:
        for v in [v for v in variants if v.startswith('camS2')]:
            for lag in (32, 64):
                d = np.asarray([R[k]['imag'][v][0]['div'][str(lag)]-ref[k]['imag'][v][0]['div'][str(lag)] for k in keys
                                if not ref[k]['label'].endswith('_th') and ref[k]['imag'].get(v) and R[k]['imag'].get(v)
                                and ref[k]['imag'][v][0]['div'].get(str(lag)) is not None
                                and R[k]['imag'][v][0]['div'].get(str(lag)) is not None], np.float64)
                if len(d):
                    res['paired']['%s:%s_d%d' % (n, v, lag)] = dict(mean=float(d.mean()), better=int((d < -0.25).sum()),
                                                                 worse=int((d > 0.25).sum()), n=int(len(d)))
                    lines.append('  paired %s d%d: %s - %s mean %+.3f m, better by > 0.25 m %d, worse %d, n=%d' % (
                        v, lag, n, runs[0][0], d.mean(), int((d < -0.25).sum()), int((d > 0.25).sum()), len(d)))
    print('\n'.join(lines))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('train', 'score', 'select', 'table', 'fidelity'))
    ap.add_argument('--data', default='output/agents/stage2/s2ft/data_train')
    ap.add_argument('--out', default=None)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--init', default=RELEASE)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--micro', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=0.)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--ema', type=float, default=0.)
    ap.add_argument('--wall-weight', type=float, default=3.)
    ap.add_argument('--pos-weight', type=float, default=3.)
    ap.add_argument('--blocked-weight', type=float, default=2.)
    ap.add_argument('--p-cv', type=float, default=0.4)
    ap.add_argument('--p-true', type=float, default=0.3)
    ap.add_argument('--p-real', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split', default='val')
    ap.add_argument('--ckpts', default='release')
    ap.add_argument('--kinds', default='cv,true,real')
    ap.add_argument('--windows', default='32,64')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stats', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--kind', default='cv')
    ap.add_argument('--window', type=int, default=32)
    ap.add_argument('--slack', type=float, default=0.)
    ap.add_argument('--probes', default=None, help='fidelity: name=shortprobe dir,... (the first is the reference)')
    a = ap.parse_args()
    if a.cmd == 'train':
        train(a)
    elif a.cmd == 'score':
        evaluate(a)
    elif a.cmd == 'table':
        table(json.loads(Path(a.stats).read_text()), a.out)
    elif a.cmd == 'fidelity':
        fidelity(a)
    else:
        select(a)


if __name__ == '__main__':
    main()
