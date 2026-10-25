"""Bounded segment fine-tuning; unaltered proper-score/height objective."""
import argparse
from collections import defaultdict
import json
import random
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from eventwm_graph.stage2_quality.model import GraphDeltaNet
from eventwm_graph.stage2_quality.common import SCHEMA as MODEL_SCHEMA
from .common import BASE, OUT, CHECKPOINT, CHECKPOINT_SHA, dump, sha, snapshot


def load_model(checkpoint, device, features='base'):
    saved = torch.load(checkpoint, map_location='cpu')
    assert saved['schema'] == MODEL_SCHEMA and saved['mode'] == 'both'
    config = dict(saved['model_config'])
    if features.startswith('ray'):
        config['input_features'] = 96 if features=='ray_end' else 104
    model = GraphDeltaNet(**config)
    if features.startswith('ray') and saved['model']['net.0.weight'].shape[1] == 92:
        weights = dict(saved['model'])
        first = torch.zeros((48, config['input_features'])); first[:, :92] = weights['net.0.weight']; weights['net.0.weight'] = first
        model.load_state_dict(weights)
    else:
        model.load_state_dict(saved['model'])
    model.to(device)
    return model


def input_features(x,model):
    if model.net[0].in_features==96:
        out=np.empty((len(x),96),np.float32);out[:,:92]=x[:,:92];out[:,92:]=x[:,100:104];return out
    return x.astype(np.float32)


def holdout(model, manifest, device):
    """Scene -> Root -> unique window proper scores; inverse cell sampling."""
    model.eval(); roots = {}; pooled = np.zeros((3, 3)); scene = defaultdict(list)
    with torch.inference_mode():
        for shard in manifest['shards']:
            assert sha(shard['path']) == shard['sha256']
            with np.load(shard['path'], allow_pickle=False) as a:
                x = input_features(a['x'],model); y = a['y'].astype(np.int64); w = a['weight'].astype(np.float64)
            pred = []
            for start in range(0, len(x), 8192):
                logits, _ = model(torch.from_numpy(x[start:start+8192]).to(device))
                pred.append(torch.softmax(logits, -1).cpu().numpy())
            p = np.concatenate(pred); states = p.argmax(-1)
            ce = -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))
            br = np.square(p).sum(-1)-2*p[np.arange(len(y)), y]+1
            for m in shard['segments']:
                sl = slice(m['start'], m['end']); rid = m['root_id']
                if rid not in roots:
                    roots[rid] = dict(scene=m['scene_id'], cm=np.zeros((3, 3)), scores=np.zeros(3))
                r = roots[rid]
                cm = np.bincount(3*y[sl]+states[sl], weights=w[sl]*m['candidate_cells'], minlength=9).reshape(3, 3)
                r['cm'] += cm; pooled += cm*m['segments_in_root']
                r['scores'] += [np.sum(ce[sl]*w[sl]), np.sum(br[sl]*w[sl]), np.sum(w[sl])]
    for r in roots.values():
        scene[r['scene']].append(r)
    cms = np.array([np.mean([r['cm'] for r in rs], axis=0) for rs in scene.values()])
    scores = np.array([np.mean([r['scores'] for r in rs], axis=0) for rs in scene.values()]).mean(0)
    from .evaluate import rates
    return dict(roots=len(roots), scenes=len(scene), scene_balanced=rates(cms.mean(0)),
                logloss=float(scores[0]/scores[2]), brier=float(scores[1]/scores[2]),
                scope='inverse-inclusion cell estimates on fitting-only scene holdout; not final development')


def fit(split, strategy, name, device='cuda:0', features='base'):
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    random.seed(2424); np.random.seed(2424); torch.manual_seed(2424); torch.cuda.manual_seed_all(2424)
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    suffix = '_ray' if features.startswith('ray') else ''
    source = OUT / 'prepared' / (split+'_'+strategy+suffix) / 'manifest.json'
    m = json.loads(source.read_text())
    dest = OUT / 'training' / name; dest.mkdir(parents=True, exist_ok=False)
    cfg = dict(task='segment_conditioned_new_legal_observations', epochs=3, seed=2424, learning_rate=.0003,
               weight_decay=.0001, batch_size=8192, height_huber_weight=.2, gradient_clip=5.,
               device=device, strategy=strategy, features=features, split=split, warm_start=str(CHECKPOINT), warm_start_sha256=CHECKPOINT_SHA,
               git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE, text=True).strip(),
               source=snapshot(), protocol_sha256=sha(OUT/'protocol.json'),
               manifest=str(source), manifest_sha256=sha(source), roots=m['roots'], scenes=m['scenes'], rows=m['sampled_cells'])
    dump(dest/'config.json', cfg)
    model = load_model(CHECKPOINT, device, features)
    # Both training strategies are judged on exactly the same frozen windows.
    holdpath = OUT/'prepared'/('pilot_hold_mixed'+suffix)/'manifest.json'
    if split == 'pilot_fit':
        h = json.loads(holdpath.read_text()); dump(dest/'baseline_holdout.json', holdout(model, h, device))
    optim = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    rng = np.random.default_rng(2424); scale = m['sampled_cells']/m['roots']; learning = []; tick = time.perf_counter()
    for epoch in range(1, 4):
        model.train(); scores = np.zeros(3); seen = 0
        for index in rng.permutation(len(m['shards'])):
            shard = m['shards'][int(index)]; assert sha(shard['path']) == shard['sha256']
            with np.load(shard['path'], allow_pickle=False) as a:
                x = input_features(a['x'],model); y = a['y'].astype(np.int64); hgt = a['height'].astype(np.float32); w = a['weight']
            order = rng.permutation(len(y))
            for start in range(0, len(order), 8192):
                at = order[start:start+8192]
                logits, height = model(torch.from_numpy(x[at]).to(device))
                labels = torch.from_numpy(y[at]).to(device); weights = torch.from_numpy(w[at]).to(device)
                ce = F.cross_entropy(logits, labels, reduction='none')
                huber = F.smooth_l1_loss(height, torch.from_numpy(hgt[at]).to(device), reduction='none')*(labels>0)
                loss = ((ce+.2*huber)*weights).sum()*scale/8192
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite fit objective')
                optim.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optim.step()
                scores += [float((ce*weights).sum()), float((huber*weights).sum()), float(weights.sum())]; seen += len(at)
        assert seen == m['sampled_cells']
        record = dict(epoch=epoch, rows=seen, logloss=float(scores[0]/scores[2]), height_huber=float(scores[1]/scores[2]), seconds=time.perf_counter()-tick)
        learning.append(record); dump(dest/'learning.json', learning); print(json.dumps(record), flush=True)
    torch.save(dict(schema=MODEL_SCHEMA, mode='both', model_config=dict(width=48, input_features=model.net[0].in_features),
                    model={k: v.cpu() for k, v in model.state_dict().items()}, config=cfg, epoch=3), dest/'last.pt')
    dump(dest/'complete.json', dict(checkpoint_sha256=sha(dest/'last.pt'), seconds=time.perf_counter()-tick,
                                    parameters=sum(p.numel() for p in model.parameters()), peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    if split == 'pilot_fit':
        dump(dest/'holdout.json', holdout(model, json.loads(holdpath.read_text()), device))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--split', choices=['pilot_fit', 'full'], required=True)
    p.add_argument('--strategy', choices=['mixed', 'partition'], required=True); p.add_argument('--name', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--features', choices=['base', 'ray', 'ray_end'], default='base')
    fit(**vars(p.parse_args()))
