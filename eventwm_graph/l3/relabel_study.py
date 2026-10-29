"""Representation study on labelled critical states: does continuation imagination (Compiler v2) predict which
candidates succeed better than v1b?  (Offline; the labels are the on-policy counterfactual outcomes of cfonline.py.)

``build``: replays labelled episodes of a run (BranchEpisode + an in-process Sandbox with continuation imagination,
the recorded decisions; actions checked against the record) and compiles the v2 graph (v1b = v2 without the
continuation columns) at every labelled decision; writes one .npz per episode.
``fit``: trains small critics to predict P(success of candidate, then the default policy) from v1b vs v2 inputs on
an episode-level split; reports held-out within-state AUC and top-1 success on states where the choice matters.

    L3_LAYOUT=v2 python scripts/l3_run.py eventwm_graph.l3.relabel_study build --run output/runs/R5 --out DIR --shard i --shards n
    L3_LAYOUT=v2 python scripts/l3_run.py eventwm_graph.l3.relabel_study fit --out DIR
"""
import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np


def build(run, out, shard, shards, continuation=24, limit=0):
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.sandbox.core import WorldModelSandbox, SandboxConfig, Candidate
    from eventwm_graph.sandbox.episode import BranchEpisode, generator
    from eventwm_graph.querygen.oracle import horizon_for
    from eventwm_graph.querygen.native import EpisodeSpec, MAX_EPISODE_STEPS
    from .actor import read_episode
    from .env import ConfigSpec
    from .cfonline import _query
    from . import compiler as C
    from .batch import pack
    assert C.LAYOUT == 'v2', 'run with L3_LAYOUT=v2'
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sb = WorldModelSandbox.from_release(SandboxConfig(materialize=False, continuation=continuation),
                                        map_device='cuda:%d' % (shard % 2))
    comp = C.GraphCompiler(brake_tail=True)
    files = sorted(Path(run, 'cf_online').glob('*.json'))
    files = [f for f in files if zlib.crc32(f.name.encode()) % shards == shard]
    if limit:
        files = files[:limit]
    for f in files:
        name = f.name[:-len('.json')]
        if (out/(name+'.npz')).exists():
            continue
        lab = json.loads(f.read_text())
        sts = {int(k): v for k, v in lab['states'].items() if 'candidates' in v}
        if not sts:
            continue
        d = read_episode(Path(run, 'spool', name))
        m = d['metrics']
        cfg = ConfigSpec.parse(d['config'])
        gen = generator(cfg.gen, cfg.dynamics)
        ep = BranchEpisode(EpisodeSpec(**d['spec']), cfg.planner, cfg.dynamics, range_sensor='fast-driver', gc_freeze=True)
        rows, ok = [], True
        try:
            for i, dec in enumerate(m['decision_list']):
                if i > max(sts) or ep.over:
                    break
                if i in sts:
                    queries, diag = gen.propose(ep.legal, ep.execution)
                    labels = [q.label for q in queries]
                    if labels != sts[i]['labels']:
                        ok = False
                        break
                    cands = [Candidate(None, cfg.planner, cfg.dynamics, execution=ep.execution, level=ep.level)
                             if q.kind == 'continue' else Candidate(q, cfg.planner, cfg.dynamics, level=ep.level)
                             for q in queries]
                    res = sb.evaluate(ep.decision_state(), ep.legal, cands)
                    g = pack(comp.compile(res, ep.legal, queries, diag, ep.steps))
                    y = np.asarray([[c['outcome'] == 'success', c['outcome'] == 'human_collision', c['outcome'] is None]
                                    for c in sts[i]['candidates']], np.float32)
                    rows.append((i, g, y, int(d['rows']['actions'][i]), int(d['rows']['default'][i])))
                ep.run(None if dec['cont'] else _query(dec), horizon=int(dec['horizon']))
            acts = [x if isinstance(x, str) else [float(v) for v in x] for x in ep.state.actions]
            if acts != m['actions'][:len(acts)]:
                ok = False
        finally:
            ep.close()
        if not ok or not rows:
            print(json.dumps(dict(episode=name, skipped=True)), flush=True)
            continue
        arr = {k: np.stack([r[1][k] for r in rows]) for k in rows[0][1]}
        K = arr['cand'].shape[1]
        Y = np.zeros((len(rows), K, 3), np.float32)
        for j, r in enumerate(rows):
            Y[j, :len(r[2])] = r[2]
        np.savez_compressed(out/(name+'.npz'), decisions=np.asarray([r[0] for r in rows]), y=Y,
                            chosen=np.asarray([r[3] for r in rows]), default=np.asarray([r[4] for r in rows]),
                            scene=d['spec']['scene'], config=d['config'], outcome=m['outcome'], **arr)
        print(json.dumps(dict(episode=name, states=len(rows))), flush=True)
    sb.close()


def fit(out, device='cuda:1', steps=3000, seeds=3):
    import torch
    import torch.nn.functional as Fn
    from . import compiler as C
    from .model import ConsequenceGraphNet, ModelConfig
    from .batch import to_torch, GRAPH_KEYS
    files = sorted(Path(out).glob('*.npz'))
    data = [dict(np.load(f, allow_pickle=True)) for f in files]
    v2cols = [C.CAND.index(c) for c in C.CAND_V2]
    res = {}
    for variant in ('v1b', 'v2'):
        per_seed = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            scenes = sorted({str(x['scene']) for x in data})
            test_scenes = set(rng.choice(scenes, max(1, len(scenes)//4), replace=False))
            tr = [x for x in data if str(x['scene']) not in test_scenes]
            te = [x for x in data if str(x['scene']) in test_scenes]

            def cat(xs):
                g = {k: np.concatenate([x[k] for x in xs]) for k in GRAPH_KEYS}
                if variant == 'v1b':
                    g['cand'] = g['cand'].copy(); g['cand'][..., v2cols] = 0
                return g, np.concatenate([x['y'] for x in xs])
            gtr, ytr = cat(tr); gte, yte = cat(te)
            torch.manual_seed(seed)
            net = ConsequenceGraphNet(ModelConfig()).to(device)
            head = torch.nn.Linear(1, 1).to(device)          # unused; the success head of the net is trained directly
            opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
            G = to_torch(gtr, device); Y = torch.as_tensor(ytr, device=device)
            Gt = to_torch(gte, device); Yt = torch.as_tensor(yte, device=device)
            n = len(Y)
            for it in range(steps):
                b = torch.as_tensor(rng.integers(0, n, 128), device=device)
                gb = {k: v[b] for k, v in G.items()}
                o = net(gb)
                mk = gb['cand_mask']
                loss = Fn.binary_cross_entropy_with_logits(o['succ'][mk], Y[b][..., 0][mk])+\
                    Fn.binary_cross_entropy_with_logits(o['coll'][mk], Y[b][..., 1][mk])
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                o = net(Gt)
                p = torch.sigmoid(o['succ']).cpu().numpy()
            mk = gte['cand_mask']
            aucs, top1, top1_def, top1_rand = [], [], [], []
            for j in range(len(yte)):
                m_ = mk[j]; s = yte[j, :, 0] > 0.5
                pos = p[j][m_ & s]; neg = p[j][m_ & ~s]
                if len(pos) and len(neg):
                    aucs.append(float((pos[:, None] > neg[None, :]).mean()+0.5*(pos[:, None] == neg[None, :]).mean()))
                    k = int(np.argmax(np.where(m_, p[j], -1)))
                    top1.append(float(s[k])); top1_rand.append(float(s[m_].mean()))
            per_seed.append(dict(train_states=int(len(ytr)), test_states=int(len(yte)), mixed=len(aucs),
                                 auc=float(np.mean(aucs)) if aucs else None, top1=float(np.mean(top1)) if top1 else None,
                                 random_top1=float(np.mean(top1_rand)) if top1_rand else None))
        res[variant] = dict(seeds=per_seed, auc=float(np.mean([r['auc'] for r in per_seed if r['auc'] is not None])),
                            top1=float(np.mean([r['top1'] for r in per_seed if r['top1'] is not None])))
    (Path(out)/'fit.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['build', 'fit'])
    ap.add_argument('--run', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--device', default='cuda:1')
    a = ap.parse_args()
    if a.mode == 'build':
        build(a.run, a.out, a.shard, a.shards, limit=a.limit)
    else:
        fit(a.out, a.device, a.steps)


if __name__ == '__main__':
    main()
