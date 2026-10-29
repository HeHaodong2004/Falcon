"""Approximate policy iteration, step 1: a critic of Q^default learned by supervised regression on the on-policy
counterfactual labels (every candidate of a labelled state, then the default to the episode end), exported as a
normal policy checkpoint (``evaluate.py --policy``).  Greedy selection over it is the learned counterpart of the
rollout policy (``rolloutpolicy.py``: the same improvement with an exact evaluator).

Inputs are the legal graphs the actors recorded (v4); targets per labelled candidate: success, human collision,
steps / path / social of the continuation (decomposed heads, network units as in the learner); continuations cut by
the labeller's cap train the collision head only.  Scene-level hold-out for model selection.

    L3_LAYOUT=v4 python scripts/l3_run.py eventwm_graph.l3.apicritic --runs output/runs/R9,output/runs/R10 \
        --out output/api/critic_v1.pt [--init output/runs/R10/learner_state.pt --steps 20000]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def collect(runs):
    from .actor import read_episode
    from .batch import KMAX, CUM_SCALE
    X, L, M, S = [], [], [], []
    for run in runs:
        for f in sorted(Path(run, 'cf_online').glob('*.json')):
            try:
                rec = json.loads(f.read_text())
            except Exception:
                continue
            sts = {int(k): v for k, v in rec['states'].items() if 'candidates' in v}
            if not sts:
                continue
            try:
                d = read_episode(Path(run, 'spool', rec['episode']))
            except Exception:
                continue
            rows = d['rows']
            for t, st in sts.items():
                if t >= len(rows['packed']) or st['labels'] != rows['labels'][t]:
                    continue
                lab = np.zeros((KMAX, 5), np.float32)
                msk = np.zeros((KMAX, 2), bool)          # [complete continuation, any outcome]
                for k, c in enumerate(st['candidates'][:KMAX]):
                    alive = bool(c.get('alive'))
                    lab[k] = [float(c['outcome'] == 'success'), float(c['outcome'] == 'human_collision'),
                              c['steps']/CUM_SCALE[2], c['path']/CUM_SCALE[3], c['social']/CUM_SCALE[4]]
                    msk[k] = [not alive, True]
                X.append(rows['packed'][t]); L.append(lab); M.append(msk); S.append(d['spec']['scene'])
    return X, np.stack(L), np.stack(M), np.asarray(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--init', default=None)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--holdout', type=float, default=0.15)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as Fn
    from dataclasses import asdict
    from .model import ConsequenceGraphNet, ModelConfig, Weights, HEADS
    from .batch import to_torch, GRAPH_KEYS
    X, L, M, S = collect(a.runs.split(','))
    G = {k: np.stack([x[k] for x in X]) for k in GRAPH_KEYS}
    rng = np.random.default_rng(a.seed)
    scenes = sorted(set(S))
    test = set(rng.choice(scenes, max(1, int(len(scenes)*a.holdout)), replace=False))
    te = np.asarray([s in test for s in S]); tr = ~te
    print(json.dumps(dict(states=len(S), train=int(tr.sum()), holdout=int(te.sum()), scenes=len(scenes))), flush=True)
    dev = torch.device(a.device)
    net = ConsequenceGraphNet(ModelConfig()).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        sd = ck.get('net') or ck.get('state_dict')
        own = net.state_dict()
        net.load_state_dict({k: (sd[k] if k in sd and sd[k].shape == v.shape else v) for k, v in own.items()})
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    Gt = to_torch(G, dev)
    Lt = torch.as_tensor(L, device=dev); Mt = torch.as_tensor(M, device=dev)
    itr, ite = np.flatnonzero(tr), np.flatnonzero(te)

    def loss_of(idx, train=True):
        b = torch.as_tensor(idx, device=dev)
        o = net({k: v[b] for k, v in Gt.items()})
        cm = Gt['cand_mask'][b]
        full = Mt[b][..., 0] & cm
        anyo = Mt[b][..., 1] & cm
        lab = Lt[b]
        ls = Fn.binary_cross_entropy_with_logits(o['succ'][full], lab[..., 0][full]) if bool(full.any()) else 0.
        lc = Fn.binary_cross_entropy_with_logits(o['coll'][anyo], lab[..., 1][anyo]) if bool(anyo.any()) else 0.
        lr = sum(Fn.smooth_l1_loss(o[h][full], lab[..., j][full], beta=0.5) for j, h in ((2, 'steps'), (3, 'path'), (4, 'soc'))) \
            if bool(full.any()) else 0.
        return ls+lc+0.5*lr, o, b

    best, best_sd = 1e9, None
    for it in range(a.steps):
        net.train()
        loss, _, _ = loss_of(rng.choice(itr, a.batch))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it+1) % 1000 == 0:
            net.eval()
            with torch.no_grad():
                tot, n = 0., 0
                for j in range(0, len(ite), 256):
                    l_, _, _ = loss_of(ite[j:j+256], False)
                    tot += float(l_)*len(ite[j:j+256]); n += len(ite[j:j+256])
            v = tot/max(n, 1)
            print(json.dumps(dict(step=it+1, train_loss=round(float(loss), 4), holdout_loss=round(v, 4))), flush=True)
            if v < best:
                best, best_sd = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
    ck = dict(state_dict=best_sd, model_config=asdict(ModelConfig()), weights=asdict(Weights(collision=2., path=0.15)),
              version=1, updates=a.steps, tau=0., epsilon=0., anchor=0., time=0., cfg_ids={}, explore=None,
              note='Q^default critic from counterfactual labels (apicritic.py)', holdout_loss=best)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.out)
    print('saved', a.out, 'holdout', best, flush=True)


if __name__ == '__main__':
    main()
