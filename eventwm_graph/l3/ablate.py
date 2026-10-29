"""Representation ablation with the data held fixed (pre-registered E3, offline part).

Critics (the shared ConsequenceGraphNet, success and collision heads) are trained on the SAME decisions -- a snapshot
of a run's spool -- to predict the Monte Carlo outcome of the taken candidate (episode success / human collision
after it), once with every input and once per removed feature group (``compiler.GROUPS``: zeroed at the input).
Scene-level split; held-out log-loss and AUC of the taken candidate's success / collision, and (where on-policy
counterfactual labels exist) within-state ranking AUC over the labelled candidates.  A group whose removal does not
hurt held-out prediction is not used by the critic (e.g. Stage 2 maps: dropping them would save their per-decision
cost).

    L3_LAYOUT=v4 python scripts/l3_run.py eventwm_graph.l3.ablate --run output/runs/R9 --groups none,map,human,info,query,route \
        [--max-episodes 4000 --steps 6000 --seeds 2]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(run, max_episodes):
    from .actor import read_episode
    from .batch import pack, GRAPH_KEYS, KMAX
    files = sorted(Path(run, 'spool').glob('*.ep'), key=lambda p: p.stat().st_mtime)[-max_episodes:]
    cfd = Path(run, 'cf_online')
    X, A, Y, S, CF = [], [], [], [], []
    for p in files:
        try:
            d = read_episode(p)
        except Exception:
            continue
        rows = d['rows']
        if not rows.get('packed'):
            continue
        succ = float(d['metrics']['outcome'] == 'success')
        coll = float(d['metrics']['outcome'] == 'human_collision')
        cf = {}
        f = cfd/(p.name+'.json')
        if f.exists():
            try:
                rec = json.loads(f.read_text())
                cf = {int(k): v for k, v in rec['states'].items() if 'candidates' in v}
            except Exception:
                cf = {}
        for t, g in enumerate(rows['packed']):
            X.append(g); A.append(int(rows['actions'][t])); Y.append((succ, coll)); S.append(d['spec']['scene'])
            lab = None
            st = cf.get(t)
            if st is not None and st['labels'] == rows['labels'][t]:
                lab = np.full(KMAX, -1., np.float32)
                for k, c in enumerate(st['candidates'][:KMAX]):
                    if not c.get('alive'):
                        lab[k] = float(c['outcome'] == 'success')
            CF.append(lab)
    G = {k: np.stack([x[k] for x in X]) for k in GRAPH_KEYS}
    return G, np.asarray(A), np.asarray(Y, np.float32), np.asarray(S), CF


def auc(score, y):
    pos, neg = score[y > 0.5], score[y < 0.5]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order)+1)
    return float((ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--groups', default='none,map,human,info,query,route')
    ap.add_argument('--max-episodes', type=int, default=4000)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as Fn
    from .model import ConsequenceGraphNet, ModelConfig
    from .batch import to_torch
    G, A, Y, S, CF = load(a.run, a.max_episodes)
    print('decisions', len(A), 'scenes', len(set(S)), 'cf states', sum(c is not None for c in CF), flush=True)
    results = {}
    for grp in a.groups.split(','):
        drop = () if grp == 'none' else tuple(grp.split('+'))
        rs = []
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            scenes = sorted(set(S))
            test = set(rng.choice(scenes, max(1, len(scenes)//5), replace=False))
            te = np.asarray([s in test for s in S]); tr = ~te
            torch.manual_seed(seed)
            net = ConsequenceGraphNet(ModelConfig(drop=drop)).to(a.device)
            opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
            itr = np.flatnonzero(tr)
            for it in range(a.steps):
                bi = rng.choice(itr, 256)
                g = to_torch({k: v[bi] for k, v in G.items()}, a.device)
                o = net(g)
                act = torch.as_tensor(A[bi], device=a.device)
                y = torch.as_tensor(Y[bi], device=a.device)
                ls = Fn.binary_cross_entropy_with_logits(o['succ'].gather(1, act[:, None]).squeeze(1), y[:, 0])
                lc = Fn.binary_cross_entropy_with_logits(o['coll'].gather(1, act[:, None]).squeeze(1), y[:, 1])
                opt.zero_grad(); (ls+lc).backward(); opt.step()
            net.eval()
            ite = np.flatnonzero(te)
            ps, pc, pr = [], [], []
            with torch.no_grad():
                for j in range(0, len(ite), 512):
                    bi = ite[j:j+512]
                    o = net(to_torch({k: v[bi] for k, v in G.items()}, a.device))
                    ps.append(torch.sigmoid(o['succ']).cpu().numpy()); pc.append(torch.sigmoid(o['coll']).cpu().numpy())
            ps = np.concatenate(ps); pc = np.concatenate(pc)
            at = A[ite]
            s_taken = ps[np.arange(len(ite)), at]; c_taken = pc[np.arange(len(ite)), at]
            eps = 1e-6
            ll = float(-np.mean(Y[ite, 0]*np.log(s_taken+eps)+(1-Y[ite, 0])*np.log(1-s_taken+eps)))
            within = []
            for j, i in enumerate(ite):
                lab = CF[i]
                if lab is None:
                    continue
                m = lab >= 0
                if m.sum() >= 2:
                    u = auc(ps[j][m], lab[m])
                    if u is not None:
                        within.append(u)
            rs.append(dict(seed=seed, succ_logloss=ll, succ_auc=auc(s_taken, Y[ite, 0]), coll_auc=auc(c_taken, Y[ite, 1]),
                           cf_within_auc=float(np.mean(within)) if within else None, cf_states=len(within)))
            print(grp, json.dumps(rs[-1]), flush=True)
        results[grp] = {k: float(np.mean([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None
                        for k in ('succ_logloss', 'succ_auc', 'coll_auc', 'cf_within_auc')}
        print('MEAN', grp, json.dumps(results[grp]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
