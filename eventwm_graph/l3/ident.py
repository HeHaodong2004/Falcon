"""Identifiability of critical decisions from the stored consequence graphs of a run (offline; labels from cfonline).

For every labelled decision of a run's episodes (``<run>/cf_online``), the episode's own packed graph at that
decision is the input and the counterfactual outcomes of its candidates the labels.  Small critics (the shared
network, success / collision heads only) are trained on a scene-level split; held-out metrics on CRITICAL states
(the default fails, some candidate succeeds): top-1 success of the critic vs a random alternative, within-state AUC.

    python scripts/l3_run.py eventwm_graph.l3.ident --run output/runs/R7 [--seeds 3 --steps 3000]
"""
import argparse
import json
from pathlib import Path

import numpy as np


SEM_BASE = ('goal', 'st_goal', 'continue', 'wait_short', 'wait_long', 'wait_clear', 'turn', 'advance', 'alt_route',
            'frontier', 'creep', 'rot', 'sidestep', 'retreat', 'yield', 'look', 'pass', 'approach', 'stop', 'goal_slow',
            'kturn')
SEM = list(SEM_BASE)+['mod_tc', 'mod_th', 'mod_vc', 'mod_s', 'dir_left', 'dir_right', 'dir_behind', 'magnitude']


def query_semantics(label):
    """Legal query-conditioned descriptor from the generator's label: category, execution-style modifiers (tight +
    close copies, wall-band thru copies, contact-level copies, short intervals), direction, magnitude."""
    import re
    v = np.zeros(len(SEM), np.float32)
    lab = label or ''
    base = None
    for b in sorted(SEM_BASE, key=len, reverse=True):
        if lab == b or lab.startswith(b+'_') or lab.startswith(b) and b in ('rot', 'creep', 'alt_route', 'frontier', 'turn', 'look', 'pass', 'sidestep'):
            base = b
            break
    if base is None and lab.startswith('st_goal'):
        base = 'st_goal'
    if base is not None:
        v[SEM.index(base)] = 1.
    for m, col in (('_tc', 'mod_tc'), ('_th', 'mod_th'), ('_vc', 'mod_vc'), ('_s', 'mod_s')):
        if lab.endswith(m):
            v[SEM.index(col)] = 1.
    for dname in ('left', 'right', 'behind'):
        if dname in lab:
            v[SEM.index('dir_'+dname)] = 1.
    num = re.findall(r'(\d+)$', lab)
    if num and base in ('rot', 'creep'):
        v[SEM.index('magnitude')] = float(num[0])/10.
    elif base in ('wait_short', 'wait_long', 'wait_clear'):
        v[SEM.index('magnitude')] = {'wait_short': 6, 'wait_long': 16, 'wait_clear': 48}[base]/48.
    return v


def _row(g, labels, cands, default):
    K = g['cand'].shape[0]
    y = np.zeros((K, 3), np.float32)
    sem = np.zeros((K, len(SEM)), np.float32)
    for k, c in enumerate(cands[:K]):
        if c.get('skipped'):
            y[k] = [0., 0., 1.]                      # unlabelled placeholder: excluded like a cut continuation
            continue
        y[k] = [c['outcome'] == 'success', c['outcome'] == 'human_collision', bool(c.get('alive'))]
    for k, l in enumerate(labels[:K]):
        sem[k] = query_semantics(l)
    return y, sem


def dataset_oracle(warm, cf):
    """Oracle-state graphs (warm-up replays) with their counterfactual labels (cflabel.py)."""
    from .actor import read_episode
    from .batch import GRAPH_KEYS
    from .warmup import cf_index
    cfi = cf_index(cf)
    X, Y, M, D, S = [], [], [], [], []
    for p in sorted(Path(warm).glob('*.ep')):
        d = read_episode(p)
        sp = d['spec']
        lab = cfi.get('%s/%s/%s/%s|%s' % (sp['dataset'], sp['split'], sp['scene'], sp['episode_id'], d['config']))
        if lab is None or not d['rows'].get('packed'):
            continue
        rows = d['rows']
        for t, st in lab.items():
            if t >= len(rows['packed']) or st['labels'] != rows['labels'][t]:
                continue
            y, sem = _row(rows['packed'][t], st['labels'], st['candidates'], rows['default'][t])
            X.append(rows['packed'][t]); Y.append(y); M.append(sem); D.append(int(rows['default'][t])); S.append(sp['scene'])
    return X, Y, M, D, S


def dataset(run):
    from .actor import read_episode
    from .batch import GRAPH_KEYS
    X, Y, M, D, S = [], [], [], [], []
    for f in sorted(Path(run, 'cf_online').glob('*.json')):
        rec = json.loads(f.read_text())
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
            g = rows['packed'][t]
            y, sem = _row(g, st['labels'], st['candidates'], rows['default'][t])
            X.append(g); Y.append(y); D.append(int(rows['default'][t])); S.append(d['spec']['scene']); M.append(sem)
    return X, Y, M, D, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None)
    ap.add_argument('--warm', default=None)
    ap.add_argument('--cf', default=None)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--semantics', action='store_true', help='append label-derived query semantics to the candidates')
    ap.add_argument('--crit-frac', type=float, default=0., help='share of each batch drawn from critical states')
    ap.add_argument('--rank', type=float, default=0., help='within-state ranking loss weight (success > failure)')
    ap.add_argument('--tag', default='ident')
    a = ap.parse_args()
    import torch
    import torch.nn.functional as Fn
    from .model import ConsequenceGraphNet, ModelConfig, mlp
    from .batch import to_torch, GRAPH_KEYS
    from . import compiler as C
    X, Y, M, D, S = [], [], [], [], []
    if a.run:
        for lst, new in zip((X, Y, M, D, S), dataset(a.run)):
            lst.extend(new)
    if a.warm:
        for lst, new in zip((X, Y, M, D, S), dataset_oracle(a.warm, a.cf)):
            lst.extend(new)
    G = {k: np.stack([x[k] for x in X]) for k in GRAPH_KEYS}
    Y = np.stack(Y); M = np.stack(M); D = np.asarray(D); S = np.asarray(S)
    if a.semantics:
        G['cand'] = np.concatenate([G['cand'].astype(np.float32), M], -1)
    K = Y.shape[1]
    ok = (Y[..., 2] < 0.5) & G['cand_mask']
    crit = np.asarray([ok[j, D[j]] and Y[j, D[j], 0] < 0.5 and ((Y[j, :, 0] > 0.5) & ok[j]).any() for j in range(len(D))])
    print('labelled states', len(D), 'critical', int(crit.sum()), 'scenes', len(set(S)), 'semantics', a.semantics, flush=True)
    out = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        scenes = sorted(set(S))
        test = set(rng.choice(scenes, max(1, len(scenes)//4), replace=False))
        te = np.asarray([s in test for s in S]); tr = ~te
        torch.manual_seed(seed)
        net = ConsequenceGraphNet(ModelConfig()).to(a.device)
        if a.semantics:
            net.enc['cand'] = mlp(G['cand'].shape[-1], net.cfg.hidden, net.cfg.hidden).to(a.device)
            net.keep_cand = torch.ones(G['cand'].shape[-1], device=a.device)
        opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
        Gtr = to_torch({k: v[tr] for k, v in G.items()}, a.device); Ytr = torch.as_tensor(Y[tr], device=a.device)
        ctr = np.flatnonzero(crit[tr]); n = int(tr.sum())
        for it in range(a.steps):
            nb = 128
            if a.crit_frac > 0 and len(ctr):
                nc = int(nb*a.crit_frac)
                bi = np.concatenate([rng.integers(0, n, nb-nc), rng.choice(ctr, nc)])
            else:
                bi = rng.integers(0, n, nb)
            b = torch.as_tensor(bi, device=a.device)
            o = net({k: v[b] for k, v in Gtr.items()})
            mk = Gtr['cand_mask'][b] & (Ytr[b][..., 2] < 0.5)
            loss = Fn.binary_cross_entropy_with_logits(o['succ'][mk], Ytr[b][..., 0][mk]) + \
                Fn.binary_cross_entropy_with_logits(o['coll'][mk], Ytr[b][..., 1][mk])
            if a.rank > 0:
                sc = o['succ']; ys = Ytr[b][..., 0]
                pos = mk & (ys > 0.5); neg = mk & (ys < 0.5)
                diff = sc.unsqueeze(2)-sc.unsqueeze(1)                       # [B, K, K]: score(i) - score(j)
                pair = pos.unsqueeze(2) & neg.unsqueeze(1)
                if bool(pair.any()):
                    loss = loss+a.rank*Fn.softplus(1.-diff)[pair].mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(to_torch({k: v[te] for k, v in G.items()}, a.device))['succ']).cpu().numpy()
        Yt, Dt, Mk = Y[te], D[te], ok[te]
        top, rnd, auc, dflt_mixed, top_mixed = [], [], [], [], []
        for j in range(len(Yt)):
            m = Mk[j]; s_ = Yt[j, :, 0] > 0.5
            if not ((s_ & m).any() and ((~s_) & m).any()):
                continue
            k = int(np.argmax(np.where(m, p[j], -1)))
            top_mixed.append(float(s_[k])); dflt_mixed.append(float(s_[Dt[j]]) if m[Dt[j]] else 0.)
            if not (m[Dt[j]] and not s_[Dt[j]]):
                continue
            alt = m.copy(); alt[Dt[j]] = False
            top.append(float(s_[k])); rnd.append(float(s_[alt].mean()))
            pos, neg = p[j][m & s_], p[j][m & ~s_]
            auc.append(float((pos[:, None] > neg[None, :]).mean()+0.5*(pos[:, None] == neg[None, :]).mean()))
        # across states: does the critic know WHEN the default fails?  (AUC of the default's predicted success) and a
        # gated policy: deviate to the critic's best (or a random) alternative only when P(default succeeds) < theta
        dsc, dlab, gate = [], [], {}
        rng_g = np.random.default_rng(100+seed)
        for j in range(len(Yt)):
            m = Mk[j]
            if not m[Dt[j]] or m.sum() < 2:
                continue
            s_ = Yt[j, :, 0] > 0.5
            dsc.append(p[j, Dt[j]]); dlab.append(float(s_[Dt[j]]))
            alt = m.copy(); alt[Dt[j]] = False
            kb = int(np.argmax(np.where(alt, p[j], -1)))
            kr = int(rng_g.choice(np.flatnonzero(alt)))
            for th in (0.0, 0.2, 0.35, 0.5, 0.65):
                dev = p[j, Dt[j]] < th
                g = gate.setdefault(th, [[], [], 0])
                g[0].append(float(s_[kb] if dev and p[j, kb] > p[j, Dt[j]] else s_[Dt[j]]))
                g[1].append(float(s_[kr] if dev else s_[Dt[j]]))
                g[2] += int(dev)
        dsc, dlab = np.asarray(dsc), np.asarray(dlab)
        if len(dlab) and 0 < dlab.sum() < len(dlab):
            pos, neg = dsc[dlab > 0.5], dsc[dlab < 0.5]
            d_auc = float((pos[:, None] > neg[None, :]).mean()+0.5*(pos[:, None] == neg[None, :]).mean())
        else:
            d_auc = None
        out.append(dict(seed=seed, critical=len(top), critic_top1=float(np.mean(top)) if top else None,
                        random_alt=float(np.mean(rnd)) if rnd else None, auc=float(np.mean(auc)) if auc else None,
                        mixed=len(top_mixed), mixed_top1=float(np.mean(top_mixed)) if top_mixed else None,
                        mixed_default=float(np.mean(dflt_mixed)) if dflt_mixed else None,
                        default_auc=d_auc, n_default=int(len(dlab)), default_success=float(dlab.mean()) if len(dlab) else None,
                        gate={str(th): dict(critic=round(float(np.mean(g[0])), 4), random=round(float(np.mean(g[1])), 4),
                                            deviate=round(g[2]/max(len(g[0]), 1), 3)) for th, g in gate.items()}))
        print(json.dumps(out[-1]), flush=True)
    agg = {k: float(np.mean([o[k] for o in out if o.get(k) is not None])) for k in
           ('critic_top1', 'random_alt', 'auc', 'mixed_top1', 'mixed_default', 'default_auc', 'default_success')}
    agg['gate'] = {th: {k: float(np.mean([o['gate'][th][k] for o in out])) for k in ('critic', 'random', 'deviate')}
                   for th in out[0]['gate']}
    agg['critical_states'] = int(sum(o['critical'] for o in out))
    print('MEAN', json.dumps(agg), flush=True)


if __name__ == '__main__':
    main()
