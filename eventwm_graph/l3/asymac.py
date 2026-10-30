"""Asymmetric actor-critic for query selection: privileged critic, legal actor (analysis / training).

Critic (privileged, training only): the EXACT short-horizon evaluation of every candidate recorded by the short-horizon
probe (``shortprobe.py``: the candidate's interval, then the default policy, executed in querygen's branch simulator
for 64 steps = 3.1 s).  It is turned into a utility per candidate: contact or STOP failure within the window 0, a goal
reached within the window 1, otherwise 0.5 + 0.4 x the candidate's legal route progress relative to the best and worst
running candidates of the state.  ``--critic-w`` chooses the window (32 or 64).

Actor (legal): a per-candidate MLP over ``shortsel.features`` (legal imagination only: each candidate imagined in 6
Stage 1 worlds with the camera model and with the released Sandbox's people, contact statistics, progress, STOPs over
16 / 32 / 64 steps, family, differences to the default) with a softmax over the candidate set.  Trained by the
all-action policy gradient  grad sum_a pi(a|o) (Q(s, a) - sum_b pi(b|o) Q(s, b))  + entropy -- the actor maximises the
critic's expected utility; it never sees privileged inputs.  Iteration 0 uses the floor's states; later iterations
add states visited by the actor itself (DAgger-style state distribution, the critic re-evaluating every candidate).

    python scripts/l3_run.py eventwm_graph.l3.asymac --train output/shortprobe/orca_train --eval output/shortprobe/orca_dev \
        [--save output/asymac/v1]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def utilities(cands, w=64):
    """Privileged critic: utility per candidate from the exact window outcome."""
    from .shortsel import _fails
    ctg = []
    for c in cands:
        v = c['real']['ctg'].get(str(w))
        ctg.append(v if v is not None else math.nan)
    run = [i for i, c in enumerate(cands) if not _fails(c, w) and not (c['real']['outcome'] == 'success' and c['real']['steps'] <= w)
           and not math.isnan(ctg[i])]
    lo = min([ctg[i] for i in run], default=0.)
    hi = max([ctg[i] for i in run], default=0.)
    out = []
    for i, c in enumerate(cands):
        if _fails(c, w):
            out.append(0.)
        elif c['real']['outcome'] == 'success' and c['real']['steps'] <= w:
            out.append(1.)
        elif i in run:
            out.append(0.5+0.4*((hi-ctg[i])/(hi-lo) if hi > lo else 1.))
        else:
            out.append(0.5)
    return np.asarray(out, np.float32)


def build(states, w):
    from .shortsel import features
    X, U, D = [], [], []
    for s in states:
        di = s['labels'].index(s['default'])
        X.append(features(s['cands'], s['ctg0'], di).astype(np.float32))
        U.append(utilities(s['cands'], w))
        D.append(di)
    return X, U, D


class Actor:
    def __init__(self, n_in, hidden=128, seed=0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.mu = None
        self.sd = None

    def scores(self, X):
        import torch
        with torch.no_grad():
            return self.net(torch.as_tensor((X-self.mu)/self.sd, dtype=torch.float32)).squeeze(-1).numpy()


def train(Xs, Us, epochs=60, lr=1e-3, beta=0.01, hidden=128, seed=0, batch=64):
    import torch
    allX = np.concatenate(Xs)
    actor = Actor(allX.shape[1], hidden, seed)
    actor.mu = allX.mean(0)
    actor.sd = allX.std(0)+1e-6
    opt = torch.optim.Adam(actor.net.parameters(), lr=lr, weight_decay=1e-4)
    Xt = [torch.as_tensor((x-actor.mu)/actor.sd, dtype=torch.float32) for x in Xs]
    Ut = [torch.as_tensor(u) for u in Us]
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        order = rng.permutation(len(Xt))
        tot = 0.
        for b in range(0, len(order), batch):
            loss = 0.
            for i in order[b:b+batch]:
                logit = actor.net(Xt[i]).squeeze(-1)
                pi = torch.softmax(logit, 0)
                u = Ut[i]
                adv = u-(pi.detach()*u).sum()
                ent = -(pi*torch.log(pi+1e-9)).sum()
                loss = loss-(pi*adv).sum()-beta*ent
            loss = loss/max(len(order[b:b+batch]), 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
    return actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--eval', default=None)
    ap.add_argument('--critic-w', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--beta', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save', default=None)
    ap.add_argument('--margins', default='0,0.5,1,2')
    a = ap.parse_args()
    from .shortsel import load_states, evaluate_scores
    tr = load_states(a.train)
    Xs, Us, _ = build(tr, a.critic_w)
    print('train states %d candidates %d' % (len(Xs), sum(len(x) for x in Xs)), flush=True)
    actor = train(Xs, Us, a.epochs, beta=a.beta, seed=a.seed)
    if a.eval:
        ev = load_states(a.eval)
        Xe, Ue, De = build(ev, a.critic_w)
        sc = np.concatenate([actor.scores(x) for x in Xe])
        for m in (float(x) for x in a.margins.split(',')):
            for w in (32, 64):
                r = evaluate_scores(ev, sc, m, w)
                print('  margin %.1f window %d: exact category = best %.3f  ctg regret %.3f  critical %d fixed %d  breaks %d  '
                      'deviations %.2f' % (m, w, r['category_ok'], r['ctg_regret'], r['critical'], r['fixed'], r['breaks'],
                                           r['deviations']))
    if a.save:
        import torch
        Path(a.save).mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=actor.net.state_dict(), mu=actor.mu, sd=actor.sd, n_in=len(actor.mu)), Path(a.save, 'actor.pt'))
        Path(a.save, 'meta.json').write_text(json.dumps(dict(critic_w=a.critic_w, train=a.train, epochs=a.epochs, beta=a.beta)))


if __name__ == '__main__':
    main()
