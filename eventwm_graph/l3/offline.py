"""Offline diagnostics of the representation on counterfactually labelled Oracle states (no environment needed).

Episode-level split of the warm-up states (Oracle replays with counterfactual labels).  A critic is fitted on the
training episodes with the warm-up losses (counterfactual regression of every labelled candidate, TD on the Oracle
demonstrations); on the held-out episodes we measure, per state with at least one successful and one failing
candidate (``mixed`` states -- where the choice matters for success):
  * top-1 success: the chosen candidate's "candidate, then default policy" outcome is a success;
  * within-state AUC: P(a successful candidate scores above a failing one);
and on all states the step regret (remaining steps of the chosen candidate minus the best successful one).
Baselines: the default (no-selection) choice, the hand-written imagination heuristic, the Oracle's own choice, random.
Feature-group ablations (``--drop``) and the flat set model (``--flat``) reuse the same split and budget.

    python scripts/l3_run.py eventwm_graph.l3.offline --warm output/warm/replay/spool --cf output/warm/cf --out DIR \
        [--drop human,map] [--flat] [--steps 4000]
"""
import argparse
import json
from pathlib import Path
import zlib

import numpy as np
import torch

from .batch import Ring, CUM_SCALE
from .model import ModelConfig, Weights


def split_rings(warm, cf, test_frac=0.25, seed=0):
    from .actor import read_episode
    from .warmup import cf_index
    from .learner import Learner
    cfi = cf_index(cf)
    tr, te = [], []
    for p in sorted(Path(warm).glob('*.ep')):
        d = read_episode(p)
        m = d['metrics']
        if m.get('diverged') or m['outcome'] != 'success':
            continue
        key = '%s/%s/%s/%s' % (d['spec']['dataset'], d['spec']['split'], d['spec']['scene'], d['spec']['episode_id'])
        lab = cfi.get('%s|%s' % (key, d['config']))
        if lab is None:
            continue
        (te if (zlib.crc32(d['spec']['scene'].encode()) % 1000)/1000. < test_frac else tr).append((p, lab))
    return tr, te


def evaluate(net, w, ring, idx, device):
    from .batch import to_torch
    from .heuristics import greedy_index
    from . import compiler as C
    res = dict(states=0, mixed=0, top1=[], auc=[], regret=[], default_top1=[], greedy_top1=[], oracle_top1=[],
               random_top1=[], default_regret=[], oracle_regret=[], greedy_regret=[])
    with torch.no_grad():
        for s in range(0, len(idx), 512):
            ii = idx[s:s+512]
            g = to_torch(ring.gather(ii), device)
            q = w.scalarize(net(g)).masked_fill(~g['cand_mask'], -1e9).cpu().numpy()
            for j, i in enumerate(ii):
                m = ring.cf_mask[i]
                if not m.any():
                    continue
                succ = ring.cf[i, :, 0] > 0.5
                steps = ring.cf[i, :, 2]*CUM_SCALE[2]
                n = int(ring.ncand[i])
                lab = m[:n]
                res['states'] += 1
                qq = q[j, :n]
                pick = int(np.argmax(np.where(lab, qq, -1e9)))
                # default = continue if present else goal: recover from the packed intent columns
                cand = ring.g['cand'][i][:n].astype(np.float32)
                cont = cand[:, C.CAND.index('intent_continue')] > 0.5
                goal = cand[:, C.CAND.index('intent_goal')] > 0.5
                dflt = int(np.flatnonzero(cont)[0]) if cont.any() else (int(np.flatnonzero(goal)[0]) if goal.any() else 0)
                graph = dict(cand=cand)
                gr = greedy_index(graph)
                orc = int(ring.oracle[i]) if ring.oracle[i] >= 0 else dflt
                best = steps[:n][succ[:n] & lab].min() if (succ[:n] & lab).any() else None

                def reg(k):
                    return float(steps[k]-best) if (best is not None and succ[k]) else None
                for name, k in (('top1', pick), ('default_top1', dflt), ('greedy_top1', gr), ('oracle_top1', orc)):
                    pass
                mixed = (succ[:n] & lab).any() and ((~succ[:n]) & lab).any()
                if mixed:
                    res['mixed'] += 1
                    res['top1'].append(float(succ[pick])); res['default_top1'].append(float(succ[dflt]))
                    res['greedy_top1'].append(float(succ[gr])); res['oracle_top1'].append(float(succ[orc]))
                    res['random_top1'].append(float(succ[:n][lab].mean()))
                    pos = qq[succ[:n] & lab]; neg = qq[(~succ[:n]) & lab]
                    res['auc'].append(float((pos[:, None] > neg[None, :]).mean()+0.5*(pos[:, None] == neg[None, :]).mean()))
                for name, k in (('regret', pick), ('default_regret', dflt), ('oracle_regret', orc), ('greedy_regret', gr)):
                    r = reg(k)
                    if r is not None:
                        res[name].append(r)
    out = dict(states=res['states'], mixed=res['mixed'])
    for k, v in res.items():
        if isinstance(v, list):
            out[k] = round(float(np.mean(v)), 4) if v else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warm', required=True)
    ap.add_argument('--cf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--drop', default='')
    ap.add_argument('--flat', action='store_true')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--td', type=float, default=1.0, help='weight of the TD loss on the demonstrations')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    from .learner import Learner
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr, te = split_rings(a.warm, a.cf, seed=a.seed)
    mc = ModelConfig(flat=a.flat, drop=tuple(x for x in a.drop.split(',') if x))
    L = Learner(out, a.device, capacity=200_000, model_cfg=mc, warm_steps=a.steps, cf_w=(1.0, 1.0, 1), margin_w=(0.5, 0.5, 1),
                demo_frac=1.0, seed=a.seed)
    for p, lab in tr:
        L.ingest_file(p, source=1, cf=lab)
    test = Ring(100_000, 3)
    L2 = Learner.__new__(Learner)
    L2.ring, L2.cfg_ids, L2.ep_stats, L2.warm_slots, L2.warm_inserted = test, {}, [], {}, 0
    for p, lab in te:
        Learner.ingest_file(L2, p, source=1, cf=lab)
    if a.td == 0:
        L.head_w = {k: 0. for k in L.head_w}
    log = []
    for step in range(a.steps):
        info = L.train_step()
        if (step+1) % 1000 == 0:
            L.net.eval()
            ev = evaluate(L.net, L.w, test, np.flatnonzero(test.valid), L.dev)
            evt = evaluate(L.net, L.w, L.ring, np.flatnonzero(L.ring.valid)[:3000], L.dev)
            L.net.train()
            rec = dict(step=step+1, loss=info['loss'], test=ev, train=dict(top1=evt['top1'], auc=evt['auc']))
            log.append(rec)
            print(json.dumps(rec), flush=True)
    summary = dict(drop=a.drop, flat=a.flat, steps=a.steps, train_episodes=len(tr), test_episodes=len(te),
                   train_states=int(L.ring.valid.sum()), test_states=int(test.valid.sum()), final=log[-1] if log else None)
    (out/'offline.json').write_text(json.dumps(dict(summary=summary, log=log), indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
