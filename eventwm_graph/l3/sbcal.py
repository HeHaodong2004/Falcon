"""Calibrated candidate collision risk for the closed-loop Sandbox (analysis / Layer 2.5 component).

Inputs per banked decision (``statebank.py``) and candidate: the closed-loop imagination of that candidate in the 6
heaviest Stage 1 worlds (``sbeval.py`` variant ``cl_frozen_p``: per-world body clearance to that world's people at
0.77 / 1.5 / 3.1 s, legal progress, STOPs, pose checkpoints every 4 steps), and the near-field people OCCUPANCY model
(``occupancy.py``, legal inputs) evaluated along each world's imagined trajectory (robot centre and a point 0.45 m
ahead, at the checkpoint's horizon) -- the hazard from people the closed loop cannot simulate because nobody tracks
them.  A calibrator (LightGBM, legal features, trained on the TRAIN bank with the exact outcomes as labels) maps them
to P(real human collision within W).  Evaluated on the DEV bank: AUC / Brier vs the uncalibrated closed loop, and the
selection rule driven by the calibrated risk.

    python scripts/l3_run.py eventwm_graph.l3.sbcal --occupancy output/occupancy/v1
"""
import argparse
import glob
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np

WINDOWS = (16, 32, 64)
CHECKS = list(range(4, 65, 4))


def hazard_rows(item, pred, occ_model):
    """Per candidate and window: occupancy hazard features along its imagined trajectories (world-weighted)."""
    from .occupancy import features
    from .infofeat import view_age, _blocked_lookup
    legal = item['legal']
    blocked = _blocked_lookup(legal.map)
    out = []
    # gather every checkpoint point of every candidate / world, evaluate by horizon
    pts = {}           # step -> list of (cand, world, x, z)
    for k, p in enumerate(pred['preds']):
        for j, cps in enumerate(p['poses']):
            for c, (x, z, yaw) in zip(CHECKS, cps):
                pts.setdefault(c, []).append((k, j, x, z))
                pts[c].append((k, j, x+0.45*math.cos(yaw), z-0.45*math.sin(yaw)))
    q = {}
    for c, lst in pts.items():
        px = np.asarray([v[2] for v in lst]); pz = np.asarray([v[3] for v in lst])
        va = view_age(legal, px, pz, blocked)
        F = features(legal, item['bank'], px, pz, c, view_age_fn=lambda a, b, va=va: va)
        pr = occ_model.predict(F)
        for (k, j, _, _), v in zip(lst, pr):
            q[(k, j, c)] = max(q.get((k, j, c), 0.), float(v))
    for k, p in enumerate(pred['preds']):
        ws = np.asarray(p['weights'])
        rec = {}
        for w in WINDOWS:
            mx, sm, nc = [], [], []
            for j in range(len(p['poses'])):
                vals = [q.get((k, j, c), 0.) for c in CHECKS if c <= w]
                mx.append(max(vals) if vals else 0.)
                sm.append(1.-float(np.prod([1.-v for v in vals])) if vals else 0.)
            rec[w] = (float((ws*np.asarray(mx)).sum()), float((ws*np.asarray(sm)).sum()), float(max(mx) if mx else 0.))
        out.append(rec)
    return out


def exposure_rows(item, pred):
    """Legal exposure features: the decision's information state (infofeat.InfoState.ego) and, per candidate, its
    imagined interval's yaw sweep / stale area covered / Stage 1 residual and lost slots seen (InfoState.cand), from the
    heaviest world's pose checkpoints (steps 4, 8, 12, 16)."""
    from .shortprobe import expo_state
    from .sbeval import StoredBank
    legal = item['legal']
    info = expo_state(legal, StoredBank(item['bank']))
    ego = [float(x) for x in info.ego()]
    cands = []
    for p in pred['preds']:
        cps = p['poses'][0]
        poses = [list(map(float, legal.pose))]
        for (x, z, yaw) in cps[:4]:
            poses.append([x, float(legal.pose[1]), z, yaw])
        cands.append([float(x) for x in info.cand(np.asarray(poses, np.float64))])
    return ego, cands


def candidate_features(pred_k, haz_k, w, ctg0, expo=None):
    clr = np.asarray(pred_k['clr'][w] if w in pred_k['clr'] else pred_k['clr'][str(w)], np.float64)
    ws = np.asarray(pred_k['weights'], np.float64)
    stop = np.asarray(pred_k['stop'][w], np.int64)
    f = []
    mc = pred_k.get('mclr', {}).get(w) if isinstance(pred_k.get('mclr'), dict) else None
    if mc:
        mw = np.asarray([a for a, _ in mc], np.float64); mcl = np.asarray([b for _, b in mc], np.float64)
        mw = mw/max(mw.sum(), 1e-9)
        for t in (0.2, 0.5):
            f.append(float((mw*(mcl < t)).sum()))
    else:
        for t in (0.2, 0.5):
            f.append(float((ws*(clr < t)).sum()))
    for t in (0.1, 0.2, 0.3, 0.5, 0.75):
        f.append(float((ws*(clr < t)).sum()))
    f += [float((ws*np.minimum(clr, 3.)).sum()), float(clr.min()), float(np.median(clr))]
    f += [float(pred_k['prog'][w]), ctg0]
    f += [float((ws*(stop == 1)).sum()), float((ws*(stop == 2)).sum())]
    f += list(haz_k[w])
    if expo is not None:
        ego, cand = expo
        f += list(cand)+list(ego)
    f.append(w*0.048)
    return f


def build(bank_dir, sb_dir, occ_model, cache=None):
    if cache and Path(cache).exists():
        d = np.load(cache, allow_pickle=True)
        return d['X'], d['Y'], d['G'], d['meta'].tolist()
    X, Y, G, meta = [], [], [], []
    files = sorted(glob.glob(os.path.join(sb_dir, '*.json')))
    for gi, f in enumerate(files):
        stem = Path(f).stem
        pkl = Path(bank_dir, stem+'.pkl')
        if not pkl.exists():
            continue
        item = pickle.load(open(pkl, 'rb'))
        pred = json.loads(Path(f).read_text())
        if 'poses' not in pred['preds'][0]:
            continue
        for p in pred['preds']:
            for key in ('clr', 'prog', 'stop', 'mclr'):
                if key in p:
                    p[key] = {int(a): b for a, b in p[key].items()}
        haz = hazard_rows(item, pred, occ_model) if occ_model is not None else [
            {w: (0., 0., 0.) for w in WINDOWS} for _ in pred['preds']]
        ego, cexp = exposure_rows(item, pred)
        from .compiler import LegalField
        ctg0 = float(LegalField(item['legal']).start)
        probe = item['probe']
        for k, c in enumerate(probe['cands']):
            for w in WINDOWS:
                X.append(candidate_features(pred['preds'][k], haz[k], w, ctg0, (ego, cexp[k])))
                r = c['real']
                Y.append(int(r['outcome'] == 'human_collision' and r['steps'] <= w))
                G.append(gi)
                meta.append((stem, k, w))
    X, Y, G = np.asarray(X, np.float32), np.asarray(Y, np.int8), np.asarray(G)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, Y=Y, G=G, meta=np.asarray(meta, dtype=object))
    return X, Y, G, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--occupancy', default='output/occupancy/v1', help="occupancy model dir, or 'none'")
    ap.add_argument('--train-bank', default='output/statebank/orca_train')
    ap.add_argument('--dev-bank', default='output/statebank/orca_dev')
    ap.add_argument('--variant', default='cl_frozen_p')
    ap.add_argument('--out', default='output/sbeval')
    ap.add_argument('--save', default=None)
    a = ap.parse_args()
    import lightgbm as lgb
    from .leafvalue import ValueModel
    from .occupancy import auc
    occ = None if a.occupancy == 'none' else ValueModel(a.occupancy)
    tag = Path(a.occupancy).name
    Xt, Yt, Gt, _ = build(a.train_bank, Path(a.out, Path(a.train_bank).name, a.variant), occ,
                          cache='output/sbcal/cache_train_%s_%s.npz' % (a.variant, tag))
    Xd, Yd, Gd, md = build(a.dev_bank, Path(a.out, Path(a.dev_bank).name, a.variant), occ,
                           cache='output/sbcal/cache_dev_%s_%s.npz' % (a.variant, tag))
    print('train rows %d (collisions %d), dev rows %d (collisions %d)' % (len(Yt), Yt.sum(), len(Yd), Yd.sum()), flush=True)
    params = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, seed=0,
                  num_threads=int(os.environ.get('L3_LGB_THREADS', '4')))
    b = lgb.train(params, lgb.Dataset(Xt, Yt), 300)
    pd_ = b.predict(Xd)
    for i, w in enumerate(WINDOWS):
        m = np.isclose(Xd[:, -1], w*0.048)
        base = Yd[m].mean() if m.any() else float('nan')
        raw = Xd[m, 0]                     # contact probability over every scored world (body clearance < 0.2 m)
        haz = Xd[m, 14]                    # world-weighted max occupancy hazard
        print('%.2f s: collisions %d / %d | AUC calibrated %.3f  closed-loop contact %.3f  occupancy hazard %.3f | '
              'Brier calibrated %.4f  constant %.4f' % (w*0.048, Yd[m].sum(), m.sum(), auc(pd_[m], Yd[m]), auc(raw, Yd[m]),
                                                     auc(haz, Yd[m]), ((pd_[m]-Yd[m])**2).mean(),
                                                     ((base-Yd[m])**2).mean()), flush=True)
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
        b.save_model(str(Path(a.save, 'model.txt')))
        Path(a.save, 'model.json').write_text(json.dumps(b.dump_model()))
        Path(a.save, 'meta.json').write_text(json.dumps(dict(keys=['sbcal'], occupancy=a.occupancy, variant=a.variant)))


if __name__ == '__main__':
    main()
