"""Learned re-weighting of Stage 1's worlds for TRACKED people (Layer 2 improvement; legal inputs, privileged labels).

Stage 1 returns 40 equally weighted worlds.  For the people the robot tracks, the world nearest the truth is very
accurate (0.07 m median at 0.77 s) while an average world is 0.6 m off and the mean of worlds 0.37 m: the right
future is usually present, but nothing ranks it.  This model scores each world's predicted path for a tracked person
from legal cues -- agreement with the track's recently observed motion (tracker velocity and a fit of the legal track
log), the path's speed / acceleration / turn, clearance of its points from the legal map's BLOCKED cells, distance to
the robot -- and is trained (train bank) to rank the worlds by closeness to the true position at 0.77 / 1.5 s.  The
softmax of the scores replaces the uniform weights for that person; evaluated on the dev bank as the weighted error.

    python scripts/l3_run.py eventwm_graph.l3.worldweights --train output/statebank/orca_train --eval output/statebank/orca_dev
"""
import argparse
import glob
import math
import os
import pickle

import numpy as np

H_EVAL = (16, 32)


def _person_features(legal, path, e, hist_fit, pose, blocked, steps=(8, 16, 24, 32)):
    """World features [W, F] of one tracked person's predicted paths ``path`` [W, Tb, 2] (world x, z)."""
    p0 = np.array([e.position[0], -e.position[1]])
    v_tr = np.array([e.velocity[0], -e.velocity[1]])
    v_fit = hist_fit if hist_fit is not None else v_tr
    F = []
    for w in range(len(path)):
        f = []
        for st in steps:
            q = path[w, st-1]
            f += [float(np.hypot(*(q-(p0+v_tr*st*0.048)))), float(np.hypot(*(q-(p0+v_fit*st*0.048))))]
        v1 = (path[w, 7]-p0)/(8*0.048)
        v2 = (path[w, 23]-path[w, 7])/(16*0.048)
        sp1, sp2 = float(np.hypot(*v1)), float(np.hypot(*v2))
        turn = math.atan2(v1[0]*v2[1]-v1[1]*v2[0], float(v1 @ v2)) if sp1 > 0.05 and sp2 > 0.05 else 0.
        f += [sp1, sp2, sp2-sp1, abs(turn), float(np.hypot(*v_tr)), float(np.hypot(*v_fit))]
        pts = path[w, :32:4]
        f.append(float(blocked(pts[:, 0], pts[:, 1]).mean()))
        f.append(float(np.hypot(p0[0]-pose[0], p0[1]-pose[2])))
        F.append(f)
    return np.asarray(F, np.float32)


def _history_fit(legal, track_id, now):
    hs = []
    for tm, trs in (legal.track_log or []):
        for t in trs:
            if t.get('visible_now') and str(t['track_id']) == str(track_id) and now-float(tm) <= 1.0:
                hs.append((float(tm), float(t['position'][0]), float(t['position'][2])))
    if len(hs) < 3:
        return None
    tt = np.array([h[0] for h in hs])-now
    A = np.c_[tt, np.ones(len(tt))]
    return np.array([np.linalg.lstsq(A, np.array([h[1] for h in hs]), rcond=None)[0][0],
                     np.linalg.lstsq(A, np.array([h[2] for h in hs]), rcond=None)[0][0]])


class WorldWeighter:
    """Deployment: learned weights of a Stage 1 bank's worlds from the tracked people's predicted paths (legal)."""

    def __init__(self, path, temp=0.5):
        from .leafvalue import ValueModel
        self.model = ValueModel(path)
        self.temp = float(temp)

    def weights(self, legal, bank):
        from eventwm_graph.querygen.control import human_estimates
        from .infofeat import _blocked_lookup
        W = np.asarray(bank.weights, np.float64)
        W = W/W.sum() if W.sum() > 0 else np.full(len(W), 1./max(len(W), 1))
        keys = [str(k) for k in getattr(bank, 'track_keys', ())]
        if not keys:
            return W
        P = np.asarray(bank.sandbox_positions(np.arange(1, 33)*0.048), np.float64)
        pose = np.asarray(legal.pose, np.float64)
        blocked = _blocked_lookup(legal.map)
        now = float(legal.time)
        logw = np.log(np.maximum(W, 1e-12))
        for e in human_estimates(legal.tracks, legal.time):
            k = 'track:%s' % e.track_id
            if k not in keys:
                continue
            path = P[:, :, keys.index(k)]
            if not np.isfinite(path[:, :32]).all():
                continue
            F = _person_features(legal, path, e, _history_fit(legal, e.track_id, now), pose, blocked)
            sc = self.model.predict_raw(F)
            z = (sc-sc.max())/self.temp
            logw += z-np.log(np.exp(z).sum())
        w = np.exp(logw-logw.max())
        return w/w.sum()


def person_rows(item, steps=(8, 16, 24, 32)):
    """Per tracked person matched to a true person: world features [W, F] and true errors at H_EVAL [W, len(H_EVAL)]."""
    from eventwm_graph.querygen.control import human_estimates
    from .infofeat import _blocked_lookup
    legal = item['legal']
    T = np.asarray(item['truth_people'], np.float64)
    b = item['bank']
    P = np.asarray(b['positions'], np.float64)
    keys = list(b['track_keys'])
    pose = np.asarray(legal.pose, np.float64)
    blocked = _blocked_lookup(legal.map)
    now = float(legal.time)
    hist = {}
    for tm, trs in (legal.track_log or []):
        for t in trs:
            if t.get('visible_now'):
                hist.setdefault(str(t['track_id']), []).append((float(tm), float(t['position'][0]), float(t['position'][2])))
    out = []
    for e in human_estimates(legal.tracks, legal.time):
        k = 'track:%s' % e.track_id
        if k not in keys:
            continue
        s = keys.index(k)
        d0 = np.hypot(T[0, :, 0]-e.position[0], T[0, :, 1]+e.position[1])
        if not len(d0) or np.nanmin(d0) > 0.75:
            continue
        j = int(np.nanargmin(d0))
        path = P[:, :, s]                                                     # [W, Tb, 2]
        if not np.isfinite(path[:, :max(steps)]).all():
            continue
        p0 = np.array([e.position[0], -e.position[1]])                      # world x, z
        v_tr = np.array([e.velocity[0], -e.velocity[1]])                     # world
        hs = [(t, x, z) for t, x, z in hist.get(str(e.track_id), []) if now-t <= 1.0]
        if len(hs) >= 3:
            tt = np.array([h[0] for h in hs])-now
            A = np.c_[tt, np.ones(len(tt))]
            v_fit = np.array([np.linalg.lstsq(A, np.array([h[1] for h in hs]), rcond=None)[0][0],
                              np.linalg.lstsq(A, np.array([h[2] for h in hs]), rcond=None)[0][0]])
        else:
            v_fit = v_tr
        F = []
        for w in range(len(path)):
            f = []
            prev = p0
            for st in steps:
                q = path[w, st-1]
                cv = p0+v_tr*st*0.048
                fit = p0+v_fit*st*0.048
                f += [float(np.hypot(*(q-cv))), float(np.hypot(*(q-fit)))]
            v1 = (path[w, 7]-p0)/(8*0.048)
            v2 = (path[w, 23]-path[w, 7])/(16*0.048)
            sp1, sp2 = float(np.hypot(*v1)), float(np.hypot(*v2))
            turn = math.atan2(v1[0]*v2[1]-v1[1]*v2[0], float(v1 @ v2)) if sp1 > 0.05 and sp2 > 0.05 else 0.
            f += [sp1, sp2, sp2-sp1, abs(turn), float(np.hypot(*v_tr)), float(np.hypot(*v_fit))]
            pts = path[w, :32:4]
            f.append(float(blocked(pts[:, 0], pts[:, 1]).mean()))
            f.append(float(np.hypot(p0[0]-pose[0], p0[1]-pose[2])))
            F.append(f)
        F = np.asarray(F, np.float32)
        err = np.stack([np.hypot(path[:, h-1, 0]-T[h, j, 0], path[:, h-1, 1]-T[h, j, 1]) for h in H_EVAL], 1)
        if not np.isfinite(err).all():
            continue
        out.append((F, err.astype(np.float32)))
    return out


def collect(bank_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(bank_dir, '*.pkl'))):
        rows += person_rows(pickle.load(open(p, 'rb')))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='output/statebank/orca_train')
    ap.add_argument('--eval', default='output/statebank/orca_dev')
    ap.add_argument('--temp', default='0.5,1,2')
    ap.add_argument('--save', default=None)
    a = ap.parse_args()
    import lightgbm as lgb
    tr, ev = collect(a.train), collect(a.eval)
    print('tracked people: train %d, eval %d' % (len(tr), len(ev)), flush=True)
    # graded relevance per person: rank of the world's 1.5 s error among the 40 worlds (closest = highest)
    X = np.concatenate([F for F, _ in tr]); grp = [len(F) for F, _ in tr]
    rel = np.concatenate([np.clip(4-np.floor(np.argsort(np.argsort(e[:, 1]))/(len(e)/5.)), 0, 4).astype(int) for _, e in tr])
    b = lgb.train(dict(objective='lambdarank', learning_rate=0.05, num_leaves=15, min_data_in_leaf=40, lambda_l2=1.,
                       verbose=-1, seed=0, num_threads=4, label_gain=[0, 1, 3, 7, 15]), lgb.Dataset(X, rel, group=grp), 300)
    if a.save:
        import json
        from pathlib import Path
        Path(a.save).mkdir(parents=True, exist_ok=True)
        b.save_model(str(Path(a.save, 'model.txt')))
        Path(a.save, 'model.json').write_text(json.dumps(b.dump_model()))
        Path(a.save, 'meta.json').write_text(json.dumps(dict(keys=['worldweights'], train=a.train, temp=0.5)))
    for t in (float(x) for x in a.temp.split(',')):
        res = {h: dict(uni=[], learned=[], top1=[], best=[], mean_uni=[], mean_learned=[]) for h in range(len(H_EVAL))}
        for F, e in ev:
            sc = b.predict(F)
            wl = np.exp((sc-sc.max())/t); wl /= wl.sum()
            for i in range(len(H_EVAL)):
                res[i]['uni'].append(float(e[:, i].mean()))
                res[i]['learned'].append(float((wl*e[:, i]).sum()))
                res[i]['top1'].append(float(e[int(np.argmax(sc)), i]))
                res[i]['best'].append(float(e[:, i].min()))
        print('temperature %.1f: ' % t + ' | '.join(
            '%.2f s: weighted error uniform %.2f -> learned %.2f (median); top-scored world %.2f; best world %.2f' % (
                H_EVAL[i]*0.048, np.median(res[i]['uni']), np.median(res[i]['learned']), np.median(res[i]['top1']),
                np.median(res[i]['best'])) for i in res), flush=True)


if __name__ == '__main__':
    main()
