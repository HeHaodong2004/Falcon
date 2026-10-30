"""Calibrated official-contact model for the Sandbox's imagined rollouts (Layer 2.5 option, default off).

The benchmark's human collision is a physics contact between the articulated robot and the human mesh (a stick
skeleton), not a distance: real contacts occur with centres up to ~1 m apart (docs/l3/LAYER2_V2_DESIGN.md 1.2).  The
multi-world selector (``heuristics.multiworld_index``) scored an imagined contact with a hard rule (imagined centre
distance < 0.75 m).  This module replaces the rule by P(official contact within W steps | imagined geometry over the
first W steps), W = 16 / 32 / 64 (0.77 / 1.5 / 3.1 s), fitted on the TRAIN part of the bank imagination dataset
(``bankimag.py``: TRUE-people imagination; labels = the exact branch outcomes).  Privileged data (true people, exact
outcomes) are used only as labels on TRAIN and for evaluation; at run time the features are computed from the
imagined robot path and the people the selector scores (legal CV / Stage 1 people in the primary benchmark).

Features (``geometry_features``):
  rc_min  min body-rectangle clearance (``querygen.native.ROBOT_BOUNDS``) to any person's centre over the window
  cd_min  min centre distance
  ex      exposure: sum over the window's steps of dt * exp(-c_i / 0.2 m), c_i = the step's min rectangle clearance
  i_ca    step of the closest approach (min rectangle clearance; the outcome rule's "first contact" tie-break)
Model (``FEATURES``): monotone logistic regression (coefficient signs constrained: closer / longer exposure never
lowers P) on log(min(rc, 1.5) + 0.05), log(min(cd, 2) + 0.05), [rc < 0.01 m], log(ex + 1e-3).  Diagnostics computed
with ``full=True`` (time of closest approach, robot speed / turn rate, person speed, approach speed, bearing and
relative heading at the closest approach, dwell) were tested in episode-grouped cross-validation on TRAIN and did not
improve the held-out log-loss or the within-state ranking; neither did a monotone GBM (LightGBM).

    python scripts/l3_run.py eventwm_graph.l3.contactmodel fit --train output/agents/bankimag/orca_train.npz \
        --out output/agents/contact/contact_model.json
    python scripts/l3_run.py eventwm_graph.l3.contactmodel evaluate --model output/agents/contact/contact_model.json \
        --data output/agents/bankimag/orca_dev.npz --out output/agents/contact/eval_dev.json

Selector (default off): ``L3_MW_CONTACT_MODEL=<model json>`` (multiworld_index); ``L3_MW_CONTACT_THR=<p>`` overrides
the threshold chosen on TRAIN (maximal within-state discrimination), ``L3_MW_CONTACT_THR=soft`` uses the probability
itself (expected contact share / expected outcome value); ``L3_MW_CONTACT_W=16|32|64`` overrides the model window
(default: the smallest model window covering the selector's imagined window, else the largest).

Result (ORCA native k10 bank, 2026-10-30; output/agents/contact/{contact_model,eval_dev}.json): TRAIN thresholds
0.62 / 0.18 / 0.24 (W 16 / 32 / 64).  Dev, TRUE-people imagination: AUC 0.990 / 0.973 / 0.912, ECE 0.002 / 0.014 /
0.025, precision 0.81 / 0.45 / 0.49 at recall 0.54 / 0.88 / 0.75 (0.75 m rule: 0.39 / 0.39 / 0.42 at 0.96 / 0.94 /
0.80); within-state discrimination of the thresholded model is not better than the 0.75 m rule (n.s.; candidates of
one state often share their imagined geometry but not their real outcome -- the robot-loop fidelity limit), the
probability ranks candidates of a state better than the centre distance at 3.1 s only (+0.04 within-state AUC).
Legal CV people: AUC 0.65-0.70, within-state ~0.5, over-confident (contacts with untracked people are invisible to
any geometry model of the CV world: the people model is the binding constraint there).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

STEP_S = 0.048
WINDOWS = (16, 32, 64)
TAU = 0.2                    # exposure length scale (m)
RC_CAP, CD_CAP = 3., 4.      # feature caps (m); also the values without any person
FEATURES = ('lrc', 'lcd', 'rc0', 'lex')
SIGNS = (-1, -1, 1, 1)       # monotone constraints on the (standardised) coefficients
RULE_M = 0.75                # the hard rule this model replaces (imagined centre distance < 0.75 m)
VERSION = 1


# ---------------------------------------------------------------------------------------------- features
def geometry_features(seq, people, pose0=None, step_s=STEP_S, full=False):
    """Features of one imagined rollout window.  ``seq`` [n, 4]: the imagined robot after 1..n steps (x, y, z, yaw);
    ``people`` [n, N, 2]: the scored people's centres (world x, z) at the same steps (non-finite = absent); ``pose0``:
    the robot pose at t (``full`` only).  Returns a dict (no person in the window: caps, ``none`` = 1)."""
    from .heuristics import _rect_clearance
    seq = np.asarray(seq, np.float64).reshape(-1, 4)
    n = len(seq)
    out = dict(rc_min=RC_CAP, cd_min=CD_CAP, ex=0., i_ca=max(n-1, 0), none=1.)
    P = np.asarray(people, np.float64)
    if not n or P.ndim != 3 or not P.shape[1]:
        return _diag(out, n, step_s) if full else out
    P = P[:n]
    rc = _rect_clearance(seq[:len(P)], P)
    rc = np.where(np.isfinite(rc), rc, np.inf)
    if not np.isfinite(rc).any():
        return _diag(out, n, step_s) if full else out
    with np.errstate(invalid='ignore'):
        cd = np.hypot(P[..., 0]-seq[:len(P), None, 0], P[..., 1]-seq[:len(P), None, 2])
    cd = np.where(np.isfinite(cd), cd, np.inf)
    i, p = divmod(int(np.argmin(rc)), rc.shape[1])
    step = rc.min(axis=1)
    out.update(rc_min=float(min(rc[i, p], RC_CAP)), cd_min=float(min(cd.min(), CD_CAP)),
               ex=float(np.exp(-np.minimum(step, 5.)/TAU).sum()*step_s), i_ca=int(i), none=0.)
    if full:
        out.update(_dynamics(seq, P, rc, cd, i, p, pose0, step_s))
    return out


def _diag(out, n, step_s):
    out.update(t_ca=n*step_s, cd_ca=CD_CAP, v_rob=0., w_rob=0., v_per=0., v_close=0., cos_b=0., sin_b=0., cos_h=0.,
               dwell=0.)
    return out


def _dynamics(seq, P, rc, cd, i, p, pose0, step_s, k=2):
    """Diagnostics at the closest approach (step i, person p): time, robot speed / turn rate, person speed, approach
    speed (positive = closing), bearing of the person in the robot frame, relative heading, dwell below 0.25 m."""
    n = len(seq)
    R = np.concatenate([np.asarray(pose0, np.float64).reshape(1, 4) if pose0 is not None else seq[:1], seq])
    a, b = max(i+1-k, 0), i+1                        # rows of R: robot after a .. b steps
    vr = (R[b, [0, 2]]-R[a, [0, 2]])/max((b-a)*step_s, 1e-9)
    dyaw = math.atan2(math.sin(R[b, 3]-R[a, 3]), math.cos(R[b, 3]-R[a, 3]))
    pa, pb = max(i-k, 0), i
    if pb == pa:
        pb = min(pa+k, len(P)-1)
    if pb > pa and np.isfinite(P[pa, p]).all() and np.isfinite(P[pb, p]).all():
        vp = (P[pb, p]-P[pa, p])/((pb-pa)*step_s)
    else:
        vp = np.zeros(2)
    u = P[i, p]-seq[i, [0, 2]]
    du = float(np.hypot(*u))
    u = u/du if du > 1e-9 else np.zeros(2)
    yaw = seq[i, 3]
    f = np.array([math.cos(yaw), -math.sin(yaw)])
    r = np.array([math.sin(yaw), math.cos(yaw)])
    sp = float(np.hypot(*vp))
    return dict(t_ca=(i+1)*step_s, cd_ca=float(min(cd[i, p], CD_CAP)), v_rob=float(np.hypot(*vr)),
                w_rob=abs(dyaw)/max((b-a)*step_s, 1e-9), v_per=sp, v_close=float(-np.dot(vp-vr, u)),
                cos_b=float(np.dot(u, f)), sin_b=float(abs(np.dot(u, r))),
                cos_h=float(np.dot(vp, f)/sp) if sp > 0.05 else 0., dwell=float((rc.min(axis=1) < 0.25).sum()*step_s))


def design(F):
    """Model inputs [n, len(FEATURES)] from feature arrays (dict of arrays or of scalars)."""
    rc = np.atleast_1d(np.asarray(F['rc_min'], np.float64))
    cd = np.atleast_1d(np.asarray(F['cd_min'], np.float64))
    ex = np.atleast_1d(np.asarray(F['ex'], np.float64))
    return np.stack([np.log(np.minimum(rc, 1.5)+0.05), np.log(np.minimum(cd, 2.)+0.05), (rc < 0.01).astype(np.float64),
                     np.log(ex+1e-3)], 1)


# ---------------------------------------------------------------------------------------------- model
def fit_logistic(Z, y, lam=1.0, signs=SIGNS):
    """L2-regularised logistic regression on standardised inputs, coefficient signs constrained (L-BFGS-B bounds)."""
    from scipy.optimize import minimize
    Z = np.asarray(Z, np.float64)
    y = np.asarray(y, np.float64)
    mu = Z.mean(0)
    sd = Z.std(0)
    sd = np.where(sd > 1e-9, sd, 1.)
    A = np.c_[np.ones(len(Z)), (Z-mu)/sd]

    def f(b):
        z = A@b
        val = (np.logaddexp(0., z)-y*z).sum()+0.5*lam*(b[1:]**2).sum()
        g = A.T@(1./(1.+np.exp(-z))-y)+np.r_[0., lam*b[1:]]
        return val, g
    bounds = [(None, None)]+[((0., None) if s > 0 else (None, 0.)) if s else (None, None) for s in signs]
    r = minimize(f, np.zeros(A.shape[1]), jac=True, method='L-BFGS-B', bounds=bounds)
    return dict(mu=mu.tolist(), sd=sd.tolist(), intercept=float(r.x[0]), coef=r.x[1:].tolist(), lam=float(lam))


def predict_logistic(m, Z):
    Z = np.asarray(Z, np.float64)
    z = m['intercept']+((Z-np.asarray(m['mu']))/np.asarray(m['sd']))@np.asarray(m['coef'])
    return 1./(1.+np.exp(-z))


class ContactModel:
    """Loaded model file: per window W the logistic coefficients and the threshold chosen on TRAIN."""

    def __init__(self, path):
        self.path = str(path)
        self.doc = json.loads(Path(path).read_text())
        if tuple(self.doc['features']) != FEATURES:
            raise ValueError('contact model %s: features %s, expected %s' % (path, self.doc['features'], FEATURES))
        self.models = {int(w): m for w, m in self.doc['windows'].items()}
        self.windows = tuple(sorted(self.models))

    def window(self, steps, override=None):
        """Model window for a selector scoring ``steps`` imagined steps: ``override`` if given, else the smallest
        model window >= steps, else the largest."""
        if override:
            w = int(override)
            if w not in self.models:
                raise ValueError('contact model %s has no window %d (%s)' % (self.path, w, self.windows))
            return w
        for w in self.windows:
            if w >= int(steps):
                return w
        return self.windows[-1]

    def threshold(self, w):
        return float(self.models[int(w)]['threshold'])

    def features(self, seq, people, pose0=None):
        return geometry_features(seq, people, pose0)

    def prob(self, feats, w):
        return float(predict_logistic(self.models[int(w)], design(feats))[0])

    def prob_many(self, F, w):
        return predict_logistic(self.models[int(w)], design(F))


# ---------------------------------------------------------------------------------------------- dataset
SOURCES = {'true': ('b', 'truth', 'state_n_people'),       # TRUE-people imagination scored against the true people
           'cv': ('a', 'cv_paths', 'state_n_cv')}           # LEGAL CV imagination scored against the CV people


def load(path):
    D = np.load(path)
    return {k: D[k] for k in D.files}


def dataset_features(D, source, w, full=False):
    """Per-candidate features of the window's first ``w`` imagined steps, labels (real contact within w) and ids."""
    v, people, count = SOURCES[source]
    S = D['cand_state']
    rows = []
    for r in range(len(S)):
        s = int(S[r])
        n = int(min(D['%s_n' % v][r], w))
        seq = np.asarray(D['%s_poses' % v][r, :n], np.float64)
        P = np.asarray(D[people][s, 1:n+1, :int(D[count][s])], np.float64)
        rows.append(geometry_features(seq, P, D['state_robot_pose'][s], full=full))
    F = {k: np.asarray([x[k] for x in rows], np.float64) for k in rows[0]}
    y = (D['cand_real_outcome'] == 'human_collision') & (D['cand_real_steps'] <= w)
    return F, y.astype(bool), S


# ---------------------------------------------------------------------------------------------- metrics
def auc(score, y):
    """ROC AUC with ties counted 1/2 (NaN if one class is empty)."""
    y = np.asarray(y, bool)
    s = np.asarray(score, np.float64)
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j+1 < len(ss) and ss[j+1] == ss[i]:
            j += 1
        ranks[order[i:j+1]] = 0.5*(i+j)+1.
        i = j+1
    return float((ranks[y].sum()-npos*(npos+1)/2.)/(npos*nneg))


def mixed_states(y, S):
    return [s for s in np.unique(S) if y[S == s].any() and (~y[S == s]).any()]


def within_auc(score, y, S):
    """Mean AUC over the states with both outcomes (the selection problem: candidates of the same state)."""
    ms = mixed_states(y, S)
    if not ms:
        return float('nan'), 0
    return float(np.mean([auc(score[S == s], y[S == s]) for s in ms])), len(ms)


def within_hard(hit, y, S):
    """Within-state discrimination of a binary rule: mean over mixed states of 0.5 * (1 + TPR - FPR) (= the
    within-state AUC of the binary score)."""
    hit = np.asarray(hit, bool)
    ms = mixed_states(y, S)
    if not ms:
        return float('nan')
    return float(np.mean([0.5*(1.+hit[(S == s) & y].mean()-hit[(S == s) & ~y].mean()) for s in ms]))


def precision_recall(hit, y):
    hit = np.asarray(hit, bool)
    tp = int((hit & y).sum()); fp = int((hit & ~y).sum()); fn = int((~hit & y).sum())
    return tp/max(tp+fp, 1), tp/max(tp+fn, 1)


def calibration(p, y, bins=(0., .01, .02, .05, .1, .2, .3, .5, .7, .9, 1.0001)):
    """Brier, log-loss, ECE (10 equal-mass bins), recalibration intercept / slope (logit(y) ~ a + b logit(p)) and a
    reliability table on fixed bins (mean p, observed rate, count)."""
    from scipy.optimize import minimize
    y = np.asarray(y, np.float64)
    q = np.clip(np.asarray(p, np.float64), 1e-6, 1-1e-6)
    out = dict(brier=float(((q-y)**2).mean()), logloss=float(-(y*np.log(q)+(1-y)*np.log(1-q)).mean()),
               mean_p=float(q.mean()), rate=float(y.mean()))
    edges = np.quantile(q, np.linspace(0, 1, 11))
    h = np.clip(np.searchsorted(edges, q, side='right')-1, 0, 9)
    out['ece'] = float(sum(abs(q[h == i].mean()-y[h == i].mean())*(h == i).sum() for i in range(10) if (h == i).any())/len(q))
    lg = np.log(q/(1-q))
    A = np.c_[np.ones(len(q)), lg]

    def f(b):
        z = A@b
        return (np.logaddexp(0., z)-y*z).sum(), A.T@(1./(1.+np.exp(-z))-y)
    r = minimize(f, np.array([0., 1.]), jac=True, method='L-BFGS-B')
    out['recal_intercept'], out['recal_slope'] = float(r.x[0]), float(r.x[1])
    h = np.clip(np.searchsorted(np.asarray(bins), q, side='right')-1, 0, len(bins)-2)
    out['reliability'] = [dict(lo=float(bins[i]), hi=float(min(bins[i+1], 1.)), n=int((h == i).sum()),
                               mean_p=float(q[h == i].mean()) if (h == i).any() else None,
                               rate=float(y[h == i].mean()) if (h == i).any() else None) for i in range(len(bins)-1)]
    return out


def folds(groups, k=5, seed=0):
    u = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    f = {g: i % k for i, g in enumerate(u)}
    return np.asarray([f[g] for g in groups])


# ---------------------------------------------------------------------------------------------- selection replica
def simulate_selection(D, source, w, hit, first, p=None, soft=False, eps=0.):
    """Offline replica of multiworld_index's outcome rule (one world, EPS ``eps``) on the dataset's imagined rollouts:
    per state, the chosen candidate and its EXACT outcome within ``w`` steps.  Critical = the default really collides
    (or STOP-fails) within w and some candidate does not; fixed = the choice does not; broken = the default does not
    and the choice does."""
    v = SOURCES[source][0]
    n_all, stop_all, poses = D['%s_n' % v], D['%s_stop' % v], D['%s_poses' % v]
    ctg = D['%s_ctg' % v][:, WINDOWS.index(w)]
    out = dict(states=0, critical=0, fixed=0, broken=0, collisions=0, default_collisions=0, deviations=0)
    for s in range(len(D['state_n_cands'])):
        a = int(D['state_first_cand'][s])
        rows = range(a, a+int(D['state_n_cands'][s]))
        vals, cat = [], []
        for r in rows:
            n = int(min(n_all[r], w))
            stp = bool(stop_all[r]) and n_all[r] <= w
            near = n > 0 and math.hypot(poses[r, n-1, 0]-D['state_goal'][s, 0], poses[r, n-1, 2]-D['state_goal'][s, 2]) <= 0.3
            if stp and not near:
                vn = 0.
            elif stp and near:
                vn = 3.+0.5*(1.-max(n, 1)/w)
            else:
                g = D['state_ctg0'][s]-ctg[r] if np.isfinite(ctg[r]) else 0.
                vn = 2.+0.5*float(np.clip(g/max(D['state_ctg0'][s], 1e-6), 0., 0.999))
            vh = 0.5*max(int(first[r]), 0)/max(n, 1)
            if soft:
                vals.append(p[r]*vh+(1.-p[r])*vn)
            else:
                vals.append(vh if hit[r] else vn)
            o, st = D['cand_real_outcome'][r], D['cand_real_steps'][r]
            cat.append(0 if (o in ('human_collision', 'stop_fail') and st <= w) else (3 if (o == 'success' and st <= w) else 2))
        vals, cat = np.asarray(vals), np.asarray(cat)
        d = int(D['state_default'][s])
        best = int(np.argmax(vals))
        pick = d if vals[d] >= vals[best]-eps else best
        out['states'] += 1
        out['deviations'] += int(pick != d)
        out['collisions'] += int(cat[pick] == 0)
        out['default_collisions'] += int(cat[d] == 0)
        if cat[d] == 0 and cat.max() > 0:
            out['critical'] += 1
            out['fixed'] += int(cat[pick] > 0)
        if cat[d] > 0 and cat[pick] == 0:
            out['broken'] += 1
    return out


def rule_first(F_cd_steps, radius=RULE_M):
    """(hit, first step) of the centre-distance rule from per-step min centre distances [R, w]."""
    d = np.where(np.isfinite(F_cd_steps), F_cd_steps, np.inf)
    h = d < radius
    return h.any(1), np.where(h.any(1), h.argmax(1), 0)


def _rule_arrays(D, source, w):
    v = SOURCES[source][0]
    arr = np.asarray(D['%s_cd_%s' % (v, 'true' if source == 'true' else 'cv')][:, :w], np.float64)
    return rule_first(arr)


# ---------------------------------------------------------------------------------------------- fit / evaluate
GRID = np.round(np.arange(0.02, 0.905, 0.01), 2)


def _check_train(D, path, allow):
    banks = set(np.unique(D['state_bank']).tolist())
    if not allow and not all('train' in b for b in banks):
        raise SystemExit('contactmodel fit: %s holds banks %s; fit on the TRAIN split only' % (path, sorted(banks)))


def fit(train, out, source='true', lam=1.0, seeds=3, k=5, smooth=2, allow_nontrain=False):
    t0 = time.time()
    D = load(train)
    _check_train(D, train, allow_nontrain)
    keys = D['state_key']
    doc = dict(version=VERSION, kind='logistic', features=list(FEATURES), signs=list(SIGNS), tau=TAU, step_s=STEP_S,
               caps=dict(rc=RC_CAP, cd=CD_CAP), source=source, rule_m=RULE_M,
               data=dict(path=str(train), states=int(len(D['state_key'])), candidates=int(len(D['cand_state'])),
                         banks=sorted(set(np.unique(D['state_bank']).tolist()))),
               threshold_rule='max over a 0.01 grid of the within-state discrimination of p >= thr (mean over states '
                              'with both outcomes of 0.5 (1 + TPR - FPR)), out-of-fold predictions (%d-fold, grouped by '
                              'episode, %d seeds), moving average over +-%d grid points' % (k, seeds, smooth),
               windows={})
    for w in WINDOWS:
        F, y, S = dataset_features(D, source, w)
        Z = design(F)
        g = keys[S]
        oofs = []
        for seed in range(seeds):
            fo = folds(g, k, seed)
            o = np.zeros(len(y))
            for i in range(k):
                m = fit_logistic(Z[fo != i], y[fo != i], lam)
                o[fo == i] = predict_logistic(m, Z[fo == i])
            oofs.append(o)
        curve = np.asarray([np.mean([within_hard(o >= t, y, S) for o in oofs]) for t in GRID])
        sm = np.asarray([curve[max(0, i-smooth):i+smooth+1].mean() for i in range(len(curve))])
        thr = float(GRID[int(np.argmax(sm))])
        m = fit_logistic(Z, y, lam)
        p = predict_logistic(m, Z)
        rh, _ = _rule_arrays(D, source, w)
        oof = np.mean(oofs, 0)
        m.update(threshold=thr, n=int(len(y)), positives=int(y.sum()), mixed_states=len(mixed_states(y, S)),
                 oof=dict(logloss=float(np.mean([calibration(o, y)['logloss'] for o in oofs])),
                          auc=float(np.mean([auc(o, y) for o in oofs])),
                          within_auc=float(np.mean([within_auc(o, y, S)[0] for o in oofs])),
                          within_hard_at_thr=float(curve[int(np.argmax(sm))]),
                          within_hard_curve={'%.2f' % t: round(float(c), 4) for t, c in zip(GRID, curve)}),
                 train_in_sample=dict(within_hard_at_thr=within_hard(p >= thr, y, S),
                                      precision_recall_at_thr=precision_recall(p >= thr, y), auc=auc(p, y),
                                      within_auc=within_auc(p, y, S)[0], logloss=calibration(p, y)['logloss']),
                 train_rule=dict(within_hard=within_hard(rh, y, S), precision_recall=precision_recall(rh, y)),
                 oof_spread=float(np.std([calibration(o, y)['logloss'] for o in oofs])))
        doc['windows'][str(w)] = m
        print('W=%d: %d candidates, %d contacts, %d mixed states | OOF logloss %.4f AUC %.3f within-state AUC %.3f | '
              'threshold %.2f: within-state (hard) OOF %.3f, in-sample %.3f (rule centre < %.2f m: %.3f) | coef %s' % (
                  w, len(y), y.sum(), m['mixed_states'], m['oof']['logloss'], m['oof']['auc'], m['oof']['within_auc'],
                  thr, m['oof']['within_hard_at_thr'], m['train_in_sample']['within_hard_at_thr'], RULE_M,
                  m['train_rule']['within_hard'], ' '.join('%s=%.3f' % (f, c) for f, c in zip(FEATURES, m['coef']))),
              flush=True)
    doc['seconds'] = round(time.time()-t0, 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1))
    print('wrote', out)
    return doc


def _regimes(D, S):
    tracked = D['state_n_est'][S] > 0
    step1 = D['cand_real_steps'] == 1
    thru = np.char.endswith(D['cand_label'].astype(str), '_th')
    return dict(all=np.ones(len(S), bool), no_step1=~step1, someone_tracked=tracked, nobody_tracked=~tracked,
                thru=thru, non_thru=~thru)


def evaluate(model, data, out=None, thr=None):
    cm = ContactModel(model)
    D = load(data)
    res = dict(model=str(model), data=str(data), banks=sorted(set(np.unique(D['state_bank']).tolist())), windows={})
    for w in cm.windows:
        th = float(thr) if thr is not None else cm.threshold(w)
        rw = res['windows'][str(w)] = dict(threshold=th)
        for source in ('true', 'cv'):
            F, y, S = dataset_features(D, source, w)
            p = cm.prob_many(F, w)
            reg = _regimes(D, S)
            rh, rf = _rule_arrays(D, source, w)
            hit = p >= th
            e = dict(n=int(len(y)), positives=int(y.sum()), mixed_states=len(mixed_states(y, S)))
            e['auc'] = {k: auc(p[m], y[m]) for k, m in reg.items()}
            e['auc_rc_min'] = auc(-F['rc_min'], y)
            e['auc_cd_min'] = auc(-F['cd_min'], y)
            e['within_auc'] = within_auc(p, y, S)[0]
            e['within_auc_rc_min'] = within_auc(-F['rc_min'], y, S)[0]
            e['within_auc_cd_min'] = within_auc(-F['cd_min'], y, S)[0]
            e['calibration'] = calibration(p, y)
            e['calibration_regimes'] = {k: {kk: vv for kk, vv in calibration(p[m], y[m]).items() if kk != 'reliability'}
                                        for k, m in reg.items() if k in ('someone_tracked', 'nobody_tracked', 'no_step1')}
            e['model_hard'] = dict(within_hard=within_hard(hit, y, S), precision_recall=precision_recall(hit, y))
            e['rule_hard'] = dict(within_hard=within_hard(rh, y, S), precision_recall=precision_recall(rh, y))
            e['selection'] = dict(rule=simulate_selection(D, source, w, rh, rf),
                                  model=simulate_selection(D, source, w, hit, F['i_ca']),
                                  model_soft=simulate_selection(D, source, w, hit, F['i_ca'], p=p, soft=True))
            rw[source] = e
            c = e['calibration']
            print('W=%d %-4s (%d cand, %d contacts, %d mixed): AUC %.3f (no step-1 %.3f; tracked %.3f / nobody %.3f) '
                  'within %.3f [rc_min %.3f, cd_min %.3f] | Brier %.4f logloss %.4f ECE %.4f slope %.2f | '
                  'p>=%.2f: within %.3f P %.2f R %.2f | rule cd<%.2f: within %.3f P %.2f R %.2f' % (
                      w, source, e['n'], e['positives'], e['mixed_states'], e['auc']['all'], e['auc']['no_step1'],
                      e['auc']['someone_tracked'], e['auc']['nobody_tracked'], e['within_auc'],
                      e['within_auc_rc_min'], e['within_auc_cd_min'], c['brier'], c['logloss'], c['ece'],
                      c['recal_slope'], th, e['model_hard']['within_hard'], *e['model_hard']['precision_recall'],
                      RULE_M, e['rule_hard']['within_hard'], *e['rule_hard']['precision_recall']), flush=True)
            for name in ('rule', 'model', 'model_soft'):
                x = e['selection'][name]
                print('        selection replica %-10s critical %d fixed %d broken %d | chosen collisions %d (default %d) '
                      'deviations %d / %d' % (name, x['critical'], x['fixed'], x['broken'], x['collisions'],
                                              x['default_collisions'], x['deviations'], x['states']))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1))
        print('wrote', out)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['fit', 'evaluate'])
    ap.add_argument('--train', default='output/agents/bankimag/orca_train.npz')
    ap.add_argument('--data', default='output/agents/bankimag/orca_dev.npz')
    ap.add_argument('--model', default='output/agents/contact/contact_model.json')
    ap.add_argument('--out', default=None)
    ap.add_argument('--source', default='true', choices=sorted(SOURCES))
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--thr', type=float, default=None)
    a = ap.parse_args()
    if a.mode == 'fit':
        fit(a.train, a.out or a.model, source=a.source, lam=a.lam)
    else:
        evaluate(a.model, a.data, a.out, thr=a.thr)


if __name__ == '__main__':
    main()
