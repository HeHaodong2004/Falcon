"""Short-horizon imagination features and a learned legal scorer trained against a privileged critic.

Asymmetric training (critic privileged, actor legal):
  * critic  = the EXACT short-horizon evaluation of every candidate (querygen's branch simulator: the candidate's
              interval, then the default policy, to 64 steps = 3.1 s; contact first, then legal route progress) --
              ``shortprobe.py`` records it at the decisions of training episodes (privileged supervision only);
  * actor   = a ranking model over candidates whose inputs are LEGAL imagination outputs only: each candidate imagined
              in the 6 heaviest Stage 1 worlds with the camera model (perceive.VisibleWorld), and with the released
              Sandbox's nominal people, over 16 / 32 / 64 steps -- contact statistics, legal route progress, STOPs --
              plus the candidate family and its differences to the default candidate.

``features(cands, ctg0, default_index)`` turns per-candidate imagination records (the probe's ``imag`` format) into
the actor's input rows; ``imagine_candidates`` produces those records online (selector ``learnedshort``).

    python scripts/l3_run.py eventwm_graph.l3.shortsel fit --train output/shortprobe/orca_train \
        --eval output/shortprobe/orca_dev --save output/shortsel/v1
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

WINDOWS = (16, 32, 64)
EXPO = os.environ.get('L3_SHORT_EXPO') == '1'     # exposure features (needs probe records with L3_PROBE_EXPO=1)
THRESH = (0.1, 0.2, 0.3, 0.5)
W_MAX = 64
N_WORLDS = 6


def _clr(v):
    return 3. if v is None else min(float(v), 3.)


def _cand_vector(c, ctg0):
    """Legal imagination summary of one candidate (probe ``imag`` records of 'cam6' and 'rel')."""
    from .leafvalue import FAMILY_KEYS, family
    f = []
    runs = c['imag'].get('cam6') or []
    wsum = sum(r['w'] for r in runs) or 1.
    for w in WINDOWS:
        k = str(w)
        clr = np.asarray([_clr(r['clr'].get(k, r['clr'].get(w))) for r in runs]) if runs else np.full(1, 3.)
        ww = np.asarray([r['w']/wsum for r in runs]) if runs else np.ones(1)
        for t in THRESH:
            f.append(float((ww*(clr < t)).sum()))
        f += [float((ww*clr).sum()), float(clr.min()), float(clr.max())]
        ctg = np.asarray([(r['ctg'].get(k, r['ctg'].get(w)) if r['ctg'].get(k, r['ctg'].get(w)) is not None else ctg0)
                          for r in runs]) if runs else np.full(1, ctg0)
        prog = ctg0-ctg
        f += [float((ww*prog).sum()), float(prog.min()), float(prog.max())]
        f.append(float(sum(r['w']/wsum for r in runs if r['stop'] and r['n'] <= w)))
        f.append(float(sum(r['w']/wsum for r in runs if r['stop'] and r['n'] <= w and r['near_goal'])))
    rel = (c['imag'].get('rel') or [None])[0]
    for w in WINDOWS:
        k = str(w)
        if rel is None:
            f += [3., 0., 0.]
        else:
            v = rel['ctg'].get(k, rel['ctg'].get(w))
            f += [_clr(rel['clr'].get(k, rel['clr'].get(w))), ctg0-(v if v is not None else ctg0),
                  float(rel['stop'] and rel['n'] <= w)]
    fam = family(c['label'], c.get('kind'))
    f += [float(fam == k) for k in FAMILY_KEYS]
    f.append(float(c.get('h', 16))/16.)
    return f


N_BASE = None
DIFF_INDEX = None


def features(cands, ctg0, default_index, ego=None):
    """Rows [K, F]: candidate summaries, their differences to the default's, is-default flag, decision context;
    with exposure features (probe L3_PROBE_EXPO=1 / online) the candidates' CAND_INFO and the state's EGO_INFO."""
    base = np.asarray([_cand_vector(c, ctg0)+(list(c['expo']) if ego is not None else []) for c in cands], np.float64)
    d = base[int(default_index)]
    diff = base-d[None]
    ctx = np.tile(np.asarray([ctg0, len(cands)], np.float64), (len(cands), 1))
    isd = np.zeros((len(cands), 1))
    isd[int(default_index)] = 1.
    parts = [base, diff, isd, ctx]
    if ego is not None:
        parts.append(np.tile(np.asarray(ego, np.float64), (len(cands), 1)))
    return np.concatenate(parts, 1)


def relevance(cands, w=64):
    """Graded relevance of each candidate under the exact key at window ``w`` (the privileged critic): contact or STOP
    failure 0, running 1..3 by legal route cost-to-go relative to the best running candidate, success 4."""
    out = []
    best = min([c['real']['ctg'].get(str(w)) for c in cands
                if c['real']['ctg'].get(str(w)) is not None and not _fails(c, w)] or [math.inf])
    for c in cands:
        r = c['real']
        if _fails(c, w):
            out.append(0)
        elif r['outcome'] == 'success' and r['steps'] <= w:
            out.append(4)
        else:
            v = r['ctg'].get(str(w))
            gap = (v-best) if v is not None else 99.
            out.append(3 if gap <= 0.25 else 2 if gap <= 1.0 else 1)
    return out


def _fails(c, w):
    r = c['real']
    return r['outcome'] in ('human_collision', 'stop_fail') and r['steps'] <= w


def load_states(path):
    states = []
    for f in sorted(Path(path).glob('*.json')):
        d = json.loads(f.read_text())
        for s in d.get('probe', []):
            if s['cands'] and s['default'] in s['labels'] and (not EXPO or ('ego' in s and all('expo' in c for c in s['cands']))):
                s['episode'] = d['key']
                states.append(s)
    return states


def dataset(states, w=64):
    X, y, g, meta = [], [], [], []
    for s in states:
        di = s['labels'].index(s['default'])
        F = features(s['cands'], s['ctg0'], di, s.get('ego') if EXPO else None)
        X.append(F); y += relevance(s['cands'], w); g.append(len(s['cands']))
        meta.append((s['episode'], s['step'], di))
    return np.concatenate(X), np.asarray(y), g, meta


def fit_ranker(X, y, g, rounds=400, seed=0):
    import lightgbm as lgb
    params = dict(objective='lambdarank', learning_rate=0.05, num_leaves=31, min_data_in_leaf=30, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed,
                  lambdarank_truncation_level=3, label_gain=[0, 1, 3, 7, 15], num_threads=int(os.environ.get('L3_LGB_THREADS', '2')))
    return lgb.train(params, lgb.Dataset(X, y, group=g), rounds)


def evaluate_scores(states, scores, margin=0., w=64):
    """Exact-window metrics of picking the top-scored candidate (default kept within ``margin``)."""
    from .shortprobe import _key_real
    n = cat_ok = crit = fixes = breaks = dev = 0
    reg = []
    pos = 0
    for s in states:
        k = len(s['cands'])
        sc = scores[pos:pos+k]; pos += k
        di = s['labels'].index(s['default'])
        top = int(np.argmax(sc))
        pick = top if sc[top] > sc[di]+margin else di
        kr = [_key_real(c, w) for c in s['cands']]
        best = max(kr)
        n += 1
        cat_ok += int(kr[pick][0] == best[0])
        if kr[pick][0] == best[0] == 2.:
            reg.append(float(best[1]-kr[pick][1]))
        dev += int(pick != di)
        if kr[di][0] == 0. and best[0] > 0.:
            crit += 1
            fixes += int(kr[pick][0] > 0.)
        if kr[di][0] > 0. and kr[pick][0] == 0.:
            breaks += 1
    return dict(states=n, category_ok=cat_ok/max(n, 1), ctg_regret=float(np.mean(reg)) if reg else 0., critical=crit,
                fixed=fixes, breaks=breaks, deviations=dev/max(n, 1))


# ------------------------------------------------------------------------------------------------ online
def imagine_candidates(sb, legal, d, cfg, steps_index):
    """The legal imagination records of every candidate at a live decision (selector use): 'rel' (released Sandbox
    people: Stage 1 mean of the tracked people) and 'cam6' (camera model in each of the 6 heaviest Stage 1 worlds),
    interval + default continuation to 64 steps with continuation fidelity -- the probe's legal variants."""
    from eventwm_graph.sandbox.imagine import imagine, NominalHumans, frozen_map
    from eventwm_graph.sandbox.episode import default_query
    from eventwm_graph.sandbox import accel
    from eventwm_graph.querygen.dynamics import FAMILIES
    from .compiler import LegalField
    from .perceive import VisibleWorld
    from .shortprobe import _world_clearance, _min_upto, _ctg_at
    dyn = FAMILIES[cfg.dynamics]
    goal = default_query()
    field = LegalField(legal)
    gmap = frozen_map(legal.map)
    bank = d.result.humans
    W = np.asarray(bank.weights, np.float64)
    W = W/W.sum() if W.sum() > 0 else np.full(len(W), 1./max(len(W), 1))
    worlds = np.argsort(-W)[:N_WORLDS] if len(np.unique(W)) > 1 else np.arange(min(N_WORLDS, len(W)))
    ww = W[worlds]/W[worlds].sum()
    rel = np.arange(1, W_MAX+1)*0.048
    HP = None
    if len(getattr(bank, 'track_keys', ())):
        inside = rel[rel <= float(bank.times[-1])+1e-9]
        HP = np.asarray(bank.sandbox_positions(inside), np.float64)
        if len(inside) < W_MAX and HP.shape[1] >= 2:
            vel = (HP[:, -1]-HP[:, -2])/0.048
            extra = np.arange(1, W_MAX-len(inside)+1)*0.048
            HP = np.concatenate([HP, HP[:, -1][:, None]+extra[None, :, None, None]*vel[:, None]], 1)
    from eventwm_graph.querygen.oracle import horizon_for
    H = type('H', (), {'horizon': 16})()
    info = None
    if EXPO:
        from .shortprobe import expo_state
        info = expo_state(legal, bank)
    out = []
    cache = accel.FrozenMapCache() if sb.config.memoize else None
    with accel.active(cache):
        for q, cand in zip(d.queries, d.candidates):
            cont = q.kind == 'continue'
            h = min(horizon_for(q, H) if not cont else 16, 500-int(steps_index))
            rec = dict(label=q.label, kind=q.kind, h=int(h), imag={})
            for v in ('rel', 'cam6'):
                runs = []
                specs = ([(int(w), float(x)) for w, x in zip(worlds, ww)] if HP is not None else []) if v == 'cam6' \
                    else [(None, 1.)]
                for w, wt in specs:
                    ex, _ = sb._execution(cand)
                    if isinstance(ex, Exception):
                        break
                    hum = NominalHumans(legal.tracks, legal.time, W_MAX, bank=bank, mode='mean') if v == 'rel' \
                        else VisibleWorld(legal, bank, w, W_MAX)
                    fid = v != 'rel'
                    ro = imagine(legal, ex, dyn, hum, h, level=int(cand.level), gmap=gmap,
                                 continuation=max(0, W_MAX-h), cont_query=goal, planner=cfg.planner,
                                 cont_boundary=16 if fid else 0, cont_inherit=fid)
                    c = getattr(ro, 'cont', None)
                    im = np.concatenate([ro.poses[1:], c]) if c is not None and len(c) else np.asarray(ro.poses[1:])
                    im = np.asarray(im, np.float64).reshape(-1, 4)
                    if v == 'cam6':
                        ic = _world_clearance(im, HP[w])
                    else:
                        ic = _world_clearance(im, np.asarray(hum.positions[1:, :, :], np.float64)[:, :, [0, 2]])
                        if info is not None:
                            from .shortprobe import expo_cand
                            rec['expo'] = expo_cand(info, legal, im)
                    runs.append(dict(w=wt, n=len(im), stop=bool(ro.stop), ended=ro.ended_by,
                                     near_goal=bool(len(im) and math.hypot(im[-1, 0]-legal.goal[0], im[-1, 2]-legal.goal[2]) <= 0.3),
                                     clr={str(w2): _min_upto(ic, w2) for w2 in WINDOWS},
                                     ctg={str(w2): _ctg_at(field, im, w2) for w2 in WINDOWS}))
                rec['imag'][v] = runs
            out.append(rec)
    ego = [float(x) for x in info.ego()] if info is not None else None
    return out, float(field.start), ego


_MODEL = {}


def select(sb, legal, d, cfg, steps_index):
    """Selector ``learnedshort``: the legal scorer's top candidate; the default is kept unless the top score exceeds
    the default's by L3_SHORT_MARGIN (default 0.3)."""
    path = os.environ['L3_SHORT_MODEL']
    if path not in _MODEL:
        from .leafvalue import ValueModel
        _MODEL[path] = ValueModel(path)
    m = _MODEL[path]
    cands, ctg0, ego = imagine_candidates(sb, legal, d, cfg, steps_index)
    X = features(cands, ctg0, d.default, ego)
    raw = m.predict_raw(X) if hasattr(m, 'predict_raw') else m.predict(X)
    top = int(np.argmax(raw))
    margin = float(os.environ.get('L3_SHORT_MARGIN', m.meta.get('margin', 0.3)))
    return top if raw[top] > raw[d.default]+margin else int(d.default)


def select_from_records(rec, d):
    """``select`` on the legal parts of a probe record (its 'rel' / 'cam6' imagination and exposure features)."""
    path = os.environ['L3_SHORT_MODEL']
    if path not in _MODEL:
        from .leafvalue import ValueModel
        _MODEL[path] = ValueModel(path)
    X = features(rec['cands'], rec['ctg0'], d.default, rec.get('ego') if EXPO else None)
    raw = _MODEL[path].predict_raw(X)
    top = int(np.argmax(raw))
    margin = float(os.environ.get('L3_SHORT_MARGIN', _MODEL[path].meta.get('margin', 0.3)))
    return top if raw[top] > raw[d.default]+margin else int(d.default)


def crossval(states, folds=5, w=64, rounds=300, margins=(0., 0.3, 1.0), seed=0):
    """Episode-grouped cross-validation of the legal ranker on probe states (analysis): learned ranker vs the raw
    legal imagination rule (shortprobe._key_imag on 'cam6') vs the default."""
    from .shortprobe import _key_imag
    eps = sorted({s['episode'] for s in states})
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    fold = {e: i % folds for i, e in enumerate(eps)}
    scores = [None]*len(states)
    for k in range(folds):
        tr = [s for s in states if fold[s['episode']] != k]
        te = [i for i, s in enumerate(states) if fold[s['episode']] == k]
        if not tr or not te:
            continue
        X, y, g, _ = dataset(tr, w)
        b = fit_ranker(X, y, g, rounds, seed+k)
        for i in te:
            s = states[i]
            di = s['labels'].index(s['default'])
            scores[i] = b.predict(features(s['cands'], s['ctg0'], di, s.get('ego') if EXPO else None))
    flat = np.concatenate(scores)
    for m in margins:
        for win in (32, 64):
            r = evaluate_scores(states, flat, m, win)
            print('  ranker margin %.1f window %d: category = best %.3f  ctg regret %.3f  critical %d fixed %d  breaks %d  '
                  'deviations %.2f' % (m, win, r['category_ok'], r['ctg_regret'], r['critical'], r['fixed'], r['breaks'],
                                       r['deviations']))
    for thr in (0.2, 0.5):
        raw = []
        for s in states:
            ki = [_key_imag(c['imag'].get('cam6', []), 64, thr) for c in s['cands']]
            order = sorted(range(len(ki)), key=lambda i: ki[i])
            rank = np.empty(len(ki)); rank[order] = np.arange(len(ki))
            raw.append(rank)
        flat = np.concatenate(raw)
        for win in (32, 64):
            r = evaluate_scores(states, flat, 0.5, win)
            print('  cam6 rule thr %.1f window %d: category = best %.3f  ctg regret %.3f  critical %d fixed %d  breaks %d  '
                  'deviations %.2f' % (thr, win, r['category_ok'], r['ctg_regret'], r['critical'], r['fixed'], r['breaks'],
                                       r['deviations']))
    for win in (32, 64):
        r = evaluate_scores(states, np.zeros(sum(len(s['cands']) for s in states)), 1e9, win)
        print('  default window %d: category = best %.3f  critical %d fixed %d' % (win, r['category_ok'], r['critical'], r['fixed']))


def choose_margin(states, w=64, rounds=300, folds=5, margins=(0., 0.25, 0.5, 1., 2., 4.), seed=0):
    """Margin maximising the exact window utility (asymac.utilities) of the chosen candidate, by episode-grouped
    cross-validation on TRAINING states only."""
    from .asymac import utilities
    eps = sorted({s['episode'] for s in states})
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    fold = {e: i % folds for i, e in enumerate(eps)}
    scores = [None]*len(states)
    for k in range(folds):
        tr = [s for s in states if fold[s['episode']] != k]
        te = [i for i, s in enumerate(states) if fold[s['episode']] == k]
        if not tr or not te:
            continue
        X, y, g, _ = dataset(tr, w)
        b = fit_ranker(X, y, g, rounds, seed+k)
        for i in te:
            s = states[i]
            scores[i] = b.predict(features(s['cands'], s['ctg0'], s['labels'].index(s['default']),
                                           s.get('ego') if EXPO else None))
    best = None
    for m in margins:
        u = []
        for s, sc in zip(states, scores):
            if sc is None:
                continue
            di = s['labels'].index(s['default'])
            top = int(np.argmax(sc))
            pick = top if sc[top] > sc[di]+m else di
            u.append(float(utilities(s['cands'], w)[pick]))
        mu = float(np.mean(u)) if u else 0.
        print('  train-CV margin %.2f: mean exact utility of the choice %.4f' % (m, mu), flush=True)
        if best is None or mu > best[1]:
            best = (m, mu)
    du = float(np.mean([utilities(s['cands'], w)[s['labels'].index(s['default'])] for s in states]))
    print('  train-CV default: %.4f -> chosen margin %.2f' % (du, best[0]), flush=True)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['fit', 'cv'])
    ap.add_argument('--train', required=True)
    ap.add_argument('--eval', default=None)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--rounds', type=int, default=400)
    ap.add_argument('--save', default=None)
    ap.add_argument('--margins', default='0,0.1,0.3,0.5,1.0')
    a = ap.parse_args()
    if a.mode == 'cv':
        st = load_states(a.train)
        print('cv states %d episodes %d' % (len(st), len({s['episode'] for s in st})), flush=True)
        crossval(st, w=a.window, rounds=a.rounds)
        return
    tr = load_states(a.train)
    X, y, g, _ = dataset(tr, a.window)
    print('train states %d rows %d relevance %s' % (len(g), len(y), np.bincount(y, minlength=5).tolist()), flush=True)
    margin = choose_margin(tr, a.window, a.rounds)
    b = fit_ranker(X, y, g, a.rounds)
    if a.eval:
        ev = load_states(a.eval)
        Xe, ye, ge, _ = dataset(ev, a.window)
        sc = b.predict(Xe)
        print('eval states %d' % len(ge))
        for m in (float(x) for x in a.margins.split(',')):
            r = evaluate_scores(ev, sc, m, a.window)
            print('  margin %.2f  exact category = best %.3f  ctg regret %.3f  critical %d fixed %d  breaks %d  deviations %.2f'
                  % (m, r['category_ok'], r['ctg_regret'], r['critical'], r['fixed'], r['breaks'], r['deviations']))
        r = evaluate_scores(ev, np.zeros(len(ye)), 1e9, a.window)
        print('  default        exact category = best %.3f  ctg regret %.3f  critical %d fixed %d  breaks %d' % (
            r['category_ok'], r['ctg_regret'], r['critical'], r['fixed'], r['breaks']))
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
        b.save_model(str(Path(a.save, 'model.txt')))
        Path(a.save, 'model.json').write_text(json.dumps(b.dump_model()))
        Path(a.save, 'meta.json').write_text(json.dumps(dict(keys=['shortsel'], window=a.window, rows=int(len(y)),
                                                             train=a.train, objective='lambdarank', margin=margin,
                                                             expo=EXPO)))


if __name__ == '__main__':
    main()
