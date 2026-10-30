"""Offline reliability benchmark of the closed-loop Sandbox on the state bank (``statebank.py``).

Every banked decision holds the legal inputs and the EXACT outcome of every candidate (interval + default policy for
16 / 32 / 64 steps in the branch simulator).  A Sandbox variant (legal) is re-run on the banked inputs and scored:

* collision prediction: AUC of the variant's contact score for a real collision within 0.77 / 1.5 / 3.1 s, and
  precision / recall of contact-probability >= 0.5;
* selection: the outcome rule on the variant's predictions (contact first, then legal progress; default kept unless
  another candidate is better by ``margin``) -- exact category agreement with the best candidate, critical states
  fixed / broken (critical: the default collides within the window and some candidate does not);
* people (mode ``people``): Layer 2's Stage 1 bank vs the true people -- tracked: best-of-worlds and weighted-mean
  displacement at 0.77 / 1.5 / 3.1 s; untracked people that come within 3 m of the robot within 3.1 s: coverage
  (some world places SOME person within 1 m of them at that time) and the weighted mass of such worlds.

    python scripts/l3_run.py eventwm_graph.l3.sbeval run --bank output/statebank/orca_dev --variant cl_frozen \
        --out output/sbeval --shard i --shards n
    python scripts/l3_run.py eventwm_graph.l3.sbeval summary --out output/sbeval --bank-name orca_dev
    python scripts/l3_run.py eventwm_graph.l3.sbeval people --bank output/statebank/orca_dev
"""
import argparse
import glob
import json
import math
import os
import pickle
import time
import zlib
from pathlib import Path

import numpy as np

WINDOWS = (16, 32, 64)


class _NoLayer2:
    """Placeholder Layer 2 for variants that never call it."""
    def close(self):
        pass

    def reset(self, stream=None):
        pass


class TrueTracksCV:
    """ANALYSIS ONLY (privileged): the legal tracks of ``oraclebank.CVBank`` at their TRUE future (the person nearest
    to the constant-velocity estimate within 0.75 m), constant velocity when unmatched; nobody else -- the ceiling of a
    tracked-people forecaster on the same states (``cl_frozen_truetrk``)."""

    def __init__(self, item):
        from .oraclebank import CVBank
        self.cv = CVBank(item['legal'])
        for k in ('track_keys', 'unseen', 'weights', 'present', 'times', 'long_steps'):
            setattr(self, k, getattr(self.cv, k))
        self.T = np.asarray(item['truth_people'], np.float64)                  # [65, N, 2], row j = t + j steps
        self.match = {}
        for n, p0 in enumerate(self.cv.P0):
            d = np.hypot(self.T[0, :, 0]-p0[0], self.T[0, :, 1]-p0[1]) if self.T.shape[1] else np.zeros(0)
            if len(d) and np.isfinite(d).any() and np.nanmin(d) < 0.75:
                self.match[n] = int(np.nanargmin(d))

    def sandbox_positions(self, rel):
        P = np.array(self.cv.sandbox_positions(rel), np.float64)
        idx = np.clip(np.round(np.asarray(rel, np.float64)/0.048).astype(int), 0, len(self.T)-1)
        for n, j in self.match.items():
            P[0, :, n] = self.T[idx, j]
        return P


class TrueTrackedBank:
    """ANALYSIS ONLY: the banked Stage 1 bank with every TRACKED slot replaced by its matched person's true future."""

    def __init__(self, item):
        from eventwm_graph.querygen.control import human_estimates
        self.s1 = StoredBank(item['bank'])
        for k in ('track_keys', 'weights', 'present', 'unseen', 'times', 'long_steps'):
            setattr(self, k, getattr(self.s1, k))
        T = np.asarray(item['truth_people'], np.float64)                      # [65, N, 2], row j = t + j steps
        legal = item['legal']
        keys = list(self.track_keys)
        self.true_slot = {}
        for e in human_estimates(legal.tracks, legal.time):
            k = 'track:%s' % e.track_id
            if k not in keys or not T.shape[1]:
                continue
            d0 = np.hypot(T[0, :, 0]-e.position[0], T[0, :, 1]+e.position[1])
            if np.nanmin(d0) < 0.75:
                self.true_slot[keys.index(k)] = T[:, int(np.nanargmin(d0))]
        self.T = T

    def sandbox_positions(self, rel):
        rel = np.asarray(rel, np.float64)
        out = np.asarray(self.s1.sandbox_positions(rel), np.float64).copy()
        idx = np.clip(np.round(rel/0.048).astype(int), 0, len(self.T)-1)
        for slot, path in self.true_slot.items():
            out[:, :, slot] = path[idx][None]
        return out


class StoredBank:
    """The banked Stage 1 bank with the interface the Sandbox reads (track_keys, weights, present, unseen, times,
    sandbox_positions)."""

    def __init__(self, arrays):
        self.track_keys = tuple(arrays['track_keys'])
        self.weights = np.asarray(arrays['weights'], np.float64)
        self.present = np.asarray(arrays['present'], bool)
        self.unseen = np.asarray(arrays['unseen'], bool)
        self.times = np.asarray(arrays['times'], np.float64)
        self.P = np.asarray(arrays['positions'], np.float64)            # [W, T, N, 2] at rel = (t+1) * 0.048
        self.long_steps = 0

    def sandbox_positions(self, rel):
        rel = np.asarray(rel, np.float64)
        idx = np.clip(np.round(rel/0.048).astype(int)-1, 0, self.P.shape[1]-1)
        out = self.P[:, idx]
        # rel ~ 0: the current positions -> first row backed off by one step of the first displacement
        if np.any(rel < 0.024) and self.P.shape[1] >= 2:
            first = self.P[:, 0]-(self.P[:, 1]-self.P[:, 0])
            out = out.copy()
            out[:, rel < 0.024] = first[:, None]
        return out


VARIANTS = {
    # name: (ClosedLoopConfig kwargs, people source)
    'cl_frozen': (dict(window=64, map_loop='off'), 'stage1'),
    'cl_frozen_p': (dict(window=64, map_loop='off'), 'stage1'),            # + pose checkpoints (occupancy hazard)
    'cl_s2_p': (dict(window=64, map_loop='stage2_union'), 'stage1'),
    'cl_cluster8_p': (dict(window=64, map_loop='off', worlds=8, selection='cluster'), 'stage1'),
    'cl_cluster8_rw': (dict(window=64, map_loop='off', worlds=8, selection='cluster', reweight='output/worldweights/v1'), 'stage1'),
    'cl_heavy8_rw': (dict(window=64, map_loop='off', worlds=8, selection='heaviest', reweight='output/worldweights/v1'), 'stage1'),
    # Layer 2 v2 people module: tracked people from the learned multimodal forecaster (l3/peoplefc.py), hidden people
    # from Stage 1; 12 sampled worlds, all imagined in closed loop
    'cl_v2': (dict(window=64, map_loop='off', worlds=12), 'v2:output/peoplefc/v1'),
    # ANALYSIS ONLY (privileged): tracked people follow their TRUE future, everyone else stays Stage 1's hidden slots
    'cl_truetracked': (dict(window=64, map_loop='off', worlds=8, selection='cluster'), 'truetracked'),
    'cl_s2': (dict(window=64, map_loop='stage2_union'), 'stage1'),
    'cl_frozen_cv': (dict(window=64, map_loop='off', worlds=1), 'cv'),
    # map-aware multimodal forecaster of the tracked people (l3/peoplemap.py, model L3_FC_MODEL; legal): one world
    # (weighted mean / top mode) or the 6 most probable joint mode combinations, weighted
    'cl_frozen_fc1': (dict(window=64, map_loop='off', worlds=1), 'fc1'),
    'cl_frozen_fck': (dict(window=64, map_loop='off', worlds=6), 'fck'),
    # ANALYSIS ONLY (privileged): the same tracked people at their TRUE future (TrueTracksCV), nobody else
    'cl_frozen_truetrk': (dict(window=64, map_loop='off', worlds=1), 'truetrk'),
    'cl_frozen_all': (dict(window=64, map_loop='off', perception='all'), 'stage1'),
    'cl_frozen_w12': (dict(window=64, map_loop='off', worlds=12), 'stage1'),
}


_WEIGHTER = {}
_FORECASTER = {}


def candidates_of(item):
    from eventwm_graph.sandbox.core import Candidate
    from .env import ConfigSpec
    cfg = ConfigSpec.parse(item['config'])
    ex = item['execution']
    out = []
    for q in item['queries']:
        if q.kind == 'continue':
            out.append(Candidate(None, cfg.planner, cfg.dynamics, execution=ex, level=item['level'], query_id='continue'))
        else:
            out.append(Candidate(q, cfg.planner, cfg.dynamics, level=item['level'], query_id=q.label, execution=ex))
    return out


def run_variant(sb, item, name):
    from eventwm_graph.sandbox.closedloop import imagine_closed_loop, ClosedLoopConfig
    from .compiler import LegalField
    kw, people = VARIANTS[name]
    legal = item['legal']
    if people == 'cv':
        from .oraclebank import CVBank
        bank = CVBank(legal)
    elif people == 'truetracked':
        bank = TrueTrackedBank(item)
    elif people == 'truetrk':
        bank = TrueTracksCV(item)
    elif people in ('fc1', 'fck'):
        from .peoplemap import FCBank
        bank = FCBank(legal, mode=people, worlds=int(kw.get('worlds', 6)))
    elif people.startswith('v2:'):
        from .peoplefc import Forecaster, PeopleBankV2
        fpath = people[3:]
        fc = _FORECASTER.setdefault(fpath, Forecaster(fpath))
        bank = PeopleBankV2(legal, StoredBank(item['bank']), fc, worlds=int(kw.get('worlds', 12)))
    else:
        bank = StoredBank(item['bank'])
    kw = dict(kw)
    if isinstance(kw.get('reweight'), str):
        from .worldweights import WorldWeighter
        kw['reweight'] = _WEIGHTER.setdefault(kw['reweight'], WorldWeighter(kw['reweight']))
    res = imagine_closed_loop(sb, item['state'], legal, candidates_of(item), bank, ClosedLoopConfig(**kw))
    field = LegalField(legal)
    ctg0 = float(field.start)
    preds = []
    for k in range(len(item['queries'])):
        rec = dict(contact={}, clr={}, prog={}, stop={})
        for w in WINDOWS:
            c = res.clearance(k, w)
            ws = np.asarray([x.weight for x in res.worlds])
            ws = ws/max(ws.sum(), 1e-9)
            rec['clr'][w] = [float(min(v, 9.)) for v in c]
            mc = res.member_clearance(k, w)
            rec.setdefault('mclr', {})[w] = [[float(a), float(min(b, 9.))] for a, b in mc]
            rec['contact'][w] = res.contact(k, w, 0.2)
            prog, stop = [], []
            for x in res.worlds:
                ro = x.rollouts[k]
                seq = res._seq(ro)
                p = seq[min(w, len(seq))-1]
                prog.append(ctg0-float(field.at(p[0], p[2])))
                near = math.hypot(seq[-1, 0]-legal.goal[0], seq[-1, 2]-legal.goal[2]) <= 0.3
                stop.append((1 if near else 2) if (ro.stop and len(seq) <= w) else 0)
            rec['prog'][w] = float((ws*np.asarray(prog)).sum())
            rec['stop'][w] = [int(s) for s in stop]
            rec['weights'] = [float(v) for v in ws]
        # pose checkpoints of every world's imagined trajectory (steps 4, 8, ..., 64): x, z, yaw
        cps = []
        for x in res.worlds:
            seq = res._seq(x.rollouts[k])
            cps.append([[float(v) for v in seq[min(j, len(seq))-1][[0, 2, 3]]] for j in range(4, 65, 4)])
        rec['poses'] = cps
        preds.append(rec)
    from eventwm_graph.querygen.control import human_estimates
    return dict(preds=preds, profile=res.profile, tracked=len(human_estimates(legal.tracks, legal.time)))


def _real_coll(c, w):
    r = c['real']
    return r['outcome'] == 'human_collision' and r['steps'] <= w


def _real_key(c, w):
    from .shortprobe import _key_real
    return _key_real(c, w)


def score_states(results, margin=0.25, contact=0.5):
    """Collision prediction and selection metrics over a list of (probe record, predictions)."""
    out = {}
    for w in WINDOWS:
        y, s = [], []
        tp = fp = fn = 0
        agree = n = crit = fixed = broke = dev = 0
        for probe, pr in results:
            cands = probe['cands']
            preds = pr['preds']
            for c, p in zip(cands, preds):
                y.append(_real_coll(c, w))
                s.append(p['contact'][w])
                hit = p['contact'][w] >= contact
                tp += int(hit and y[-1]); fp += int(hit and not y[-1]); fn += int((not hit) and y[-1])
            # selection: outcome rule on the predictions
            di = probe['labels'].index(probe['default'])
            val = []
            for p in preds:
                pc = p['contact'][w]
                stp = p['stop'][w]
                goal = sum(wt for wt, st in zip(p['weights'], stp) if st == 1)
                bad = sum(wt for wt, st in zip(p['weights'], stp) if st == 2)
                val.append((1.-pc-bad)*2.+goal*1.+0.05*p['prog'][w])
            top = int(np.argmax(val))
            pick = top if val[top] > val[di]+margin else di
            kr = [_real_key(c, w) for c in cands]
            best = max(kr)
            n += 1
            agree += int(kr[pick][0] == best[0])
            dev += int(pick != di)
            if kr[di][0] == 0. and best[0] > 0.:
                crit += 1
                fixed += int(kr[pick][0] > 0.)
            if kr[di][0] > 0. and kr[pick][0] == 0.:
                broke += 1
        y = np.asarray(y, bool); s = np.asarray(s, float)
        auc = float('nan')
        if y.any() and (~y).any():
            pos, neg = s[y], s[~y]
            auc = float((pos[:, None] > neg[None]).mean()+0.5*(pos[:, None] == neg[None]).mean())
        out[w] = dict(auc=auc, precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1), collisions=int(y.sum()),
                      candidates=int(len(y)), category_ok=agree/max(n, 1), critical=crit, fixed=fixed, broke=broke,
                      deviations=dev/max(n, 1), states=n)
    return out


def people_audit(paths, horizons=(16, 32, 64), near_m=3.0, cover_m=1.0):
    """Stage 1 bank vs the true people (privileged evaluation)."""
    from eventwm_graph.querygen.control import human_estimates
    tr_err = {h: [] for h in horizons}
    tr_best = {h: [] for h in horizons}
    un_cover = {h: [] for h in horizons}
    un_mass = {h: [] for h in horizons}
    n_states = n_tracked_states = 0
    for p in paths:
        item = pickle.load(open(p, 'rb'))
        legal = item['legal']
        truth = np.asarray(item['truth_people'], np.float64)           # [65, N, 2], row 0 = t
        bank = item['bank']
        P = np.asarray(bank['positions'], np.float64)                  # [W, T, M, 2], row j = t + (j+1) steps
        W = np.asarray(bank['weights'], np.float64); W = W/max(W.sum(), 1e-9)
        keys = list(bank['track_keys'])
        pose = np.asarray(legal.pose, np.float64)
        n_states += 1
        # tracked people at t: legal estimates -> nearest true person at t
        est = human_estimates(legal.tracks, legal.time)
        tracked_true = set()
        for h in est:
            x, z = float(h.position[0]), -float(h.position[1])
            d = np.hypot(truth[0, :, 0]-x, truth[0, :, 1]-z)
            if len(d) and np.nanmin(d) < 0.75:
                j = int(np.nanargmin(d))
                tracked_true.add(j)
                slot = keys.index('track:%s' % h.track_id) if ('track:%s' % h.track_id) in keys else None
                for hz in horizons:
                    hp = min(hz, P.shape[1])                              # the bank's horizon (~3 s) caps 64 steps
                    if slot is None or hp < 1:
                        continue
                    pred = P[:, hp-1, slot]                              # [W, 2]
                    ok = np.isfinite(pred).all(1)
                    if not ok.any():
                        continue
                    e = np.hypot(pred[ok, 0]-truth[hp, j, 0], pred[ok, 1]-truth[hp, j, 1])
                    tr_best[hz].append(float(e.min()))
                    tr_err[hz].append(float((W[ok]*e).sum()/max(W[ok].sum(), 1e-9)))
        if est:
            n_tracked_states += 1
        # untracked people that come within near_m of the robot's CURRENT position within the horizon
        for j in range(truth.shape[1]):
            if j in tracked_true:
                continue
            for hz in horizons:
                hp = min(hz, P.shape[1])
                if hp < 1:
                    continue
                tp = truth[hp, j]
                if not np.isfinite(tp).all() or math.hypot(tp[0]-pose[0], tp[1]-pose[2]) > near_m:
                    continue
                pred = P[:, hp-1]                                       # [W, M, 2]
                d = np.hypot(pred[..., 0]-tp[0], pred[..., 1]-tp[1])
                d = np.where(np.isfinite(d), d, np.inf).min(axis=1) if pred.shape[1] else np.full(len(W), np.inf)
                un_cover[hz].append(float((d < cover_m).any()))
                un_mass[hz].append(float((W*(d < cover_m)).sum()))
    print('states %d, with someone tracked %d' % (n_states, n_tracked_states))
    for hz in horizons:
        def q(v):
            return ('%.2f / %.2f (n=%d)' % (np.median(v), np.percentile(v, 90), len(v))) if v else 'n/a'
        print('  %.2f s  tracked: best-of-worlds err %s m, weighted err %s m | untracked near the robot: coverage %.2f, '
              'world mass %.2f (n=%d)' % (hz*0.048, q(tr_best[hz]), q(tr_err[hz]),
                                          np.mean(un_cover[hz]) if un_cover[hz] else float('nan'),
                                          np.mean(un_mass[hz]) if un_mass[hz] else float('nan'), len(un_cover[hz])))


def occupancy_calibration(paths, horizons=(16, 32, 62), radius=0.5, extent=3.0, spacing=0.5):
    """Near-field people occupancy: Stage 1's predicted probability that a person is within ``radius`` of a grid point
    around the robot at t + h, against the truth (privileged evaluation); a constant-velocity-tracks baseline."""
    from eventwm_graph.querygen.control import human_estimates
    g = np.arange(-extent, extent+1e-9, spacing)
    gx, gz = np.meshgrid(g, g)
    keep = np.hypot(gx, gz) <= extent
    gx, gz = gx[keep], gz[keep]
    rows = {h: dict(p=[], cv=[], y=[]) for h in horizons}
    for path in paths:
        item = pickle.load(open(path, 'rb'))
        legal = item['legal']
        pose = np.asarray(legal.pose, np.float64)
        px, pz = pose[0]+gx, pose[2]+gz
        truth = np.asarray(item['truth_people'], np.float64)
        bank = item['bank']
        P = np.asarray(bank['positions'], np.float64)
        W = np.asarray(bank['weights'], np.float64); W = W/max(W.sum(), 1e-9)
        est = human_estimates(legal.tracks, legal.time)
        for h in horizons:
            hp = min(h, P.shape[1])
            tp = truth[hp]
            tp = tp[np.isfinite(tp).all(1)]
            y = (np.hypot(px[:, None]-tp[None, :, 0], pz[:, None]-tp[None, :, 1]) < radius).any(1) if len(tp) else np.zeros(len(px), bool)
            if P.shape[2]:
                pr = P[:, hp-1]                                        # [W, M, 2]
                d = np.hypot(px[None, :, None]-pr[:, None, :, 0], pz[None, :, None]-pr[:, None, :, 1])
                occ = (np.where(np.isfinite(d), d, np.inf) < radius).any(2)   # [W, G]
                p = (W[:, None]*occ).sum(0)
            else:
                p = np.zeros(len(px))
            cv = np.zeros(len(px))
            for e in est:
                x, z = e.at(hp*0.048)
                cv = np.maximum(cv, (np.hypot(px-x, pz-(-z)) < radius).astype(float))
            rows[h]['p'].append(p); rows[h]['cv'].append(cv); rows[h]['y'].append(y.astype(float))
    for h in horizons:
        p = np.concatenate(rows[h]['p']); cv = np.concatenate(rows[h]['cv']); y = np.concatenate(rows[h]['y'])
        base = y.mean()
        print('%.2f s: points %d, occupied %.4f | Brier Stage 1 %.5f  CV tracks %.5f  constant %.5f' % (
            h*0.048, len(y), base, ((p-y)**2).mean(), ((cv-y)**2).mean(), ((base-y)**2).mean()))
        for lo, hi in ((0, 0.001), (0.001, 0.05), (0.05, 0.15), (0.15, 0.35), (0.35, 0.65), (0.65, 1.01)):
            m = (p >= lo) & (p < hi)
            if m.any():
                print('     Stage 1 p in [%.3f, %.2f): n %7d  mean p %.3f  observed %.3f' % (lo, hi, m.sum(), p[m].mean(), y[m].mean()))
        m = cv > 0.5
        print('     CV tracks occupied: n %d observed %.3f; CV empty: observed %.4f' % (m.sum(), y[m].mean() if m.any() else float('nan'), y[~m].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'summary', 'people', 'calib'])
    ap.add_argument('--bank', default='output/statebank/orca_dev')
    ap.add_argument('--variant', default='cl_frozen')
    ap.add_argument('--out', default='output/sbeval')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default=None)
    ap.add_argument('--variants', default=None)
    ap.add_argument('--margin', type=float, default=0.25)
    ap.add_argument('--only-tracked', action='store_true', help='run: only states where the robot tracks someone')
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.bank, '*.pkl')))
    if a.mode == 'people':
        people_audit(paths)
        return
    if a.mode == 'calib':
        occupancy_calibration(paths)
        return
    bank_name = Path(a.bank).name
    if a.mode == 'run':
        os.environ.setdefault('QG_FAST_SENSE', '1')
        from eventwm_graph.sandbox.core import WorldModelSandbox, SandboxConfig
        out = Path(a.out, bank_name, a.variant); out.mkdir(parents=True, exist_ok=True)
        if VARIANTS[a.variant][0].get('map_loop', 'stage2_union') == 'off':
            # frozen-map variants never call Layer 2: a CPU-only Sandbox (executor construction and config only)
            sb = WorldModelSandbox(_NoLayer2(), SandboxConfig(materialize=False, route=True))
        else:
            sb = WorldModelSandbox.from_release(SandboxConfig(materialize=False, route=True),
                                               map_device=a.device or 'cuda:%d' % (a.shard % 2))
        mine = [p for p in paths if zlib.crc32(Path(p).name.encode()) % a.shards == a.shard]
        for p in mine:
            target = out/(Path(p).stem+'.json')
            if target.exists():
                continue
            item = pickle.load(open(p, 'rb'))
            if a.only_tracked:
                from eventwm_graph.querygen.control import human_estimates
                if not human_estimates(item['legal'].tracks, item['legal'].time):
                    continue
            t0 = time.perf_counter()
            try:
                r = run_variant(sb, item, a.variant)
            except Exception as exc:
                print(json.dumps(dict(state=Path(p).name, error=repr(exc))), flush=True)
                continue
            r['seconds'] = round(time.perf_counter()-t0, 2)
            target.write_text(json.dumps(r))
        sb.close()
        return
    # summary: every variant found under out/bank_name
    names = a.variants.split(',') if a.variants else sorted(os.listdir(Path(a.out, bank_name)))
    probes = {}
    for p in paths:
        probes[Path(p).stem] = None
    for name in names:
        res = []
        secs = []
        for f in sorted(Path(a.out, bank_name, name).glob('*.json')):
            stem = f.stem
            if stem not in probes:
                continue
            if probes[stem] is None:
                probes[stem] = pickle.load(open(Path(a.bank, stem+'.pkl'), 'rb'))['probe']
            r = json.loads(f.read_text())
            r['preds'] = [dict(p, contact={int(k): v for k, v in p['contact'].items()},
                               prog={int(k): v for k, v in p['prog'].items()},
                               stop={int(k): v for k, v in p['stop'].items()}) for p in r['preds']]
            res.append((probes[stem], r))
            secs.append(r.get('seconds', 0.))
        if not res:
            continue
        m = score_states(res, a.margin)
        print('%-16s states %d  s/state %.1f' % (name, len(res), float(np.mean(secs))))
        for w in WINDOWS:
            x = m[w]
            print('   %.2f s: collision AUC %.3f  P %.2f R %.2f (%d / %d)  | rule: category = best %.3f  critical %d '
                  'fixed %d broke %d  deviations %.2f' % (w*0.048, x['auc'], x['precision'], x['recall'], x['collisions'],
                                                        x['candidates'], x['category_ok'], x['critical'], x['fixed'],
                                                        x['broke'], x['deviations']))


if __name__ == '__main__':
    main()
