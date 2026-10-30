"""ANALYSIS: checks of the imagined camera + tracker (perceive.py; tracker emulation L3_CAM_TRACKER) against the robot's
real perception pipeline.  Ground truth (the recorded people, the real branch's track log) is used only here.

    python scripts/l3_run.py eventwm_graph.l3.trackcheck seed --bank output/statebank/orca_train
    python scripts/l3_run.py eventwm_graph.l3.trackcheck openloop --bank output/statebank/orca_train
    python scripts/l3_run.py eventwm_graph.l3.trackcheck ladder --probe output/agents/tracker/ladder_tk

* seed: the seeded real ``LegalHumanTracker`` (perceive.seeded_tracker, from the legal state at t), fed the REAL
  detections (the real track log's visible tracks) at the REAL timestamps, must reproduce the real track log.
* openloop: the REAL robot poses of a state-bank episode and the TRUE people are fed to the camera model and to the
  emulated camera + tracker; their tracks are compared step by step with the real track log
  (``shortprobe.track_agreement``).  Isolates perception from the imagined planner.
* ladder: summary of a ``shortprobe`` run with L3_PROBE_TRACKS=1 (camTPc / camTkPc / camRPc / camRMPc / camTkMPc):
  track agreement (closed and open loop), imagined-vs-real divergence, contact precision / recall (imagined centre
  distance < 0.75 m vs official contact), decisive decisions, seed check.

State-bank modes pair every banked decision with the episode's last banked decision, whose legal history (poses, track
log) holds the real continuation of the floor episode.
"""
import argparse
import collections
import glob
import json
import math
import os
import pickle

import numpy as np

K = 64


def _episodes(bank):
    byep = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(bank, '*.pkl'))):
        byep[os.path.basename(f).split('__d')[0]].append(f)
    return byep


def _states(bank):
    """(item, history legal state, n0, b0, steps) for every banked decision followed by a later one."""
    for ep, fs in sorted(_episodes(bank).items()):
        if len(fs) < 2:
            continue
        H = pickle.load(open(fs[-1], 'rb'))['legal']
        for f in fs[:-1]:
            it = pickle.load(open(f, 'rb'))
            L = it['legal']
            n0, b0 = len(L.poses), len(L.track_log)-1
            if not np.allclose(np.asarray(H.poses[n0-1]), np.asarray(L.pose)):
                continue
            yield it, H, n0, b0, min(K, len(H.poses)-n0, len(it['truth_people'])-1)


def seed_check(bank):
    from scripts.collector.observation import HumanDetection
    from .perceive import seeded_tracker
    n = steps = bad = 0
    for it, H, n0, b0, kk in _states(bank):
        n += 1
        tr = seeded_tracker(it['legal'])
        for k in range(1, kk+1):
            t, R = H.track_log[b0+k]
            dets = [HumanDetection(float(t), tuple(float(v) for v in x['position'])) for x in R if x.get('visible_now')]
            steps += 1
            bad += int(tr.update(float(t), sorted(dets, key=lambda d: d.position)) != R)
    print('states %d, steps %d: seeded tracker vs real track log mismatches %d' % (n, steps, bad))


def _sum(S, rec):
    for w, d in rec.items():
        if w == 'ex':
            continue
        for k, v in d.items():
            S[w][k] += v


def _line(name, S):
    return ('  %-34s exact-set steps %.3f | tracks P %.3f R %.3f | near(2.5 m) P %.3f R %.3f exact %.3f | '
            'vel-valid agree %.3f | pos err %.3f m | |dv| %.3f m/s  (steps %d)' % (
                name, S['exact']/max(S['steps'], 1), S['nm']/max(S['ni'], 1), S['nm']/max(S['nr'], 1),
                S['nnm']/max(S['nni'], 1), S['nnm']/max(S['nnr'], 1), S['nexact']/max(S['steps'], 1),
                S['vv']/max(S['nm'], 1), S['perr']/max(S['nm'], 1), S['verr']/max(S['vn'], 1), S['steps']))


def openloop(bank):
    from .oraclebank import OracleBank
    from .perceive import VisibleWorld
    from .shortprobe import track_agreement, WINDOWS
    models = (('camera model (today)', False), ('emulated camera + tracker', True))
    S = {m: {w: collections.Counter() for w in WINDOWS} for m, _ in models}
    n = 0
    for it, H, n0, b0, kk in _states(bank):
        if kk < 1:
            continue
        n += 1
        L = it['legal']
        T = np.asarray(it['truth_people'], np.float64)                    # [65, N, 2] true people, row j = t + j steps
        tape = np.zeros((T.shape[0], T.shape[1], 3))
        tape[..., 0], tape[..., 2] = T[..., 0], T[..., 1]
        bank_t = OracleBank(tape, 0, L, 'all', horizon_steps=K+1, long_steps=K)
        pose = lambda k: np.asarray(H.poses[n0-1+k], np.float64)
        for name, emu in models:
            hum = VisibleWorld(L, bank_t, 0, K, camera=it['state'], tracker=emu)
            hum.rec = []
            for k in range(kk+1):
                hum.tracks_at(k, pose(k))
            _sum(S[name], track_agreement(hum.rec, H.track_log, b0, pose, pose))
    print('%s: %d states (real poses, true people, vs the real track log)' % (bank, n))
    for w in WINDOWS:
        print('-- steps 1..%d' % w)
        for name, _ in models:
            print(_line(name, S[name][w]))


VARIANTS = ('camTPc', 'camTkPc', 'camRPc', 'camRMPc', 'camTkMPc')
NAMES = dict(camTPc='camera model (today), frozen map', camTkPc='EMULATED camera + tracker, frozen map',
             camRPc='REAL tracker output, frozen map', camRMPc='REAL tracker + REAL map stream',
             camTkMPc='EMULATED tracker + REAL map stream')


def ladder(path):
    recs, seeds = [], []
    for f in sorted(glob.glob(os.path.join(path, '*.json'))):
        d = json.load(open(f))
        pr = sorted(d['probe'], key=lambda r: r['decision'])
        recs.append((d['key'], pr))
        seeds += [r['seed'] for r in pr if 'seed' in r]
    cands = [x for _, pr in recs for r in pr for x in r['cands']]
    V = tuple(v for v in VARIANTS if any(v in x['imag'] for x in cands))
    print('%s: episodes %d, decisions %d, candidates %d' % (path, len(recs), sum(len(pr) for _, pr in recs), len(cands)))
    if seeds:
        ok = sum(s['tracks_equal'] and s['next_id'][0] == s['next_id'][1] and s['timestamp'][0] == s['timestamp'][1]
                 for s in seeds)
        print('seeded tracker == the runtime tracker at t (tracks, next id, clock): %d / %d decisions' % (ok, len(seeds)))
    for key, title in (('trk', 'closed loop (imagined pose)'), ('trk_ol', 'open loop (real poses)')):
        for w in (16, 32):
            rows = []
            for v in V:
                S = collections.Counter()
                for x in cands:
                    t = ((x['imag'].get(v) or [{}])[0].get(key) or {}).get(str(w))
                    if t:
                        S.update(t)
                if S['steps']:
                    rows.append(_line(NAMES[v], S))
            if rows:
                print('-- track agreement, %s, steps 1..%d' % (title, w))
                print('\n'.join(rows))
    print('-- imagined vs real divergence (m), median / p90')
    for sel, name in ((False, 'non-thru'), (True, 'thru')):
        for v in V:
            s = '  %-8s %-40s' % (name, NAMES[v])
            for lag in ('16', '32'):
                a = [run['div'][lag] for x in cands if x['label'].endswith('_th') == sel for run in (x['imag'].get(v) or [])
                     if (run.get('div') or {}).get(lag) is not None]
                if a:
                    s += '  lag %s: %.2f / %.2f (n=%d)' % (lag, np.median(a), np.quantile(a, .9), len(a))
            print(s)
    for W in (16, 32):
        y, cd = [], {v: [] for v in V}
        for x in cands:
            re = x['real']
            if (re['outcome'] == 'running' and re['steps'] < W) or not all(x['imag'].get(v) for v in V):
                continue
            y.append(re['outcome'] == 'human_collision' and re['steps'] <= W)
            for v in V:
                c = (x['imag'][v][0].get('cd') or {}).get(str(W))
                cd[v].append(c if c is not None else 9.)
        y = np.asarray(y, bool)
        print('-- contact within %.2f s (imagined centre distance < 0.75 m vs official contact): %d candidates, %d contacts'
              % (W*0.048, len(y), int(y.sum())))
        for v in V:
            p = np.asarray(cd[v]) < 0.75
            tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
            print('  %-40s precision %.2f recall %.2f' % (NAMES[v], tp/max(tp+fp, 1), tp/max(tp+fn, 1)))
    print('-- decisive decisions (the chosen candidate collides within 1.5 s, a safe alternative existed): contact foreseen')
    dec = {v: [] for v in V}
    for key, pr in recs:
        T = pr[0]['replay']['steps'] if pr and 'replay' in pr[0] else None
        decisive = None
        for r in pr:
            c = r['labels'].index(r['replay']['label']) if r.get('replay', {}).get('label') in r['labels'] else None
            exc = [x['real']['outcome'] == 'human_collision' and x['real']['steps'] <= 32 for x in r['cands']]
            if c is not None and T is not None and T-r['step'] <= 32 and exc[c] and not all(exc):
                decisive = (r, c)
        if decisive:
            r, c = decisive
            for v in V:
                dec[v].append((key, ((r['cands'][c]['imag'].get(v) or [{}])[0].get('cd') or {}).get('32')))
    for v in V:
        a = [x for _, x in dec[v] if x is not None]
        print('  %-40s %d / %d  %s' % (NAMES[v], sum(x < 0.75 for x in a), len(dec[v]),
                                       [(k.split('/')[-2], None if x is None else round(x, 2)) for k, x in dec[v]]))
    if 'camRPc' in V:
        print('-- paired with the real-tracker replay: identical imagined runs')
        for a, b in (('camTPc', 'camRPc'), ('camTkPc', 'camRPc'), ('camTkMPc', 'camRMPc')):
            if a in V and b in V:
                same = sum(int(x['imag'][a][0]['div'] == x['imag'][b][0]['div'] and x['imag'][a][0]['n'] == x['imag'][b][0]['n']
                               and x['imag'][a][0].get('cd') == x['imag'][b][0].get('cd')) for x in cands)
                big = [abs(x['imag'][a][0]['div']['32']-x['imag'][b][0]['div']['32']) > 0.1 for x in cands
                       if '32' in x['imag'][a][0]['div'] and '32' in x['imag'][b][0]['div']]
                print('  %-9s vs %-8s %d / %d; 1.5 s divergence differs by > 0.1 m: %.3f' % (a, b, same, len(cands),
                                                                                            float(np.mean(big))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['seed', 'openloop', 'ladder'])
    ap.add_argument('--bank', default='output/statebank/orca_train')
    ap.add_argument('--probe', default='output/agents/tracker/ladder_tk')
    ap.add_argument('--nice', type=int, default=10)
    a = ap.parse_args()
    if a.nice:
        os.nice(a.nice)
    if a.mode == 'seed':
        seed_check(a.bank)
    elif a.mode == 'openloop':
        openloop(a.bank)
    else:
        ladder(a.probe)


if __name__ == '__main__':
    main()
