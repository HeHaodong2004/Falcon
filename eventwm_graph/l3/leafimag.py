"""ANALYSIS: IMAGINED candidate leaves at the labelled decisions of a run (the legal counterpart of ``leafdata.py``).

Each labelled episode of ``<run>/cf_online`` is replayed in the Layer 3 environment with a local Sandbox (same
generator, Stage 1 bank, camera model; the recorded decisions are executed).  At a labelled decision every labelled
candidate is imagined in each of the 6 heaviest Stage 1 worlds (perceive.VisibleWorld: the camera model in that
world) for 32 steps -- its interval, then the default policy with continuation fidelity -- exactly as the selector
``mwroll`` with ``L3_MW_CONT=16`` imagines it.  Per world: the world weight, the imagined contact flag (centre within
0.75 m of one of the world's people, the selector's rule) and min body-rectangle clearance, STOP flags, and
``leafvalue.legal_features`` of the imagined leaf state.  Labels as in ``leafdata.py``.

Output ``<out>/<episode>.npz``: Xi [n, 6, F], wi / ci / mi [n, 6] (weight, contact, min clearance), si [n, 6] (0 running,
1 STOP at the goal, 2 STOP elsewhere), y, steps, label, decision, is_default.

    L3_INHERIT=1 L3_LAYOUT=v4 python scripts/l3_run.py eventwm_graph.l3.leafimag --run output/runs/R10 \
        --out output/leafimag/R10 [--shard i --shards n]
"""
import argparse
import json
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np

N_WORLDS = 6
CONT = 16
RADIUS = 0.75


def imagined_rows(sb, ep, d, labelled):
    """Per labelled candidate index -> per-world imagined leaf records at the current decision ``d``."""
    from eventwm_graph.querygen.dynamics import FAMILIES
    from .heuristics import _leaf_rollouts, _rect_clearance
    from .compiler import LegalField
    from .leafvalue import legal_features
    from .perceive import VisibleWorld
    legal = ep.ep.legal
    cfg = ep.cfg
    dyn = FAMILIES[cfg.dynamics]
    bank = d.result.humans
    W = np.asarray(bank.weights, np.float64)
    W = W/W.sum() if W.sum() > 0 else np.full(len(W), 1./max(len(W), 1))
    worlds = np.argsort(-W)[:N_WORLDS] if len(np.unique(W)) > 1 else np.arange(min(N_WORLDS, len(W)))
    ww = W[worlds]/W[worlds].sum()
    steps = max(sb.config.steps_for(c) for c in d.candidates)
    T = steps+CONT
    HP = None
    if len(getattr(bank, 'track_keys', ())):
        rel = np.arange(1, T+1)*0.048
        inside = rel[rel <= float(bank.times[-1])+1e-9]
        HP = np.asarray(bank.sandbox_positions(inside), np.float64)
        if len(inside) < T and HP.shape[1] >= 2:
            vel = (HP[:, -1]-HP[:, -2])/0.048
            extra = np.arange(1, T-len(inside)+1)*0.048
            HP = np.concatenate([HP, HP[:, -1][:, None]+extra[None, :, None, None]*vel[:, None]], 1)
    field = LegalField(legal)
    F = None
    out = {k: dict(X=[], w=[], c=[], m=[], s=[]) for k in labelled}
    for j, w in enumerate(worlds):
        hum = VisibleWorld(legal, bank, int(w), T)
        ros = _leaf_rollouts(sb, legal, d, hum, CONT)
        for k in labelled:
            ro = ros[k]
            poses = np.asarray(ro.poses, np.float64)
            c = getattr(ro, 'cont', None)
            seq = np.concatenate([poses[1:], c]) if c is not None and len(c) else poses[1:]
            if not len(seq):
                seq = poses[:1]
            hit, mclr = False, math.inf
            if HP is not None:
                n = min(len(seq), HP.shape[1])
                with np.errstate(invalid='ignore'):
                    dd = np.hypot(HP[w, :n, :, 0]-seq[:n, None, 0], HP[w, :n, :, 1]-seq[:n, None, 2])
                hit = bool(dd.size and np.nanmin(np.where(np.isfinite(dd), dd, np.inf)) < RADIUS)
                rc = _rect_clearance(seq[:n], HP[w, :n])
                if rc.size:
                    mclr = float(np.nanmin(np.where(np.isfinite(rc), rc, np.inf)))
            near = math.hypot(seq[-1, 0]-legal.goal[0], seq[-1, 2]-legal.goal[2]) <= 0.3
            sflag = (1 if near else 2) if ro.stop else 0
            lf = ro.leaf
            if lf is not None:
                cand = d.candidates[k]
                x = legal_features(lf['poses'][-1], lf['poses'][-8:], lf['times'][-8:], lf['time'], legal.map, field,
                                   legal.goal, lf['tracks'], lf['execution'], lf['level'], dyn, cand.planner,
                                   int(ep.ep.steps)+int(lf['steps']), int(lf['steps']), None)
                F = len(x)
            else:
                x = None
            out[k]['X'].append(x); out[k]['w'].append(float(ww[j])); out[k]['c'].append(float(hit))
            out[k]['m'].append(min(mclr, 9.)); out[k]['s'].append(sflag)
    return out, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=100000)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.sandbox.core import WorldModelSandbox, SandboxConfig
    from eventwm_graph.querygen.native import EpisodeSpec
    from .actor import read_episode
    from .env import L3Episode, ConfigSpec, default_index
    from .compiler import GraphCompiler
    from .cfonline import _query
    from .leafdata import OUTCOME
    from .leafvalue import LEGAL_NAMES
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = a.device or 'cuda:%d' % (a.shard % 2)
    sb = WorldModelSandbox.from_release(SandboxConfig(materialize=False, route=True), map_device=dev)
    comp = GraphCompiler(brake_tail=True)
    files = sorted(Path(a.run, 'cf_online').glob('*.json'))
    files = [f for f in files if zlib.crc32(f.name.encode()) % a.shards == a.shard][:a.limit]
    for f in files:
        name = f.name[:-len('.ep.json')] if f.name.endswith('.ep.json') else f.stem
        target = out/(name+'.npz')
        if target.exists():
            continue
        t0 = time.perf_counter()
        rec = json.loads(f.read_text())
        targets = {int(k): v for k, v in rec['states'].items() if 'candidates' in v}
        if not targets:
            continue
        try:
            d0 = read_episode(Path(a.run, 'spool', rec['episode']))
        except Exception:
            continue
        cfg = ConfigSpec.parse(d0['config'])
        ep = L3Episode(EpisodeSpec(**d0['spec']), cfg, sb, compiler=comp, scan=bool(d0.get('scan')))
        rows = d0['rows']
        res = dict(Xi=[], wi=[], ci=[], mi=[], si=[], y=[], steps=[], label=[], decision=[], is_default=[])
        ok = True
        try:
            for i, dec in enumerate(d0['metrics']['decision_list']):
                if ep.over or i > max(targets):
                    break
                if i in targets:
                    dd = ep.observe()
                    if dd.labels != rows['labels'][i] or dd.labels != targets[i]['labels']:
                        ok = False
                        break
                    labelled = [k for k, c in enumerate(targets[i]['candidates']) if not c.get('skipped') and 'outcome' in c]
                    recs, F = imagined_rows(sb, ep, dd, labelled)
                    dflt = dd.labels[default_index(dd.labels)]
                    for k in labelled:
                        r = recs[k]
                        X = np.zeros((N_WORLDS, len(LEGAL_NAMES)), np.float32)
                        wv = np.zeros(N_WORLDS); cv = np.zeros(N_WORLDS); mv = np.full(N_WORLDS, 9.); sv = np.zeros(N_WORLDS)
                        for j, x in enumerate(r['X'][:N_WORLDS]):
                            if x is not None:
                                X[j] = x
                            wv[j], cv[j], mv[j], sv[j] = r['w'][j], r['c'][j], r['m'][j], r['s'][j]
                        c = targets[i]['candidates'][k]
                        res['Xi'].append(X); res['wi'].append(wv); res['ci'].append(cv); res['mi'].append(mv)
                        res['si'].append(sv); res['y'].append(OUTCOME.get(c['outcome'], 3)); res['steps'].append(int(c['steps']))
                        res['label'].append(dd.labels[k]); res['decision'].append(i); res['is_default'].append(dd.labels[k] == dflt)
                    ep.act(int(rows['actions'][i]))
                else:
                    ep.ep.run(None if dec['cont'] else _query(dec), horizon=int(dec['horizon']))
                    ep.decisions += 1
        except Exception as exc:
            print(json.dumps(dict(episode=f.name, error=repr(exc))), flush=True)
            ok = False
        finally:
            ep.close()
        if not res['y']:
            continue
        np.savez_compressed(target, Xi=np.asarray(res['Xi'], np.float32), wi=np.asarray(res['wi'], np.float32),
                            ci=np.asarray(res['ci'], np.float32), mi=np.asarray(res['mi'], np.float32),
                            si=np.asarray(res['si'], np.int8), y=np.asarray(res['y'], np.int8),
                            steps=np.asarray(res['steps'], np.int16), label=np.asarray(res['label']),
                            decision=np.asarray(res['decision'], np.int16), is_default=np.asarray(res['is_default'], bool),
                            config=np.asarray(d0['config']), episode=np.asarray(rec['episode']), complete=np.asarray(ok))
        print(json.dumps(dict(episode=f.name, rows=len(res['y']), complete=ok, seconds=round(time.perf_counter()-t0, 1))),
              flush=True)
    sb.close()


if __name__ == '__main__':
    main()
