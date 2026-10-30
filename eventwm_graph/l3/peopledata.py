"""Large-scale people dataset for the redesigned Layer 2 people module (legal inputs; privileged labels).

People do not react to the robot in this benchmark, so a short-horizon people model needs no counterfactual rollouts:
every decision of every recorded episode is a training example -- the legal observations at t and the true people
over the next 3 s (the recorded tape, used as labels only).  Episodes of a run's spool (diverse robot behaviour) are
replayed in the Layer 3 environment with a local Sandbox (Stage 1's bank is an input feature and the baseline); at
every ``--every``-th decision this stores, per episode (``<out>/<episode>.npz``):

* occupancy: robot-centric grid points x horizons (0.77 / 1.5 / 3 s) -- ``occupancy.features`` rows and labels;
* tracked people: per person, Stage 1 world features (``worldweights``) and each world's true error at 0.77 / 1.5 s;
* tracked people: the legal track history (last 2 s of detections, robot frame-free world coordinates relative to the
  last detection) and the true displacement at 0.77 / 1.5 / 3 s, for a direct forecaster.

    L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.peopledata --run output/runs/R10 --out output/peopledata/R10 \
        [--shard i --shards n --every 2]
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

HIST_S = 2.0
HIST_N = 10                       # resampled history points (every 0.2 s)


def track_histories(item, horizons=(16, 32, 62)):
    """Per tracked person matched to a true person: history [HIST_N, 3] (dx, dz, valid) relative to the last
    detection, last tracker velocity, and the true displacement at ``horizons`` (relative to the last detection)."""
    from eventwm_graph.querygen.control import human_estimates
    legal = item['legal']
    T = np.asarray(item['truth_people'], np.float64)
    now = float(legal.time)
    det = {}
    for tm, trs in (legal.track_log or []):
        if now-float(tm) > HIST_S:
            continue
        for t in trs:
            if t.get('visible_now'):
                det.setdefault(str(t['track_id']), []).append((float(tm), float(t['position'][0]), float(t['position'][2])))
    out = []
    for e in human_estimates(legal.tracks, legal.time):
        d0 = np.hypot(T[0, :, 0]-e.position[0], T[0, :, 1]+e.position[1])
        if not len(d0) or np.nanmin(d0) > 0.75:
            continue
        j = int(np.nanargmin(d0))
        hs = sorted(det.get(str(e.track_id), []))
        if not hs:
            continue
        t_last, x_last, z_last = hs[-1]
        H = np.zeros((HIST_N, 3), np.float32)
        for i, tq in enumerate(np.linspace(now-HIST_S, now, HIST_N)):
            cand = [h for h in hs if abs(h[0]-tq) <= 0.15]
            if cand:
                h = min(cand, key=lambda h: abs(h[0]-tq))
                H[i] = (h[1]-x_last, h[2]-z_last, 1.)
        fut = []
        for h in horizons:
            tp = T[min(h, len(T)-1), j]
            fut.append((tp[0]-x_last, tp[1]-z_last) if np.isfinite(tp).all() else (np.nan, np.nan))
        out.append(dict(hist=H, vel=np.array([e.velocity[0], -e.velocity[1]], np.float32), age=float(now-t_last),
                        fut=np.asarray(fut, np.float32), track_id=e.track_id, ref=(x_last, z_last), j=j, est=e))
    return out


def episode_rows(sb, d0, rec_spec, cfg, every, comp):
    from .env import L3Episode
    from .cfonline import _query
    from .statebank import bank_arrays, W_MAX
    from .occupancy import state_rows
    from .worldweights import person_rows
    ep = L3Episode(rec_spec, cfg, sb, compiler=comp, scan=bool(d0.get('scan')))
    occX, occY, occS, wwF, wwE, wwS, th, tv, ta, tf, ts = [], [], [], [], [], [], [], [], [], [], []
    k = 0
    try:
        tape = np.asarray(ep.rt.tape.positions, np.float64)
        for i, dec in enumerate(d0['metrics']['decision_list']):
            if ep.over:
                break
            if i % every == 0:
                d = ep.observe()
                i0 = int(ep.ep.steps)
                idx = np.minimum(np.arange(i0, i0+W_MAX+1), len(tape)-1)
                item = dict(legal=ep.ep.legal, bank=bank_arrays(d.result.humans),
                            truth_people=tape[idx][:, :, [0, 2]].astype(np.float32))
                x, y = state_rows(item)
                occX.append(x); occY.append(y); occS.append(np.full(len(y), k))
                for F, e in person_rows(item):
                    wwF.append(F); wwE.append(e); wwS.append(k)
                for h in track_histories(item):
                    th.append(h['hist']); tv.append(h['vel']); ta.append(h['age']); tf.append(h['fut']); ts.append(k)
                k += 1
                # execute the recorded decision exactly (the recorded label must be among the candidates)
                lab = (dec.get('label') if not dec['cont'] else 'continue')
                a = d.labels.index(lab) if lab in d.labels else d.default
                ep.act(a)
            else:
                ep.ep.run(None if dec['cont'] else _query(dec), horizon=int(dec['horizon']))
                ep.decisions += 1
    finally:
        ep.close()
    out = dict(states=k)
    if occX:
        out.update(occ_X=np.concatenate(occX), occ_Y=np.concatenate(occY).astype(np.int8), occ_S=np.concatenate(occS))
    if wwF:
        out.update(ww_F=np.concatenate(wwF), ww_E=np.concatenate(wwE), ww_G=np.asarray([len(f) for f in wwF]),
                   ww_S=np.asarray(wwS))
    if th:
        out.update(t_hist=np.asarray(th), t_vel=np.asarray(tv), t_age=np.asarray(ta, np.float32), t_fut=np.asarray(tf),
                   t_S=np.asarray(ts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='output/runs/R10')
    ap.add_argument('--out', default='output/peopledata/R10')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--limit', type=int, default=100000)
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.sandbox.core import WorldModelSandbox, SandboxConfig
    from eventwm_graph.querygen.native import EpisodeSpec
    from .actor import read_episode
    from .env import ConfigSpec
    from .compiler import GraphCompiler
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.run, 'spool').glob('*.ep'))
    files = [f for f in files if zlib.crc32(f.name.encode()) % a.shards == a.shard][:a.limit]
    sb = WorldModelSandbox.from_release(SandboxConfig(materialize=False, route=True),
                                       map_device=a.device or 'cuda:%d' % (a.shard % 2))
    comp = GraphCompiler(brake_tail=True)
    for f in files:
        target = out/(f.stem+'.npz')
        if target.exists():
            continue
        t0 = time.perf_counter()
        try:
            d0 = read_episode(f)
            if 'decision_list' not in d0['metrics']:
                continue
            rows = episode_rows(sb, d0, EpisodeSpec(**d0['spec']), ConfigSpec.parse(d0['config']), a.every, comp)
        except Exception as exc:
            print(json.dumps(dict(episode=f.name, error=repr(exc))), flush=True)
            continue
        np.savez_compressed(target, **rows)
        print(json.dumps(dict(episode=f.name, states=rows['states'], seconds=round(time.perf_counter()-t0, 1))), flush=True)
    sb.close()


if __name__ == '__main__':
    main()
