"""Stage 2 fine-tuning data for the Sandbox's imagined map stream (data collection; inputs LEGAL, labels PRIVILEGED).

The Sandbox (``heuristics._discovery_rollouts``) asks Stage 2, at a decision state, what the legal camera will certify
along every candidate's imagined path, and reveals those cells to the imagined planner.  This collector records, on the
floor's own episodes, what Stage 2 should have answered:

* legal inputs at the decision (Stage 2's ``predict_segments`` inputs): the legal partial graph, and -- once per
  episode -- the measured range frames and legal track rows up to the last recorded decision (each decision uses the
  prefix up to its own frame);
* every candidate executed FOR REAL in the exact branch simulator (its interval, then the default policy, to 64 steps
  = 3.1 s; the legal stream is bypassed, as in ``shortprobe.probe_decision``): its real poses / times, and the cells
  UNKNOWN in the decision-time legal map that the real branch's legal map certified in the window, with their state
  and FIRST certification time (labels: privileged evaluation / supervision only);
* every candidate IMAGINED on the legal map frozen at t, as the Sandbox's first discovery pass does (interval + default
  continuation with decision boundaries every 16 steps and goal-query inheritance, core contact physics, perception
  through the camera model): with the legal constant-velocity people (``oraclebank.CVBank``; the best legal selector's
  people) and with the true people (``OracleBank``; the probe's ``camS2`` conditioning; privileged, used only as an
  alternative conditioning path).

The episode itself follows the floor (default candidate).  Output per episode ``<out>/<episode>__<config>.pkl``:
dict(key, config, scene, frames=dict(ranges, origins, rotations, timestamps, tracks), decisions=[...]).

    L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.s2data --suite train --out output/agents/stage2/s2ft/data \
        [--shard i --shards n --every 2 --max-dec 8 --per-scene 1 --keys FILE]
"""
import argparse
import json
import os
import pickle
import time
import zlib
from pathlib import Path

import numpy as np

W_MAX = 64


def _keys_state(g, keys):
    i, j = keys[:, 0]-g.ox, keys[:, 1]-g.oz
    ok = (i >= 0) & (i < g.state.shape[0]) & (j >= 0) & (j < g.state.shape[1])
    out = np.zeros(len(keys), np.uint8)
    out[ok] = g.state[i[ok], j[ok]]
    return out


def new_cells(legal_map, final):
    """Cells of the real branch's legal map ``final`` certified (FREE 1 / BLOCKED 2) that are UNKNOWN in the
    decision-time legal map: keys [n, 2] int32, final states [n] uint8, first certification times [n] float64."""
    ii, jj = np.nonzero((final.state != 0) & np.isfinite(final.first_seen))
    keys = np.stack([ii+final.ox, jj+final.oz], 1).astype(np.int64)
    new = _keys_state(legal_map, keys) == 0
    return (keys[new].astype(np.int32), final.state[ii[new], jj[new]].astype(np.uint8),
            final.first_seen[ii[new], jj[new]].astype(np.float64))


def collect_decision(ep, rt, cands, queries, labels, default, cfg, sb, imagine_people=('cv', 'true')):
    from eventwm_graph.querygen.native import MAX_EPISODE_STEPS
    from eventwm_graph.querygen.oracle import horizon_for
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.sandbox.imagine import imagine, frozen_map
    from eventwm_graph.sandbox.episode import default_query
    from .cfonline import default_continue
    from .oraclebank import OracleBank, CVBank
    from .perceive import VisibleWorld
    st = ep.state
    dyn = FAMILIES[cfg.dynamics]
    goal = default_query()
    H = type('H', (), {'horizon': 16})()
    i0 = int(st.index)
    n0 = len(st.legal.poses)
    end = min(MAX_EPISODE_STEPS, i0+W_MAX)
    t0 = float(st.legal.time)
    rec = dict(step=i0, time=t0, labels=list(labels), kinds=[q.kind for q in queries], default=int(default),
               pose=np.asarray(st.legal.pose, np.float64), cands=[])
    tick = time.perf_counter()
    raw, hooked = ep._observe, rt.sensor.observe
    rt.sensor.observe = raw                        # exact branches: the legal stream never sees them
    rt.prefix_cache = 48                           # exact prefix sharing among the branches (outputs unchanged)
    reals = []
    try:
        for q in queries:
            cont = q.kind == 'continue'
            h = min(horizon_for(q, H) if not cont else 16, MAX_EPISODE_STEPS-i0)
            child = rt.run(st, None if cont else q, horizon=h, cont=cont, geodesic=False)
            fin = child if child.terminal else default_continue(rt, child, goal, end)
            reals.append((h, fin))
    finally:
        rt.sensor.observe = hooked
        # the episode's own steps must go through the legal stream: no cached branch step may be reused by them
        rt.prefix_cache = 0
        for p in rt._pc_parents:
            p._qg_pcache = None
        rt._pc_parents = []
    rec['real_s'] = round(time.perf_counter()-tick, 2)
    tick = time.perf_counter()
    gmap = frozen_map(st.legal.map)
    tape = np.asarray(rt.tape.positions, np.float64)
    banks = {}
    if 'cv' in imagine_people:
        banks['cv'] = CVBank(st.legal, horizon_steps=W_MAX+1)
    if 'true' in imagine_people:
        banks['true'] = OracleBank(tape, i0, st.legal, 'all', horizon_steps=W_MAX+1, long_steps=W_MAX)
    for k, ((h, fin), c) in enumerate(zip(reals, cands)):
        real = np.asarray(fin.legal.poses[n0:], np.float64).reshape(-1, 4)
        times = np.asarray(fin.legal.times[n0:], np.float64)
        keys, states, first = new_cells(st.legal.map, fin.legal.map)
        item = dict(label=labels[k], h=int(h), outcome=fin.outcome or 'running', real=real.astype(np.float32),
                    real_t=times-t0, keys=keys, states=states, first=(first-t0).astype(np.float32), imag={})
        for name, bank in banks.items():
            ex, _ = sb._execution(c)
            if isinstance(ex, Exception):
                continue
            ro = imagine(st.legal, ex, dyn, VisibleWorld(st.legal, bank, 0, W_MAX), h, level=int(c.level), gmap=gmap,
                         continuation=max(0, W_MAX-h), cont_query=goal, planner=cfg.planner, cont_boundary=16,
                         cont_inherit=True, physics='core')
            cc = getattr(ro, 'cont', None)
            P = np.asarray(ro.poses, np.float64)
            if cc is not None and len(cc):
                P = np.concatenate([P, np.asarray(cc, np.float64)])
            item['imag'][name] = P.astype(np.float64)
        rec['cands'].append(item)
    rec['imag_s'] = round(time.perf_counter()-tick, 2)
    return rec


def collect_episode(spec, cfg, out, every=2, max_dec=8, rng=None, people=('cv', 'true')):
    from eventwm_graph.sandbox.episode import BranchEpisode, generator
    from eventwm_graph.sandbox.core import WorldModelSandbox, SandboxConfig, Candidate
    from eventwm_graph.querygen.native import MAX_EPISODE_STEPS
    from eventwm_graph.querygen.oracle import horizon_for
    from .env import default_index, _H, HORIZON
    from . import sensorcache
    gen = generator(cfg.gen, cfg.dynamics)          # before the runtime: sets the process-wide executor options
    ep = BranchEpisode(spec, cfg.planner, cfg.dynamics, range_sensor='fast-driver', stream=None)
    rt = ep.rt
    sensorcache.attach(rt)                         # read-only use of the exact support-certificate cache
    sb = WorldModelSandbox(None, SandboxConfig(materialize=False))
    sb.config.interval = HORIZON
    rng = rng or np.random.default_rng(0)
    offset = int(rng.integers(max(every, 1)))
    decisions = []
    k = 0
    t_start = time.perf_counter()
    try:
        while not ep.over and len(decisions) < max_dec:
            queries, diag = gen.propose(ep.legal, ep.execution)
            labels = [q.label for q in queries]
            running = ep.execution
            cands = [Candidate(None, cfg.planner, cfg.dynamics, execution=ep.execution, level=ep.level, query_id='continue')
                     if q.kind == 'continue' else Candidate(q, cfg.planner, cfg.dynamics, level=ep.level, query_id=q.label,
                                                            execution=running)
                     for q in queries]
            d = default_index(labels)
            if k % max(every, 1) == offset % max(every, 1):
                state = ep.decision_state()
                rec = collect_decision(ep, rt, cands, queries, labels, d, cfg, sb, people)
                rec['decision'] = k
                rec['frames'] = len(state.frames)
                rec['graph'] = state.partial_graph
                decisions.append(rec)
            q = queries[d]
            cont = q.kind == 'continue'
            h = min(horizon_for(q, _H(HORIZON)), MAX_EPISODE_STEPS-ep.state.index)
            ep.run(None if cont else q, horizon=h)
            k += 1
        if decisions:
            nf = max(r['frames'] for r in decisions)
            pk = ep.stream.range_packet()
            frames = dict(ranges=np.array(pk['ranges'][:nf], np.float32), origins=np.array(pk['origins'][:nf]),
                          rotations=np.array(pk['rotations'][:nf]), timestamps=np.array(pk['timestamps'][:nf]),
                          tracks=[f['tracks'] for f in ep.stream.frames[:nf]])
            item = dict(key=spec.key, config=cfg.name, scene=spec.scene, outcome=ep.state.outcome or 'running',
                        steps=int(ep.state.index), frames=frames, decisions=decisions)
            name = '%s__%s.pkl' % (spec.key.replace('/', '_'), cfg.name)
            tmp = Path(out, name+'.tmp')
            with open(tmp, 'wb') as f:
                f.write(zlib.compress(pickle.dumps(item, protocol=4), 3))
            os.replace(tmp, Path(out, name))
    finally:
        ep.close()
    return len(decisions), time.perf_counter()-t_start


def load_episode(path):
    with open(path, 'rb') as f:
        return pickle.loads(zlib.decompress(f.read()))


def episode_list(suite, per_scene=1, seed=0, keys=None):
    from . import suites
    specs = suites.load(suite)
    if keys:
        want = [x.strip() for x in Path(keys).read_text().split() if x.strip()]
        by = {s.key: s for s in specs}
        return [by[k] for k in want if k in by]
    rng = np.random.default_rng(seed)
    by_scene = {}
    for s in specs:
        by_scene.setdefault((s.dataset, s.scene), []).append(s)
    out = []
    for sc in sorted(by_scene):
        eps = by_scene[sc]
        idx = rng.permutation(len(eps))[:per_scene]
        out += [eps[i] for i in idx]
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='train')
    ap.add_argument('--config', default='orca-native-k10')
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--max-dec', type=int, default=8)
    ap.add_argument('--per-scene', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--keys', default=None)
    ap.add_argument('--people', default='cv,true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if os.environ.get('L3_INHERIT') != '1':
        raise SystemExit('Layer 3 processes run with L3_INHERIT=1')
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from .env import ConfigSpec
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ConfigSpec.parse(a.config)
    specs = episode_list(a.suite, a.per_scene, a.seed, a.keys)
    if a.limit:
        specs = specs[:a.limit]
    specs = [s for i, s in enumerate(specs) if i % a.shards == a.shard]
    for s in specs:
        name = '%s__%s.pkl' % (s.key.replace('/', '_'), cfg.name)
        if Path(out, name).exists():
            continue
        rng = np.random.default_rng(zlib.crc32(s.key.encode()))
        try:
            n, secs = collect_episode(s, cfg, out, a.every, a.max_dec, rng, tuple(a.people.split(',')))
        except Exception as exc:
            import traceback
            print(json.dumps(dict(key=s.key, error=repr(exc), tb=traceback.format_exc()[-1200:])), flush=True)
            continue
        print(json.dumps(dict(key=s.key, decisions=n, seconds=round(secs, 1))), flush=True)


if __name__ == '__main__':
    main()
