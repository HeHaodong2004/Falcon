"""Expert episode generation for Layer 3 (Oracle demonstrations; privileged selection, legal generation/execution).

One persistent worker process runs many episodes, so imports, the native ray-batch validation and the composed
benchmark configs are paid once.  Per episode:

  setup    EpisodeRuntime (exact human-tape cache ``QG_TAPE_CACHE``; cached benchmark config)
  search   the Oracle (any OracleConfig; ``--fast`` = the expert-generation budget, see FAST_ORACLE)
  replay   lean authoritative replay with the unchanged ``Env.step`` (no shadow simulator; the search runtime's pure
           sensor caches).  PASS iff the replayed actions equal the branch's and the official ``success`` is 1.
           The replay also emits the training records at every decision boundary: its legal state is bitwise the
           branch's, so the generator's candidate set there is exactly the one the Oracle chose from.
  data     per decision: the K candidate queries, the Oracle's choice (index), optional legal consequence features
           (``l3_interface``, M predicted futures), and rewards (geodesic progress, steps; success at the end)

Every phase is timed; ``summarize`` reports P50 / mean / P95 per phase and the throughput.

  python scripts/querygen_run.py eventwm_graph.querygen.expert run --suite unseen --out DIR [--only a,b] [--fast]
         [--features] [--gen JSON] [--oracle JSON] [--shard i --shards n]
  python scripts/querygen_run.py eventwm_graph.querygen.expert summarize --out DIR
"""
import argparse
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np

# expert-generation search budget: the evaluation portfolio, capped (the long tail is handled by the cap; episodes
# that exceed it are reported, never silently dropped)
FAST_ORACLE = dict(heading_h=True, escape_penalty=80, prefix_cache=48, max_edges=1500, max_seconds=900,
                   portfolio=[{'weight': 3.5}, {'st_radius': 0.6, 'st_radius_opt': 0}], portfolio_shares=[2, 1])
# member 0 (weight 3.5) wins 90% of solved development episodes (139 / 155 C1-K10); a 2:1 edge share cuts the
# median search by ~20% (94 -> 74 edges) and costs member-1 wins 1.5x
CROP = 64                     # local legal-map crop per decision (cells of 0.1017 m: 6.5 m x 6.5 m)
V12_GEN = dict(tc_variants=('goal', 'st_goal'), vc_variants=('goal', 'st_goal'), st_goal=True,
               maneuver_spawn_only=False, goal_unstick=6, bump_memory_m=0.25, evade=True, unstick_keep=True,
               bump_front=True, static_humans=True, stop_strict=True, unstick_progress=True, goal_relax_m=0.5,
               stop_closest_m=1.2)


def map_crop(legal, n=CROP):
    """[n, n] uint8 legal-map states (UNKNOWN / FREE / BLOCKED) centred on the robot (world x rows, z columns)."""
    g = legal.map
    ci, cj = g.index(float(legal.pose[0]), float(legal.pose[2]))
    out = np.zeros((n, n), dtype=np.uint8)
    i0, j0 = ci-n//2, cj-n//2
    a0, a1 = max(i0, 0), min(i0+n, g.state.shape[0])
    b0, b1 = max(j0, 0), min(j0+n, g.state.shape[1])
    if a1 > a0 and b1 > b0:
        out[a0-i0:a1-i0, b0-j0:b1-j0] = g.state[a0:a1, b0:b1]
    return out


def run_episode(spec, out_dir, planner='orca', dynamics='native', gen_kw=None, oracle_kw=None, features='raw', M=8,
                feat_steps=24):
    from .episode import EpisodeRuntime, replay
    from .generator import QueryGenerator, GeneratorConfig
    from .oracle import Oracle, OracleConfig
    from .dynamics import FAMILIES
    T = time.perf_counter
    t_all = T()
    dyn = FAMILIES[dynamics]
    ph = {}
    t = T()
    rt = EpisodeRuntime(spec, planner, dyn, strict=False)
    rt.sim.check_static = False
    gen = QueryGenerator(dyn, GeneratorConfig(**(gen_kw or {})))
    okw = dict(oracle_kw or {}); okw['escape_h'] = False
    ph['setup'] = T()-t
    t = T()
    res = Oracle(rt, gen, OracleConfig(**okw)).search()
    ph['search'] = T()-t
    st = res['state']
    rec = dict(key=spec.key, spec=spec.asdict(), planner=planner, dynamics=dynamics, gen=gen_kw, oracle=okw,
               search=dict(success=res['success'], edges=res['stats']['children'], steps=int(st.index),
                           final_geodesic=float(st.geodesic)), start_geodesic=float(rt.start_geodesic))
    private_cache = rt.private_cache
    rt.close()
    decisions_out, crops = [], []
    verdict = 'FAIL_SEARCH'
    if res['success']:
        from .sandbox import cv_futures
        from .l3_interface import consequence_features
        feat_s = [0.]

        def on_decision(i, legal, ex, d, step):
            qs, diag = gen.propose(legal, ex)
            labels = [q.label for q in qs]
            chosen = 'continue' if d['cont'] else d['query'].label
            row = dict(decision=i, step=int(step), chosen=labels.index(chosen) if chosen in labels else -1,
                       chosen_label=chosen, candidates=[dict(q.describe(), signature=str(q.signature())) for q in qs],
                       pose=[float(x) for x in legal.pose], goal=[float(legal.goal[0]), float(legal.goal[2])],
                       tracks=[dict(id=t['track_id'], position=[float(v) for v in t['position']],
                                    velocity=[float(v) for v in (t.get('estimated_velocity') or (0., 0., 0.))],
                                    age=float(legal.time-float(t['last_seen_time'])), visible=bool(t.get('visible_now')))
                               for t in legal.tracks],
                       stuck=bool(diag.get('stuck')))
            if features in ('raw', 'sandbox'):
                crops.append(map_crop(legal))
            if features == 'sandbox':
                tf = T()
                F = cv_futures(legal.tracks, legal.time, feat_steps, M=M, seed=int(step))
                X, _ = consequence_features(legal, qs, planner, dyn, F, feat_steps)
                row['features'] = np.asarray(X, dtype=np.float32).round(4).tolist()
                feat_s[0] += T()-tf
            decisions_out.append(row)
        t = T()
        rp = replay(spec, planner, dyn, st.decisions, private_cache=private_cache, lean=True, on_decision=on_decision)
        ph['replay'] = T()-t-feat_s[0]
        ph['features'] = feat_s[0]
        m = rp['metrics']
        ok = rp['actions'] == st.actions and m.get('success') == 1.0
        verdict = 'PASS' if ok else 'FAIL_REPLAY'
        # rewards per decision: geodesic progress over the decision's steps, step count; success at the end
        geo = [row[3] for row in rp['trajectory']]
        k = 0
        for row, n in zip(decisions_out, rp['per_decision_steps']):
            row['steps'] = int(n)
            row['progress_m'] = float(geo[k]-geo[min(k+n, len(geo)-1)])
            k += n
        rec['official'] = dict(success=m.get('success'), spl=m.get('spl'), psc=m.get('psc'),
                               human_collision=m.get('human_collision'), steps=rp['steps'])
        rec['chosen_in_candidates'] = all(r['chosen'] >= 0 for r in decisions_out)
    rec['verdict'] = verdict
    rec['decisions'] = decisions_out
    ph['total'] = T()-t_all
    rec['timing'] = {k: round(v, 2) for k, v in ph.items()}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    name = spec.key.replace('/', '_')
    if crops:
        np.savez_compressed(out/f'{name}.npz', map_crops=np.stack(crops))
        rec['arrays'] = f'{name}.npz'
    (out/f'{name}.json').write_text(json.dumps(rec, default=float))
    return rec


def summarize(out_dir):
    recs = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob('*.json')) if p.name != 'summary.json']
    if not recs:
        return {}
    tot = np.asarray([r['timing']['total'] for r in recs])
    phases = {}
    for k in ('setup', 'search', 'replay', 'features', 'total'):
        v = np.asarray([r['timing'].get(k, 0.) for r in recs])
        phases[k] = dict(p50=round(float(np.median(v)), 1), mean=round(float(v.mean()), 1),
                         p95=round(float(np.percentile(v, 95)), 1), max=round(float(v.max()), 1))
    ok = [r for r in recs if r['verdict'] == 'PASS']
    s = dict(episodes=len(recs), passed=len(ok), sr=len(ok)/len(recs),
             spl_mean=float(np.mean([r['official']['spl'] for r in ok])) if ok else None,
             psc_mean=float(np.mean([r['official']['psc'] for r in ok])) if ok else None,
             decisions_per_episode=float(np.mean([len(r['decisions']) for r in ok])) if ok else None,
             chosen_in_candidates=all(r.get('chosen_in_candidates', True) for r in ok),
             timing=phases, episodes_per_hour_per_worker=round(3600/float(tot.mean()), 1),
             failures=[(r['key'].split('/')[2], r['verdict'], r['search']['edges']) for r in recs
                       if r['verdict'] != 'PASS'])
    (Path(out_dir)/'summary.json').write_text(json.dumps(s, indent=1))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'summarize'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--only', default='')
    ap.add_argument('--planner', default='orca')
    ap.add_argument('--gen', default='')
    ap.add_argument('--oracle', default='')
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--features', default='raw', choices=['none', 'raw', 'sandbox'],
                    help='raw = legal map crop + tracks + candidate descriptors (cheap); sandbox = + legal consequence '
                         'features (one executor rollout per candidate; stand-in for Layer 2.5 outputs)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    if a.mode == 'summarize':
        print(json.dumps(summarize(a.out), indent=1))
        return
    from .suites import load
    from . import native
    native._install()
    specs = load(a.suite)
    if a.only:
        keep = set(a.only.split(','))
        specs = [s for s in specs if s.scene in keep]
    specs = [s for s in specs if zlib.crc32(s.key.encode()) % a.shards == a.shard]
    gen_kw = json.loads(a.gen) if a.gen else dict(V12_GEN)
    oracle_kw = json.loads(a.oracle) if a.oracle else (dict(FAST_ORACLE) if a.fast else
                                                         dict(FAST_ORACLE, max_edges=12000, max_seconds=40000))
    for spec in specs:
        f = Path(a.out)/f"{spec.key.replace('/', '_')}.json"
        if f.exists():
            continue
        r = run_episode(spec, a.out, a.planner, 'native', gen_kw, oracle_kw, features=a.features)
        print(json.dumps(dict(key=r['key'], verdict=r['verdict'], edges=r['search']['edges'], timing=r['timing'])),
              flush=True)


if __name__ == '__main__':
    main()
