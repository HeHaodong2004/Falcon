"""Reference experiment (ANALYSIS ONLY, privileged): the ROLLOUT POLICY over the default.

At every decision, every candidate is executed in querygen's exact branch simulator (recorded human tape) for its
interval and then continued by the default policy (keep the running query, else the goal query) to the episode end
or ``cap`` steps; the candidate with the best exact outcome is taken (success, fewer steps first; then still running
at the cap, more legal goal progress first; then timeout; collisions / STOP failures last).  This is one step of
policy iteration from the floor with a PERFECT evaluator: its success rate says how far exact one-step evaluation of
"candidate, then default" can carry Layer 3 (vs. deeper sequential planning).

    python scripts/l3_run.py eventwm_graph.l3.rolloutpolicy --out output/rollout_policy --configs orca-native-k10 \
        --suite dev --limit 24 [--cap 320] [--shard i --shards n]

Short-horizon variant (``--cap`` counts steps from the decision, so ``--cap 16`` = the candidate's interval only,
``--cap 32`` = interval + one default interval): a perfect simulator over a short window.  ``--leaf field`` ranks
candidates still running at the cap by the legal route cost-to-go of their end pose on the legal map AT THE DECISION
(compiler.LegalField: what a legal selector can compute for an imagined end pose) instead of the Euclidean goal
distance; ``--margin`` (m) keeps the default unless another candidate of the same category gains more than that.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def outcome_key(fin, st0, goal_xz, field=None, value=None):
    o = fin.outcome
    if o == 'success':
        return (3, -int(fin.index-st0.index))
    if o is None:
        p = fin.legal.pose
        if value is not None:
            return (2, float(value(fin)))
        if field is not None:
            return (2, -float(field.at(p[0], p[2])))
        return (2, -math.hypot(p[0]-goal_xz[0], p[2]-goal_xz[1]))
    if o == 'timeout':
        return (1, 0)
    return (0, int(fin.index-st0.index))                 # collision / STOP failure: later is (slightly) better


def run_episode(spec, cfg, cap, scan=False, shifts=(), leaf='euclid', margin=0.):
    from eventwm_graph.querygen.native import EpisodeSpec, MAX_EPISODE_STEPS
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.oracle import horizon_for
    from eventwm_graph.sandbox.episode import generator, default_query
    from .cfonline import default_continue
    from . import sensorcache
    t0 = time.perf_counter()
    gen = generator(cfg.gen, cfg.dynamics)           # process-wide executor options before the runtime
    rt = EpisodeRuntime(EpisodeSpec(**spec), cfg.planner, FAMILIES[cfg.dynamics], strict=False)
    rt.sim.check_static = False
    rt.prefix_cache = 48
    sensorcache.attach(rt)
    goal = default_query()
    H = type('H', (), {'horizon': 16})()
    st = rt.root
    gx, gz = float(st.legal.goal[0]), float(st.legal.goal[2])
    log = []
    try:
        while not st.terminal and st.index < MAX_EPISODE_STEPS:
            queries, _ = gen.propose(st.legal, st.execution)
            if scan:
                from .env import scan_queries
                queries = list(queries)+scan_queries(st.legal, gen.dyn, [q.label for q in queries])
            best = None
            scores = []
            field = value = None
            if leaf == 'field' or leaf.startswith('vmodel:'):
                from .compiler import LegalField
                field = LegalField(st.legal)
            if leaf.startswith('vmodel:'):
                # learned leaf value at the exact leaf (leafvalue.py; legal features of the leaf state on the
                # decision-time legal map): perfect short-horizon simulation + learned long-horizon value
                from .heuristics import _value_model
                from .leafvalue import legal_features
                vm = _value_model(leaf[len('vmodel:'):])
                L0 = st.legal
                xs = None
                if 'Xs' in vm.keys:
                    xs = legal_features(L0.pose, L0.poses[-8:], L0.times[-8:], L0.time, L0.map, field, L0.goal, L0.tracks,
                                        st.execution, st.dyn_state, gen.dyn, cfg.planner, st.index, 0, None)

                def value(fin, L0=L0, field=field, i0=int(st.index), xs=xs):
                    C = fin.legal
                    x = legal_features(C.pose, C.poses[-8:], C.times[-8:], C.time, L0.map, field, L0.goal, C.tracks,
                                       fin.execution, fin.dyn_state, gen.dyn, cfg.planner, fin.index, fin.index-i0, None)
                    if xs is not None:
                        x = np.concatenate([x, xs])
                    return float(vm.predict(x)[0])
            for q in queries:
                cont = q.kind == 'continue'
                h = min(horizon_for(q, H) if not cont else 16, MAX_EPISODE_STEPS-st.index)
                child = rt.run(st, None if cont else q, horizon=h, cont=cont, geodesic=False)
                fin = default_continue(rt, child, goal, min(MAX_EPISODE_STEPS, st.index+cap))
                key = outcome_key(fin, st, (gx, gz), field, value)
                if shifts:
                    # robust variant: the candidate's outcome category averaged over the recorded people timing and
                    # the shifted ones (exogenous timing noise of +-k steps); the recorded outcome breaks ties
                    cats = [key[0]]
                    for dlt in shifts:
                        cats.append(_shifted_outcome(rt, st, q, h, cont, goal, cap, dlt, (gx, gz))[0])
                    key = (sum(cats)/len(cats),)+tuple(key)
                scores.append((q.label, key))
                if best is None or key > best[0]:
                    best = (key, q, child, cont)
            default = 'continue' if 'continue' in [s[0] for s in scores] else 'goal'
            if margin > 0 and default in [s[0] for s in scores] and best[1].label != default:
                # keep the default unless the best candidate is in a better category or gains more than the margin
                kd = dict(scores)[default]
                if kd[0] == best[0][0] and best[0][1]-kd[1] <= margin:
                    q = queries[[s[0] for s in scores].index(default)]
                    cont = q.kind == 'continue'
                    h = min(horizon_for(q, H) if not cont else 16, MAX_EPISODE_STEPS-st.index)
                    best = (kd, q, rt.run(st, None if cont else q, horizon=h, cont=cont, geodesic=False), cont)
            log.append(dict(step=int(st.index), chosen=best[1].label, default=default,
                            keys=[[lab, [round(float(v), 3) for v in k]] for lab, k in scores],
                            n_success=sum(k[-2] == 3 if shifts else k[0] == 3 for _, k in scores), n=len(scores),
                            **(dict(robust=[[lab, round(k[0], 3)] for lab, k in scores]) if shifts else {})))
            st = best[2]                                   # the chosen candidate's interval (already simulated)
        out = dict(key=spec['key'] if 'key' in spec else None, outcome=st.outcome or 'running', steps=int(st.index),
                   decisions=len(log), deviations=sum(d['chosen'] != d['default'] for d in log),
                   seconds=round(time.perf_counter()-t0, 1), log=log)
        return out
    finally:
        try:
            sensorcache.save(rt)
        except Exception:
            pass
        rt.close()


def _shifted_outcome(rt, st, q, h, cont, goal, cap, dlt, goal_xz):
    """Exact outcome key of candidate ``q`` + default from ``st`` with the people's timing shifted by ``dlt`` steps
    (fidelity.shifted_tape; the prefix cache is off for the shifted run)."""
    from eventwm_graph.querygen.native import MAX_EPISODE_STEPS
    from .cfonline import default_continue
    from .fidelity import shifted_tape
    orig, pc = rt.tape, rt.prefix_cache
    rt.tape = rt.sim.tape = shifted_tape(orig, dlt)
    rt.prefix_cache = 0
    rt._loaded = None
    try:
        child = rt.run(st, None if cont else q, horizon=h, cont=cont, geodesic=False)
        fin = default_continue(rt, child, goal, min(MAX_EPISODE_STEPS, st.index+cap))
        return outcome_key(fin, st, goal_xz)
    finally:
        rt.tape = rt.sim.tape = orig
        rt.prefix_cache = pc
        rt._loaded = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--configs', default='orca-native-k10')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--limit', type=int, default=24)
    ap.add_argument('--cap', type=int, default=320)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--scan', action='store_true', help='also evaluate the Layer 3 scan queries')
    ap.add_argument('--shifts', default='', help='robust variant: comma-separated people timing shifts (steps)')
    ap.add_argument('--episodes', default='', help='comma-separated <scene>_<episode_id>: only these episodes')
    ap.add_argument('--leaf', default='euclid',
                    help="euclid | field | vmodel:<dir>: rank running candidates by Euclidean goal distance, legal route "
                         "cost-to-go at the decision, or a learned leaf value (leaffit.py --save)")
    ap.add_argument('--margin', type=float, default=0., help='deviation margin (m) within the same outcome category')
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.querygen import native
    native._install()
    from . import suites
    from .env import ConfigSpec
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    specs = suites.load(a.suite)[:a.limit]
    if a.episodes:
        want = set(a.episodes.split(','))
        specs = [s for s in specs if '%s_%s' % tuple(s.key.split('/')[2:4]) in want]
    jobs = [(s, c) for s in specs for c in a.configs.split(',')]
    for i, (s, c) in enumerate(jobs):
        if i % a.shards != a.shard:
            continue
        name = '%s__%s.json' % (s.key.replace('/', '_'), c)
        if (out/name).exists():
            continue
        try:
            r = run_episode(dict(s.asdict()), ConfigSpec.parse(c), a.cap, scan=a.scan,
                            shifts=tuple(int(x) for x in a.shifts.split(',') if x), leaf=a.leaf, margin=a.margin)
        except Exception as exc:
            import traceback
            r = dict(error=repr(exc), tb=traceback.format_exc()[-1500:])
        r.update(config=c, key=s.key)
        (out/name).write_text(json.dumps(r))
        print(json.dumps({k: v for k, v in r.items() if k != 'log'}), flush=True)


if __name__ == '__main__':
    main()
