"""Layer-3 interface: legal consequence features per candidate query.

At a decision state the generator emits <= 10 executable queries.  A learned
selector (Layer 3, not trained here) would choose among them from *legal*
information only.  This module defines that interface: for every candidate,
the kinematic sandbox executes the query once (same executor, planner and
dynamics realiser as native) against the nominal predicted human future, and
the trajectory is scored against all M predicted futures at once
(``sandbox.score_vec``).  Inputs: legal map, legal tracks, predicted futures
(constant-velocity + noise stand-in for frozen Stage 1 samples, [M, T+1, N, 3]).

Feature vector (``FEATURE_NAMES``), one row per query:
  descriptors   kind one-hot, generator category one-hot, via distance/bearing,
                requested duration
  progress      legal cost-to-go reduction to the goal over the horizon (m),
                final legal cost-to-go, query completed, stopped at goal
  effort        steps used, path length, in-place turn steps
  social risk   P[min rectangle distance < 0.25 m] over futures, P5/P50 of the
                minimum rectangle distance, mean steps with a human < 1 m
  static risk   max legal body cost (BLOCKED body-axis samples) and its increase
                over the start pose, share of poses whose body axis touches UNKNOWN
Validation (``run``/``summarize``, privileged and offline only): the same
queries are executed natively (branch simulator == Env.step) at matched
states; we report how well the legal features predict the native failures of
the query's own execution (separately for states with / without legally
tracked humans: humans not yet observed cannot be predicted legally) and the
native equal-horizon value (query + goal continuation).

  python scripts/querygen_run.py eventwm_graph.querygen.l3_interface run RESULT_JSON --out DIR
  python scripts/querygen_run.py eventwm_graph.querygen.l3_interface summarize --out DIR
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np

KINDS = ('goal', 'goto', 'wait', 'rotate', 'stop', 'continue')
CATEGORIES = ('goal', 'pass', 'yield', 'sidestep', 'approach', 'turn', 'rot', 'wait', 'retreat', 'advance',
              'alt', 'frontier', 'look', 'stop', 'continue')
FEATURE_NAMES = ([f'kind_{k}' for k in KINDS]+[f'cat_{c}' for c in CATEGORIES]+
                 ['via_dist_m', 'via_bearing_cos', 'via_bearing_sin', 'duration_steps',
                  'progress_m', 'final_cost_to_go_m', 'completed', 'stopped_at_goal',
                  'steps_used', 'path_len_m', 'turn_steps',
                  'p_overlap', 'dmin_p05', 'dmin_p50', 'psc_steps_mean',
                  'body_cost_max', 'body_cost_increase', 'unknown_body_frac'])


def category(label):
    for c in CATEGORIES:
        if label.startswith(c):
            return c
    return 'goal' if label.startswith('goal') else ''


def descriptor(q, legal):
    pose = legal.pose
    v = np.zeros(len(KINDS)+len(CATEGORIES)+4)
    if q.kind in KINDS:
        v[KINDS.index(q.kind)] = 1.
    c = category(q.label)
    if c:
        v[len(KINDS)+CATEGORIES.index(c)] = 1.
    o = len(KINDS)+len(CATEGORIES)
    if q.via:
        dx, dz = q.via[0][0]-pose[0], q.via[0][1]-pose[2]
        d = math.hypot(dx, dz)
        b = math.atan2(-dz, dx)-pose[3]
        v[o:o+3] = (d, math.cos(b), math.sin(b))
    v[o+3] = float(q.duration or 0)
    return v


def consequence_features(legal, queries, planner_name, dyn, F, steps=48):
    """[Q, D] legal features (``FEATURE_NAMES``) + per-query sandbox seconds."""
    from .gridplan import PlanningView, INF
    from .control import to_plane, body_cells_state
    from .legal import BLOCKED, UNKNOWN
    from .sandbox import rollout, score_vec
    goal_xz = (float(legal.goal[0]), float(legal.goal[2]))
    pose = legal.pose
    view = PlanningView(legal.map, (float(pose[0]), float(pose[2])), goal_xz)
    gfield = view.field([goal_xz])

    def ctg(x, z):
        c = view.cost_to_go(gfield, (x, z))
        return c if c < INF/2 else 99.
    c0 = ctg(float(pose[0]), float(pose[2]))
    rows, secs = [], []
    for q in queries:
        t = time.perf_counter()
        d = descriptor(q, legal)
        if q.kind in ('stop', 'continue'):
            traj = np.asarray([np.asarray(pose, dtype=float)])
            completed = 1. if q.kind == 'stop' else 0.
        else:
            traj = rollout(legal, q, planner_name, dyn, F[0], steps)
            completed = float(len(traj)-1 < steps)      # executor finished before the horizon
        S = score_vec(traj, F) if F.shape[2] else np.tile([np.inf, 0., 0.], (F.shape[0], 1))
        cfin = ctg(float(traj[-1][0]), float(traj[-1][2]))
        seg = np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 2])) if len(traj) > 1 else np.zeros(0)
        dyaw = np.abs(np.angle(np.exp(1j*np.diff(traj[:, 3])))) if len(traj) > 1 else np.zeros(0)
        turn_steps = float(np.sum((dyaw > 1e-3) & (seg < 1e-3)))
        bc, unk = [], []
        for p in traj:
            P = to_plane([p[0], p[2]])
            s = body_cells_state(legal.map, P, float(p[3]))
            bc.append(int((s == BLOCKED).sum())); unk.append(bool((s == UNKNOWN).any()))
        stopped = float(q.kind in ('goal',) and math.hypot(traj[-1][0]-goal_xz[0], traj[-1][2]-goal_xz[1]) < 0.2)
        dmin = S[:, 0]
        feat = np.concatenate([d, [c0-cfin, cfin, completed, stopped, float(len(traj)-1), float(seg.sum()),
                                   turn_steps, float(S[:, 1].mean()),
                                   float(np.percentile(dmin, 5)) if np.isfinite(dmin).all() else 9.,
                                   float(np.percentile(dmin, 50)) if np.isfinite(dmin).all() else 9.,
                                   float(S[:, 2].mean()), float(max(bc)), float(max(bc)-bc[0]), float(np.mean(unk))]])
        rows.append(feat)
        secs.append(time.perf_counter()-t)
    return np.asarray(rows), secs


def run(result_json, out_dir, steps=48, M=16, max_states=8):
    from .native import EpisodeSpec
    from .episode import EpisodeRuntime
    from .generator import QueryGenerator
    from .dynamics import FAMILIES
    from .diversity import reconstruct_states, FAIL_VALUE
    from .executor import Query
    from .sandbox import cv_futures
    r = json.loads(Path(result_json).read_text())
    spec = EpisodeSpec(**r['spec'])
    planner, dyn = r['planner'], FAMILIES[r['dynamics']]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rt = EpisodeRuntime(spec, planner, dyn, strict=True)
    gen = QueryGenerator(dyn)
    states = reconstruct_states(rt, r['decisions'], every=1, max_states=max_states)
    goal = Query('goal', label='goal')
    rows = []
    for st in states:
        queries, _ = gen.propose(st.legal, None)
        F = cv_futures(st.legal.tracks, st.legal.time, steps, M=M, seed=st.index)
        X, secs = consequence_features(st.legal, queries, planner, dyn, F, steps)
        native = []
        for q in queries:
            if q.kind == 'continue':
                native.append(None)
                continue
            base = st.copy(); base.execution = None; base.dyn_state = 0
            t = time.perf_counter()
            ch = rt.run(base, q, horizon=steps)
            q_outcome, q_steps = ch.outcome, ch.index-st.index     # the query's own execution
            while not ch.terminal and ch.index-st.index < steps:
                rem = steps-(ch.index-st.index)
                cont = ch.execution is not None and ch.execution.status == 'running' and ch.execution.query.kind == 'goal'
                ch = rt.run(ch, None if cont else goal, horizon=rem, cont=cont)
            n = ch.index-st.index
            if ch.outcome == 'success':
                v = 0.17*(steps-n)
            elif ch.outcome is None:
                v = -float(ch.geodesic)
            else:
                v = FAIL_VALUE
            native.append(dict(outcome=ch.outcome, value=v, geodesic=float(ch.geodesic), steps=n,
                               query_outcome=q_outcome, query_steps=q_steps, seconds=time.perf_counter()-t))
        rows.append(dict(state_index=st.index, geodesic=float(st.geodesic), tracks=len(st.legal.tracks),
                         labels=[q.label for q in queries], features=X.tolist(), sandbox_s=secs, native=native))
    rt.close()
    rec = dict(key=spec.key, steps=steps, M=M, feature_names=FEATURE_NAMES, states=rows)
    (out_dir/f"{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def auc(scores, labels):
    """Probability a positive scores above a negative (ties 1/2)."""
    s = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt+0.5*eq)


def spearman(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize(out_dir):
    recs = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob('*.json')) if p.name != 'summary.json']
    if not recs:
        return {}
    names = recs[0]['feature_names']
    ix = {n: k for k, n in enumerate(names)}
    X, fail_h, fail_s, fail_any, val, secs, nat_s, tracked = [], [], [], [], [], [], [], []
    sel_regret, sel_fail, n_sel = [], 0, 0
    for rec in recs:
        for st in rec['states']:
            F = np.asarray(st['features'])
            Vs = []
            for k, nat in enumerate(st['native']):
                if nat is None:
                    continue
                X.append(F[k])
                oc = nat.get('query_outcome', nat['outcome'])      # label: the query's own execution
                fail_h.append(oc in ('human_collision', 'body_overlap'))
                fail_s.append(oc == 'static_collision')
                tracked.append(st['tracks'] > 0)
                fail_any.append(nat['value'] <= -999.)
                val.append(nat['value'])
                nat_s.append(nat['seconds'])
                Vs.append((k, nat['value']))
            secs += st['sandbox_s']
            # a fixed, untuned legal rule (illustration of signal, not a policy):
            # among queries predicted safe (p_overlap < 0.1, no body cost increase), max progress
            safe = [k for k, _ in Vs if F[k, ix['p_overlap']] < 0.1 and F[k, ix['body_cost_increase']] <= 0]
            pool = safe or [k for k, _ in Vs]
            if pool and Vs:
                pick = max(pool, key=lambda k: F[k, ix['progress_m']])
                V = dict(Vs)
                best = max(V.values())
                if best > -999.:
                    n_sel += 1
                    if V[pick] <= -999.:
                        sel_fail += 1
                    else:
                        sel_regret.append(best-V[pick])
    X = np.asarray(X); val = np.asarray(val)
    tr = np.asarray(tracked, dtype=bool)
    fh = np.asarray(fail_h, dtype=bool)
    ok = val > -999.
    summ = dict(
        n_queries=len(X), n_states=sum(len(r['states']) for r in recs), episodes=len(recs),
        native_fail_rate=float(np.mean(fail_any)), human_fail_rate=float(np.mean(fail_h)),
        static_fail_rate=float(np.mean(fail_s)),
        tracked_state_query_share=float(tr.mean()) if len(tr) else None,
        human_fail_rate_tracked=float(fh[tr].mean()) if tr.any() else None,
        human_fail_rate_untracked=float(fh[~tr].mean()) if (~tr).any() else None,
        auc_p_overlap_vs_human_fail=auc(X[:, ix['p_overlap']], fail_h),
        auc_p_overlap_vs_human_fail_tracked=auc(X[tr, ix['p_overlap']], fh[tr]) if tr.any() else None,
        auc_neg_dmin_p05_vs_human_fail_tracked=auc(-X[tr, ix['dmin_p05']], fh[tr]) if tr.any() else None,
        auc_neg_dmin_p05_vs_human_fail=auc(-X[:, ix['dmin_p05']], fail_h),
        auc_body_cost_max_vs_static_fail=auc(X[:, ix['body_cost_max']], fail_s),
        auc_body_cost_increase_vs_static_fail=auc(X[:, ix['body_cost_increase']], fail_s),
        auc_unknown_frac_vs_static_fail=auc(X[:, ix['unknown_body_frac']], fail_s),
        spearman_progress_vs_native_value=spearman(X[ok, ix['progress_m']], val[ok]),
        illustrative_rule=dict(states=n_sel, fail_rate=sel_fail/max(1, n_sel),
                               mean_regret_m=float(np.mean(sel_regret)) if sel_regret else None),
        sandbox_ms_per_query_p50=1000*float(np.percentile(secs, 50)), sandbox_ms_per_query_p95=1000*float(np.percentile(secs, 95)),
        native_ms_per_query_p50=1000*float(np.percentile(nat_s, 50)), native_ms_per_query_p95=1000*float(np.percentile(nat_s, 95)))
    (Path(out_dir)/'summary.json').write_text(json.dumps(summ, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'summarize'])
    ap.add_argument('args', nargs='*')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.mode == 'run':
        rec = run(a.args[0], a.out)
        print(json.dumps(dict(key=rec['key'], states=len(rec['states']))))
    else:
        print(json.dumps(summarize(a.out), indent=1))


if __name__ == '__main__':
    main()
