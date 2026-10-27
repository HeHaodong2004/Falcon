"""Aggregate evaluation records into the declared metrics (Markdown + JSON).

  python scripts/querygen_run.py eventwm_graph.querygen.report RUN_DIR [RUN_DIR ...] [--cert DIR]

Behaviour metrics are computed from the replayed trace (records written by
``evaluate`` since the trace field was added; older records report them as
unavailable).  Definitions (declared, applied identically to every run):

* ``time_s``           replay steps x 0.048 s world time.
* ``wait_steps``       steps executed under ``wait`` queries.
* ``turn_steps``       in-place rotation steps (native ``left``/``right``).
* ``oscillations``     turn-direction reversals (l->r / r->l between consecutive
                       turn actions with no forward step in between) outside
                       ``wait`` queries, per 100 steps.
* ``retreat_m``        summed geodesic-distance increase over the episode.
* ``deadlock``         some 120-step (5.76 s) window farther than 1 m from the
                       goal with less than 0.25 m geodesic progress.
* runtime              generator ms per call, Oracle edge ms, Oracle seconds per
                       episode, replay seconds; P50/P95 over all samples.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

STEP_S = 0.048
DEADLOCK_WINDOW = 120
DEADLOCK_PROGRESS_M = 0.25


def pct(xs, q):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.percentile(xs, q)) if xs else None


def behaviour(rec):
    rep = rec.get('replay') or {}
    tr = rep.get('trace')
    if not tr:
        return None
    acts = tr['actions']
    traj = np.array([[np.nan if v is None else v for v in row] for row in tr['trajectory']], dtype=float)
    labels = []
    for d, n in zip(rec.get('decisions', []), tr['per_decision_steps']):
        labels += [d.get('kind', '')]*n
    labels += ['']*(len(acts)-len(labels))
    wait = sum(1 for k in labels if k == 'wait')
    turns = sum(1 for a in acts if a in 'lr')
    osc, last = 0, None
    for a, k in zip(acts, labels):
        if a == 'f':
            last = None
        elif a in 'lr':
            if k != 'wait' and last is not None and a != last:
                osc += 1
            last = a if k != 'wait' else None
    geo = traj[:, 3]
    dg = np.diff(geo)
    retreat = float(dg[dg > 0].sum()) if len(dg) else 0.
    dead = False
    for i in range(0, max(0, len(geo)-DEADLOCK_WINDOW)):
        if geo[i] > 1.0 and geo[i]-geo[i:i+DEADLOCK_WINDOW+1].min() < DEADLOCK_PROGRESS_M:
            dead = True
            break
    steps = len(acts)
    return dict(steps=steps, time_s=steps*STEP_S, wait_steps=wait, turn_steps=turns,
                forward_steps=sum(1 for a in acts if a in 'fv'),
                oscillations_per_100=100.*osc/max(1, steps), retreat_m=retreat, deadlock=dead,
                min_clearance=float(np.nanmin(traj[:, 4])) if np.isfinite(traj[:, 4]).any() else None)


def effective_branching(n_nodes, depth):
    """b* with n_nodes = b + b^2 + ... + b^depth (bisection; None if undefined)."""
    if not depth or n_nodes < depth:
        return None
    f = lambda b: sum(b**i for i in range(1, depth+1))-n_nodes
    lo, hi = 1e-6, max(2., float(n_nodes))
    if f(lo) > 0:
        return None
    for _ in range(100):
        mid = 0.5*(lo+hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)


def search_stats(r):
    """Search-tree quality of one Oracle run."""
    o = r.get('oracle') or {}
    st = o.get('stats', {})
    cfg = r.get('oracle_config', {})
    oc = st.get('outcomes', {})
    edges = int(st.get('children', 0))
    failed = sum(v for k, v in oc.items() if k != 'success')
    solved = bool(o.get('success'))
    max_edges = cfg.get('max_edges', 6000)
    max_s = cfg.get('max_seconds', 1e9)
    if solved:
        status = 'solved'
    elif edges >= max_edges or o.get('seconds', 0) >= max_s-1:
        status = 'budget'
    else:
        status = 'exhausted'           # the induced search space ran out (up to state merging)
    nodes = int(st.get('expansions', 0))
    depth = len(r.get('decisions', [])) if solved else None
    return dict(status=status, edges=edges, nodes=nodes, depth=depth,
                b_eff=effective_branching(nodes, depth) if solved else None,
                dup_rate=st.get('duplicates', 0)/max(1, edges-failed),
                human_edge_rate=(oc.get('human_collision', 0)+oc.get('body_overlap', 0))/max(1, edges),
                static_edge_rate=oc.get('static_collision', 0)/max(1, edges),
                kept_per_node=st.get('kept_candidates', 0)/max(1, nodes))


def path_length(rec):
    tr = (rec.get('replay') or {}).get('trace')
    if not tr:
        return None
    T = np.array([[np.nan if v is None else v for v in row] for row in tr['trajectory']], dtype=float)
    return float(np.hypot(np.diff(T[:, 0]), np.diff(T[:, 1])).sum())


def search_summary(recs, budgets=(500, 1000, 2000, 3000, 6000)):
    S = [(r, search_stats(r)) for r in recs]
    solved = [(r, x) for r, x in S if x['status'] == 'solved']
    med = lambda v: float(np.median(v)) if v else None
    out = dict(n=len(S), solved=len(solved),
               status=dict(Counter(x['status'] for _, x in S)),
               success_at_budget={b: sum(1 for _, x in solved if x['edges'] <= b) for b in budgets},
               edges_to_solution_median=med([x['edges'] for _, x in solved]),
               nodes_to_solution_median=med([x['nodes'] for _, x in solved]),
               solution_decisions_median=med([x['depth'] for _, x in solved]),
               b_eff_median=med([x['b_eff'] for _, x in solved if x['b_eff']]),
               dup_rate_mean=float(np.mean([x['dup_rate'] for _, x in S])) if S else None,
               human_edge_rate_mean=float(np.mean([x['human_edge_rate'] for _, x in S])) if S else None,
               static_edge_rate_mean=float(np.mean([x['static_edge_rate'] for _, x in S])) if S else None,
               kept_per_node_mean=float(np.mean([x['kept_per_node'] for _, x in S])) if S else None)
    P = [r for r in recs if r.get('verdict') == 'PASS']
    if P:
        m = [r['replay']['metrics'] for r in P]
        lens = [path_length(r) for r in P]
        out.update(path_length_m_mean=float(np.mean([v for v in lens if v is not None])) if any(v is not None for v in lens) else None,
                   time_s_mean=float(np.mean([r['replay']['steps']*STEP_S for r in P])),
                   spl_mean=float(np.mean([x['spl'] for x in m])), psc_mean=float(np.mean([x['psc'] for x in m])))
    return out


def load_run(d):
    d = Path(d)
    recs = [json.loads(p.read_text()) for p in sorted(d.glob('*.json')) if p.name not in ('summary.json', 'run.json', 'report.json')]
    return [r for r in recs if 'verdict' in r]


def summarize_run(d, certs=None):
    recs = load_run(d)
    rows = []
    gen_ms, edge_ms, oracle_s, replay_s, raw, kept = [], [], [], [], [], []
    labels = Counter()
    for r in recs:
        o = r.get('oracle') or r.get('branch') or {}
        st = o.get('stats', {})
        gen_ms += st.get('gen_ms', []); edge_ms += st.get('edge_ms', [])
        if 'seconds' in o:
            oracle_s.append(o['seconds'])
        rep = r.get('replay') or {}
        if 'seconds' in rep:
            replay_s.append(rep['seconds'])
        if st.get('expansions'):
            raw.append(st.get('raw_candidates', 0)/st['expansions']); kept.append(st.get('kept_candidates', 0)/st['expansions'])
        if r.get('verdict') == 'PASS':
            for dec in r.get('decisions', []):
                labels[dec.get('label', dec.get('kind')).rstrip('0123456789')] += 1
        m = rep.get('metrics', {})
        b = behaviour(r)
        cert = (certs or {}).get(r['key'])
        rows.append(dict(key=r['key'], verdict=r['verdict'], success=m.get('success'), spl=m.get('spl'),
                         psc=m.get('psc'), steps=rep.get('steps'), human_collision=m.get('human_collision'),
                         static_steps=rep.get('static_contact_steps'), escape_steps=rep.get('escape_contact_steps'),
                         overlap_steps=rep.get('body_overlap_steps'), min_rect=rep.get('min_rect_distance'),
                         start_geo=r.get('start_geodesic'), final_geo=rep.get('final_geodesic', o.get('final_geodesic')),
                         decisions=len(r.get('decisions', [])), oracle_s=o.get('seconds'),
                         expansions=st.get('expansions'), spawn_contact=r.get('spawn_contact'),
                         certified_infeasible=bool(cert and cert['infeasible']),
                         certified_by=(cert or {}).get('by', []),
                         behaviour=b))
    n = len(rows)
    passed = [x for x in rows if x['verdict'] == 'PASS']
    infeasible = [x for x in rows if x['certified_infeasible']]
    feas = [x for x in rows if not x['certified_infeasible']]
    B = [x['behaviour'] for x in rows if x['behaviour'] and x['verdict'] == 'PASS']
    summ = dict(
        run=str(d), n=n, passed=len(passed), pass_rate=len(passed)/max(1, n),
        search=search_summary(recs),
        certified_infeasible=[x['key'] for x in infeasible],
        feasible_n=len(feas), feasible_passed=sum(x['verdict'] == 'PASS' for x in feas),
        verdicts=dict(Counter(x['verdict'] for x in rows)),
        official_success=sum(1 for x in rows if x['success'] == 1.0),
        human_collision_eps=sum(1 for x in rows if x['human_collision']),
        mean_spl_all=float(np.mean([x['spl'] or 0. for x in rows])) if rows else None,
        mean_spl_pass=float(np.mean([x['spl'] for x in passed])) if passed else None,
        mean_psc_pass=float(np.mean([x['psc'] for x in passed if x['psc'] is not None])) if passed else None,
        mean_time_s_pass=float(np.mean([x['steps']*STEP_S for x in passed])) if passed else None,
        min_clearance_pass=min((x['min_rect'] for x in passed if x['min_rect'] is not None), default=None),
        behaviour_pass=(dict(
            wait_steps_mean=float(np.mean([b['wait_steps'] for b in B])),
            turn_steps_mean=float(np.mean([b['turn_steps'] for b in B])),
            oscillations_per_100_mean=float(np.mean([b['oscillations_per_100'] for b in B])),
            retreat_m_mean=float(np.mean([b['retreat_m'] for b in B])),
            deadlock_eps=sum(b['deadlock'] for b in B), n=len(B)) if B else None),
        queries=dict(raw_per_node_mean=float(np.mean(raw)) if raw else None,
                     kept_per_node_mean=float(np.mean(kept)) if kept else None,
                     selected_label_counts=dict(labels.most_common())),
        runtime=dict(gen_ms_p50=pct(gen_ms, 50), gen_ms_p95=pct(gen_ms, 95),
                     edge_ms_p50=pct(edge_ms, 50), edge_ms_p95=pct(edge_ms, 95),
                     oracle_s_p50=pct(oracle_s, 50), oracle_s_p95=pct(oracle_s, 95),
                     replay_s_p50=pct(replay_s, 50), replay_s_p95=pct(replay_s, 95),
                     n_gen_calls=len(gen_ms), n_edges=len(edge_ms)),
        episodes=rows)
    return summ


def markdown(summ):
    L = [f"### {summ['run']}", '',
         f"PASS {summ['passed']}/{summ['n']}  (certified-infeasible: {len(summ['certified_infeasible'])}; "
         f"PASS on feasible {summ['feasible_passed']}/{summ['feasible_n']}); official success {summ['official_success']}; "
         f"human-collision episodes {summ['human_collision_eps']}", '',
         '| episode | verdict | steps | SPL | PSC | min clr | static | esc | overlap | decisions | oracle s | wait | osc/100 | retreat m | deadlock |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    f = lambda v, k=2: '-' if v is None else (f'{v:.{k}f}' if isinstance(v, float) else str(v))
    for x in summ['episodes']:
        b = x['behaviour'] or {}
        cert = f" (cert. infeasible: {'+'.join(x['certified_by'])})" if x['certified_infeasible'] else ''
        L.append(f"| {x['key']} | {x['verdict']}{cert} | {f(x['steps'])} | "
                 f"{f(x['spl'])} | {f(x['psc'])} | {f(x['min_rect'])} | {f(x['static_steps'])} | {f(x['escape_steps'])} | "
                 f"{f(x['overlap_steps'])} | {x['decisions']} | {f(x['oracle_s'], 0)} | {f(b.get('wait_steps'))} | "
                 f"{f(b.get('oscillations_per_100'), 1)} | {f(b.get('retreat_m'))} | {f(b.get('deadlock'))} |")
    rt = summ['runtime']
    L += ['', f"runtime: generator {f(rt['gen_ms_p50'], 1)}/{f(rt['gen_ms_p95'], 1)} ms (P50/P95, {rt['n_gen_calls']} calls); "
          f"oracle edge {f(rt['edge_ms_p50'], 1)}/{f(rt['edge_ms_p95'], 1)} ms ({rt['n_edges']} edges); "
          f"oracle per episode {f(rt['oracle_s_p50'], 0)}/{f(rt['oracle_s_p95'], 0)} s; replay {f(rt['replay_s_p50'], 0)}/{f(rt['replay_s_p95'], 0)} s",
          f"queries: raw/node {f(summ['queries']['raw_per_node_mean'], 1)}, kept/node {f(summ['queries']['kept_per_node_mean'], 1)}",
          f"selected query labels in PASS branches: {summ['queries']['selected_label_counts']}"]
    sr = summ['search']
    L.append(f"search: status {sr['status']}; success@edges {sr['success_at_budget']}; edges/nodes to solution (median) "
             f"{f(sr['edges_to_solution_median'], 0)}/{f(sr['nodes_to_solution_median'], 0)}; solution decisions {f(sr['solution_decisions_median'], 0)}; "
             f"b_eff {f(sr['b_eff_median'], 2)}; duplicate rate {f(sr['dup_rate_mean'], 2)}; edges ending in human contact/overlap "
             f"{f(sr['human_edge_rate_mean'], 3)}, static {f(sr['static_edge_rate_mean'], 3)}; kept queries/node {f(sr['kept_per_node_mean'], 1)}")
    if sr.get('spl_mean') is not None:
        L.append(f"paths (PASS): length {f(sr['path_length_m_mean'], 2)} m, time {f(sr['time_s_mean'], 1)} s, SPL {f(sr['spl_mean'], 3)}, PSC {f(sr['psc_mean'], 3)}")
    if summ['behaviour_pass']:
        b = summ['behaviour_pass']
        L.append(f"behaviour (PASS episodes with trace, n={b['n']}): wait {b['wait_steps_mean']:.1f} steps, "
                 f"turns {b['turn_steps_mean']:.1f}, osc/100 {b['oscillations_per_100_mean']:.2f}, "
                 f"retreat {b['retreat_m_mean']:.2f} m, deadlock episodes {b['deadlock_eps']}")
    return '\n'.join(L)


def load_certs(d):
    """Per episode: {'infeasible': bool, 'by': [certificate kinds proving it], 'files': [...]}.

    start escape (``feasible_escape``): complete search (no time-out, not depth-limited) without
    escape; progress (``drop_m``): ``certified_infeasible``; goal region (``feasible_goal``): no
    contact-free final pose."""
    out = {}
    if d:
        for p in sorted(Path(d).glob('*.json')):
            c = json.loads(p.read_text())
            e = out.setdefault(c['key'], dict(infeasible=False, by=[], files=[]))
            e['files'].append(p.name)
            if 'feasible_escape' in c:
                proven = (not c['feasible_escape']) and c.get('search_complete') and not c.get('depth_limited', False)
                kind = 'start_escape'
            elif 'drop_m' in c:
                proven = bool(c.get('certified_infeasible'))
                kind = 'progress'
            elif 'feasible_goal' in c:
                # an empty test set (no navmesh point found in the success region) proves nothing
                proven = c['feasible_goal'] is False and c.get('goal_points', 0) > 0
                kind = 'goal_region'
            else:
                continue
            if proven:
                e['infeasible'] = True
                e['by'].append(kind)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--cert', default=None, help='feasibility certificate directory')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    certs = load_certs(a.cert)
    parts = []
    for d in a.runs:
        s = summarize_run(d, certs)
        (Path(d)/'report.json').write_text(json.dumps(s, indent=1, default=float))
        parts.append(markdown(s))
    text = '\n\n'.join(parts)
    if a.md:
        Path(a.md).write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
