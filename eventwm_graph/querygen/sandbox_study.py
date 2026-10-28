"""Sandbox approximation study driver (see sandbox.py).

  python scripts/querygen_run.py eventwm_graph.querygen.sandbox_study run RESULT_JSON --out DIR
  python scripts/querygen_run.py eventwm_graph.querygen.sandbox_study summarize --out DIR
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np


def run(result_json, out_dir, steps=48, M=16, planner='orca', dynamics='native'):
    from .native import EpisodeSpec
    from .episode import EpisodeRuntime
    from .generator import QueryGenerator
    from .dynamics import FAMILIES
    from .diversity import reconstruct_states
    from .sandbox import cv_futures, compare, rollout, score
    r = json.loads(Path(result_json).read_text())
    spec = EpisodeSpec(**r['spec'])
    dyn = FAMILIES[dynamics]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rt = EpisodeRuntime(spec, planner, dyn, strict=True, velocity_action=True)
    gen = QueryGenerator(dyn)
    states = reconstruct_states(rt, r['decisions'], every=1, max_states=40)
    rows = []
    for st in states:
        if not st.legal.tracks:
            continue
        queries, _ = gen.propose(st.legal, None)
        F = cv_futures(st.legal.tracks, st.legal.time, steps, M=M, seed=st.index)
        comp = [asdict(c) for c in compare(st.legal, queries, planner, dyn, F, steps)]
        # sandbox vs native with the TRUE future (privileged validation only)
        tape = rt.tape
        Ftrue = tape.positions[st.index:min(len(tape), st.index+steps+1)]
        val = []
        for q in queries:
            if q.kind in ('continue', 'stop'):
                continue
            base = st.copy(); base.execution = None; base.dyn_state = 0
            t = time.perf_counter()
            nat = rt.run(base, q, horizon=steps)
            t_nat = time.perf_counter()-t
            t = time.perf_counter()
            sb = rollout(st.legal, q, planner, dyn, Ftrue, steps)
            t_sb = time.perf_counter()-t
            npos = np.asarray(nat.legal.poses[len(st.legal.poses)-1:])
            n = min(len(npos), len(sb))
            dev = np.hypot(npos[:n, 0]-sb[:n, 0], npos[:n, 2]-sb[:n, 2])
            sb_s = score(sb, Ftrue[None])[0]
            val.append(dict(query=q.label, native_outcome=nat.outcome, native_min_rect=float(nat.min_rect_dist),
                            sandbox_overlap=bool(sb_s[1] > 0), sandbox_min_rect=float(sb_s[0]),
                            mean_dev=float(dev.mean()), end_dev=float(dev[-1]), len_native=len(npos), len_sandbox=len(sb),
                            native_s=t_nat, sandbox_s=t_sb))
        rows.append(dict(state_index=st.index, tracks=len(st.legal.tracks), comparisons=comp, native_validation=val))
        if len(rows) >= 8:
            break
    rt.close()
    rec = dict(key=spec.key, planner=planner, dynamics=dynamics, steps=steps, M=M, states=rows)
    (out_dir/f"{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def summarize(out_dir):
    recs = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob('*.json')) if p.name != 'summary.json']
    C, V = [], []
    top_agree = []
    for rec in recs:
        for st in rec['states']:
            C += st['comparisons']; V += st['native_validation']
            cs = st['comparisons']
            if len(cs) >= 2:
                ko = min(range(len(cs)), key=lambda i: (cs[i]['open_overlap'], -cs[i]['open_dmin']))
                kc = min(range(len(cs)), key=lambda i: (cs[i]['closed_overlap'], -cs[i]['closed_dmin']))
                top_agree.append(ko == kc)
    if not C:
        return {}
    oo = np.array([c['open_overlap'] for c in C]); co = np.array([c['closed_overlap'] for c in C])
    od = np.array([c['open_dmin'] for c in C]); cd = np.array([c['closed_dmin'] for c in C])
    fin = np.isfinite(od) & np.isfinite(cd)
    summ = dict(
        n_queries=len(C), n_states=sum(len(r['states']) for r in recs),
        overlap_prob_mae=float(np.abs(oo-co).mean()),
        overlap_prob_corr=float(np.corrcoef(oo, co)[0, 1]) if oo.std() > 0 and co.std() > 0 else None,
        open_overlap_mean=float(oo.mean()), closed_overlap_mean=float(co.mean()),
        dmin_mae=float(np.abs(od[fin]-cd[fin]).mean()) if fin.any() else None,
        end_deviation_mean_m=float(np.mean([c['end_dev_mean'] for c in C])),
        top_choice_agreement=float(np.mean(top_agree)) if top_agree else None,
        open_ms_per_query=1000*float(np.mean([c['open_s'] for c in C])),
        closed_ms_per_query=1000*float(np.mean([c['closed_s'] for c in C])),
        native_validation=dict(
            n=len(V), mean_dev_m=float(np.mean([v['mean_dev'] for v in V])),
            end_dev_m=float(np.median([v['end_dev'] for v in V])),
            overlap_agreement=float(np.mean([(v['native_outcome'] in ('body_overlap', 'human_collision')) == v['sandbox_overlap'] for v in V])),
            native_ms=1000*float(np.mean([v['native_s'] for v in V])), sandbox_ms=1000*float(np.mean([v['sandbox_s'] for v in V]))))
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
