"""Diagnostic: can the generator's queries track a verified witness?

Beam search over decision depth from the reset; children are scored by the witness lateness
(index + h_w - T, 0 = on the witness schedule; same h_w as the guided Oracle).  Prints per depth
the best children (label chain, lateness, deviation from the witness pose at the same index) and
all candidate outcomes of the best node, so the first decision where every generator query falls
behind the witness is visible.  Privileged; never used by the generator.

  python scripts/querygen_run.py tests.querygen.witness_track KEY --witness DIR [--beam 3] [--gen JSON]
"""
import argparse
import json
import math

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--witness', required=True)
    ap.add_argument('--beam', type=int, default=3)
    ap.add_argument('--depth', type=int, default=40)
    ap.add_argument('--gen', default='{}')
    ap.add_argument('--planner', default='orca')
    ap.add_argument('--dynamics', default='native')
    a = ap.parse_args()
    from eventwm_graph.querygen.suites import load
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
    from eventwm_graph.querygen.oracle import Oracle, OracleConfig, horizon_for
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.native import MAX_EPISODE_STEPS, SUCCESS_DISTANCE
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    dyn = FAMILIES[a.dynamics]
    rt = EpisodeRuntime(spec, a.planner, dyn, strict=True)
    gen = QueryGenerator(dyn, GeneratorConfig(**json.loads(a.gen)))
    orc = Oracle(rt, gen, OracleConfig(witness_dir=a.witness))
    orc._witness_h(rt.root)
    W = orc._witness
    T = len(W)-1
    print(f'witness {T} steps; start geodesic {rt.start_geodesic:.2f}', flush=True)

    def late(st):
        return st.index+orc._witness_h(st)-T

    def dev(st):
        k = min(st.index, T)
        p = st.legal.pose
        return math.hypot(W[k, 0]-p[0], W[k, 1]-p[2])

    def chain(st):
        return ' '.join(f"{d['query'].label}@{d['start']}" for d in st.decisions)

    beam = [rt.root]
    for depth in range(a.depth):
        kids = []
        for node in beam:
            qs, diag = gen.propose(node.legal, node.execution)
            rows = []
            for q in qs:
                ch = rt.run(node, q, horizon=min(horizon_for(q, orc.cfg), MAX_EPISODE_STEPS-node.index))
                if ch.outcome == 'success':
                    print(f'SUCCESS depth {depth} steps {ch.index}: {chain(ch)}', flush=True)
                    return
                if ch.outcome is None and ch.geodesic < SUCCESS_DISTANCE:
                    stop_q = next((x for x in gen.propose(ch.legal, ch.execution)[0] if x.kind == 'stop'), None)
                    if stop_q is not None:
                        done = rt.run(ch, stop_q, horizon=1)
                        if done.outcome == 'success':
                            print(f'SUCCESS depth {depth} steps {done.index}: {chain(done)}', flush=True)
                            return
                rows.append((q.label, ch.outcome, ch.index, None if ch.outcome else round(late(ch), 1),
                             None if ch.outcome else round(dev(ch), 2), round(float(ch.geodesic), 2)))
                if ch.outcome is None:
                    kids.append(ch)
            if node is beam[0]:
                print(f'  d{depth} node t={node.index} late={late(node):.1f} dev={dev(node):.2f} stuck={diag.get("stuck")}'
                      f' cands: ' + '; '.join(f'{r[0]}:{r[1] or ""}t{r[2]} L{r[3]} d{r[4]} g{r[5]}' for r in rows), flush=True)
        if not kids:
            print('no surviving children', flush=True)
            return
        uniq = {}
        for ch in kids:
            p = ch.legal.pose
            sig = (round(p[0]/0.1), round(p[2]/0.1), int((p[3] % (2*math.pi))/(2*math.pi)*24), ch.index//4)
            if sig not in uniq or late(ch) < late(uniq[sig]):
                uniq[sig] = ch
        beam = sorted(uniq.values(), key=late)[:a.beam]
        b = beam[0]
        print(f'depth {depth}: best t={b.index} late={late(b):.1f} dev={dev(b):.2f} geo={b.geodesic:.2f} | {chain(b)}',
              flush=True)
        if b.index >= MAX_EPISODE_STEPS-1:
            break
    rt.close()


if __name__ == '__main__':
    main()
