"""Inspect generator candidates and child outcomes at the first few decisions."""
import math
import sys
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec
from eventwm_graph.querygen.episode import EpisodeRuntime
from eventwm_graph.querygen.generator import QueryGenerator
from eventwm_graph.querygen.oracle import horizon_for, OracleConfig
from eventwm_graph.querygen.dynamics import NATIVE

key = sys.argv[1] if len(sys.argv) > 1 else 'hm3d/val/3t8DB4Uzvkt/1'
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ds, sp, sc, ep = key.split('/')
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 101
rt = EpisodeRuntime(EpisodeSpec(ds, sp, sc, ep, seed), 'orca', NATIVE, strict=True)
gen = QueryGenerator(NATIVE)
cfg = OracleConfig()
node = rt.root
print('root geo', rt.start_geodesic, 'pose', np.round(node.legal.pose, 2))
for d in range(depth):
    qs, diag = gen.propose(node.legal, node.execution)
    print(f'--- decision {d} at step {node.index}: diag {diag}')
    best = None
    for q in qs:
        ch = rt.run(node, q, horizon=horizon_for(q, cfg))
        print(f'  {q.label:14s} {q.kind:8s} via={[tuple(round(v, 2) for v in p) for p in q.via]} -> steps {ch.index-node.index:3d} '
              f'outcome {ch.outcome} geo {ch.geodesic:.3f} pose {np.round(ch.legal.pose, 2).tolist()} '
              f'status {ch.execution.status if ch.execution else None}/{ch.execution.reason if ch.execution else None} '
              f'minrect {ch.min_rect_dist:.2f} static {ch.static_contacts} acts {"".join(a[0] for a in ch.actions[node.index:])}')
        if ch.outcome is None and (best is None or ch.geodesic < best.geodesic):
            best = ch
    if best is None:
        break
    node = best
rt.close()
