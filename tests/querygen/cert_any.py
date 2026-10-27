"""Certificates for an arbitrary episode spec (outside the frozen suites)."""
import json, sys
from eventwm_graph.querygen.native import EpisodeSpec
from eventwm_graph.querygen.feasibility import certify, certify_progress
ds, split, scene, ep = sys.argv[1].split('/'); seed = int(sys.argv[2])
spec = EpisodeSpec(ds, split, scene, ep, seed)
r = certify(spec, 60)
print(json.dumps({k: r[k] for k in ('key', 'feasible_escape', 'escape_depth', 'escape_actions', 'states', 'certified_infeasible', 'spawn_contact_links', 'first_primitive')}, default=str))
if len(sys.argv) > 3:
    p = certify_progress(spec, float(sys.argv[3]), budget_s=900, exact_gait=False)
    print(json.dumps({k: p[k] for k in ('found', 'witness', 'best_geodesic', 'start_geodesic', 'states', 'exhausted')}))
