"""Where does an Oracle branch lose time against a verified witness?  (privileged diagnostic)

Replays the branch's query sequence from a result JSON (PASS or FAIL) with ``episode.replay`` and the
witness's primitive actions with the branch simulator, then prints, per geodesic-to-goal level, the first
step each trajectory reaches it and the lag, plus the branch's decision timeline (geodesic, distance to the
nearest same-floor human, rectangle clearance) and the nearest human at the witness's corresponding step.

  python scripts/querygen_run.py tests.querygen.branch_lag RESULT_JSON WITNESS_JSON [--suite dev] [--every 0.5]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def first_reach(geo, levels):
    out = []
    for g in levels:
        idx = np.nonzero(geo <= g)[0]
        out.append(int(idx[0]) if len(idx) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result')
    ap.add_argument('witness')
    ap.add_argument('--every', type=float, default=0.5)
    a = ap.parse_args()
    import habitat_sim
    from eventwm_graph.querygen.native import EpisodeSpec, NativeTask, HumanTape, MAX_EPISODE_STEPS
    from eventwm_graph.querygen.episode import replay
    from eventwm_graph.querygen.executor import Query
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.querygen.witness import load_witness_poses
    r = json.loads(Path(a.result).read_text())
    w = json.loads(Path(a.witness).read_text())
    s = r['spec']
    spec = EpisodeSpec(s['dataset'], s['split'], s['scene'], s['episode_id'], s['seed'])
    dyn = FAMILIES[r['dynamics']]
    decisions = []
    for x in r['decisions']:
        q = Query(x['kind'], via=tuple(tuple(v) for v in x['via']), heading=x['heading'], duration=x['duration'],
                  speed=x['speed'], label=x['label'], meta=x['meta'] or {})
        decisions.append(dict(query=q, cont=x['cont'], horizon=x['horizon']))
    try:
        rep = replay(spec, r['planner'], dyn, decisions)
        B = np.asarray([row[:4] for row in rep['trajectory']], dtype=float)       # x, z, yaw, geodesic
    except RuntimeError as exc:
        print('branch replay diverged:', exc)
        return
    P = load_witness_poses(spec, w['actions'], w.get('dynamics') or 'native')
    task = NativeTask(spec)
    try:
        tape = HumanTape.record(task, MAX_EPISODE_STEPS)
        pf = task.sim.pathfinder
        goal = np.asarray(task.goal, dtype=np.float32)

        def geod(p):
            sp = habitat_sim.ShortestPath()
            sp.requested_start = np.asarray(p, dtype=np.float32)
            sp.requested_end = goal
            return float(sp.geodesic_distance) if pf.find_path(sp) else math.nan
        Wg = np.asarray([geod(p[:3]) for p in P])
        y = float(P[0, 1])
        H = tape.positions

        def near_h(x, z, t):
            h = H[min(t, len(H)-1)]
            if len(h) == 0:
                return math.inf
            same = np.abs(h[:, 1]-y) < 0.5
            d = np.hypot(h[:, 0]-x, h[:, 2]-z)
            return float(d[same].min()) if same.any() else math.inf
    finally:
        task.close()
    g0 = max(float(B[0, 3]), float(Wg[0]))
    levels = np.arange(math.floor(g0/a.every)*a.every, -1e-9, -a.every)
    tb, tw = first_reach(B[:, 3], levels), first_reach(Wg, levels)
    print(f"{r['key']} {r['planner']}/{r['dynamics']} verdict {r['verdict']}: branch {len(B)-1} steps (final geodesic "
          f"{B[-1, 3]:.2f}), witness {len(P)-1} steps")
    print(' geodesic<=   witness   branch   lag')
    for g, x, yb in zip(levels, tw, tb):
        lag = (yb-x) if (x is not None and yb is not None) else None
        print(f'   {g:5.1f}     {str(x):>6}   {str(yb):>6}   {str(lag):>5}')
    print(' branch decisions: label [start-end] geodesic start->end | nearest human at end (m)')
    for d in r['decisions']:
        s0, s1 = int(d['start']), min(int(d['end']), len(B)-1)
        print(f"   {d['label']:<16} [{s0:3d}-{s1:3d}] {B[s0, 3]:5.2f}->{B[s1, 3]:5.2f} | "
              f"{near_h(B[s1, 0], B[s1, 1], s1):4.2f}")
    print(' witness every 10 steps: t geodesic nearest-human')
    print('   '+'  '.join(f'{t}:{Wg[t]:.1f}/{near_h(P[t, 0], P[t, 2], t):.1f}' for t in range(0, len(P), 10)))


if __name__ == '__main__':
    main()
