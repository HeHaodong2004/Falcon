"""Start-feasibility certificate (privileged diagnostic, never used online).

Exhaustive breadth-first search over native primitive sequences from the
episode reset with the exact branch simulator and canonical contacts, under
the declared collision rules (official human contact, body-proxy overlap,
static contact with the spawn-contact escape rule).  States are de-duplicated
on (2 cm position, exact turn index, gait phase, escape flag), so the search
is complete up to ``max_depth`` primitives.  Reports whether the robot can get
``escape_m`` metres from the start (and the best geodesic progress) without a
declared collision, plus the contacting links that block it.

  python scripts/querygen_run.py eventwm_graph.querygen.feasibility KEY --out DIR [--depth 60]
"""
import argparse
import json
import math
import time
from collections import deque, Counter
from pathlib import Path


def certify(spec, max_depth=60, escape_m=1.0, budget_s=1800., native_actions=True):
    from .native import (NativeTask, HumanTape, BranchSimulator, classify_contacts, contact_links,
                         escape_exempt)
    task = NativeTask(spec)
    sim = BranchSimulator(task, None)
    root = sim.root()
    p0 = task.robot_pose()
    g0 = task.geodesic_to_goal()
    c0 = classify_contacts(task, gate_humans=False)
    C0 = contact_links(c0)
    sim.tape = HumanTape.record(task, max_depth+2)
    names = {lid: task.robot.sim_obj.get_link_name(lid) for lid in task.robot.sim_obj.get_link_ids()}
    t0 = time.perf_counter()
    q = deque([(root, 0, 0, p0, bool(C0), 0.)])
    seen = {(round(p0[0]/.02), round(p0[2]/.02), 0, 0, bool(C0))}
    best_disp, best_geo, escape, sims = 0., g0, None, 0
    blockers = Counter()
    complete = True
    while q:
        if time.perf_counter()-t0 > budget_s:
            complete = False
            break
        st, depth, ti, pose, esc, trav = q.popleft()
        if depth >= max_depth:
            continue
        for a in ('forward', 'left', 'right'):
            sim.load(st, depth)
            r = sim.step(a, compute_geodesic=True)
            sims += 1
            trav2 = trav+math.hypot(r.pose[0]-pose[0], r.pose[2]-pose[2])
            if r.human_collision or r.body_overlap:
                blockers['human'] += 1
                continue
            if r.static_contact and not escape_exempt(r.static_links, C0, depth+1, trav2):
                for l in r.static_links:
                    blockers[names.get(l, str(l))] += 1
                continue
            esc2 = esc and bool(r.static_contact)
            ti2 = ti+(1 if a == 'left' else -1 if a == 'right' else 0)
            key = (round(r.pose[0]/.02), round(r.pose[2]/.02), ti2 % 75, sim.state.gait % 756, esc2)
            if key in seen:
                continue
            seen.add(key)
            disp = math.hypot(r.pose[0]-p0[0], r.pose[2]-p0[2])
            if not esc2:
                best_disp = max(best_disp, disp)
                best_geo = min(best_geo, r.geodesic)
                if disp >= escape_m and escape is None:
                    escape = depth+1
            q.append((sim.state, depth+1, ti2, r.pose, esc2, trav2))
        if escape is not None:
            break
    task.close()
    return dict(key=spec.key, seed=spec.seed, spawn_contact_links=sorted(names.get(l, str(l)) for l in C0),
                max_depth=max_depth, escape_m=escape_m, states=len(seen), simulated_steps=sims,
                escape_depth=escape, feasible_escape=escape is not None, best_displacement_m=best_disp,
                start_geodesic=g0, best_geodesic=best_geo, search_complete=complete,
                blocking_contacts=dict(blockers.most_common(12)), seconds=time.perf_counter()-t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--out', required=True)
    ap.add_argument('--depth', type=int, default=60)
    ap.add_argument('--suite', default='dev')
    a = ap.parse_args()
    from .suites import load
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    rec = certify(spec, a.depth)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/f"{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec))


if __name__ == '__main__':
    main()
