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
import numpy as np
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
    names[-1] = 'base'

    def depth_by_link(witnesses):
        out = {}
        for w in witnesses:
            k = names.get(w['robot_link'], str(w['robot_link']))
            out[k] = round(min(out.get(k, 0.), float(w['distance'])), 4)
        return out
    spawn_depth = depth_by_link(c0.get('static_witnesses', []))
    # one-primitive evidence from reset: contacting links, deepest penetration, translation
    first = {}
    for a in ('forward', 'left', 'right'):
        sim.load(root, 0)
        r = sim.step(a, compute_geodesic=True)
        sim.clear_robot_manifolds()
        cc = classify_contacts(task, gate_humans=False)
        first[a] = dict(links=depth_by_link(cc.get('static_witnesses', [])),
                        new_links=sorted(names.get(l, str(l)) for l in r.static_links-C0),
                        translation_m=round(math.hypot(r.pose[0]-p0[0], r.pose[2]-p0[2]), 4))
    t0 = time.perf_counter()
    k0 = (round(p0[0]/.02), round(p0[2]/.02), 0, 0, bool(C0))
    q = deque([(root, 0, 0, p0, bool(C0), 0., k0)])
    seen = {k0}
    parent = {k0: None}
    escape_key = None
    best_disp, best_geo, escape, sims = 0., g0, None, 0
    blockers = Counter()
    complete = True
    depth_limited = False
    while q:
        if time.perf_counter()-t0 > budget_s:
            complete = False
            break
        st, depth, ti, pose, esc, trav, kparent = q.popleft()
        if depth >= max_depth:
            depth_limited = True
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
            parent[key] = (kparent, a[0])
            disp = math.hypot(r.pose[0]-p0[0], r.pose[2]-p0[2])
            if not esc2:
                best_disp = max(best_disp, disp)
                best_geo = min(best_geo, r.geodesic)
                if disp >= escape_m and escape is None:
                    escape = depth+1
                    escape_key = key
            q.append((sim.state, depth+1, ti2, r.pose, esc2, trav2, key))
        if escape is not None:
            break
    task.close()
    seq = ''
    k = escape_key
    while k is not None and parent.get(k) is not None:
        k, a = parent[k]
        seq = a+seq
    return dict(key=spec.key, seed=spec.seed, escape_actions=seq, spawn_contact_links=sorted(names.get(l, str(l)) for l in C0),
                spawn_contact_depth_m=spawn_depth, first_primitive=first,
                max_depth=max_depth, escape_m=escape_m, states=len(seen), simulated_steps=sims,
                escape_depth=escape, feasible_escape=escape is not None, best_displacement_m=best_disp,
                start_geodesic=g0, best_geodesic=best_geo, search_complete=complete, depth_limited=depth_limited,
                certified_infeasible=bool(escape is None and complete and not depth_limited),
                blocking_contacts=dict(blockers.most_common(12)), seconds=time.perf_counter()-t0)


def certify_progress(spec, drop_m=1.5, max_depth=200, budget_s=1800., depth_weight=0.002, exact_gait=False):
    """Witness search (privileged diagnostic): best-first over native primitives by
    true geodesic distance, exact branch simulation with canonical contacts and
    the declared rules (official human contact, body proxy, static contact with
    the spawn escape rule).  Looks for a sequence that reduces the geodesic
    distance by ``drop_m`` (e.g. turns around in a corridor).  De-duplication on
    (2 cm, turn index, escape flag[, gait phase]): with ``exact_gait`` the key
    also holds the leg-gait phase, so exhausting the queue (``exhausted``)
    within ``max_depth`` is a certificate that no such sequence exists up to
    the 2 cm position merge; a found witness is exact (every step simulated)."""
    import heapq
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
    t0 = time.perf_counter()
    k0 = (round(p0[0]/.02), round(p0[2]/.02), 0, bool(C0), 0)
    heap = [(g0, 0, 0, root, 0, p0, bool(C0), 0., k0)]
    max_reached = 0
    parent = {k0: None}
    seen = {k0}
    counter, sims, best_geo, found = 0, 0, g0, None
    depth_limited = False
    while heap and time.perf_counter()-t0 < budget_s:
        _, _, depth, st, ti, pose, esc, trav, kp = heapq.heappop(heap)
        if depth >= max_depth:
            depth_limited = True
            continue
        for a in ('forward', 'left', 'right'):
            sim.load(st, depth)
            r = sim.step(a, compute_geodesic=True)
            sims += 1
            if r.human_collision or r.body_overlap:
                continue
            trav2 = trav+math.hypot(r.pose[0]-pose[0], r.pose[2]-pose[2])
            if r.static_contact and not escape_exempt(r.static_links, C0, depth+1, trav2):
                continue
            esc2 = esc and bool(r.static_contact)
            ti2 = ti+(1 if a == 'left' else -1 if a == 'right' else 0)
            key = (round(r.pose[0]/.02), round(r.pose[2]/.02), ti2 % 75, esc2,
                   (sim.state.gait % 756) if exact_gait else 0)
            if key in seen:
                continue
            seen.add(key)
            parent[key] = (kp, a[0])
            max_reached = max(max_reached, depth+1)
            best_geo = min(best_geo, r.geodesic)
            if r.geodesic <= g0-drop_m:
                found = key
                break
            counter += 1
            heapq.heappush(heap, (r.geodesic+depth_weight*(depth+1), counter, depth+1, sim.state, ti2, r.pose, esc2, trav2, key))
        if found is not None:
            break
    task.close()
    seq = ''
    k = found
    while k is not None and parent.get(k) is not None:
        k, a = parent[k]
        seq = a+seq
    return dict(key=spec.key, drop_m=drop_m, start_geodesic=g0, best_geodesic=best_geo, found=found is not None,
                witness=seq, witness_len=len(seq), states=len(seen), simulated_steps=sims,
                exhausted=not heap, exact_gait=exact_gait, max_depth=max_depth, max_depth_reached=max_reached,
                depth_limited=depth_limited,
                certified_infeasible=bool(found is None and not heap and exact_gait and not depth_limited),
                seconds=time.perf_counter()-t0)


def certify_goal(spec, cell=0.04, heading_bins=75, gaits=(None, 0, 189, 378, 567)):
    """Necessary condition at the goal end: is any final pose inside the success
    region (geodesic < 0.2 m) free of static contact?

    Every 4 cm navmesh point within 0.2 m of the goal with geodesic < 0.2 m,
    every heading bin and several leg-gait phases (None = reset legs) is placed
    kinematically and checked canonically (robot moved far away, contacts
    flushed, placed, detected).  Humans are ignored (static question only).
    If no combination is contact-free the episode cannot end in a collision-free
    success; otherwise the free fraction and best headings are reported.
    """
    import magnum as mn
    from .native import NativeTask, classify_contacts, FAR_AWAY, SUCCESS_DISTANCE
    from .native import BranchSimulator
    task = NativeTask(spec)
    sim = BranchSimulator(task, None)
    robot = task.robot
    obj = robot.sim_obj
    pf = task.sim.pathfinder
    M0 = mn.Matrix4(obj.transformation)
    base0 = np.asarray(robot.base_pos, dtype=float)
    off = np.asarray(M0.translation, dtype=float)-base0
    snap0 = np.asarray(pf.snap_point(base0.astype(np.float32)), dtype=float)
    dy = base0[1]-snap0[1]                    # base height above the navmesh surface
    R0 = mn.Matrix4.from_(M0.rotation(), mn.Vector3(0, 0, 0))
    joints0 = np.asarray(obj.joint_positions, dtype=np.float32).copy()
    lower, upper = obj.joint_position_limits
    joints0 = np.minimum(np.maximum(joints0, np.asarray(lower, np.float32)), np.asarray(upper, np.float32))
    goal = np.asarray(task.goal, dtype=float)
    t0 = time.perf_counter()
    points = []
    r = int(math.ceil(SUCCESS_DISTANCE/cell))
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            q = goal+[i*cell, 0., j*cell]
            if math.hypot(i*cell, j*cell) > SUCCESS_DISTANCE+1e-9:
                continue
            sp = np.asarray(pf.snap_point(q.astype(np.float32)), dtype=float)
            if not np.isfinite(sp).all() or math.hypot(sp[0]-q[0], sp[2]-q[2]) > 0.01:
                continue
            if task.geodesic_to_goal(sp+[0., dy, 0.]) >= SUCCESS_DISTANCE:
                continue
            points.append(sp)
    free = 0; total = 0
    free_by_heading = np.zeros(heading_bins, dtype=int)
    gait_free = {str(g): 0 for g in gaits}
    leg = sim._leg_data
    for g in gaits:
        J = joints0.copy()
        obj.joint_positions = J
        if g is not None:
            robot.leg_joint_pos = leg[int(g) % sim._leg_length]
        for sp in points:
            for h in range(heading_bins):
                yaw = 2*math.pi*h/heading_bins
                obj.transformation = mn.Matrix4.translation(mn.Vector3(*FAR_AWAY))
                task.sim.perform_discrete_collision_detection()
                b = sp+[0., dy, 0.]
                M = mn.Matrix4.translation(mn.Vector3(*(b+off))) @ mn.Matrix4.rotation_y(mn.Rad(yaw)) @ R0
                obj.transformation = M
                robot.update()
                c = classify_contacts(task, gate_humans=False)
                total += 1
                if not c['static']:
                    free += 1; free_by_heading[h] += 1; gait_free[str(g)] += 1
    task.close()
    best = [int(h) for h in np.argsort(-free_by_heading)[:5] if free_by_heading[h] > 0]
    return dict(key=spec.key, goal_points=len(points), checks=total, contact_free=free,
                contact_free_fraction=free/max(1, total), free_by_gait=gait_free,
                best_heading_bins=best, heading_bins=heading_bins, feasible_goal=free > 0,
                seconds=time.perf_counter()-t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--out', required=True)
    ap.add_argument('--depth', type=int, default=60)
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--goal', action='store_true', help='goal-region certificate instead of start escape')
    ap.add_argument('--progress', type=float, default=0., help='witness search for this geodesic drop (m)')
    ap.add_argument('--budget', type=float, default=1800.)
    ap.add_argument('--exact-gait', action='store_true')
    ap.add_argument('--max-depth', type=int, default=200)
    a = ap.parse_args()
    from .suites import load
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    if a.progress:
        rec = certify_progress(spec, a.progress, max_depth=a.max_depth, budget_s=a.budget, exact_gait=a.exact_gait)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out/f"progress{'_exact' if a.exact_gait else ''}_{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
        print(json.dumps(rec))
        return
    if a.goal:
        rec = certify_goal(spec)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out/f"goal_{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
        print(json.dumps(rec))
        return
    rec = certify(spec, a.depth)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/f"{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec))


if __name__ == '__main__':
    main()
