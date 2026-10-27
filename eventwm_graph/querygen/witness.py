"""Privileged feasibility witnesses (diagnostic + Oracle guidance; never used by the generator).

A witness is a complete native action sequence (forward / left / right / stop) from the episode
reset to a successful STOP that obeys every declared rule: no official human collision, no
body-proxy overlap (robot rectangle >= 0.25 m from same-floor human centres), no non-exempt static
contact (canonical contacts, spawn escape rule), inside the 500-step limit.  It is found by a
weighted A* over exactly simulated primitives (branch simulator == Env.step; no legal sensor, so a
step costs ~6 ms) with the privileged space-time cost-to-go (true human tape) as heuristic.

A found witness *proves* the episode feasible (every step was simulated exactly; it is replayed
again from reset before being reported).  Exhausting the queue is not a proof of infeasibility here
(states are merged on 5 cm x turn index x 4-step time buckets); use feasibility.py certificates
for that.

  python scripts/querygen_run.py eventwm_graph.querygen.witness KEY --suite dev --out DIR [--budget 1800]
"""
import argparse
import heapq
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ACTIONS = ('forward', 'left', 'right')


def find_witness(spec, budget_s=1800., max_sims=400_000, weight=3.0, cell=0.05, t_bucket=4):
    from .native import (NativeTask, HumanTape, BranchSimulator, classify_contacts, contact_links,
                         escape_exempt, MAX_EPISODE_STEPS, SUCCESS_DISTANCE)
    from .oracle import SpaceTimeHeuristic
    t0 = time.perf_counter()
    task = NativeTask(spec)
    sim = BranchSimulator(task, None)
    root = sim.root()
    p0 = task.robot_pose()
    g0 = task.geodesic_to_goal()
    c0 = classify_contacts(task, gate_humans=False)
    C0 = contact_links(c0)
    sim.tape = HumanTape.record(task, MAX_EPISODE_STEPS+2)
    st = SpaceTimeHeuristic(SimpleNamespace(task=task, tape=sim.tape))

    import habitat_sim
    pf = task.sim.pathfinder
    goal = np.asarray(task.goal, dtype=np.float32)
    from .dynamics import NATIVE_TURN_RAD

    def heading_steps(pose):
        """In-place turn steps needed to face the geodesic path (look-ahead 0.6 m)."""
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(pose[:3], dtype=np.float32)
        sp.requested_end = goal
        if not pf.find_path(sp) or len(sp.points) < 2:
            return 0.
        P = np.asarray(sp.points, dtype=float)
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 2]))
        cum = np.concatenate([[0.], np.cumsum(seg)])
        L = min(0.6, float(cum[-1]))
        x = np.interp(L, cum, P[:, 0]); z = np.interp(L, cum, P[:, 2])
        if math.hypot(x-pose[0], z-pose[2]) < 1e-6:
            return 0.
        want = math.atan2(-(z-pose[2]), x-pose[0])
        err = abs(math.atan2(math.sin(want-pose[3]), math.cos(want-pose[3])))
        return err/NATIVE_TURN_RAD

    def h(pose, t, geo):
        v = st.value(float(pose[0]), float(pose[2]), t)
        base = geo/0.208
        hs = heading_steps(pose) if geo > 0.3 else 0.
        return (max(base, v) if v is not None else base+300.)+hs

    counter = 0
    k0 = (round(p0[0]/cell), round(p0[2]/cell), 0, 0, bool(C0))
    heap = [(weight*h(p0, 0, g0), counter, 0, root, 0, p0, bool(C0), 0., k0)]
    parent = {k0: None}
    seen = {k0}
    sims, best_geo, found = 0, g0, None
    while heap and time.perf_counter()-t0 < budget_s and sims < max_sims:
        _, _, t, state, ti, pose, esc, trav, key = heapq.heappop(heap)
        if t >= MAX_EPISODE_STEPS-1:
            continue
        for a in ACTIONS:
            sim.load(state, t)
            r = sim.step(a, compute_geodesic=True)
            sims += 1
            if r.human_collision or r.body_overlap:
                continue
            trav2 = trav+math.hypot(r.pose[0]-pose[0], r.pose[2]-pose[2])
            if r.static_contact and not escape_exempt(r.static_links, C0, t+1, trav2):
                continue
            esc2 = esc and bool(r.static_contact)
            ti2 = ti+(1 if a == 'left' else -1 if a == 'right' else 0)
            k2 = (round(r.pose[0]/cell), round(r.pose[2]/cell), ti2 % 75, (t+1)//t_bucket, esc2)
            if k2 in seen:
                continue
            seen.add(k2)
            parent[k2] = (key, a)
            best_geo = min(best_geo, r.geodesic)
            if r.geodesic < SUCCESS_DISTANCE:
                # STOP here (one more step): success by the official definition
                st2 = sim.state
                sim.load(st2, t+1)
                rs = sim.step('stop', compute_geodesic=True)
                sims += 1
                if rs.success and not rs.human_collision and not rs.body_overlap:
                    found = k2
                    break
            counter += 1
            heapq.heappush(heap, ((t+1)+weight*h(r.pose, t+1, r.geodesic), counter, t+1, sim.state, ti2, r.pose,
                                  esc2, trav2, k2))
        if found is not None:
            break
    seq = []
    k = found
    while k is not None and parent.get(k) is not None:
        k, a = parent[k]
        seq.append(a)
    seq.reverse()
    task.close()
    rec = dict(key=spec.key, found=found is not None, actions=''.join(a[0] for a in seq)+('s' if found else ''),
               steps=len(seq)+(1 if found else 0), start_geodesic=g0, best_geodesic=best_geo, sims=sims,
               states=len(seen), exhausted=not heap, seconds=time.perf_counter()-t0)
    if found:
        rec['verified'] = verify_witness(spec, rec['actions'])
    return rec


def verify_witness(spec, actions):
    """Re-execute the witness from reset with the authoritative Env.step and a lockstep shadow
    branch simulator (canonical contacts); check every declared rule."""
    from .native import (NativeTask, HumanTape, BranchSimulator, classify_contacts, contact_links,
                         escape_exempt, MAX_EPISODE_STEPS)
    names = {'f': 'forward', 'l': 'left', 'r': 'right', 's': 'stop'}
    task = NativeTask(spec)
    shadow = NativeTask(spec)
    try:
        ssim = BranchSimulator(shadow, None)
        sroot = ssim.root()
        c0 = classify_contacts(shadow, gate_humans=False)
        C0 = contact_links(c0)
        ssim.tape = HumanTape.record(shadow, MAX_EPISODE_STEPS)
        ssim.load(sroot, 0)
        statics = overlaps = mismatch = 0
        trav = 0.
        for ch in actions:
            a = names[ch]
            prev = task.robot_pose()
            task.step(a)
            res = ssim.step(a)
            pose = task.robot_pose()
            mismatch += int(not np.array_equal(pose, res.pose))
            trav += math.hypot(pose[0]-prev[0], pose[2]-prev[2])
            overlaps += int(res.body_overlap)
            if res.static_contact and not escape_exempt(res.static_links, C0, task.steps, trav):
                statics += 1
            if task.episode_over:
                break
        m = task.metrics()
        ok = (m.get('success') == 1.0 and m.get('human_collision') == 0.0 and statics == 0 and overlaps == 0
              and mismatch == 0)
        return dict(ok=bool(ok), metrics=m, static_steps=statics, overlap_steps=overlaps, shadow_mismatch=mismatch)
    finally:
        task.close()
        shadow.close()


def load_witness_poses(spec, actions):
    """Exact pose trajectory [T+1, 4] of a witness (branch simulator from reset)."""
    from .native import NativeTask, HumanTape, BranchSimulator, MAX_EPISODE_STEPS
    names = {'f': 'forward', 'l': 'left', 'r': 'right', 's': 'stop'}
    task = NativeTask(spec)
    try:
        sim = BranchSimulator(task, None)
        root = sim.root()
        sim.tape = HumanTape.record(task, MAX_EPISODE_STEPS)
        sim.load(root, 0)
        poses = [task.robot_pose()]
        for ch in actions:
            if ch == 's':
                break
            r = sim.step(names[ch])
            poses.append(r.pose)
        return np.asarray(poses)
    finally:
        task.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget', type=float, default=1800.)
    ap.add_argument('--weight', type=float, default=3.0)
    a = ap.parse_args()
    from .suites import load
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    rec = find_witness(spec, budget_s=a.budget, weight=a.weight)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/f"{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: v for k, v in rec.items() if k != 'actions'}, default=float))


if __name__ == '__main__':
    main()
