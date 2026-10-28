"""Native-infeasibility certificate under the official Falcon rules (privileged accounting, never used online).

Official rules (verified in this repository): native actions forward / turn left / turn right / stop; the
episode ends at the first official robot-human contact (HumanCollision -> task.should_end), at STOP, or at
step 500; success = STOP with geodesic distance to the goal < 0.2 m.  Static contact is not a failure.

Exhaustive breadth-first search over *all* native action sequences from the reset with the exact branch
simulator (the Env.step mirror, bitwise shadow-verified in every replay).  Two sequences are merged only if
they reach a bitwise-identical robot state (base transform, joint positions, gait phase) at the same step --
their futures are identical because the humans follow the recorded tape, which does not depend on the robot
(every authoritative replay so far reproduced the tape bit for bit).  A branch is closed only by an official
human contact (canonical detection, validated against the official measure in every replay).  STOP is a
success iff the geodesic distance is < 0.2 m, so a state inside the success region is a feasibility witness.

If the frontier becomes empty, no action sequence can succeed: the episode is natively infeasible
(``certified_infeasible``).  If the depth or time budget ends first, the result is inconclusive.

  python scripts/querygen_run.py eventwm_graph.querygen.native_cert KEY --suite unseen2 --out DIR
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


def verify_official(spec, sequences):
    """Replay every closed sequence in the official environment (unchanged Env.step) and read the official
    human_collision / success measures: removes the robot-independent-humans assumption for small trees."""
    from .native import NativeTask
    out = dict(sequences=len(sequences), official_human_collision=0, official_success=0, mismatches=[])
    for seq in sequences:
        task = NativeTask(spec)
        try:
            for a in seq:
                task.step(a)
                if task.episode_over:
                    break
            m = task.metrics()
            hit = bool(m.get('human_collision'))
            out['official_human_collision'] += int(hit)
            out['official_success'] += int(m.get('success') == 1.0)
            if not hit or len(seq) != task.steps:
                out['mismatches'].append(dict(seq=''.join(x[0] for x in seq), steps=task.steps, metrics=m))
        finally:
            task.close()
    out['ok'] = out['official_human_collision'] == len(sequences) and not out['mismatches']
    return out


def certify_native(spec, max_depth=500, budget_s=3600., max_frontier=2_000_000, verify_max=200):
    from .native import NativeTask, HumanTape, BranchSimulator, MAX_EPISODE_STEPS, SUCCESS_DISTANCE
    t0 = time.perf_counter()
    task = NativeTask(spec)
    try:
        sim = BranchSimulator(task, None)
        sim.check_static = False
        root = sim.root()
        g0 = task.geodesic_to_goal()
        sim.tape = HumanTape.record(task, min(max_depth, MAX_EPISODE_STEPS)+2)

        def key(state, t):
            h = hashlib.sha1()
            h.update(state.matrix.tobytes()); h.update(state.joints.tobytes())
            if getattr(state, 'target', None) is not None:
                h.update(state.target.tobytes() if hasattr(state.target, 'tobytes') else repr(state.target).encode())
            h.update(str(int(state.gait)).encode()); h.update(str(int(t)).encode())
            return h.digest()
        frontier = [(root, ())]
        closed = []                          # action sequences closed by an official human contact
        sims, closed_by_human, best_geo, depth, witness = 0, 0, g0, 0, None
        sizes = []
        status = 'inconclusive'
        while frontier:
            if depth >= min(max_depth, MAX_EPISODE_STEPS-1):
                status = 'depth_limit'
                break
            if time.perf_counter()-t0 > budget_s:
                status = 'time_limit'
                break
            nxt, seen = [], set()
            for state, seq in frontier:
                for a in ('forward', 'left', 'right'):
                    sim.load(state, depth)
                    r = sim.step(a, compute_geodesic=True)
                    sims += 1
                    if r.human_collision:
                        closed_by_human += 1
                        if len(closed) <= 1000:
                            closed.append(seq+(a,))
                        continue
                    best_geo = min(best_geo, float(r.geodesic))
                    if r.geodesic < SUCCESS_DISTANCE:
                        witness = dict(depth=depth+1, geodesic=float(r.geodesic))
                        break
                    k = key(sim.state, depth+1)
                    if k in seen:
                        continue
                    seen.add(k)
                    nxt.append((sim.state, seq+(a,)))
                if witness is not None:
                    break
            if witness is not None:
                status = 'feasible_state_found'
                break
            depth += 1
            sizes.append(len(nxt))
            frontier = nxt
            if len(frontier) > max_frontier:
                status = 'frontier_limit'
                break
        else:
            status = 'certified_infeasible'
        official = None
        if status == 'certified_infeasible' and len(closed) <= verify_max:
            official = verify_official(spec, closed)
        return dict(key=spec.key, rules='official Falcon (native actions; human contact ends; STOP < 0.2 m)',
                    official_verification=official,
                    status=status, certified_infeasible=status == 'certified_infeasible',
                    depth=depth, frontier_sizes=sizes, sims=sims, closed_by_human_contact=closed_by_human,
                    start_geodesic=float(g0), best_geodesic=float(best_geo), witness=witness,
                    seconds=time.perf_counter()-t0,
                    assumption='humans follow the recorded tape independently of the robot (bitwise-verified in '
                               'every authoritative replay)')
    finally:
        task.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-depth', type=int, default=500)
    ap.add_argument('--budget', type=float, default=3600.)
    ap.add_argument('--max-frontier', type=int, default=2_000_000)
    a = ap.parse_args()
    from .suites import load
    spec = [s for s in load(a.suite) if s.key == a.key][0]
    rec = certify_native(spec, a.max_depth, a.budget, a.max_frontier)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/f"native_{spec.key.replace('/', '_')}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != 'frontier_sizes'}))


if __name__ == '__main__':
    main()
