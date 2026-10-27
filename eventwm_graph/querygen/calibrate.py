"""Dynamics calibration: model vs native simulator, per family.

For every family the robot is placed at an open navmesh point (>= 2 m from
obstacles) and driven through (a) full acceleration, 6 cruise steps and full
braking with the family's own level schedule, (b) 12 in-place turn steps
each way.  Commands go through the same interface the executor uses (native
discrete actions, or the base-velocity action through the unchanged
``update_base`` path).  Reported: per-step translation and yaw measured in the
simulator vs the model's level distances / turn step, stopping distance
measured vs modelled, maximum absolute errors.

  python scripts/querygen_run.py eventwm_graph.querygen.calibrate --out DIR [--key KEY]
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def open_line(task, rng, length=3.6, clearance=0.4, tries=40000):
    """A navigable start point and yaw whose straight forward line of ``length``
    keeps ``clearance`` from navmesh obstacles (no step-filter sliding)."""
    pf = task.sim.pathfinder
    y0 = float(task.robot.base_pos[1])
    for _ in range(tries):
        p = np.asarray(pf.get_random_navigable_point(), dtype=float)
        if not np.isfinite(p).all() or abs(p[1]-y0) > 0.3:
            continue
        yaw = float(rng.uniform(-math.pi, math.pi))
        d = np.array([math.cos(yaw), 0., -math.sin(yaw)])
        ok = True
        for t in np.arange(-0.8, length+1e-9, 0.1):
            q = (p+t*d).astype(np.float32)
            if not pf.is_navigable(q) or pf.distance_to_closest_obstacle(q, 2.0) < clearance:
                ok = False
                break
        if ok:
            return p, yaw
    raise RuntimeError('no open line')


def place(task, sim, p, yaw):
    """Kinematic placement (same construction as feasibility.certify_goal); returns a RobotState."""
    import magnum as mn
    from .native import RobotState
    obj = task.robot.sim_obj
    M0 = mn.Matrix4(obj.transformation)
    base0 = np.asarray(task.robot.base_pos, dtype=float)
    off = np.asarray(M0.translation, dtype=float)-base0
    snap0 = np.asarray(task.sim.pathfinder.snap_point(base0.astype(np.float32)), dtype=float)
    dy = base0[1]-snap0[1]
    R0 = mn.Matrix4.from_(M0.rotation(), mn.Vector3(0, 0, 0))
    from .native import matrix_yaw
    yaw0 = matrix_yaw(np.asarray(M0, dtype=np.float32))       # rotation_y(a) @ R0 has yaw yaw0 + a
    M = mn.Matrix4.translation(mn.Vector3(*(p+[0., dy, 0.]+off))) @ mn.Matrix4.rotation_y(mn.Rad(yaw-yaw0)) @ R0
    obj.transformation = M
    task.robot.update()
    st = task.robot_state()
    return RobotState(st.matrix, None, st.joints, st.gait)


def calibrate(spec, seed=0):
    from .native import NativeTask, BranchSimulator, HumanTape
    from .dynamics import FAMILIES
    from .executor import command_for
    rng = np.random.default_rng(seed)
    out = {}
    for name, dyn in FAMILIES.items():
        task = NativeTask(spec, velocity_action=not dyn.native_actions)
        sim = BranchSimulator(task, None, check_static=False)
        sim.root()
        sim.tape = HumanTape.record(task, 200)
        p, yaw = open_line(task, rng)
        st = place(task, sim, p, yaw)
        sim.load(st, 1)
        top = len(dyn.levels)-1
        # (a) translation: accelerate, cruise, brake (native: 12 forward actions)
        if dyn.stateful:
            sched, l = [], 0
            while l < top:
                l = min(top, l+dyn.accel); sched.append(l)
            sched += [top]*6
            while l > 0:
                l = max(0, l-dyn.brake); sched.append(l)
            prims = [('forward', l) if l > 0 else 'idle' for l in sched]
            model = [dyn.levels[l] for l in sched]
        else:
            prims = ['forward']*12
            model = [dyn.step_m]*12
        poses = [task.robot_pose()]
        for pr in prims:
            r = sim.step(command_for(dyn, pr))
            poses.append(r.pose)
        P = np.asarray(poses)
        meas = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 2]))
        stop_model = dyn.stop_distance(top) if dyn.stateful else 0.
        stop_meas = float(meas[len(prims)-dyn.stop_steps(top):].sum()) if dyn.stateful else 0.
        # (b) rotation both ways
        st2 = place(task, sim, p, yaw)
        turns = {}
        for side in ('left', 'right'):
            sim.load(st2, 1)
            yaws = [task.robot_pose()[3]]
            for _ in range(12):
                r = sim.step(command_for(dyn, side))
                yaws.append(r.pose[3])
            dy = np.angle(np.exp(1j*np.diff(yaws)))
            turns[side] = dict(mean_rad=float(np.mean(np.abs(dy))), max_err_rad=float(np.max(np.abs(np.abs(dy)-dyn.turn_rad))))
        out[name] = dict(model=dyn.describe(), translation_model_m=[round(x, 5) for x in model],
                         translation_measured_m=[round(float(x), 5) for x in meas],
                         translation_max_abs_err_m=float(np.max(np.abs(meas-np.asarray(model)))),
                         stop_distance_model_m=stop_model, stop_distance_measured_m=stop_meas, turns=turns,
                         turn_model_rad=dyn.turn_rad)
        task.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--key', default='hm3d/val/TEEsavR23oF/12')
    a = ap.parse_args()
    from .suites import load
    spec = [s for s in load('dev') if s.key == a.key][0]
    rec = calibrate(spec)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out/'calibration.json').write_text(json.dumps(rec, indent=1, default=float))
    for name, r in rec.items():
        print(f"{name:18s} v_max {r['model']['v_max_mps']:.2f} m/s, turn {r['model']['turn_deg_per_step']:.2f} deg/step: "
              f"translation max |err| {r['translation_max_abs_err_m']:.4f} m; stop model {r['stop_distance_model_m']:.3f} "
              f"measured {r['stop_distance_measured_m']:.3f} m; turn mean L/R {r['turns']['left']['mean_rad']:.4f}/"
              f"{r['turns']['right']['mean_rad']:.4f} rad (model {r['turn_model_rad']:.4f})")


if __name__ == '__main__':
    main()
