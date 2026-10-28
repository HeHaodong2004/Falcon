"""Native equivalence: real Env.step vs exogenous-tape robot branching.

For random robot action sequences on real benchmark episodes, checks that
  1. human trajectories under real Env.step equal the robot-free tape bitwise,
  2. branch robot poses, official human-collision and static-contact flags
     equal the real run,
  3. legal observations (camera pose, tracks, map certificates) are identical,
  4. restoring any recorded mid-episode branch state reproduces the same
     continuation bitwise (exact setter-input restoration),
  5. the human contact gate never skips a step with an exact human contact.
Run: python scripts/querygen_run.py tests.querygen.native_equivalence
"""
import json
import sys
import time
from types import SimpleNamespace
import numpy as np
from eventwm_graph.querygen.native import (EpisodeSpec, NativeTask, HumanTape, BranchSimulator,
                                           classify_contacts, CONTACT_GATE_M)

DEFAULT = [('hm3d', 'val', '3t8DB4Uzvkt', '1', 11), ('hm3d', 'val', '5cdEh9F2hJL', '3', 12),
           ('mp3d', 'val', '2azQ1b91cZZ', '0', 13)]
RES = 0.1016884229464652


def random_actions(rng, n, collide_bias=False):
    out = []
    while len(out) < n:
        kind = rng.choice(['forward', 'left', 'right'], p=[.6, .2, .2])
        out.extend([kind]*int(rng.integers(1, 8)))
    return out[:n]


def chase_actions(tape, start_pose, n, human=0):
    """Privileged pursuit of one human (forces real robot-human contacts)."""
    import math
    pose = np.asarray(start_pose, dtype=float).copy(); out = []
    for k in range(n):
        target = tape.positions[min(k+1, len(tape)-1), human]
        d = target[[0, 2]]-pose[[0, 2]]
        err = math.atan2(math.sin(math.atan2(-d[1], d[0])-pose[3]), math.cos(math.atan2(-d[1], d[0])-pose[3]))
        if abs(err) > 0.09:
            a = 'left' if err > 0 else 'right'; pose[3] += 1/12 if err > 0 else -1/12
        else:
            a = 'forward'; pose[0] += .2*math.cos(pose[3]); pose[2] -= .2*math.sin(pose[3])
        out.append(a)
    return out


def make_observer(task):
    from scripts.collector.observation import ObservationConfig
    from eventwm_graph.partial.fast_observer import FastGeometryObserver
    sensor = task.config.habitat.simulator.agents.agent_0
    # The headless config strips sim_sensors; rebuild the jaw camera contract
    # from the unmodified benchmark config.
    from habitat_baselines.config.default import get_config
    from eventwm_graph.querygen.native import CONFIGS
    jaw = get_config(CONFIGS[task.spec.dataset]).habitat.simulator.agents.agent_0.sim_sensors.jaw_depth_sensor
    rt = SimpleNamespace(sim=task.sim, observation_config=ObservationConfig.from_depth_sensor(jaw),
                         count=task.human_count,
                         episode=SimpleNamespace(scene_id=task.episode.scene_id, info={'graph_navmesh': 'native'}))
    return FastGeometryObserver(rt, RES, private_cache={})


def canon(row):
    ups = sorted((tuple(u['cell']), u['state'], round(u['height'], 9), u['source']) for u in row['static_updates'])
    tracks = [(t['track_id'], tuple(t['position']), t['visible_now'], t['velocity_valid']) for t in row['tracks']]
    return (tuple(np.round(row['camera_origin'], 9)), tuple(np.round(np.asarray(row['camera_rotation']).ravel(), 9)),
            tuple(ups), tuple(tracks))


def real_run(spec, actions, observe=True):
    with NativeTask(spec) as t:
        obs = make_observer(t) if observe else None
        rows = []
        first = canon(obs.observe(t.world_time())) if observe else None
        for a in actions:
            if t.episode_over:
                break
            t.step(a)
            m = t.metrics()
            c = classify_contacts(t, gate_humans=False, need_static=True)
            rows.append(dict(pose=t.robot_pose(), humans=t.human_positions(), time=t.world_time(),
                             human_collision=bool(m['human_collision']), static=c['static'],
                             exact_human=c['human'], rect=c['min_rect_distance'],
                             obs=canon(obs.observe(t.world_time())) if observe else None))
        return rows, t.metrics(), first


def main():
    specs = [EpisodeSpec(*row) for row in DEFAULT]
    report = []
    for spec in specs:
        rng = np.random.default_rng(spec.seed)
        task = NativeTask(spec)
        sim = BranchSimulator(task, None)
        root = sim.root()
        root_pose = task.robot_pose()
        obs_root = make_observer(task)
        first_branch = canon(obs_root.observe(task.world_time()))
        t0 = time.perf_counter()
        sim.tape = HumanTape.record(task, 300)
        tape_s = time.perf_counter()-t0
        for trial in range(4):
            actions = random_actions(rng, 260) if trial < 3 else chase_actions(
                sim.tape, root_pose, 260)
            t0 = time.perf_counter()
            real, metrics, first_real = real_run(spec, actions, observe=(trial == 0))
            real_s = time.perf_counter()-t0
            sim.load(root, 0)
            obs = make_observer(task) if trial == 0 else None
            t0 = time.perf_counter()
            branch, states, observations = [], [], []
            for a in actions[:len(real)]:
                r = sim.step(a)
                branch.append(r); states.append(sim.state)
                if obs is not None:
                    observations.append(canon(obs.observe(r.time)))
                if r.human_collision:
                    break
            branch_s = time.perf_counter()-t0
            n = min(len(real), len(branch))
            pose_exact = all(np.array_equal(real[k]['pose'], branch[k].pose) for k in range(n))
            human_exact = all(np.array_equal(real[k]['humans'], sim.tape.positions[k+1]) for k in range(n))
            time_exact = all(real[k]['time'] == sim.tape.times[k+1] for k in range(n))
            coll_equal = all(real[k]['human_collision'] == branch[k].human_collision for k in range(n))
            static_equal = [k for k in range(n) if real[k]['static'] != branch[k].static_contact]
            gate_ok = all(not (row['exact_human'] and row['rect'] > CONTACT_GATE_M) for row in real)
            obs_equal = None
            if observations:
                obs_equal = bool(first_real == first_branch and
                                 all(real[k]['obs'] == observations[k] for k in range(n)))
            # Exact restoration: reload states and continue; compare to main run.
            restore_bad = 0
            for i0 in range(5, n-1, max(1, n//12)):
                sim.load(states[i0], i0+1)
                ok = True
                for k in range(i0+1, min(n, i0+25)):
                    r = sim.step(actions[k])
                    ok &= bool(np.array_equal(r.pose, branch[k].pose) and r.human_collision == branch[k].human_collision
                               and r.static_contact == branch[k].static_contact)
                restore_bad += int(not ok)
            row = dict(episode=spec.key, trial=trial, steps_real=len(real), steps_branch=len(branch),
                       pose_exact=pose_exact, human_exact=human_exact, time_exact=time_exact,
                       collision_equal=coll_equal, static_mismatch_steps=static_equal, gate_ok=gate_ok,
                       observation_equal=obs_equal, restore_mismatches=restore_bad,
                       real_collision=any(r['human_collision'] for r in real),
                       real_static_steps=int(sum(bool(r['static']) for r in real)),
                       min_rect=round(min(r['rect'] for r in real), 3) if real else None,
                       real_s=round(real_s, 2), branch_s=round(branch_s, 2), tape_s=round(tape_s, 2),
                       metrics={k: v for k, v in metrics.items() if isinstance(v, float)})
            print(json.dumps(row), flush=True)
            report.append(row)
        task.close()
    ok = all(r['pose_exact'] and r['human_exact'] and r['collision_equal'] and not r['static_mismatch_steps']
             and r['gate_ok'] and r['time_exact'] and r['restore_mismatches'] == 0
             and r['observation_equal'] in (True, None) for r in report)
    print('ALL_EQUIVALENT' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
