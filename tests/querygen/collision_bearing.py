"""For default-policy (executor-only) human collisions: where was the colliding human relative to the robot
(bearing from the heading; inside the 90-degree camera field of view?) over the last steps before contact.

  python scripts/querygen_run.py tests.querygen.collision_bearing RUN_DIR [RUN_DIR ...]
"""
import argparse
import glob
import json
import math

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    from eventwm_graph.querygen.native import EpisodeSpec, NativeTask, HumanTape
    for d in a.runs:
        for f in sorted(glob.glob(d+'/*.json')):
            if f.endswith(('run.json', 'summary.json')):
                continue
            r = json.load(open(f))
            if r.get('branch', {}).get('outcome') != 'human_collision':
                continue
            spec = EpisodeSpec(**r['spec'])
            T = int(r['branch']['steps'])
            task = NativeTask(spec)
            try:
                tape = HumanTape.record(task, T+2)
            finally:
                task.close()
            traj = r['replay']['trace']['trajectory']          # x, z, yaw, geodesic, min rect per step
            x, z, yaw = traj[T][0], traj[T][1], traj[T][2]
            H = np.asarray(tape.positions[T])
            d = np.hypot(H[:, 0]-x, H[:, 2]-z)
            k = int(np.argmin(d))
            rows = []
            for t in (T-20, T-10, T-5, T):
                if t < 0:
                    continue
                hx, hz = tape.positions[t][k][0], tape.positions[t][k][2]
                rx, rz, ryaw = traj[t][0], traj[t][1], traj[t][2]
                bearing = math.degrees(math.atan2(math.sin(math.atan2(-(hz-rz), hx-rx)-ryaw),
                                                  math.cos(math.atan2(-(hz-rz), hx-rx)-ryaw)))
                rows.append(f't-{T-t:2d}: dist {math.hypot(hx-rx, hz-rz):4.2f} m bearing {bearing:5.0f}' +
                            (' (in FOV)' if abs(bearing) <= 45 else ''))
            print(f"{r['planner']:4} {r['key'].split('/')[2]:12} T={T:3d} | " + ' | '.join(rows), flush=True)


if __name__ == '__main__':
    main()
