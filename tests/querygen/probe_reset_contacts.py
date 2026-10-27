"""Static contact at the benchmark reset pose and after one step, per suite episode."""
import json
import math
import sys
import numpy as np
from eventwm_graph.querygen.native import NativeTask, classify_contacts
from eventwm_graph.querygen.suites import load

suite = sys.argv[1] if len(sys.argv) > 1 else 'dev'
for spec in load(suite):
    rows = {}
    for first in (None, 'left', 'right', 'forward'):
        t = NativeTask(spec)
        names = {lid: t.robot.sim_obj.get_link_name(lid) for lid in t.robot.sim_obj.get_link_ids()}
        if first is not None:
            t.step(first)
        c = classify_contacts(t, gate_humans=False)
        pose = t.robot_pose()
        w = []
        for x in c['static_witnesses'][:3]:
            d = np.asarray(x['point'])-pose[:3]
            cs, sn = math.cos(pose[3]), math.sin(pose[3])
            w.append((names.get(x['robot_link'], x['robot_link']), round(d[0]*cs-d[2]*sn, 2), round(d[0]*sn+d[2]*cs, 2),
                      round(d[1], 2), round(x['distance'], 4)))
        rows[str(first)] = (c['static'], w)
        t.close()
    flag = any(v[0] for v in rows.values())
    if flag:
        print(spec.key, json.dumps({k: [v[0], v[1][:2]] for k, v in rows.items()}))
    else:
        print(spec.key, 'clear')
