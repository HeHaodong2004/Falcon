"""Privileged future supervision builder; never imported by model inference."""
import gzip
import json
import math
import numpy as np
import torch


def physical_target(branch, origin, yaw, horizons=(.25,.5,1,2,3)):
    truth = branch['truth']
    start = float(truth[0]['timestamp'])
    bytime = {round(float(f['timestamp'])-start, 8):f for f in truth}
    times = np.asarray(sorted(bytime))
    frames = [bytime[t] for t in times]
    rotation = np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    grids = np.zeros((len(horizons),24,24), dtype=np.uint8)
    mask = np.zeros(len(horizons), dtype=bool)
    for h, clock in enumerate(horizons):
        if clock < times[0]-1e-8 or clock > times[-1]+1e-8:
            continue
        mask[h] = True
        later = min(int(np.searchsorted(times, clock)),len(times)-1)
        earlier = max(0,later-1)
        fraction = 0 if later == earlier else (clock-times[earlier])/(times[later]-times[earlier])
        previous = {a['object_id']:np.asarray(a['position']) for a in frames[earlier]['humans']}
        for a in frames[later]['humans']:
            if a['object_id'] not in previous:
                continue
            p = previous[a['object_id']]+fraction*(np.asarray(a['position'])-previous[a['object_id']])
            x,z = rotation@(p-origin)[[0,2]]
            if -6 <= x < 6 and -6 <= z < 6:
                grids[h,int((z+6)/.5),int((x+6)/.5)] = 1
    return dict(physical_occupancy=torch.from_numpy(grids), physical_mask=torch.from_numpy(mask),
                observed_duration_s=torch.tensor(float(times[-1])))


def root_physical_targets(root, wanted):
    """Stream saved completed branches; no environment, stepping or native job."""
    history = json.loads((root/'legal_observations/history.json').read_text())
    origin, yaw = np.asarray(history[-1]['robot_position']), float(history[-1]['robot_yaw'])
    targets = {}
    with gzip.open(root/'branch_truth/branches.jsonl.gz','rt') as stream:
        for line in stream:
            branch = json.loads(line)
            if branch['branch_id'].startswith('b') and branch['query_id'] in wanted:
                targets[branch['query_id']] = physical_target(branch,origin,yaw)
            if len(targets) == len(wanted):
                break
    if set(targets) != set(wanted):
        raise ValueError('Missing saved physical branches')
    return targets
