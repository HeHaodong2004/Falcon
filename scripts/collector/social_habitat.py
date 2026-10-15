"""Read-only privileged PSC evidence, sampled at native recorder timestamps."""
import numpy as np
import habitat_sim
from collector.v3 import SOCIAL_METRIC


def capture_social_metric(sim, count):
    robot = sim.get_agent_state(0).position
    records = []
    for i in range(1, count+1):
        human = sim.get_agent_state(i).position
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = robot, human
        found = sim.pathfinder.find_path(path)
        # Exact same operations as FALCON PersonalSpaceCompliance(use_geo_distance=True).
        distance = sim.geodesic_distance(robot, human) if found else np.linalg.norm(human-robot, ord=2)
        records.append(dict(human_slot=i, distance=float(distance),
                            mode='geodesic' if found else 'euclidean_no_path'))
    return dict(version=SOCIAL_METRIC, distances=records)


def make_social_recorder(env, count):
    from collector.runtime import Recorder

    class SocialRecorder(Recorder):
        def capture(self, phase, requested_dt=0.):
            row = super().capture(phase, requested_dt)
            row['social_metric'] = capture_social_metric(self.sim, self.count)
            return row

    return SocialRecorder(env, count)
