"""Opt-in controlled experiments; upstream task/simulator defaults stay intact."""
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from habitat.tasks.rearrange.social_nav_v2.socialnav_task import MultiAgentPointNavTask
from falcon.rearrange_sim_v2 import RearrangeSim_v2
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy


def policy_observations(observation_space):
    """Recording cameras must never add RGB channels to the depth checkpoint."""
    return spaces.Dict(OrderedDict(
        (key, value) for key, value in observation_space.spaces.items()
        if not key.endswith(('articulated_agent_jaw_rgb', 'third_rgb'))
    ))


@baseline_registry.register_policy
class ControlledPointNavResNetPolicy(PointNavResNetPolicy):
    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        # The upstream policy filters display sensors via extra_sim_sensors.
        # Our runner installs them directly on Spot, so do the same filtering
        # here, before invoking the unchanged native network constructor.
        return super().from_config(config, policy_observations(observation_space),
                                   action_space, **kwargs)


@registry.register_simulator(name="ControlledRearrangeSim-v0")
class ControlledRearrangeSim(RearrangeSim_v2):
    def _load_navmesh(self, ep_info):
        # Experiment-owned meshes avoid overwriting the user's upstream cache.
        path = ep_info.info["controlled_navmesh"]
        if not self.pathfinder.load_nav_mesh(path):
            raise RuntimeError(f"Cannot load controlled navmesh: {path}")
        self._largest_indoor_island_idx = get_largest_island_index(
            self.pathfinder, self, allow_outdoor=False
        )


@registry.register_task(name="ControlledPointNavTask-v0")
class ControlledPointNavTask(MultiAgentPointNavTask):
    def reset(self, episode, fetch_observations=True):
        count = episode.info.get("human_num")
        expected = self._config.controlled_human_count
        if type(count) is not int or not 0 <= count <= 6 or count != expected:
            raise ValueError(f"Requested {expected} humans, episode declares {count}")
        if self._sim.num_articulated_agents != 7 or not self._use_episode_start_goal:
            raise ValueError("Controlled FALCON requires Spot + six human slots and episode starts")
        for h in range(count):
            for k in range(3):
                point = np.asarray(episode.info[f"human_{h}_waypoint_{k}_position"])
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError(f"Invalid waypoint for human {h}, waypoint {k}")
        # The upstream task retains these attributes between resets. Explicitly
        # clear unused slots so changing counts in a reused environment is safe.
        for i in range(count + 1, 7):
            setattr(self, f"agent{i}_episode_start_position", None)
            action = self.actions[f"agent_{i}_oracle_nav_randcoord_action_obstacle"]
            action.goals = [np.zeros(3, dtype=np.float32) for _ in range(action.num_goals)]
        observations = super().reset(episode, fetch_observations)
        positions = []
        for i in range(1, 7):
            actual = np.asarray(self._sim.get_agent_data(i).articulated_agent.base_pos)
            target = (episode.info[f"human_{i-1}_waypoint_0_position"]
                      if i <= count else [-100 + i] * 3)
            if not np.allclose(actual, target, atol=1e-3, rtol=0):
                raise RuntimeError(f"Human slot {i} was placed at {actual}, expected {target}")
            action = self.actions[f"agent_{i}_oracle_nav_randcoord_action_obstacle"]
            if action.human_num != count:
                raise RuntimeError(f"Human controller {i} has incorrect crowd count")
            if i <= count:
                goals = [episode.info[f"human_{i-1}_waypoint_{k}_position"] for k in (1, 2)]
                if not np.allclose(action.goals, goals):
                    raise RuntimeError(f"Human controller {i} has incorrect waypoint goals")
            positions.append(dict(slot=i, active=i <= count, position=actual.tolist()))
        if self._human_num != count:
            raise RuntimeError("Task crowd count differs from requested count")
        actual_active = sum(not np.allclose(p['position'], [-100 + p['slot']] * 3,
                                           atol=1e-3, rtol=0) for p in positions)
        if actual_active != expected:
            raise RuntimeError(f"Requested {expected} humans; observed {actual_active}")
        audit = dict(episode_id=episode.episode_id, scene_id=episode.scene_id,
                     requested_humans=expected, actual_active_humans=actual_active,
                     allocated_human_slots=6, positions=positions,
                     navmesh=episode.info["controlled_navmesh"], verified=True)
        path = Path(self._config.controlled_audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(audit, indent=2) + "\n")
        temporary.replace(path)
        print(f"CONTROLLED_CROWD requested={expected} actual_active={actual_active} "
              f"allocated_slots=6 episode={episode.episode_id}", flush=True)
        return observations
