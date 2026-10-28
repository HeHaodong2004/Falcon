"""Simulator-side registration of the continuous base-velocity robot action.

Same ``BaseVelAction.update_base`` integration, navmesh step filter and Falcon
leg animation as the discrete actions; only the commanded velocity differs.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .dynamics import VELOCITY_ARG, velocity_moves_legs

_REGISTERED = False


def _register():
    global _REGISTERED
    if _REGISTERED:
        return
    from gym import spaces
    import magnum as mn
    import csv
    from hydra.core.config_store import ConfigStore
    from habitat.core.registry import registry
    from habitat.tasks.rearrange.actions.actions import BaseVelAction
    from habitat.config.default_structured_configs import ActionConfig
    import falcon.additional_action as fa

    @registry.register_task_action
    class QGBaseVelocityAction(BaseVelAction):
        """One control interval at a commanded (linear, angular) base velocity."""

        def __init__(self, *args, config, sim, **kwargs):
            super().__init__(*args, config=config, sim=sim, **kwargs)
            self._leg_data = {}
            use_range = self._config.get('use_range')
            with open(self._config.get('leg_animation_checkpoint'), newline='') as f:
                first, t = True, 0
                for row in csv.reader(f, delimiter=' ', quotechar='|'):
                    if not first:
                        if use_range[0] <= t < use_range[1]:
                            self._leg_data[t-use_range[0]] = [float(v) for v in row[0].split(',')[1:13]]
                        t += 1
                    first = False
            self._play_length_data = len(self._leg_data)
            self._play_i_perframe = self._config.get('play_i_perframe')

        @property
        def action_space(self):
            return spaces.Dict({VELOCITY_ARG: spaces.Box(shape=(2,), low=-np.inf, high=np.inf, dtype=np.float32)})

        def step(self, *args, **kwargs):
            lin, ang = (float(v) for v in kwargs[VELOCITY_ARG])
            self.base_vel_ctrl.linear_velocity = mn.Vector3(lin, 0, 0)
            self.base_vel_ctrl.angular_velocity = mn.Vector3(0, ang, 0)
            if lin != 0. or ang != 0.:
                self.update_base(fix_leg=False)
                if velocity_moves_legs(lin, ang):
                    cur = int(fa.play_i % self._play_length_data)
                    self.cur_articulated_agent.leg_joint_pos = self._leg_data[cur]
                    fa.play_i += self._play_i_perframe

    @dataclass
    class QGBaseVelocityActionConfig(ActionConfig):
        type: str = 'QGBaseVelocityAction'
        lin_speed: float = 1.0
        ang_speed: float = 1.0
        allow_back: bool = False
        allow_dyn_slide: bool = True
        leg_animation_checkpoint: str = 'data/robots/spot_data/spot_walking_trajectory.csv'
        play_i_perframe: int = 5
        use_range: Optional[List[int]] = field(default_factory=lambda: [107, 863])

    ConfigStore.instance().store(package='habitat.task.actions.qg_base_velocity',
                                 group='habitat/task/actions', name='qg_base_velocity',
                                 node=QGBaseVelocityActionConfig)
    globals()['QGBaseVelocityActionConfig'] = QGBaseVelocityActionConfig
    _REGISTERED = True


def velocity_action_config():
    from omegaconf import OmegaConf
    _register()
    return OmegaConf.structured(globals()['QGBaseVelocityActionConfig']())

