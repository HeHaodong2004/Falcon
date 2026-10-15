"""Measured-support ablation of the learned acceleration residual bound."""
from dataclasses import replace
from eventforecastwm.model import ForecastWorldModel


class KinematicForecastWorldModel(ForecastWorldModel):
    def __init__(self,variant='kinematic_v1',max_acceleration=4.,**kwargs):
        if variant!='kinematic_v1' or not 0.<max_acceleration<=8.:raise ValueError('Explicit bounded acceleration recipe required')
        super().__init__(**kwargs)
        self.core.config=replace(self.core.config,max_acceleration=max_acceleration)
        self.recipe.update(variant=variant,max_acceleration=max_acceleration)
