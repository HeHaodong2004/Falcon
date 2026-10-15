"""Combine measured directional-map and initial-belief improvements.

This is a controlled small composition of existing components, not a new
spatial recurrent model. The initial correction runs only at history encoding.
"""
from eventforecastwm.model_map_patch import MapPatchForecastWorldModel
from eventforecastwm.model_posterior import attach_posterior,correct_initial


class MapPosteriorForecastWorldModel(MapPatchForecastWorldModel):
    def __init__(self,variant='map_posterior_v1',**kwargs):
        if variant!='map_posterior_v1':raise ValueError(variant)
        super().__init__(**kwargs);attach_posterior(self);self.recipe['variant']=variant

    def encode_history(self,*args,**kwargs):
        return correct_initial(self,super().encode_history(*args,**kwargs))
