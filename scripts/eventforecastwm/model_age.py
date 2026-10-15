"""A minimal observation-age-conditioned initial scene belief.

The four learned noise scales and all recurrent dynamics remain unchanged.
Initial position/velocity uncertainty propagates over the legally known time
since the last observation. No privileged state or human count enters this rule.
"""
import torch
from eventforecastwm.model import ForecastWorldModel


class AgeAwareForecastWorldModel(ForecastWorldModel):
    def __init__(self,variant='age_aware_v1',**kwargs):
        if variant!='age_aware_v1':raise ValueError(variant)
        kwargs.setdefault('noise','integrated')
        super().__init__(**kwargs)
        if self.noise!='integrated':raise ValueError('Age-aware variant requires coherent residual covariance')
        self.recipe['variant']=variant

    def initial_covariance_for(self,core):
        age=core.age[:,None].expand(core.positions.shape[:3]).clamp_min(0.)
        covariance=core.positions.new_zeros(*age.shape,2,2)
        position=self.log_initial_position.exp().clamp(.01,3.).square()
        velocity=self.log_initial_velocity.exp().clamp(.01,3.).square()
        spectral=self.log_acceleration.exp().clamp(.01,5.).square()*.25
        # Continuous white-acceleration approximation only over the observed
        # fractional acquisition gap; future transitions retain the canonical
        # discrete acceleration distribution and exact composition semantics.
        covariance[...,0,0]=position+age.square()*velocity+spectral*age.pow(3)/3.
        covariance[...,0,1]=covariance[...,1,0]=age*velocity+spectral*age.square()/2.
        covariance[...,1,1]=velocity+spectral*age
        return covariance

    def encode_history(self,*args,**kwargs):
        belief=super().encode_history(*args,**kwargs)
        belief.covariance=self.initial_covariance_for(belief.core)
        return belief
