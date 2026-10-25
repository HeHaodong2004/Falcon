"""Measured 7-ray visibility FEATURE approximation; certificate targets stay63.

All63frustum probes are retained. Only the learned visibility covariate uses
three center heights and four mid-height halo rays. Reconstruction effect must
be measured before selecting this variant. No candidate/output is removed.
"""
import torch
from .cuda_module import CudaModule
from .range_fast import RangeFeatureKernel,SOURCE
from .range_planar_api import PlanarRangePredictor
from .range_spatial_api import SpatialRangePredictor
from .camera_vectorized import parameters


class ProxyKernel(RangeFeatureKernel):
    camera_parameters=staticmethod(parameters)
    def __init__(self,device):
        self.device=torch.device(device);old='for(int j=0;j<63;j++){\n  int at=j%21;double ax=0.,az=0.;';new='for(int probe=0;probe<7;probe++){\n  int j=probe<3?probe*21:26+(probe-3)*4;int at=j%21;double ax=0.,az=0.;'
        if SOURCE.count(old)!=1:raise ValueError('visibility approximation source contract changed')
        self.module=CudaModule(SOURCE.replace(old,new),device)


class FastRangePredictor(PlanarRangePredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.range_kernel=ProxyKernel(self.device)

    def predict_segments(self,**kwargs):
        out=super().predict_segments(**kwargs)
        for d in out:d['provenance']['visibility_feature_approximation']='7LOS rays; all63frustum probes retained; original certificate targets and original candidate cells unchanged.'
        return out


class FastSpatialPredictor(SpatialRangePredictor,PlanarRangePredictor):
    """Spatial wrapper sets legal history; planar wrapper builds measured mesh."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.range_kernel=ProxyKernel(self.device)

    def predict_segments(self,**kwargs):
        out=super().predict_segments(**kwargs)
        for d in out:d['provenance']['visibility_feature_approximation']='7LOS rays; all63frustum probes retained; original certificate targets and original candidate cells unchanged.'
        return out
