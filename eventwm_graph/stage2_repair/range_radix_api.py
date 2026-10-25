from .range_fast_proxy import FastRangePredictor,FastSpatialPredictor
from .range_planar_api import PlanarRangePredictor
from .range_planar import PlanarRangeMesh
from .range_map_reuse import ReusedProxyKernel,ReusedExactKernel
from .range_spatial_api import SpatialRangePredictor


class DirectRadix:
    def _make_mesh(self,ranges,origins,rotations,timestamps,now,resolution,tracks):
        return PlanarRangeMesh(ranges,origins,rotations,timestamps,now,resolution,self.device,tracks=tracks,bvh='radix')


class RadixRangePredictor(DirectRadix,FastRangePredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.range_kernel=ReusedProxyKernel(self.device)


class RadixExactPredictor(DirectRadix,PlanarRangePredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.range_kernel=ReusedExactKernel(self.device)


class RadixSpatialPredictor(DirectRadix,FastSpatialPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.range_kernel=ReusedProxyKernel(self.device)


class RadixExactSpatialPredictor(DirectRadix,SpatialRangePredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.range_kernel=ReusedExactKernel(self.device)
