from .range_fast_proxy import FastRangePredictor,FastSpatialPredictor
from .range_planar_api import PlanarRangePredictor
from .radix_bvh import rebuild


class RadixRangePredictor(FastRangePredictor):
    def _make_mesh(self,*args):return rebuild(super()._make_mesh(*args))


class RadixExactPredictor(PlanarRangePredictor):
    def _make_mesh(self,*args):return rebuild(super()._make_mesh(*args))


class RadixSpatialPredictor(FastSpatialPredictor):
    def _make_mesh(self,*args):return rebuild(super()._make_mesh(*args))
