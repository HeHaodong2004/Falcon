"""Same legal63LOS evidence, shared geometry constants and shadow rejection."""
from .range_radix_api import RadixExactPredictor,RadixExactSpatialPredictor
from .range_exact_fast import FastExactKernel
from .range_canvas_fast import FastRangeCanvas


class ExactFastRangePredictor(RadixExactPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.range_kernel=FastExactKernel(self.device)


class ExactFastSpatialPredictor(RadixExactSpatialPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.range_kernel=FastExactKernel(self.device);self.canvas=FastRangeCanvas(self.device)
