"""Same legal-input predictor with an additional partial START render feature."""
from .device_pipeline import ColumnPreparedPipelinePredictor
from .initial_visibility import OptixInitialKernel
from .initial_model import InitialVisibilityNet


class InitialVisibilityPredictor(ColumnPreparedPipelinePredictor):
    model_class=InitialVisibilityNet
    checkpoint_schema='legal_range_initial_visibility_1'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.range_kernel=OptixInitialKernel(self.device,primitive_mode='triangle')
