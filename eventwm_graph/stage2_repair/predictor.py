"""Public checkpoint dispatcher for complete hypothetical Stage 2 updates."""
import torch


def make_predictor(checkpoint, *, global_features=False, **kwargs):
    schema = torch.load(checkpoint, map_location='cpu')['schema']
    if schema == 'stage2_hf_time_1':
        if global_features:
            from .global_features import GlobalTimePredictor as Predictor
        else:
            from .api import TimePredictor as Predictor
    elif schema == 'stage2_signed_pose_1':
        from .pose_api import PosePredictor as Predictor
    elif schema == 'stage2_shared_map_1':
        from .shared_api import SharedPredictor as Predictor
    else:
        raise ValueError('unsupported Stage 2 checkpoint schema: '+str(schema))
    return Predictor(checkpoint, **kwargs)
