"""Signed-pose increment model, sharing complete current-map preparation."""
import numpy as np
import torch
from .global_features import GlobalTimePredictor
from .pose_model import PoseNet
from .pose_features import signed_pose_features
from .diagnose import sha


class PosePredictor(GlobalTimePredictor):
    def __init__(self, checkpoint, *, device='cuda:0', threshold=None, increment_format='columns', **kwargs):
        saved = torch.load(checkpoint, map_location='cpu')
        if saved['schema'] != 'stage2_signed_pose_1':
            raise ValueError('unsupported signed-pose checkpoint')
        super().__init__(saved['config']['point_checkpoint'], device=device, threshold=threshold if threshold is not None else saved.get('threshold', .2), increment_format=increment_format, **kwargs)
        self.model = PoseNet().to(device).eval()
        self.model.load_state_dict(saved['model'])
        self.config = saved['config']
        self.checkpoint_sha256 = sha(checkpoint)
        self.predictor_schema = 'stage2_signed_pose_1'
        self.pose_kernel = None
        if self.device.type == 'cuda':
            from .pose_cuda import PoseKernel
            self.pose_kernel = PoseKernel(self.device)

    def _projection(self, prepared, graph, path):
        return path, prepared.resolution

    def _network(self, xs, details):
        result = [np.empty((len(x), 4), np.float32) for x in xs]
        extras = []
        for x, (keys, _, (path, resolution)) in zip(xs, details):
            if self.pose_kernel is not None:
                extras.append(self.pose_kernel(keys, path, resolution, x))
            else:
                extras.append(torch.from_numpy(signed_pose_features(keys, path, resolution, x.cpu().numpy())).to(self.device))
        pieces, used = [], 0
        def flush():
            nonlocal pieces, used
            x = torch.cat([v[3] for v in pieces])
            extra = torch.cat([v[4] for v in pieces])
            p, h = self.model.probabilities(x, extra)
            values = torch.cat([p, h[:, None]], 1).cpu().numpy()
            at = 0
            for qi, start, end, xt, et in pieces:
                result[qi][start:end] = values[at:at+len(xt)]
                at += len(xt)
            pieces, used = [], 0
        for qi, (x, extra) in enumerate(zip(xs, extras)):
            start = 0
            while start < len(x):
                end = min(len(x), start+self.tile_size-used)
                pieces.append((qi, start, end, x[start:end], extra[start:end]))
                used += end-start
                start = end
                if used == self.tile_size:
                    flush()
        if used:
            flush()
        return result
