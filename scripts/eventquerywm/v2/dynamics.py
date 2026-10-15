"""Small CV anchored forecasts, retaining prior checkpoint parameter names."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from ..cv_residual import CVResidual


class Dynamics(CVResidual):
    def __init__(self, kind='cv', background=None):
        super().__init__(torch.zeros(5, 24, 24) if background is None else background)
        if kind not in ('cv', 'residual', 'history'):
            raise ValueError(kind)
        self.kind = kind
        if kind == 'cv':
            self.net = nn.Identity()
        elif kind == 'history':
            self.history_gru = nn.GRU(10, 16, batch_first=True)
            self.net = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 48),
                                     nn.GELU(), nn.Linear(48, 3))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def encode_history(self, sequence, mask):
        b, n, s, f = sequence.shape
        flat = torch.where(mask[..., None], sequence, torch.zeros_like(sequence)).reshape(b*n, s, f)
        lengths = mask.sum(-1).reshape(-1)
        packed = pack_padded_sequence(flat, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.history_gru(packed)
        return hidden[0].reshape(b, n, 16)*(lengths > 0).reshape(b, n, 1)

    def forecast(self, tracks, valid, times, sequence=None, sequence_mask=None):
        """Query-free physical forecast at requested clocks [B,K] or [K].

        CV extrapolation remains explicit outside trained [0.25,3]s. Neural
        residual/uncertainty features clamp at 3s; future pose is still propagated
        by CV. This is an unvalidated extrapolation, never a safety certificate.
        """
        b, n, _ = tracks.shape
        if times.ndim == 1:
            times = times[None].expand(b, -1)
        if times.shape[0] != b or not torch.isfinite(times).all() or (times < 0).any():
            raise ValueError('Forecast times must be finite nonnegative clocks')
        tracks = torch.where(valid[..., None], tracks, torch.zeros_like(tracks))
        age = tracks[..., 4, None]+times[:, None]
        center = tracks[:, :, None, :2]+tracks[:, :, None, 2:4]*age[..., None]
        scale = .1+nn.functional.softplus(self.log_scale)+nn.functional.softplus(self.log_growth)*age
        sigma = scale[..., None].expand(-1, -1, -1, 2)
        confidence = torch.sigmoid(self.confidence_logit).expand(b, n, times.shape[1])
        if self.kind != 'cv':
            features = torch.cat([tracks[..., :2]/6, tracks[..., 2:4]/2,
                                  tracks[..., 4:5]/5, tracks[..., 5:7]], -1)
            features = features[:, :, None].expand(-1, -1, times.shape[1], -1)
            features = torch.cat([features, times.clamp_max(3)[:, None, :, None].expand(b, n, -1, 1)/3], -1)
            if self.kind == 'history':
                history = self.encode_history(sequence, sequence_mask)
                features = torch.cat([features, history[:, :, None].expand(-1, -1, times.shape[1], -1)], -1)
            correction = self.net(features)
            if self.kind == 'residual':
                center = center+2*torch.tanh(correction[..., :2])
                correction = correction[..., 2:]
            sigma = (sigma*torch.exp(1.5*torch.tanh(correction[..., :2]))).clamp_min(.1)
            confidence = torch.sigmoid(self.confidence_logit+6*torch.tanh(correction[..., 2]))
        return dict(center=center, sigma=sigma, confidence=confidence*valid[..., None],
                    within_trained_horizon=(times >= .25) & (times <= 3))

    def rasterize(self, forecast):
        """Human-center Gaussian cell masses; off-grid mass is not renormalized.

        A cross-track noisy-OR is an approximation for union occupancy. It does
        not assume independence of collision events at adjacent times.
        """
        center, sigma, confidence = (forecast[k] for k in ('center', 'sigma', 'confidence'))
        scaled = (self.edges[None, None, None, None]-center[..., None])/(sigma[..., None]*math.sqrt(2))
        cdf = .5*(1+torch.erf(scaled))
        mass = (cdf[..., 1:]-cdf[..., :-1]).clamp_min(0)
        per_track = mass[..., 1, :, None]*mass[..., 0, None, :]*confidence[..., None, None]
        survival = torch.log1p(-per_track.clamp_max(1-1e-6)).sum(1)+torch.log1p(-self.background)[None]
        return (-torch.expm1(survival)).clamp(1e-7, 1-1e-6)

    def forward(self, tracks, valid, sequence=None, sequence_mask=None):
        return self.rasterize(self.forecast(tracks, valid, self.horizons, sequence, sequence_mask))

    @torch.no_grad()
    def point_cv_occupancy(self, tracks, valid):
        """Deterministic center-cell CV reference, no learned residual/uncertainty."""
        age = tracks[...,4,None]+self.horizons
        positions = tracks[:,:,None,:2]+tracks[:,:,None,2:4]*age[...,None]
        cells = torch.floor((positions+6)/.5).long()
        output = tracks.new_full((len(tracks),5,24,24),1e-6)
        available = valid[...,None] & (cells >= 0).all(-1) & (cells < 24).all(-1)
        for b,n,h in available.nonzero().tolist():
            x,z = cells[b,n,h].tolist()
            output[b,h,z,x] = 1-1e-6
        return output

    def load_reference(self, state):
        """Load verified prior CV/residual/uncertainty weights, strict for used keys."""
        selected = {k: v for k, v in state.items() if k in self.state_dict()}
        self.load_state_dict(selected, strict=True)
