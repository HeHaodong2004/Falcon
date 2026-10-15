"""SR reference plus a small trajectory-belief interaction correction."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from ..model import EventQueryWM, ModelConfig
from .dynamics import Dynamics
from .history import arrived_tracks
from torch.nn import functional as F


def sample_query(query, samples=16, horizon=None):
    """Interpolate legal nominal tokens, preserving physical clocks and endpoints.

    A short-horizon ablation truncates the interaction only; the original query
    encoder still sees the full query. No executed/future robot states enter.
    """
    out = F.interpolate(query.transpose(1, 2), size=samples, mode='linear', align_corners=True).transpose(1, 2)
    if horizon is not None:
        end = query[:, -1, 2].clamp_max(horizon)
        times = torch.linspace(0, 1, samples, device=query.device)[None]*end[:, None]
        index = times/query[:, -1:, 2].clamp_min(1e-6)*(query.shape[1]-1)
        lower = index.floor().long().clamp_max(query.shape[1]-2)
        upper = lower+1
        weight = (index-lower)[..., None]
        out = query.gather(1, lower[..., None].expand(-1,-1,query.shape[-1]))*(1-weight)+query.gather(1,upper[..., None].expand(-1,-1,query.shape[-1]))*weight
    return out

INTERACTION_FEATURES = ('proximity_mass', 'minimum_clearance', 'relative_x', 'relative_z',
    'relative_vx', 'relative_vz', 'sigma_x', 'sigma_z', 'persistence', 'observation_age',
    'visible', 'velocity_valid', 'track_count', 'robot_speed', 'distance_travelled',
    'clock', 'clock_fraction', 'outside_trained_horizon', 'sin_yaw', 'cos_yaw')


def interaction_features(query, tracks, valid, forecast):
    center, sigma, confidence = (forecast[k] for k in ('center', 'sigma', 'confidence'))
    relative = center-query[:, None, :, :2]
    # Use mean-track chord velocity, including the residual motion if present.
    dt = (query[:, 1:, 2]-query[:, :-1, 2]).clamp_min(1e-6)
    velocity = (center[:, :, 1:]-center[:, :, :-1])/dt[:, None, :, None]
    velocity = torch.cat([velocity, velocity[:, :, -1:]], 2)-query[:, None, :, 3:5]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    nearest = distance.masked_fill(~valid[..., None], float('inf')).argmin(1)
    def closest(x):
        if x.ndim == 3:
            x = x[..., None]
        return x.gather(1, nearest[:, None, :, None].expand(-1, 1, -1, x.shape[-1])).squeeze(1)
    known = valid.any(1)[:, None, None]
    # Clearance footprint radius .25+.30m. Smooth spatial exposure feature,
    # deliberately not interpreted as a calibrated collision probability.
    proximity = (confidence*torch.exp(-.5*(relative.square()/(sigma+.55).square()).sum(-1))).sum(1)
    selected = torch.cat([closest(distance)[..., :1]-.55, closest(relative)/6,
                          closest(velocity)/2, closest(sigma)/6, closest(confidence),
                          closest(tracks[..., 4, None].expand_as(distance))/5,
                          closest(tracks[..., 5, None].expand_as(distance)),
                          closest(tracks[..., 6, None].expand_as(distance))], -1)
    selected = torch.where(known, selected, torch.zeros_like(selected))
    selected[..., 0] = torch.where(known.squeeze(-1), selected[..., 0]/6, torch.ones_like(selected[..., 0]))
    count = valid.sum(1)[:, None, None].expand(-1, query.shape[1], 1)/6
    legal_query = torch.cat([torch.linalg.vector_norm(query[..., 3:5], dim=-1, keepdim=True)/2,
        query[..., 7:8]/6, query[..., 2:3]/6, query[..., 9:10],
        (~forecast['within_trained_horizon'])[..., None].float(), query[..., 5:7]], -1)
    result = torch.cat([proximity[..., None], selected, count, legal_query], -1)
    if result.shape[-1] != len(INTERACTION_FEATURES) or not torch.isfinite(result).all():
        raise ValueError('Invalid dynamics/query interaction features')
    return result


@dataclass(frozen=True)
class V2Config:
    variant: str = 'history'
    sr_size: str = 'M'
    dropout: float = .2
    freeze_sr: bool = False
    freeze_dynamics: bool = True
    outcome: str = 'binary'
    interaction_samples: int = 16
    interaction_horizon: float = None
    freeze_binary: bool = False


class FactorizedEventWM(nn.Module):
    def __init__(self, config=V2Config()):
        super().__init__()
        if config.variant not in ('sr', 'cv', 'residual', 'history', 'joint'):
            raise ValueError(config.variant)
        self.config = config
        if config.outcome not in ('binary', 'direct', 'hierarchical') or config.interaction_samples < 2:
            raise ValueError('Invalid outcome/interaction configuration')
        if config.freeze_binary and config.outcome != 'hierarchical':
            raise ValueError('Binary preservation requires the hierarchical outcome head')
        if config.freeze_binary and not config.freeze_dynamics:
            raise ValueError('Binary preservation also requires frozen dynamics')
        self.backbone = EventQueryWM(ModelConfig.family(config.sr_size, config.dropout))
        # Matched SR-only affine control; identity at initialization.
        self.sr_affine = nn.Parameter(torch.zeros(2))
        self.dynamics = Dynamics('history' if config.variant == 'joint' else
                                 'cv' if config.variant == 'sr' else config.variant)
        with torch.random.fork_rng(devices=[]):
            # The same correction initialization in every paired variant, even
            # though different dynamics constructors consume different RNG draws.
            torch.manual_seed(torch.initial_seed()+33)
            self.correction = nn.Sequential(nn.Linear(2*len(INTERACTION_FEATURES), 64), nn.GELU(),
                                            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)
        for p in self.backbone.parameters():
            p.requires_grad_(not (config.freeze_sr or config.freeze_binary))
        for p in self.dynamics.parameters():
            p.requires_grad_(not config.freeze_dynamics)
        if config.variant == 'sr':
            for p in self.correction.parameters():
                p.requires_grad_(False)
        if config.outcome != 'binary':
            width = 2*self.backbone.config.width+(0 if config.variant == 'sr' else 2*len(INTERACTION_FEATURES))
            self.outcome_head = nn.Sequential(nn.Linear(width,64), nn.GELU(),
                                             nn.Linear(64,3 if config.outcome == 'direct' else 1))
            nn.init.zeros_(self.outcome_head[-1].weight)
            nn.init.zeros_(self.outcome_head[-1].bias)
            self.register_buffer('social_prior_logit',torch.tensor(0.))
        if config.freeze_binary:
            self.sr_affine.requires_grad_(False)
            for p in self.correction.parameters():p.requires_grad_(False)

    @torch.no_grad()
    def initialize_outcome_prior(self, class_prior):
        if self.config.outcome == 'binary':return
        prior = torch.as_tensor(class_prior,device=self.sr_affine.device).clamp_min(1e-7)
        logit = torch.log(prior[1]/prior[0])
        self.social_prior_logit.copy_(logit)
        if self.config.outcome == 'hierarchical':self.outcome_head[-1].bias.fill_(logit)

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_sr or self.config.freeze_binary:
            self.backbone.eval()
        if self.config.freeze_binary:self.correction.eval()
        if self.config.freeze_dynamics:
            self.dynamics.eval()
        return self

    def event_outputs(self, latent, pooled):
        """Event heads on legal encoder outputs; also used for frozen-head training.

        These tensors must be produced by the frozen encoders, never targets.
        Keeping this path shared makes cached optimization and inference identical.
        """
        sr = self.backbone.head(latent).squeeze(-1)*self.sr_affine[0].exp()+self.sr_affine[1]
        if self.config.variant != 'sr':
            sr = sr+self.correction(pooled).squeeze(-1)
        result = dict(sr_logits=sr, p_safe=sr.sigmoid(), p_no_physical_collision=sr.sigmoid())
        if self.config.outcome != 'binary':
            context = latent if self.config.variant == 'sr' else torch.cat([latent,pooled],-1)
            logits = self.outcome_head(context)
            if self.config.outcome == 'direct':
                base = torch.stack([sr+F.logsigmoid(-self.social_prior_logit),
                                    sr+F.logsigmoid(self.social_prior_logit),torch.zeros_like(sr)],-1)
                log_p = (base+logits).log_softmax(-1)
            else:
                social = logits.squeeze(-1)
                log_p = torch.stack([F.logsigmoid(sr)+F.logsigmoid(-social),
                                     F.logsigmoid(sr)+F.logsigmoid(social),F.logsigmoid(-sr)],-1)
                result.update(social_logits=social, p_social_given_noncollision=social.sigmoid())
            p = log_p.exp()
            result.update(outcome_log_probs=log_p, outcome_probabilities=p,
                          p_safe=p[:,0], p_social_violation_success=p[:,1], p_human_collision=p[:,2],
                          p_no_physical_collision=1-p[:,2], p_binary_collision=1-sr.sigmoid())
        else:
            result['p_human_collision'] = 1-sr.sigmoid()
        return result

    def forward(self, value, legal_motion=None, with_occupancy=False):
        tracks, valid, sequence, sequence_mask = arrived_tracks(value) if legal_motion is None else legal_motion
        latent = self.backbone.latent_from_scene(value,*self.backbone.encode_scene(value))
        query = sample_query(value.query,self.config.interaction_samples,self.config.interaction_horizon)
        forecast = self.dynamics.forecast(tracks, valid, query[..., 2], sequence, sequence_mask)
        features = interaction_features(query, tracks, valid, forecast)
        pooled = torch.cat([features.mean(1),features.max(1).values],-1)
        result = self.event_outputs(latent, pooled)
        result.update(interaction=features, interaction_query=query, successor_prediction=forecast)
        if with_occupancy:
            result['physical_occupancy'] = self.dynamics(tracks, valid, sequence, sequence_mask)
        return result

    @torch.no_grad()
    def predict(self, value):
        self.eval()
        result = self(value, with_occupancy=True)
        output = dict(probability_of_safe_execution=result['p_no_physical_collision'],
            p_human_collision=result['p_human_collision'],
            predictive_successor_state=dict(physical_occupancy=result['physical_occupancy'],
                horizons_s=self.dynamics.horizons, frame='fixed_decision_time_robot_xz',
                human_center_mean=result['successor_prediction']['center'],
                human_center_sigma=result['successor_prediction']['sigma'],
                persistence=result['successor_prediction']['confidence'],
                query_clocks_s=result['interaction_query'][..., 2],
                forecast_supported=result['successor_prediction']['within_trained_horizon']))
        if self.config.outcome != 'binary':
            output.update(p_safe=result['p_safe'],p_social_violation_success=result['p_social_violation_success'],
                          outcome_probabilities=result['outcome_probabilities'],p_binary_collision=result['p_binary_collision'])
        return output

    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
            trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
            components={k:sum(p.numel() for p in m.parameters()) for k, m in self.named_children()},
            config=asdict(self.config))
