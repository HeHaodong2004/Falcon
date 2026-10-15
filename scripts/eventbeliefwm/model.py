"""Small recurrent physical belief; sealed future truth is never imported here.

Coordinates remain in the original decision frame throughout composition. Query
clocks are segment-relative, with positions in that fixed frame. Categorical
scene scenarios persist across transitions; no fresh noise or history re-encoding
occurs in ``imagine``. The six scenarios are a finite stochastic approximation,
not a claim of calibrated uncertainty. Missing geometry/visibility/anonymous
supervision is represented by explicit unsupported flags.
"""
from dataclasses import dataclass, fields, replace, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BeliefConfig:
    kind: str = 'stochastic'
    width: int = 128
    modes: int = 6
    heads: int = 4
    interaction_layers: int = 2
    dt: float = .25
    max_acceleration: float = 2.
    max_speed: float = 3.
    trained_horizon_s: float = 3.
    grid_size: int = 32
    grid_extent_m: float = 8.
    human_radius_m: float = .30
    motion_window_s: float = .75

    def __post_init__(self):
        if self.kind not in ('cv', 'deterministic', 'stochastic'):
            raise ValueError(self.kind)
        if self.width % self.heads or self.dt <= 0 or self.modes < 1:
            raise ValueError('Invalid dimensions or physical timestep')


@dataclass
class PreparedHistory:
    """Reusable parameter-free legal features, never a learned history embedding.

    Pair this object with the exact same legal roots and padding as the raw input.
    Only arrived history enters preparation; query/future targets are not read.
    It is safe to reuse across optimizers, seeds and model kinds with the same
    motion window. Learned GRU/attention states are recomputed on every encode.
    """
    clean: torch.Tensor                  # [B,N,T,13], normalized legal features
    positions: torch.Tensor              # [B,N,2], last actual observation
    velocities: torch.Tensor             # [B,N,2], causal timestamp regression
    valid: torch.Tensor                  # [B,N]
    last_observed_times: torch.Tensor    # [B,N], float64 relative native clocks
    velocity_valid: torch.Tensor         # [B,N]
    lengths: torch.Tensor                # [B]
    motion_window_s: float

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device)
                               for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)})

    def select(self, indices, *, n=None, t=None):
        values = {f.name: getattr(self, f.name)[indices]
                  for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)}
        n = values['clean'].shape[1] if n is None else max(1, n)
        t = values['clean'].shape[2] if t is None else t
        if (values['lengths'] > t).any() or values['valid'][:, n:].any():
            raise ValueError('Prepared history cropping may remove only padding')
        values['clean'] = values['clean'][:, :n, :t]
        for name in ('positions', 'velocities', 'valid', 'last_observed_times', 'velocity_valid'):
            values[name] = values[name][:, :n]
        return replace(self, **values)


@dataclass
class Belief:
    hidden: torch.Tensor                 # [B,S,N,D]
    positions: torch.Tensor              # [B,S,N,2], physical even after collision
    velocities: torch.Tensor             # [B,S,N,2]
    valid: torch.Tensor                  # [B,N], positively observed identities only
    age: torch.Tensor                    # [B,N], since actual acquisition
    last_observed_positions: torch.Tensor # [B,N,2]
    last_observed_times: torch.Tensor     # [B,N], fixed-root relative clocks
    velocity_valid: torch.Tensor         # [B,N]
    scenario_weights: torch.Tensor       # [B,S], physical mixture
    survival_mass: torch.Tensor          # [B,S], unnormalised live event mass
    collision_mass: torch.Tensor         # [B], absorbing event mass
    clock: torch.Tensor                  # [B], physical clock since root
    ego_position: torch.Tensor           # [B,2]
    ego_yaw: torch.Tensor                # [B]
    geometry: torch.Tensor               # [B,2,32,32], clearance/support adapter
    geometry_valid: torch.Tensor         # [B]
    sensor: torch.Tensor                 # [B,2], horizontal FOV radians/range
    sensor_valid: torch.Tensor           # [B]
    robot_radius: torch.Tensor           # [B]
    track_ids: torch.Tensor              # [B,N], bookkeeping only

    def detach(self):
        return replace(self, **{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def to(self, device):
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)})


@dataclass
class Forecast:
    next_belief: Belief
    human_positions: torch.Tensor        # [B,S,T,N,2], shared across candidates
    human_velocities: torch.Tensor
    scenario_weights: torch.Tensor
    position_scale: torch.Tensor          # [B,S,T,N,2], explicit Gaussian mixture readout
    robot_positions: torch.Tensor        # [B,K,T,2]
    observation_probability: torch.Tensor # [B,K,S,T,N], FOV/range approximation
    collision_interval: torch.Tensor    # [B,K,T], unconditional new event mass
    cumulative_collision: torch.Tensor  # [B,K,T]
    physical_occupancy: torch.Tensor      # [B,T,H,W], tracked-center cell mass
    physical_flow: torch.Tensor          # [B,T,H,W,2], conditional center flow
    anonymous_occupancy: torch.Tensor     # [B,H,W], separate prior, uncalibrated
    uncertainty: torch.Tensor            # [B,T,N], scenario positional std
    validity: dict
    readout: dict
    batch_shape: tuple
    times: torch.Tensor


class BeliefWorldModel(nn.Module):
    def __init__(self, config=BeliefConfig()):
        super().__init__()
        self.config = config
        d = config.width
        self.scenarios = config.modes if config.kind == 'stochastic' else 1
        self.history_gru = nn.GRU(13, d, batch_first=True)
        self.scene_token = nn.Parameter(torch.zeros(1, 1, d))
        self.interaction = nn.ModuleList([nn.TransformerEncoderLayer(
            d, config.heads, 2*d, dropout=0., batch_first=True,
            activation='gelu', norm_first=True) for _ in range(config.interaction_layers)])
        self.mode_embedding = nn.Embedding(self.scenarios, 8)
        self.state_input = nn.Linear(2+2+1+1+2+8, d)
        self.transition = nn.GRUCell(d, d)
        self.acceleration = nn.Sequential(nn.Linear(d, d//2), nn.GELU(), nn.Linear(d//2, 2))
        nn.init.zeros_(self.acceleration[-1].weight)
        nn.init.zeros_(self.acceleration[-1].bias)
        self.mode_logits = nn.Linear(d, self.scenarios)
        nn.init.zeros_(self.mode_logits.weight)
        nn.init.zeros_(self.mode_logits.bias)
        self.correction = nn.GRUCell(6, d)
        self.anonymous_head = nn.Sequential(nn.Linear(d+2, 32), nn.GELU(),
                                            nn.Linear(32, config.grid_size**2))
        nn.init.zeros_(self.anonymous_head[-1].weight)
        nn.init.constant_(self.anonymous_head[-1].bias, -7.)
        self.log_contact_temperature = nn.Parameter(torch.tensor(math.log(.08)))
        self.contact_offset = nn.Parameter(torch.tensor(0.))
        self.detection_logit = nn.Parameter(torch.tensor(2.))
        self.log_position_scale = nn.Parameter(torch.tensor(math.log(.20)))
        self.register_buffer('anonymous_trained', torch.tensor(False))
        self.register_buffer('event_calibrated', torch.tensor(False))
        self.register_buffer('observation_calibrated', torch.tensor(False))
        angle = torch.arange(self.scenarios)*2*math.pi/self.scenarios
        directions = torch.stack([angle.cos(), angle.sin()], -1)
        if self.scenarios == 1:
            directions.zero_()
        self.register_buffer('mode_directions', directions)
        edge = torch.linspace(-config.grid_extent_m, config.grid_extent_m,
                              config.grid_size+1)
        self.register_buffer('grid_edges', edge)
        if config.kind == 'cv':
            for p in self.parameters():
                p.requires_grad_(False)

    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    config=asdict(self.config))

    def _interact(self, hidden, valid):
        b, s, n, d = hidden.shape
        scene = self.scene_token.expand(b*s, 1, d)
        tokens = torch.cat([scene, hidden.reshape(b*s, n, d)], 1)
        padding = torch.cat([torch.zeros(b*s, 1, dtype=torch.bool, device=hidden.device),
                             (~valid[:, None]).expand(b, s, n).reshape(b*s, n)], 1)
        for layer in self.interaction:
            tokens = layer(tokens, src_key_padding_mask=padding)
        return tokens[:, 1:].reshape(b, s, n, d), tokens[:, 0].reshape(b, s, d)

    @staticmethod
    def _config_rows(value, b):
        if value is None:
            return [{} for _ in range(b)]
        if isinstance(value, (tuple, list)):
            if len(value) != b:
                raise ValueError('Configuration batch mismatch')
            return [asdict(x) if hasattr(x, '__dataclass_fields__') else x for x in value]
        value = asdict(value) if hasattr(value, '__dataclass_fields__') else value
        return [value for _ in range(b)]

    @torch.no_grad()
    def prepare_history(self, legal_history, *, history_timestamps=None, track_acquisition_times=None):
        """Parameter-free arrived H13/E12 processing, reusable across training steps.

        The supplied velocity fields are replaced by timestamped finite differences.
        A track missing from the final tracker table remains a lost belief if it
        appeared earlier. No privileged global human count is consumed.
        Optional paired float64 absolute legal clock sidecars preserve native
        acquisition times. Subtract the decision anchor before conversion to
        learned features; absolute-time origins cannot alter motion estimates.
        """
        h = legal_history.humans
        if h.ndim != 4 or h.shape[-1] != 13:
            raise ValueError('Expected batched legal H[B,N,T,13]')
        b, n, t, _ = h.shape
        e = legal_history.robot
        if e.shape != (b, t, 12):
            raise ValueError('Expected legal E[B,T,12]')
        device = h.device
        if n == 0:
            h = h.new_zeros(b, 1, t, 13)
            padding = torch.ones(b, 1, dtype=torch.bool, device=device)
            n = 1
            if track_acquisition_times is not None and track_acquisition_times.shape == (b, 0, t):
                track_acquisition_times = torch.zeros(b, 1, t, dtype=torch.float64, device=device)
        else:
            padding = legal_history.human_padding
        lengths = (~legal_history.history_padding).sum(1)
        if (lengths == 0).any():
            raise ValueError('An arrived history is required')
        if (history_timestamps is None) != (track_acquisition_times is None):
            raise ValueError('Exact history and acquisition clocks must be supplied together')
        exact_clocks = history_timestamps is not None
        if exact_clocks:
            if history_timestamps.shape != (b, t) or track_acquisition_times.shape != (b, n, t):
                raise ValueError('Exact legal clock sidecars have incompatible shapes')
            if history_timestamps.dtype != torch.float64 or track_acquisition_times.dtype != torch.float64:
                raise ValueError('Exact legal clock sidecars must preserve float64 native timestamps')
            history_timestamps = history_timestamps.to(device)
            track_acquisition_times = track_acquisition_times.to(device)
            anchor = history_timestamps.gather(1, (lengths-1)[:, None]).squeeze(1)
            snapshot_history = history_timestamps-anchor[:, None]
            acquisition_history = track_acquisition_times-anchor[:, None, None]
        else:
            snapshot_history = e[..., 7].double()
            acquisition_history = h[..., 8].double()-h[..., 7].double()
        positive_history = ((h[..., 6] > .5) & (h[..., 10] > .5) &
                            ~padding[..., None] & ~legal_history.history_padding[:, None])
        acquisition_history = torch.where(positive_history, acquisition_history,
                                         torch.zeros_like(acquisition_history))
        for i in range(b):
            count = int(lengths[i])
            if legal_history.history_padding[i, :count].any() or not legal_history.history_padding[i, count:].all():
                raise ValueError('History must be right padded')
            clocks = e[i, :count, 7]
            if not torch.isfinite(clocks).all() or (clocks[1:] <= clocks[:-1]).any() or clocks[-1].abs() > 1e-5:
                raise ValueError('History clocks must increase and end at root time zero')
            native = snapshot_history[i, :count]
            if not torch.isfinite(native).all() or (native[1:] <= native[:-1]).any():
                raise ValueError('Exact legal history clocks must be finite and increase')
            if not torch.allclose(native, clocks.double(), atol=1e-4, rtol=1e-6):
                raise ValueError('Exact and tensorized legal history clocks disagree')
        clean = torch.zeros_like(h)
        pos = h.new_zeros(b, n, 2)
        velocity = torch.zeros_like(pos)
        seen = torch.zeros(b, n, dtype=torch.bool, device=device)
        last_time = torch.zeros(b, n, dtype=torch.float64, device=device)
        vv = seen.clone()
        position_history = torch.nan_to_num(h[..., :2])
        fresh_history = torch.zeros(b, n, t, dtype=torch.bool, device=device)
        segment_start = torch.full((b, n), -float('inf'), dtype=torch.float64, device=device)
        for j in range(t):
            row = h[:, :, j]
            ok = (~padding & ~legal_history.history_padding[:, j, None] &
                  (row[..., 6] > .5) & (row[..., 10] > .5))
            if not torch.isfinite(row[ok]).all():
                raise ValueError('Nonfinite legal observed track')
            if (ok & ((row[..., 8]-e[:, j, 7, None]).abs() > 1e-5)).any():
                raise ValueError('Human snapshot and ego history clocks disagree')
            acquired = acquisition_history[:, :, j]
            snapshot = snapshot_history[:, j, None]
            if (ok & ~torch.isfinite(acquired)).any():
                raise ValueError('Nonfinite exact acquisition clock')
            if exact_clocks and (ok & ((acquired-(row[..., 8].double()-row[..., 7].double())).abs() >
                                      1e-4+1e-6*acquired.abs())).any():
                raise ValueError('Exact and tensorized acquisition clocks disagree')
            if (ok & ((row[..., 7] < 0) | (acquired > snapshot+1e-5))).any():
                raise ValueError('An observation cannot arrive from the future')
            if (ok & seen & (acquired < last_time-1e-5)).any():
                raise ValueError('Acquisition clocks moved backward')
            fresh = ok & (~seen | (acquired > last_time+1e-5))
            gap = acquired-last_time
            segment_start = torch.where(fresh & (~seen | (gap > .50001)), acquired, segment_start)
            fresh_history[:, :, j] = fresh
            # Native articulated positions can update less often than sensor
            # frames. Causal trailing regression avoids interpreting stutter as
            # zero/impulsive speed, and is shared by CV and learned baselines.
            candidate_time = acquisition_history[:, :, :j+1]
            weights = (fresh_history[:, :, :j+1] &
                       (candidate_time >= acquired[..., None]-self.config.motion_window_s) &
                       (candidate_time >= segment_start[..., None])).to(acquisition_history.dtype)
            safe_times = torch.nan_to_num(candidate_time)
            count = weights.sum(-1).clamp_min(1)
            mean_time = (safe_times*weights).sum(-1)/count
            mean_position = (position_history[:, :, :j+1]*weights[..., None]).sum(-2)/count[..., None]
            centered_time = safe_times-mean_time[..., None]
            denominator = (centered_time.square()*weights).sum(-1)
            fd = ((position_history[:, :, :j+1]-mean_position[..., None, :])*
                  (centered_time*weights)[..., None]).sum(-2)/denominator.clamp_min(1e-8)[..., None]
            fd = fd.to(h.dtype)
            earliest = safe_times.masked_fill(weights == 0, float('inf')).amin(-1)
            fd_ok = fresh & (count >= 2) & (acquired-earliest >= .09999) & (denominator > 1e-8)
            velocity = torch.where(fresh[..., None],
                                   torch.where(fd_ok[..., None], fd, torch.zeros_like(fd)), velocity)
            vv = torch.where(fresh, fd_ok, vv)
            pos = torch.where(fresh[..., None], row[..., :2], pos)
            last_time = torch.where(fresh, acquired, last_time)
            seen |= fresh
            safe = torch.where(ok[..., None], row, torch.zeros_like(row)).clone()
            safe[..., 2:4] = torch.where(ok[..., None], velocity, torch.zeros_like(velocity))
            safe[..., 5] = vv & ok
            safe[..., 7] = torch.where(ok, snapshot-acquired, torch.zeros_like(acquired)).to(h.dtype)
            safe[..., 8] = torch.where(ok, snapshot.expand_as(acquired), torch.zeros_like(acquired)).to(h.dtype)
            exact_dt = snapshot_history[:, j]-snapshot_history[:, j-1] if j else snapshot_history[:, j]*0
            safe[..., 9] = torch.where(ok, exact_dt[:, None].expand_as(acquired), torch.zeros_like(acquired)).to(h.dtype)
            clean[:, :, j] = safe
        clean[..., :4] /= clean.new_tensor([6., 6., 2., 2.])
        clean[..., 7:10] /= 5.
        return PreparedHistory(clean, pos, velocity, seen, last_time, vv, lengths,
                               self.config.motion_window_s)

    def encode_history(self, legal_history, static_map=None, sensor_config=None,
                       robot_geometry=None, *, track_ids=None, history_timestamps=None,
                       track_acquisition_times=None, prepared_history=None):
        """Encode arrived observations with current learned GRU/attention weights.

        ``prepared_history`` skips only immutable legal timestamp regression and
        sanitization. It must be paired with the same raw legal roots; it never
        caches the learned encoding. Without it, preparation runs as before.
        """
        prepared = prepared_history
        if prepared is None:
            prepared = self.prepare_history(legal_history, history_timestamps=history_timestamps,
                                             track_acquisition_times=track_acquisition_times)
        if not isinstance(prepared, PreparedHistory) or prepared.motion_window_s != self.config.motion_window_s:
            raise ValueError('Prepared legal history/configuration mismatch')
        clean = prepared.clean
        b, n, t, _ = clean.shape
        raw_shape = legal_history.humans.shape
        if raw_shape != (b, n, t, 13) and not (raw_shape == (b, 0, t, 13) and n == 1):
            raise ValueError('Prepared legal history and raw padding shapes disagree')
        if clean.device != legal_history.humans.device or clean.dtype != legal_history.humans.dtype:
            raise ValueError('Prepared legal history device/dtype mismatch')
        h, device = clean, clean.device
        pos, velocity, seen = prepared.positions, prepared.velocities, prepared.valid
        last_time, vv, lengths = prepared.last_observed_times, prepared.velocity_valid, prepared.lengths
        encoded, _ = self.history_gru(clean.reshape(b*n, t, 13))
        idx = (lengths-1).repeat_interleave(n)
        hidden = encoded[torch.arange(b*n, device=device), idx].reshape(b, 1, n, -1)
        hidden, pooled = self._interact(hidden, seen)
        weights = self.mode_logits(pooled[:, 0]).softmax(-1)
        age = (-last_time).to(h.dtype)
        initial = pos+velocity*age[..., None]
        # A persistent joint scene mode, not independent noise at each future time.
        spread = (.025+.10*age).clamp(max=1.)
        positions = initial[:, None]+self.mode_directions[None, :, None]*spread[:, None, :, None]
        velocities = velocity[:, None].expand(-1, self.scenarios, -1, -1).clone()
        if self.config.kind == 'stochastic':
            velocities = velocities+self.mode_directions[None, :, None]*(.04+.03*age[:, None, :, None])
        hidden = hidden.expand(-1, self.scenarios, -1, -1).contiguous()
        geometry = h.new_zeros(b, 2, 32, 32)
        geometry_valid = torch.zeros(b, dtype=torch.bool, device=device)
        if static_map is not None:
            if isinstance(static_map, dict):
                geometry = static_map['features'].to(device)
                support = static_map.get('valid', static_map.get('support', False))
                geometry_valid = torch.as_tensor(support, device=device, dtype=torch.bool)
                if geometry_valid.ndim > 1:
                    geometry_valid = geometry_valid.flatten(1).all(1)
                geometry_valid = geometry_valid.expand(b)
            else:
                raise ValueError('Static map requires explicit features and validity')
            if geometry.shape != (b, 2, 32, 32):
                raise ValueError('Expected static map [B,2,32,32]')
        sensor = h.new_zeros(b, 2)
        sensor_valid = torch.zeros(b, dtype=torch.bool, device=device)
        for i, c in enumerate(self._config_rows(sensor_config, b)):
            if 'horizontal_fov_deg' in c and 'max_range_m' in c:
                sensor[i] = sensor.new_tensor([math.radians(c['horizontal_fov_deg']), c['max_range_m']])
                sensor_valid[i] = True
            else:
                sensor[i] = sensor.new_tensor([math.pi/2, 6.])
        radius = h.new_tensor([float(c.get('radius', c.get('radius_m', .25)))
                              for c in self._config_rows(robot_geometry, b)])
        if track_ids is None:
            track_ids = torch.arange(n, device=device)[None].expand(b, -1).clone()
        elif track_ids.shape != (b, n):
            raise ValueError('Track IDs must match padded legal identity slots')
        return Belief(hidden, positions, velocities, seen, age, pos, last_time, vv,
            weights, weights.clone(), h.new_zeros(b), h.new_zeros(b), h.new_zeros(b, 2),
            h.new_zeros(b), geometry, geometry_valid, sensor, sensor_valid, radius,
            track_ids.to(device))

    def _geometry_at(self, belief, positions):
        b, s, n, _ = positions.shape
        grid = positions.reshape(b, s*n, 1, 2)/self.config.grid_extent_m
        value = F.grid_sample(belief.geometry, grid, align_corners=False,
                              padding_mode='zeros').squeeze(-1).transpose(1, 2)
        value = value.reshape(b, s, n, 2)
        return value*belief.geometry_valid[:, None, None, None]

    def _step(self, belief):
        if self.config.kind == 'cv':
            return replace(belief, positions=belief.positions+self.config.dt*belief.velocities,
                           age=belief.age+self.config.dt, clock=belief.clock+self.config.dt)
        b, s, n, d = belief.hidden.shape
        interacted, _ = self._interact(belief.hidden, belief.valid)
        modes = self.mode_embedding.weight[None, :, None].expand(b, s, n, -1)
        features = torch.cat([belief.positions/6, belief.velocities/2,
                              belief.age[:, None, :, None].expand(b, s, n, 1)/5,
                              belief.velocity_valid[:, None, :, None].expand(b, s, n, 1),
                              self._geometry_at(belief, belief.positions), modes], -1)
        hidden = self.transition(self.state_input(features).reshape(-1, d),
                                 interacted.reshape(-1, d)).reshape(b, s, n, d)
        acceleration = self.config.max_acceleration*torch.tanh(self.acceleration(hidden))
        velocity = belief.velocities+self.config.dt*acceleration
        speed = velocity.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        velocity = velocity*(self.config.max_speed/speed).clamp_max(1.)
        position = belief.positions+.5*self.config.dt*(belief.velocities+velocity)
        return replace(belief, hidden=hidden, positions=position, velocities=velocity,
                       age=belief.age+self.config.dt, clock=belief.clock+self.config.dt)

    @staticmethod
    def _safe_atan2(y, x):
        # atan2(0,0) has undefined backward even if its downstream probability
        # is multiplied by an invalid-track mask. Pick an arbitrary zero bearing
        # with zero derivative for undefined vectors before calling atan2.
        undefined = x.square()+y.square() < 1e-12
        return torch.atan2(torch.where(undefined, torch.zeros_like(y), y),
                           torch.where(undefined, torch.ones_like(x), x))

    @staticmethod
    def _sample_query(query, times):
        b, k, l, f = query.shape
        flat = query.reshape(b*k, l, f)
        clocks = flat[..., 2].contiguous()
        request = times[None].expand(b*k, -1).contiguous()
        hi = torch.searchsorted(clocks, request).clamp(1, l-1)
        lo = hi-1
        a = flat.gather(1, lo[..., None].expand(-1, -1, f))
        z = flat.gather(1, hi[..., None].expand(-1, -1, f))
        fraction = ((request-a[..., 2])/(z[..., 2]-a[..., 2])).clamp(0, 1)
        return (a+fraction[..., None]*(z-a)).reshape(b, k, len(times), f)

    def _raster(self, positions, velocities, weights, valid):
        # Soft center-cell mass; no unobserved-free label and no foreground/background accuracy shortcut.
        b, s, t, n, _ = positions.shape
        sigma = self.log_position_scale.exp().clamp(.03, 3.)
        scaled = (self.grid_edges[None, None, None, None, None]-positions[..., None])/(sigma*math.sqrt(2))
        cdf = .5*(1+torch.erf(scaled))
        mass = (cdf[..., 1:]-cdf[..., :-1]).clamp_min(0)
        per = mass[..., 1, :, None]*mass[..., 0, None, :]*valid[:, None, None, :, None, None]
        occupancy = 1-(1-per).prod(3)
        occupancy = (occupancy*weights[:, :, None, None, None]).sum(1)
        weighted = per*weights[:, :, None, None, None, None]
        total = weighted.sum((1, 3))
        flow = (weighted[..., None]*velocities[..., None, None, :]).sum((1, 3))/total[..., None].clamp_min(1e-8)
        return occupancy, flow

    def anonymous_prior(self, belief):
        mask = belief.valid[:, None, :, None]
        pooled = (belief.hidden*mask).sum(2)/mask.sum(2).clamp_min(1)
        pooled = (pooled*belief.scenario_weights[..., None]).sum(1)
        geom = belief.geometry.mean((-1, -2))*belief.geometry_valid[:, None]
        return self.anonymous_head(torch.cat([pooled, geom], -1)).reshape(
            len(pooled), self.config.grid_size, self.config.grid_size).sigmoid()

    def imagine(self, belief, candidate_query, horizon_s=3., *, rasterize=True,
                allow_query_hold=False, detach_risk_states=False):
        """Reusable .25s transition. Multi-candidate successor has flattened B*K.

        Candidate Q[...,10] must cover the requested horizon and begin at the
        current ego position in the unchanged root frame. Physical rollouts are
        unconditioned by robot queries and remain alive after event absorption.
        Training may detach risk states so unassociated native contacts calibrate
        contact parameters without pulling a known track toward an unknown
        collision culprit. Observation likelihood remains physically coupled.
        Calibrating against all native contacts does not make tracked-only risk
        complete when anonymous humans and static risk are unsupported.
        """
        query = candidate_query[:, None] if candidate_query.ndim == 3 else candidate_query
        if query.ndim != 4 or query.shape[-1] != 10 or query.shape[0] != len(belief.clock):
            raise ValueError('Expected query [B,K,L,10] or [B,L,10]')
        if not torch.isfinite(query).all() or (query[..., 1:, 2] <= query[..., :-1, 2]).any():
            raise ValueError('Query clocks must be finite and strictly increasing')
        if query[..., 0, 2].abs().max() > 1e-5:
            raise ValueError('Segment query clocks must begin at zero')
        if not torch.allclose(query[:, :, 0, :2], belief.ego_position[:, None].expand(-1, query.shape[1], -1), atol=1e-4, rtol=0):
            raise ValueError('Query start must use current ego in fixed root frame')
        steps = round(horizon_s/self.config.dt)
        if steps < 1 or abs(steps*self.config.dt-horizon_s) > 1e-6:
            raise ValueError('Composition boundaries must lie on canonical .25s ticks')
        if not allow_query_hold and (query[..., -1, 2]+1e-5 < horizon_s).any():
            raise ValueError('Query does not cover requested physical horizon')
        times = torch.arange(steps+1, device=query.device, dtype=query.dtype)*self.config.dt
        robot = self._sample_query(query, times)
        states, velocities = [belief.positions], []
        current = belief
        for _ in range(steps):
            current = self._step(current)
            states.append(current.positions)
            velocities.append(current.velocities)
        all_positions = torch.stack(states, 2)
        positions = all_positions[:, :, 1:]
        velocity = torch.stack(velocities, 2)
        b, s, t, n, _ = positions.shape
        k = query.shape[1]
        query_time_valid = times[None, None, 1:] <= query[..., -1, 2, None]+1e-5
        risk_positions = all_positions.detach() if detach_risk_states else all_positions
        relative = risk_positions[:, None]-robot[:, :, None, :, None, :2]
        a, z = relative[:, :, :, :-1], relative[:, :, :, 1:]
        delta = z-a
        fraction = (-a.mul(delta).sum(-1)/delta.square().sum(-1).clamp_min(1e-10)).clamp(0, 1)
        distance = (a+fraction[..., None]*delta).norm(dim=-1)
        clearance = distance-belief.robot_radius[:, None, None, None, None]-self.config.human_radius_m
        temperature = self.log_contact_temperature.exp().clamp(.02, .5)
        contact = torch.sigmoid((self.contact_offset.clamp(-.3, .3)-clearance)/temperature)
        contact = contact*belief.valid[:, None, None, None]
        hazard = 1-(1-contact.clamp_max(1-1e-6)).prod(-1)
        hazard = hazard*query_time_valid[:, :, None]
        survive = torch.cumprod(1-hazard, -1)
        prior_mass = belief.survival_mass.detach() if detach_risk_states else belief.survival_mass
        prior_collision = belief.collision_mass.detach() if detach_risk_states else belief.collision_mass
        mass = prior_mass[:, None, :, None]*survive
        cumulative = 1-mass.sum(2)
        previous = torch.cat([prior_collision[:, None, None].expand(b, k, 1), cumulative[..., :-1]], -1)
        interval = (cumulative-previous).clamp_min(0)
        rel = all_positions[:, None, :, 1:]-robot[:, :, None, 1:, None, :2]
        yaw = self._safe_atan2(robot[:, :, 1:, 5], robot[:, :, 1:, 6])
        bearing = self._safe_atan2(-rel[..., 1], rel[..., 0])-yaw[:, :, None, :, None]
        angle = self._safe_atan2(bearing.sin(), bearing.cos()).abs()
        in_fov = torch.sigmoid((belief.sensor[:, None, None, None, None, 0]/2-angle)/.05)
        in_range = torch.sigmoid((belief.sensor[:, None, None, None, None, 1]-rel.norm(dim=-1))/.15)
        observation = in_fov*in_range*torch.sigmoid(self.detection_logit)*belief.valid[:, None, None, None]
        observation = observation*query_time_valid[:, :, None, :, None]
        # Neither expected detection nor missed expectation updates physical belief.
        def expand(tensor):
            return tensor[:, None].expand(b, k, *tensor.shape[1:]).reshape(b*k, *tensor.shape[1:])
        successor = replace(current, **{f.name: expand(getattr(current, f.name)) for f in fields(current)})
        successor = replace(successor, survival_mass=mass[..., -1].reshape(b*k, s),
            collision_mass=cumulative[..., -1].reshape(b*k),
            ego_position=robot[:, :, -1, :2].reshape(b*k, 2), ego_yaw=yaw[:, :, -1].reshape(b*k))
        if rasterize:
            occupancy, flow = self._raster(positions, velocity, belief.scenario_weights, belief.valid)
        else:
            occupancy = positions.new_empty(b, t, 0, 0)
            flow = positions.new_empty(b, t, 0, 0, 2)
        mean = (positions*belief.scenario_weights[:, :, None, None, None]).sum(1)
        variance = ((positions-mean[:, None]).square().sum(-1)*belief.scenario_weights[:, :, None, None]).sum(1)
        speed = query[..., 3:5].norm(dim=-1)
        support = (speed[..., None]-speed.new_tensor([0., .35, .70])).abs().min(-1).values < .035
        validity = dict(tracked_physical=belief.valid, anonymous_occupancy=self.anonymous_trained.expand(b),
            query_time=query_time_valid,
            static_geometry=belief.geometry_valid, static_occlusion=torch.zeros_like(belief.geometry_valid),
            observation_sensor=belief.sensor_valid, observation_calibrated=self.observation_calibrated.expand(b),
            collision_calibrated=self.event_calibrated.expand(b), track_existence_calibrated=torch.zeros_like(belief.valid),
            complete_collision_risk=torch.zeros(b, k, dtype=torch.bool, device=query.device),
            within_trained_horizon=(belief.clock[:, None]+times[None, 1:] <= self.config.trained_horizon_s+1e-6),
            speed_supported=support.all(-1), execution_residual_supported=torch.zeros(b, k, dtype=torch.bool, device=query.device))
        minimum = clearance.masked_fill(~belief.valid[:, None, None, None], float('inf')).amin((2, 3, 4))
        readout = dict(tracked_collision_probability=cumulative[..., -1],
            minimum_tracked_clearance=minimum,
            survival_conditioned_weights=mass[..., -1]/mass[..., -1].sum(-1, keepdim=True).clamp_min(1e-12),
            unknown_human_risk_supported=torch.zeros(b, k, dtype=torch.bool, device=query.device),
            nominal_execution_only=torch.ones(b, k, dtype=torch.bool, device=query.device))
        return Forecast(successor, positions, velocity, belief.scenario_weights,
                        self.log_position_scale.exp().clamp(.03, 3.).expand_as(positions),
                        robot[:, :, 1:, :2], observation, interval, cumulative, occupancy, flow,
                        self.anonymous_prior(belief), (variance+1e-12).sqrt(), validity, readout, (b, k), times[1:])

    def observe(self, belief, actual_arrived_observation):
        """Correct with actual acquired positions in the fixed root frame only.

        Existing slots may be reordered or new legal tracker IDs appended by the
        caller. IDs are bookkeeping, never learned numeric input. Missed tracks
        persist; absence is not certified physical emptiness. Measurement velocity
        is derived only from two actual observations and a <=.5s acquisition gap.
        """
        obs = actual_arrived_observation
        if obs.get('source', 'actual_arrived') != 'actual_arrived':
            raise ValueError('Expected actual arrived observation, never an imagined detection')
        position = obs['positions'].to(dtype=belief.positions.dtype, device=belief.positions.device)
        observed = obs['observed'].bool()
        acquired = obs['acquisition_times'].to(dtype=torch.float64, device=belief.clock.device)
        timestamp = torch.as_tensor(obs['timestamp_s'], device=belief.clock.device).expand_as(belief.clock)
        if not torch.allclose(timestamp, belief.clock, atol=1e-5, rtol=0):
            raise ValueError('Correct at current physical time; explicitly propagate before observation')
        if position.shape[:2] != observed.shape or acquired.shape != observed.shape:
            raise ValueError('Observation shape mismatch')
        if (observed & (acquired > timestamp[:, None]+1e-5)).any() or not torch.isfinite(position[observed]).all():
            raise ValueError('Nonfinite or future observation')
        b, m = observed.shape
        ids = obs.get('track_ids')
        if ids is None:
            if m != belief.valid.shape[1]:
                raise ValueError('Track birth requires explicit legal track IDs')
            ids = belief.track_ids
        # Expand identity table without discarding old/lost identities.
        tables = []
        for i in range(b):
            old = belief.track_ids[i].tolist()
            active_ids = ids[i, observed[i]].tolist()
            if len(set(active_ids)) != len(active_ids):
                raise ValueError('Duplicate observed identity')
            tables.append(old+[x for x in active_ids if x not in old])
        new_n = max(map(len, tables))
        old_n = belief.valid.shape[1]
        track_fields = {'hidden': 2, 'positions': 2, 'velocities': 2, 'valid': 1, 'age': 1,
                        'last_observed_positions': 1, 'last_observed_times': 1,
                        'velocity_valid': 1, 'track_ids': 1}
        updates = {}
        for name, axis in track_fields.items():
            value = getattr(belief, name)
            shape = list(value.shape); shape[axis] = new_n
            padded = value.new_zeros(shape)
            slices = [slice(None)]*value.ndim; slices[axis] = slice(0, old_n)
            padded[tuple(slices)] = value
            updates[name] = padded
        current = replace(belief, **updates)
        current.track_ids = torch.full((b, new_n), -1, device=ids.device, dtype=ids.dtype)
        for i, table in enumerate(tables):
            current.track_ids[i, :len(table)] = torch.tensor(table, device=ids.device)
        measurement = position.new_zeros(b, new_n, 2)
        clock = acquired.new_zeros(b, new_n)
        mask = torch.zeros(b, new_n, dtype=torch.bool, device=observed.device)
        for i, table in enumerate(tables):
            for j in observed[i].nonzero().flatten().tolist():
                slot = table.index(int(ids[i, j]))
                measurement[i, slot] = position[i, j]
                clock[i, slot] = acquired[i, j]
                mask[i, slot] = True
        if (mask & current.valid & (clock < current.last_observed_times-1e-5)).any():
            raise ValueError('Observation acquisition clock moved backwards')
        fresh = mask & (~current.valid | (clock > current.last_observed_times+1e-5))
        gap = clock-current.last_observed_times
        velocity_valid = fresh & current.valid & (gap > 1e-6) & (gap <= .50001)
        velocity = (measurement-current.last_observed_positions)/gap.clamp_min(1e-6)[..., None]
        velocity = torch.where(velocity_valid[..., None], velocity, torch.zeros_like(velocity)).to(current.positions.dtype)
        age = (timestamp[:, None]-clock).to(current.positions.dtype)
        # A long reacquisition gap does not justify a fresh velocity measurement.
        # Preserve the predictive velocity in that case, instead of inventing a
        # stop; truly new identities have zero-initialised predictive velocity.
        posterior_velocity = torch.where(velocity_valid[:, None, :, None], velocity[:, None], current.velocities)
        posterior_position = measurement[:, None]+posterior_velocity*age[:, None, :, None]
        # Likelihood uses the predicted state at acquisition time (local linear back-propagation).
        predicted = current.positions-current.velocities*age[:, None, :, None]
        likelihood = -.5*((predicted-measurement[:, None])/.20).square().sum(-1)
        likelihood = (likelihood*fresh[:, None]).sum(-1)
        weights = (current.scenario_weights.clamp_min(1e-12).log()+likelihood).softmax(-1)
        live_fraction = (current.survival_mass/current.scenario_weights.clamp_min(1e-12)).clamp(0, 1)
        live_total = 1-current.collision_mass
        survival = weights*live_fraction
        survival = survival/survival.sum(-1, keepdim=True).clamp_min(1e-12)*live_total[:, None]
        mean_velocity = (posterior_velocity*weights[:, :, None, None]).sum(1)
        features = torch.cat([measurement/6, mean_velocity/2, age[..., None]/5, velocity_valid[..., None]], -1)
        expanded = features[:, None].expand(-1, self.scenarios, -1, -1)
        corrected_hidden = self.correction(expanded.reshape(-1, 6), current.hidden.reshape(-1, self.config.width)).reshape_as(current.hidden)
        return replace(current,
            positions=torch.where(fresh[:, None, :, None], posterior_position, current.positions),
            velocities=torch.where(fresh[:, None, :, None], posterior_velocity, current.velocities),
            hidden=torch.where(fresh[:, None, :, None], corrected_hidden, current.hidden),
            valid=current.valid | fresh, age=torch.where(fresh, age, current.age),
            last_observed_positions=torch.where(fresh[..., None], measurement, current.last_observed_positions),
            last_observed_times=torch.where(fresh, clock, current.last_observed_times),
            velocity_valid=torch.where(fresh, velocity_valid, current.velocity_valid),
            scenario_weights=weights, survival_mass=survival)
