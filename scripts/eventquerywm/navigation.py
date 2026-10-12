"""Goal-directed timed-query navigation. No human predictor or CV dependency."""
from dataclasses import dataclass
import math
import time
import numpy as np
import torch
from collector.core import LegalRoot, compile_motion
from eventquerywm.inputs import tensorize, collate
from eventquerywm.model import EventQueryWM, ModelConfig


@dataclass(frozen=True)
class NavigationConfig:
    lookahead_m: float = 1.5
    speed_mps: float = .7
    yaw_speed_rps: float = 1.5
    execution_s: float = .5
    risk_budget: float = .2
    max_consecutive_wait_s: float = 2.
    history_window_s: float = 8.
    lateral_offset_m: float = .35
    spatial_alternatives: bool = True
    risk_hysteresis: float = .02


def prefix(path, length):
    result = [np.asarray(path[0], dtype=float)]
    for point in path[1:]:
        point = np.asarray(point, dtype=float)
        distance = float(np.linalg.norm((point-result[-1])[[0, 2]]))
        if distance > length:
            result.append(result[-1]+(point-result[-1])*length/distance)
            break
        if distance > 1e-6:
            result.append(point)
            length -= distance
        if length <= 1e-6:
            break
    return [v.tolist() for v in result]


class GoalQueryGenerator:
    """Receives only static navigation, robot pose and the episode goal."""
    def __init__(self, geometry, config):
        self.geometry, self.config = geometry, config

    def generate(self, root, goal):
        cfg = self.config
        path, distance = self.geometry.path(root.position, goal)
        if path is None:
            return [], dict(reason='no_global_static_route', goal_distance=None)
        center = prefix(path, cfg.lookahead_m)
        if len(center) < 2:
            return [], dict(reason='at_goal_or_degenerate_route', goal_distance=distance)
        paths = [('route', center)]
        if cfg.spatial_alternatives:
            d = np.asarray(center[-1])-center[0]
            normal = np.array([-d[2], 0., d[0]])/max(np.linalg.norm(d[[0, 2]]), 1e-6)
            for sign, name in ((1, 'left'), (-1, 'right')):
                end = np.asarray(center[-1])+sign*cfg.lateral_offset_m*normal
                alternative, length = self.geometry.path(root.position, end, max_projection=.15)
                if alternative is not None and length <= cfg.lookahead_m+1.:
                    paths.append((name, alternative))
        generation = dict(speed=cfg.speed_mps, yaw_speed=cfg.yaw_speed_rps, directions=3,
                          target_radius=cfg.lookahead_m, max_projection=.15,
                          max_path_length=cfg.lookahead_m+1., human_blind=True)
        candidates, rejected = [], []
        for name, p in paths:
            _, remaining = self.geometry.path(p[-1], goal)
            progress = distance-remaining if remaining is not None else -math.inf
            if progress <= .03:
                continue
            for delay, scale, timing in ((0., 1., 'normal'), (0., .5, 'slow'), (.5, 1., 'delay')):
                q = compile_motion(root, p, [0., 0.], name, delay, scale, generation).data
                if not self.geometry.feasible(q):
                    rejected.append(dict(family=name+'/'+timing, reason='nominal_path_not_executable'))
                    continue
                candidates.append(dict(query=q, family=name+'/'+timing, progress_m=progress,
                                       remaining_distance_m=remaining, waiting=False))
        if candidates:
            # WAIT has a comparable full prediction horizon, then only a short
            # prefix is executed. STOP is never an executable control here.
            horizon = max(v['query']['planned_duration'] for v in candidates)
            q = compile_motion(root, [list(root.position)], [0., 0.], 'WAIT', horizon, 0., generation).data
            candidates.append(dict(query=q, family='wait', progress_m=0.,
                                   remaining_distance_m=distance, waiting=True))
        return candidates, dict(goal_distance=distance, route=path, rejected=rejected)


class WorldModelRisk:
    def __init__(self, checkpoint, device='cuda:0'):
        self.device = device
        ck = torch.load(checkpoint, map_location='cpu')
        self.model = EventQueryWM(ModelConfig(**ck['config'])).to(device).eval()
        self.model.load_state_dict(ck['model'], strict=True)
        self.parameters = sum(p.numel() for p in self.model.parameters())
        assert self.parameters == 1035649
        self.protocol, self.step = ck['protocol'], ck['step']

    @torch.no_grad()
    def evaluate(self, history, candidates):
        begin = time.perf_counter()
        values = [tensorize(dict(history=history, history_images=None, query=v['query'])) for v in candidates]
        batch = collate(values).to(self.device)
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)
        inference_begin = time.perf_counter()
        logits = self.model(batch)
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)
        inference_s = time.perf_counter()-inference_begin
        probabilities = torch.sigmoid(logits).cpu().numpy()
        assert np.isfinite(probabilities).all()
        return probabilities.tolist(), dict(inference_s=inference_s,
                tensorization_and_inference_s=time.perf_counter()-begin,
                human_tensor_shape=list(batch.humans.shape), history_length=len(history))


class RiskAwareSelector:
    """Chance-constrained goal progress; no baseline predictor or learned policy."""
    def __init__(self, config, use_world_model=True):
        self.config, self.use_world_model = config, use_world_model
        self.last_family = None
        self.wait_s = 0.

    def select(self, candidates, safe_probabilities):
        cfg = self.config
        moving = [i for i, c in enumerate(candidates) if not c['waiting'] and
                  (c['query']['wait_duration'] == 0 or self.wait_s < cfg.max_consecutive_wait_s)]
        if not moving:
            raise ValueError('No feasible progress-making query')
        # Seconds/metre is physical cost; normalization is one metre traversed
        # at configured normal speed. It does not contain arbitrary risk units.
        efficiency = [c['query']['planned_duration']*cfg.speed_mps/max(c['progress_m'], .03) for c in candidates]
        risk = [1-float(p) for p in safe_probabilities] if self.use_world_model else [None]*len(candidates)
        reason = 'static_time_per_progress'
        if not self.use_world_model:
            chosen = min(moving, key=lambda i: (efficiency[i], candidates[i]['family']))
            previous = [i for i in moving if candidates[i]['family'] == self.last_family]
            if previous and efficiency[previous[0]] <= 1.1*efficiency[chosen]:
                chosen = previous[0]; reason += '_hysteresis'
        else:
            safe = [i for i in moving if risk[i] <= cfg.risk_budget]
            if safe:
                chosen = min(safe, key=lambda i: (efficiency[i], risk[i]))
                reason = 'risk_budget_then_time_per_progress'
            else:
                lowest = min(risk[i] for i in moving)
                near = [i for i in moving if risk[i] <= lowest+cfg.risk_hysteresis]
                chosen = min(near, key=lambda i: efficiency[i])
                reason = 'minimum_risk_band_then_time_per_progress'
                waits = [i for i, c in enumerate(candidates) if c['waiting'] and risk[i] <= cfg.risk_budget
                         and risk[i]+.05 < risk[chosen] and self.wait_s < cfg.max_consecutive_wait_s]
                if waits:
                    chosen = min(waits, key=lambda i: risk[i]); reason = 'bounded_wait_for_risk'
            previous = [i for i in moving if candidates[i]['family'] == self.last_family]
            if previous and not candidates[chosen]['waiting']:
                i = previous[0]
                if risk[i] <= risk[chosen]+cfg.risk_hysteresis and efficiency[i] <= 1.1*efficiency[chosen]:
                    chosen = i; reason += '_hysteresis'
        self.last_family = candidates[chosen]['family']
        for i, c in enumerate(candidates):
            c['selection_score'] = dict(collision_probability=risk[i], normalized_time_per_progress=efficiency[i],
                                        within_risk_budget=risk[i] is not None and risk[i] <= cfg.risk_budget)
        return chosen, reason

    def executed(self, candidate, seconds):
        self.wait_s = self.wait_s+seconds if candidate['waiting'] or candidate['query']['wait_duration'] > 0 else 0.


class TimedWaypointController:
    """Track the compiled native-clock schedule without speeding up late paths."""
    def command(self, query, index, position, yaw):
        nominal = query['nominal_timed_trajectory']
        native = np.asarray(query['executable_controls'][index], dtype=float)
        before, after = nominal[index], nominal[index+1]
        dt, integration = query['control_dt'], query['integration_dt']
        target = np.asarray(after[1:4]); pos = np.asarray(position)
        error = float(np.linalg.norm((pos-np.asarray(before[1:4]))[[0, 2]]))
        heading_error = (before[4]-yaw+math.pi)%(2*math.pi)-math.pi
        # Exact WAIT and turn-only segments cannot acquire translation through
        # feedback. Feed-forward plus bounded heading feedback preserves speed.
        if abs(native[0]) < 1e-9:
            return native.tolist(), dict(position_error_m=error, heading_error_rad=heading_error,
                                         nominal_speed_mps=0., corrected=False)
        delta = target-pos
        target_heading = math.atan2(-delta[2], delta[0])
        angle = (target_heading-yaw+math.pi)%(2*math.pi)-math.pi
        max_physical_speed = abs(native[0])*integration/dt
        forward_distance = max(0., float(np.dot(delta[[0, 2]], [math.cos(yaw), -math.sin(yaw)])))
        speed = min(max_physical_speed, forward_distance/dt)*max(0., math.cos(angle))
        omega = np.clip(angle/dt, -1.5, 1.5)
        return [speed*dt/integration, omega*dt/integration], dict(position_error_m=error,
            heading_error_rad=heading_error, nominal_speed_mps=max_physical_speed,
            corrected=bool(abs(angle) > 1e-4 or error > 1e-4))
