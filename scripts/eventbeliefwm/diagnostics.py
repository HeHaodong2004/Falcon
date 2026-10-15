"""Bounded physical-belief diagnostics; truth is evaluation-only.

Teacher forcing is explicitly privileged and is never used for deployment or
free-running composition. Synthetic straight queries test the model algebra;
they are not assigned invented V4 outcomes. Latency includes legal encoding,
physical rollout, candidate readouts and optional occupancy rasterization, not
the simulator, candidate generator, or native controller.
"""
from dataclasses import replace
import time
import numpy as np
import torch


def _encode(model, batch):
    return model.encode_history(batch['base'],
        {'features': batch['geometry'], 'support': batch['geometry_support']},
        batch['sensor_config'], batch['robot_geometry'],
        history_timestamps=batch.get('history_timestamps'),
        track_acquisition_times=batch.get('track_acquisition_times'),
        prepared_history=batch.get('prepared_history'))


def _query(belief, horizon, speed=0.):
    b = len(belief.clock)
    q = belief.positions.new_zeros(b, 2, 10)
    q[:, :, :2] = belief.ego_position[:, None]
    q[:, 1, 0] += speed*horizon
    q[:, 1, 2] = horizon
    q[:, :, 3] = speed
    q[:, :, 6] = 1.
    q[:, 1, 9] = 1.
    return q


def _root_error(prediction, truth, mask):
    distance = (prediction-truth).norm(dim=-1)
    count = mask.flatten(1).sum(-1)
    error = (distance*mask).flatten(1).sum(-1)/count.clamp_min(1)
    return [float(error[i]) if count[i] else None for i in range(len(error))]


def _mean(forecast):
    return (forecast.human_positions*forecast.scenario_weights[:, :, None, None, None]).sum(1)


def _root_max(a, b):
    return (a-b).abs().reshape(len(a), -1).amax(-1).tolist()


def _one_root(batch):
    result = {}
    for name, value in batch.items():
        if name == 'prepared_history':
            continue
        elif name == 'base':
            result[name] = value.select(torch.tensor([0], device=value.humans.device))
        elif isinstance(value, torch.Tensor):
            result[name] = value[:1]
        elif isinstance(value, dict):
            result[name] = {k: v[:1] for k, v in value.items()}
        else:
            result[name] = value[:1]
    # Pad-only cropping prevents timing unrelated root padding from dominating.
    base = result['base']
    n = max(1, int((~base.human_padding[0]).sum()))
    t = int((~base.history_padding[0]).sum())
    base.humans = base.humans[:, :n, :t]
    base.human_padding = base.human_padding[:, :n]
    base.robot = base.robot[:, :t]
    base.history_padding = base.history_padding[:, :t]
    if 'prepared_history' in batch:
        result['prepared_history'] = batch['prepared_history'].select(
            torch.tensor([0],device=base.humans.device),n=n,t=t)
    if 'history_timestamps' in result:
        result['history_timestamps'] = result['history_timestamps'][:, :t]
        result['track_acquisition_times'] = result['track_acquisition_times'][:, :n, :t]
    return result


@torch.no_grad()
def planning_latency(model, batch, horizon=3., repetitions=5):
    """8/33 candidate end-to-end model cost, synchronised on CUDA."""
    sample = _one_root(batch)
    device = sample['query'].device
    cuda = device.type == 'cuda'
    output = {}
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    for k in (8, 33):
        index = torch.arange(k, device=device) % sample['query'].shape[1]
        query = sample['query'][:, index]
        measurements = []
        for iteration in range(repetitions+1):
            if cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            belief = _encode(model, sample)
            model.imagine(belief, query, horizon, allow_query_hold=True)
            if cuda:
                torch.cuda.synchronize(device)
            if iteration:
                measurements.append((time.perf_counter()-start)*1000)
        output[str(k)] = dict(repetitions=repetitions, median_ms=float(np.median(measurements)),
            p95_ms=float(np.quantile(measurements, .95)), max_ms=float(max(measurements)),
            candidate_repetition=k > sample['query'].shape[1],
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else None)
    prepared = 'prepared_history' in sample
    return dict(device=str(device), includes='learned encode + shared physical imagination + candidate risk/observation + occupancy'+
                ('' if prepared else ' + legal history preparation'),
                excludes='global/candidate planning, simulator/controller and sensor transport'+
                ('; cached legal history preparation' if prepared else ''),
                legal_history_prepared=prepared,
                roots_timed=1, measurements=output)


@torch.no_grad()
def diagnose(model, batch, horizon=3., latency_repeats=5):
    was_training = model.training
    model.eval()
    try:
        b0 = _encode(model, batch)
        b = len(b0.clock)
        t = round(horizon/model.config.dt)
        if t < 2:
            raise ValueError('Diagnostics need at least two canonical transitions')
        split_steps = min(round(1./model.config.dt), t-1)
        first_horizon = split_steps*model.config.dt
        # Same exact physical clocks, persistent scenario identities, and zero
        # resampling noise for full versus segmented straight-line motion.
        full = model.imagine(b0, _query(b0, horizon, .35), horizon, rasterize=False)
        first = model.imagine(b0, _query(b0, first_horizon, .35), first_horizon, rasterize=False)
        second = model.imagine(first.next_belief,
            _query(first.next_belief, horizon-first_horizon, .35), horizon-first_horizon, rasterize=False)
        component_errors = {name: _root_max(getattr(full.next_belief, name), getattr(second.next_belief, name))
            for name in ('positions', 'velocities', 'hidden', 'scenario_weights', 'survival_mass', 'collision_mass')}
        wait = model.imagine(b0, _query(b0, horizon), horizon, rasterize=False)
        invariance = _root_max(wait.human_positions, full.human_positions)
        target = batch['targets']
        truth = target['track_positions'][:, :t]
        valid = target['track_valid'][:, :t]
        free = _mean(wait)
        free_error = _root_error(free, truth, valid)
        # Privileged teacher forcing only for this diagnostic. Neither this
        # state nor these trajectories are reused by any deployment planner.
        forced = b0
        forced_states = []
        previous_truth = target['initial_track_positions']
        previous_valid = target['initial_track_valid']
        for j in range(t):
            forced = model._step(forced)
            forced_states.append((forced.positions*forced.scenario_weights[:, :, None, None]).sum(1))
            step_valid = valid[:, j]
            velocity_valid = step_valid & previous_valid
            target_velocity = (truth[:, j]-previous_truth)/model.config.dt
            forced = replace(forced,
                positions=torch.where(step_valid[:, None, :, None], truth[:, j, None], forced.positions),
                velocities=torch.where(velocity_valid[:, None, :, None], target_velocity[:, None], forced.velocities))
            previous_truth, previous_valid = truth[:, j], step_valid
        teacher_error = _root_error(torch.stack(forced_states, 1), truth, valid)
        # Actual detections from an actually recorded stationary native branch.
        # The chosen row is verified from its nominal path, not assumed b032.
        stationary = batch['query'][..., :2].abs().amax((-1, -2)) < 1e-5
        stationary &= batch['query'][..., 3:5].abs().amax((-1, -2)) < 1e-5
        chosen = stationary.long().argmax(-1)
        has_stationary = stationary.any(-1)
        row = torch.arange(b, device=truth.device)
        j = split_steps-1
        observed = target['observed_tracks'][row, chosen, j] & target['observed_valid'][row, chosen, j]
        observed &= has_stationary[:, None]
        root_times = torch.tensor([m['timestamp'] for m in batch['metadata']],
                                  device=truth.device, dtype=torch.float64)
        observation = dict(positions=target['actual_observed_positions'][row, chosen, j], observed=observed,
            acquisition_times=target['actual_acquisition_times'][row, chosen, j]-root_times[:, None],
            timestamp_s=truth.new_full((b,), first_horizon), source='actual_arrived')
        prefix = model.imagine(b0, _query(b0, first_horizon), first_horizon, rasterize=False)
        corrected = model.observe(prefix.next_belief, observation)
        post = model.imagine(corrected, _query(corrected, horizon-first_horizon), horizon-first_horizon, rasterize=False)
        correction_mask = observed & valid[:, j]
        before = _root_error(_mean(prefix)[:, -1:, ...], truth[:, j:j+1], correction_mask[:, None])
        corrected_mean = (corrected.positions*corrected.scenario_weights[:, :, None, None]).sum(1)
        after = _root_error(corrected_mean[:, None], truth[:, j:j+1], correction_mask[:, None])
        post_mask = valid[:, split_steps:] & observed[:, None]
        after_future = _root_error(_mean(post), truth[:, split_steps:], post_mask)
        without_future = _root_error(free[:, split_steps:], truth[:, split_steps:], post_mask)
        # Coherent categorical mode diagnostics, with observed identities only.
        s = wait.human_positions.shape[1]
        w = wait.scenario_weights
        entropy = -(w*w.clamp_min(1e-12).log()).sum(-1)
        difference = wait.human_positions[:, :, None]-wait.human_positions[:, None, :]
        mask = b0.valid[:, None, None, None, :, None]
        pair_distance = ((difference.square()*mask).sum((3, 4, 5))/
                         (b0.valid.sum(-1).clamp_min(1)*t*2)[:, None, None]).sqrt()
        off_diagonal = ~torch.eye(s, dtype=torch.bool, device=w.device)
        diversity = pair_distance[:, off_diagonal].mean(-1) if s > 1 else w.new_zeros(b)
        # Force positively tracked physical state to robot path and zero speed;
        # preserve every learned hidden feature, weight and calibration parameter.
        perturbed = replace(b0, positions=b0.positions*0, velocities=b0.velocities*0)
        perturbed_risk = model.imagine(perturbed, _query(perturbed, horizon), horizon, rasterize=False)
        risk_delta = (perturbed_risk.cumulative_collision[:, 0, -1]-wait.cumulative_collision[:, 0, -1]).abs()
        roots = []
        for i, meta in enumerate(batch['metadata']):
            roots.append(dict(root_id=meta['root_id'], composition_max_abs={key: value[i] for key, value in component_errors.items()},
                query_invariance_max_abs=invariance[i], free_running_ADE_m=free_error[i],
                teacher_forced_one_step_ADE_m=teacher_error[i], correction_clock_s=first_horizon,
                actual_correction_tracks=int(correction_mask[i].sum()), correction_before_ADE_m=before[i],
                correction_after_ADE_m=after[i], remaining_uncorrected_ADE_m=without_future[i],
                remaining_corrected_ADE_m=after_future[i], scene_entropy_nats=float(entropy[i]),
                effective_scenarios=float(entropy[i].exp()), pairwise_joint_path_RMSE_m=float(diversity[i]),
                state_perturbation_collision_probability_delta=float(risk_delta[i]),
                perturbation_supported=bool(b0.valid[i].any())))
        return dict(parameter_counts=model.parameter_counts(), roots=roots,
            composition_pass=max(max(v) for v in component_errors.values()) <= 1e-5,
            query_invariance_pass=max(invariance) <= 1e-6,
            exact_physical_clock=True, persistent_scenario_ids=True, future_noise_resampling=False,
            composition_queries='synthetic straight motion, no native outcomes assigned',
            teacher_forcing='privileged state replacement exclusively for one-step diagnostic',
            correction='actual recorded WAIT detections only; future detections never fed into imagination',
            latency=planning_latency(model,batch,horizon,latency_repeats) if latency_repeats else None)
    finally:
        model.train(was_training)
