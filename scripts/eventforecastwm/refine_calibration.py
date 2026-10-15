"""Bounded full-root likelihood refinement with all physical means frozen.

Unlike the original minibatch calibration, every optimizer update here is the
gradient of the entire supported root-origin population. Only five global OU
noise parameters and seven persistent per-agent scale logits can change.
Completed updates resume exactly; an interrupted partial gradient is replayed.
"""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import torch
from eventbeliefwm.data import sha256
from eventforecastwm import benchmark, benchmark_reuse
from eventforecastwm.model_agent_scale import AgentScaleWorldModel, trajectory_log_probability
from eventforecastwm.train import prepared_batch, encode


NAMES = ('log_initial_position', 'log_initial_velocity', 'log_acceleration',
         'log_readout', 'raw_decay_rate', 'scale_logits')


def current_forecast(model, cached):
    """Rebuild likelihood covariance; cached means never receive gradients.

    The old time-marginal covariance is deliberately not used by this exact
    Kalman likelihood. Evaluation reimagines the model with the new parameters.
    """
    covariance = torch.zeros_like(cached.initial_covariance)
    covariance[..., 0, 0] = model.log_initial_position.exp().clamp(.01, 3.).square()
    covariance[..., 1, 1] = model.log_initial_velocity.exp().clamp(.01, 3.).square()
    return replace(cached, initial_covariance=covariance,
        acceleration_scale=model.log_acceleration.exp().clamp(.01, 5.),
        readout_scale=model.log_readout.exp().clamp(.01, 5.), decay_rate=model.decay_rate,
        agent_scale_probabilities=model.probabilities(cached.legal_context, cached.weights.shape[1]))


def root_loss_sum(model, cached, truth, mask):
    logp, count = trajectory_log_probability(current_forecast(model, cached), truth, mask)
    keep = count > 0
    return -(logp[keep] / count[keep]).sum(), int(keep.sum())


def fit(model, roots, device, guard, *, directory, identity, steps, lr):
    if not isinstance(model, AgentScaleWorldModel) or model.conditional or model.age_aware:
        raise ValueError('Use an unconditional, age-independent per-agent scale parent')
    for p in model.parameters():
        p.requires_grad_(False)
    frozen = {k: v.detach().clone() for k, v in model.state_dict().items() if k not in NAMES}
    parameters = [getattr(model, name) for name in NAMES]
    for p in parameters:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    state_path = Path(directory) / 'refinement.resume.pt'
    start, curve = 0, []
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        if state['identity'] != identity:
            raise ValueError('Refinement resume identity changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, curve = state['step'], state['curve']
    cached = []
    total = 0
    with torch.no_grad():
        for i in range(0, len(roots), 32):
            guard.check()
            batch = prepared_batch([r['windows'][0] for r in roots[i:i + 32]], device)
            forecast = model.imagine(encode(model, batch), horizon_s=3.)
            raw = batch['targets']['track_valid'][:, :12]
            mask = raw.all(1)[:, None].expand_as(raw)
            truth = batch['targets']['track_positions'][:, :12]
            total += int(mask.flatten(1).any(1).sum())
            cached.append((forecast, truth, mask))
    if not total:
        raise ValueError('No complete calibration targets')
    for step in range(start, steps + 1):
        optimizer.zero_grad()
        value = 0.
        for forecast, truth, mask in cached:
            guard.check()
            with torch.set_grad_enabled(step < steps):
                loss, count = root_loss_sum(model, forecast, truth, mask)
                if count:
                    value += float(loss.detach()) / total
                    if step < steps:
                        (loss / total).backward()
        if not torch.isfinite(torch.tensor(value)):
            raise FloatingPointError('Nonfinite full-root likelihood')
        curve.append(dict(step=step, train_nll=value))
        if step < steps:
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('Invalid covariance gradient')
            optimizer.step()
            payload = dict(identity=identity, step=step + 1, curve=curve,
                model=model.state_dict(), optimizer=optimizer.state_dict())
            temporary = state_path.with_suffix('.tmp')
            torch.save(payload, temporary)
            os.replace(temporary, state_path)
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in frozen.items()):
        raise AssertionError('A frozen physical parameter changed')
    return dict(steps=steps, roots=total, fitting_roots=len(roots), learning_rate=lr,
        parameters=sum(p.numel() for p in parameters), curve=curve,
        objective='Full-root mean of supported 3s normalized exact joint NLL',
        physical_means_and_weights_bitwise_unchanged=True,
        resume='Exact completed Adam updates; interrupted partial gradients replayed',
        identity=identity)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--refine-steps', type=int, default=32)
    parser.add_argument('--refine-lr', type=float, default=.005)
    args, remaining = parser.parse_known_args()
    if not 1 <= args.refine_steps <= 128 or not 0 < args.refine_lr <= .03:
        raise ValueError('Bounded calibration requires 1–128 steps and learning rate ≤.03')
    if '--profile' not in remaining:
        raise ValueError('Training-only calibration must be explicitly enabled')
    def argument(name):
        return remaining[remaining.index(name) + 1]
    checkpoint = argument('--checkpoint')
    identity = dict(parent_sha256=sha256(checkpoint), manifest_sha256=sha256(argument('--manifest')),
        protocol_sha256=sha256(argument('--protocol')), steps=args.refine_steps, lr=args.refine_lr,
        source_hashes={name: sha256(Path(__file__).parent / name) for name in
                      ('refine_calibration.py', 'model_agent_scale.py', 'model_damped.py')})
    parent = torch.load(checkpoint, map_location='cpu')
    def profile(model, roots, device, guard=None, objective_kind='nll'):
        if objective_kind != 'nll' or [r['root_id'] for r in roots] != parent['fit_root_ids']:
            raise ValueError('Likelihood refinement must preserve the parent fitting membership')
        return fit(model, roots, device, guard, directory=Path(argument('--output')).parent,
                   identity=identity, steps=args.refine_steps, lr=args.refine_lr)
    benchmark.profile_integrated = profile
    benchmark.EXPERIMENT_SOURCES['refine_calibration.py'] = sha256(__file__)
    sys.argv = [sys.argv[0], *remaining]
    benchmark_reuse.main()


if __name__ == '__main__':
    main()
