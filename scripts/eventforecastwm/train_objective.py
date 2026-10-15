"""Bounded single-loss controls; the original mixed-loss trainer stays pinned."""
import argparse
from pathlib import Path

from eventbeliefwm.data import read, sha256
from eventforecastwm import train, train_cached


def likelihood_loss(model, batch, objective='nll_only', nll_weight=1., horizon=3.):
    forecast = model.imagine(train.encode(model, batch), horizon_s=horizon)
    steps = forecast.positions.shape[2]
    target = batch['targets']['track_positions'][:, :steps]
    mask = batch['targets']['track_valid'][:, :steps]
    logp, count = train.trajectory_log_probability(forecast, target, mask)
    value = train.mean_valid(-logp / count.clamp_min(1), count > 0)
    # Keep the same diagnostic sample count and RNG consumption as the mixed
    # objective, but only the proper trajectory likelihood supplies gradients.
    energy, supported = train.energy_score(forecast, target, mask,
        draws=max(1, 6 // forecast.weights.shape[1]))
    energy = train.mean_valid(energy, supported)
    effective = (-(forecast.weights * forecast.weights.clamp_min(1e-12).log()).sum(1)).exp().mean()
    return value, dict(loss=float(value.detach()), nll=float(value.detach()),
        energy=float(energy.detach()), supported_roots=int(supported.sum()),
        effective_modes=float(effective.detach()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    if config.get('objective') != 'nll_only':
        raise ValueError('This control is explicitly likelihood-only')
    original = train.fingerprints
    def fingerprints():
        return dict(original(), train_objective_py=sha256(Path(__file__)))
    train.fingerprints = fingerprints
    train.loss = likelihood_loss
    train_cached.main()


if __name__ == '__main__':
    main()
