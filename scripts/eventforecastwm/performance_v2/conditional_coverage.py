"""Age-conditioned joint density ranks on the unchanged development roots.

All observations remain legal and unchanged. Subpopulation masks are applied
only to likelihood evaluation, never to the history or predicted distribution.
Each root receives equal weight after averaging its supported windows.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from eventbeliefwm.data import read, sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model, scene_interval
from eventforecastwm.benchmark_reuse import CooperativeGuard
from eventforecastwm.model_agent_scale import component_log_densities
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.sampling import joint_samples
from eventforecastwm.train import load_roots, prepare_roots, prepared_batch, encode
from eventforecastwm.window_evaluate import aggregate
from .evaluation_population import population as resolve_population


def agent_density(forecast, truth, mask):
    if hasattr(forecast,'branch_probabilities'):
        from .regime import agent_density as regime
        return regime(forecast,truth,mask)
    density, count = component_log_densities(forecast, truth, mask)
    return (density + forecast.agent_scale_probabilities.clamp_min(1e-12).log()).logsumexp(-1), count


def sample_agent_density(forecast, paths, mask, chunk=8):
    """Evaluate independent sample trajectories in a bounded vectorized batch."""
    fields = ('positions', 'velocities', 'weights', 'covariance', 'initial_covariance',
              'initial_residual_mean', 'valid', 'times', 'agent_scale_probabilities', 'legal_context')
    if hasattr(forecast,'branch_probabilities'):
        fields+=('moving_positions','moving_velocities','stopped_positions','stopped_velocities',
                 'branch_probabilities','stopped_scale_probabilities')
    b, k, t, n, _ = paths.shape
    values = []
    for start in range(0, k, chunk):
        part = paths[:, start:start+chunk]; size = part.shape[1]
        expanded = replace(forecast, **{key: getattr(forecast, key).repeat_interleave(size, 0) for key in fields})
        density, _ = agent_density(expanded, part.reshape(b*size, t, n, 2), mask.repeat_interleave(size, 0))
        values.append(density.reshape(b, size, density.shape[1], n))
    return torch.cat(values, 1)


def masked_joint(density, weights, eligible):
    if density.ndim == 4:  # B,K,S,N
        return ((density * eligible[:, None, None]).sum(-1) + weights[:, None].clamp_min(1e-12).log()).logsumexp(-1)
    return ((density * eligible[:, None]).sum(-1) + weights.clamp_min(1e-12).log()).logsumexp(-1)


@torch.no_grad()
def run(cfg):
    out = Path(cfg['run']); out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists(): raise FileExistsError('Completed diagnostic is immutable')
    guard = CooperativeGuard(out); torch.set_num_threads(1); start = time.time()
    state = torch.load(cfg['checkpoint'], map_location='cpu'); ids, population_identity = resolve_population(cfg)
    if set(ids) & set(state.get('fit_root_ids', [])): raise ValueError('Evaluation leakage')
    model = load_model(cfg['checkpoint']).eval()
    recipe=model.recipe
    while 'physical_recipe' in recipe:recipe=recipe['physical_recipe']
    if recipe.get('variant') != 'agent_scene_scale_v1': raise ValueError('This diagnostic requires the agent-scale law')
    roots = load_roots(cfg['manifest'], ids, guard)
    if cfg.get('geometry_manifest'):
        from eventforecastwm.geometry_data import attach_geometry
        attach_geometry(roots,cfg['geometry_manifest'],guard)
    prepare(model, roots, cfg['manifest'], cfg['prepared_cache'], prepare_roots, guard)
    windows = [(root, j, w) for root in roots for j, w in enumerate(root['windows'])]
    records = []; draws = 64; levels = (.1, .25, .5, .75, .9)
    for offset in range(0, len(windows), 8):
        guard.check(); selected = windows[offset:offset+8]
        batch = prepared_batch([w for _, _, w in selected], 'cpu')
        if model.recipe.get('variant')!='geometry_occupancy_v2' and not cfg.get('physical_uses_geometry',False):
            batch=dict(batch,geometry=torch.zeros_like(batch['geometry']),geometry_support=torch.zeros_like(batch['geometry_support']))
        belief = encode(model, batch)
        age = belief.core.age; visible = torch.zeros_like(age, dtype=torch.bool)
        for i, (_, _, w) in enumerate(selected):
            visible[i, :len(w['base'].humans)] = w['base'].humans[:, -1, 4] > .5
        for horizon in (1., 3.):
            length = round(horizon*4); forecast = model.imagine(belief, horizon_s=horizon)
            truth = batch['targets']['track_positions'][:, :length]
            complete = batch['targets']['track_valid'][:, :length].all(1)
            mask = complete[:, None].expand(-1, length, -1)
            actual, count = agent_density(forecast, truth, mask)
            generator = torch.Generator().manual_seed(83199+offset+int(100*horizon))
            paths, _ = joint_samples(forecast, draws, generator)
            sampled = sample_agent_density(forecast, paths, mask)
            strata = dict(all=complete, visible=complete & visible, lost=complete & ~visible,
                age_lt025=complete & (age < .25), age_025_to1=complete & (age >= .25) & (age < 1),
                age_1_to2=complete & (age >= 1) & (age < 2), age_2_to3=complete & (age >= 2) & (age < 3),
                age_ge3=complete & (age >= 3))
            for name, eligible in strata.items():
                lp = masked_joint(actual, forecast.weights, eligible)
                sample_lp = masked_joint(sampled, forecast.weights, eligible)
                ranks = (1 + (sample_lp > lp[:, None]).sum(1)).float()/(draws+1)
                for i, (root, j, _) in enumerate(selected):
                    if not eligible[i].any(): continue
                    rank = float(ranks[i])
                    records.append(dict(root_id=root['root_id'], scene=root['scene'], window=j,
                        horizon=horizon, stratum=name, agents=int(eligible[i].sum()), rank=rank,
                        nll=float(-lp[i]/count[i, eligible[i]].sum()),
                        **{f'coverage{round(level*100)}': float(rank <= level) for level in levels}))
        if offset % 128 == 0:
            save_json(out/'progress.json', dict(status='diagnosing', windows=min(offset+8, len(windows)), total_windows=len(windows)))
    keys = ('rank', 'nll', *(f'coverage{round(level*100)}' for level in levels)); summary = {}
    for population in ('origin', 'all_windows'):
        summary[population] = {}
        for horizon in (1., 3.):
            summary[population][str(horizon)] = {}
            for name in strata:
                selected = [r for r in records if r['horizon'] == horizon and r['stratum'] == name and
                            (population == 'all_windows' or r['window'] == 0)]
                root_rows = aggregate(selected, keys)
                summary[population][str(horizon)][name] = {key: scene_interval(root_rows, key) for key in keys}
    result = dict(status='conditional_coverage_complete', config=cfg, runtime_s=time.time()-start,
        checkpoint_sha256=sha256(cfg['checkpoint']), manifest_sha256=sha256(cfg['manifest']),
        protocol_sha256=sha256(cfg['protocol']), source_sha256=sha256(__file__), records=records, summary=summary,
        draws=draws, nominal_finite_rank_coverage={str(level): float(np.floor(level*(draws+1))/(draws+1)) for level in levels},
        estimand='Complete supported joint marginal for each stratum; equal windows within root, then equal roots',
        rng='Secondary conditional diagnostic: common fixed seeds across models, distinct from primary origin benchmark',
        confirmation=population_identity['confirmation'],population_identity=population_identity)
    save_json(out/'result.json', result)
    print(json.dumps(dict(status=result['status'], runtime_s=result['runtime_s'])), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    run(read(parser.parse_args().config))


if __name__ == '__main__': main()
