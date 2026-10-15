"""Re-evaluate the immutable release without fitting anything."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eventbeliefwm.data import read, sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model, extra_metrics
from eventforecastwm.distribution import enable_evaluation_dispatch
from eventforecastwm.guard import Guard
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.train import load_roots, prepare_roots, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    cfg = read(parser.parse_args().config)
    run = Path(cfg['run'])
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'result.json').exists():
        raise FileExistsError('Completed reproduction is immutable')
    torch.set_num_threads(1)
    torch.manual_seed(20261014)
    np.random.seed(20261014)
    guard = Guard(run)
    identity = {k: sha256(cfg[k]) for k in ('checkpoint', 'manifest', 'protocol', 'reference')}
    assert identity['checkpoint'] == cfg['checkpoint_sha256']
    save_json(run / 'identity.json', dict(config=cfg, hashes=identity, source_sha256=sha256(__file__)))
    start = time.time()
    enable_evaluation_dispatch()
    model = load_model(cfg['checkpoint']).eval()
    roots = load_roots(cfg['manifest'], read(cfg['protocol'])['development'], guard)
    prepare(model, roots, cfg['manifest'], cfg['prepared_cache'], prepare_roots, guard)
    scores = evaluate(model, roots, 'cpu', horizons=(1., 2., 3.), draws=48)
    state = torch.load(cfg['checkpoint'], map_location='cpu')
    extras = extra_metrics(model, roots, state['anonymous_occupancy_prior'], 'cpu', horizons=(1., 2., 3.), guard=guard)
    reference = read(cfg['reference'])
    deltas = {}
    for horizon in ('1.0', '3.0'):
        assert scores[horizon]['trajectory_roots'] == reference['scores'][horizon]['trajectory_roots']
        assert extras[horizon]['occupancy_metric_roots'] == reference['extras'][horizon]['occupancy_metric_roots']
        deltas[horizon] = {
            key: scores[horizon]['metrics'][key] - reference['scores'][horizon]['metrics'][key]
            for key in ('energy', 'nll', 'mean_ade', 'mean_fde')}
        deltas[horizon].update({
            'occupancy_' + key: extras[horizon]['occupancy'][key] - reference['extras'][horizon]['occupancy'][key]
            for key in ('average_precision', 'brier', 'nll')})
    maximum = max(abs(value) for values in deltas.values() for value in values.values())
    if maximum > 1e-6:
        raise AssertionError('Baseline reproduction discrepancy: ' + str(deltas))
    result = dict(status='reproduction_complete', scores=scores, extras=extras,
                  reference_deltas=deltas, maximum_metric_difference=maximum,
                  runtime_s=time.time()-start, identity=identity, fit_updates=0,
                  population='unchanged 256 development roots; no confirmation targets read')
    save_json(run / 'result.json', result)
    print(json.dumps(dict(status=result['status'], runtime_s=result['runtime_s'],
                          maximum_metric_difference=maximum, deltas=deltas)), flush=True)


if __name__ == '__main__':
    main()
