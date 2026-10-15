"""Bounded forecasting latency from legal history, without target-cache reads.

Repeated scenes measure batch throughput, not independent episode performance.
Run on an otherwise free research GPU for an idle-device number; contended runs
remain explicitly labelled and are not mixed with those measurements.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from eventbeliefwm.data import sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model
from eventforecastwm.benchmark_reuse import CooperativeGuard
from eventforecastwm.data import nominal_query
from eventforecastwm.inference import select_static_geometry
from eventforecastwm.readouts import tracked_occupancy
from eventforecastwm.sampling import joint_samples
from eventquerywm.inputs import collate, tensorize


def legal_batch(path, size, device):
    history = json.loads(Path(path).read_text())
    sample = tensorize(dict(history=history, history_images=None,
                            query=nominal_query(history)))
    ids = sorted({t['track_id'] for h in history for t in h['human_tracks']})
    acquisition = torch.zeros(1, max(1, len(ids)), len(history), dtype=torch.float64)
    slots = {identity: i for i, identity in enumerate(ids)}
    for j, frame in enumerate(history):
        for track in frame['human_tracks']:
            acquisition[0, slots[track['track_id']], j] = track['last_seen_time']
    times = torch.tensor([[h['timestamp'] for h in history]], dtype=torch.float64)
    return (collate([sample] * size).to(device),
            times.expand(size, -1).to(device),
            acquisition.expand(size, -1, -1).to(device), history, ids)


def measure(function, repetitions, synchronize, guard):
    samples = []
    for _ in range(repetitions):
        guard.check()
        synchronize()
        start = time.perf_counter()
        value = function()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        del value
    return dict(repetitions=repetitions, mean_ms=float(np.mean(samples)),
                median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)),
                min_ms=float(np.min(samples)), max_ms=float(np.max(samples)))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--history', required=True)
    p.add_argument('--static-map', help='Legal NPZ: features[2,32,32], support[32,32]')
    p.add_argument('--device', default='cpu')
    p.add_argument('--cpu-preparation', action='store_true',
                   help='Recompute parameter-free legal features on CPU on every encode, including transfer in timing')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8, 32])
    p.add_argument('--horizon', type=float, default=3.)
    p.add_argument('--draws', type=int, default=6)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repetitions', type=int, default=50)
    p.add_argument('--resource-condition', required=True,
                   help='Describe measured device ownership/contention; never infer idle hardware')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if min(a.batch_sizes) < 1 or not 1 <= a.repetitions <= 200 or not 0 <= a.warmup <= 20:
        raise ValueError('Invalid or unbounded benchmark workload')
    if not 1 <= a.draws <= 64:
        raise ValueError('Use 1 to 64 coherent joint draws')
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    guard = CooperativeGuard(out.parent)
    checkpoint_hash = sha256(a.checkpoint)
    model = load_model(a.checkpoint).to(a.device).eval()
    state = torch.load(a.checkpoint, map_location='cpu')
    prior = state.get('anonymous_occupancy_prior')
    if prior is not None:prior = prior.to(a.device)
    del state
    if hasattr(model.core, 'map_patch_encoder') and not a.static_map:
        raise ValueError('Map-conditioned model requires its legal static map')
    device = torch.device(a.device)
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    geometry = None
    if a.static_map:
        with np.load(a.static_map) as source:
            geometry = {k: torch.as_tensor(source[k], device=device)[None]
                        for k in ('features', 'support')}
        if geometry['features'].shape != (1, 2, 32, 32) or geometry['support'].shape != (1, 32, 32):
            raise ValueError('Invalid legal map shape')
        geometry['support'] = geometry['support'].bool()
    rows = []
    for size in a.batch_sizes:
        guard.check()
        batch, clocks, acquisition, history, ids = legal_batch(a.history, size, device)
        if a.cpu_preparation:
            cpu_batch, cpu_clocks, cpu_acquisition, _, _ = legal_batch(a.history, size, 'cpu')
        static_map = ({k: v.expand(size, *v.shape[1:]) for k, v in geometry.items()}
                      if geometry is not None else None)
        static_map = select_static_geometry(model, static_map)
        def encode():
            prepared = None
            if a.cpu_preparation:
                prepared = model.core.prepare_history(cpu_batch, history_timestamps=cpu_clocks,
                    track_acquisition_times=cpu_acquisition).to(device)
            return model.encode_history(batch, static_map, {}, {'radius': .25},
                history_timestamps=clocks, track_acquisition_times=acquisition, prepared_history=prepared)
        belief = encode()
        forecast = model.imagine(belief, horizon_s=a.horizon)
        parity = None
        if a.cpu_preparation:
            raw = model.imagine(model.encode_history(batch, static_map, {}, {'radius': .25},
                history_timestamps=clocks, track_acquisition_times=acquisition), horizon_s=a.horizon)
            parity = {name: float((getattr(raw, name)-getattr(forecast, name)).abs().max())
                      for name in ('positions', 'weights', 'position_variance')}
            if max(parity.values()) > 1e-5:
                raise AssertionError('CPU legal preprocessing changes the forecast beyond numerical tolerance')
            del raw
        generator = torch.Generator(device=device).manual_seed(20261014)
        def occupancy(future):
            tracked = tracked_occupancy(future)
            anonymous = getattr(future, 'anonymous_occupancy', None)
            if anonymous is None and prior is not None and len(prior) >= tracked.shape[1]:
                anonymous = prior[:tracked.shape[1]][None]
            return tracked, 1-(1-tracked)*(1-anonymous) if anonymous is not None else tracked
        def full():
            future = model.imagine(encode(), horizon_s=a.horizon)
            return joint_samples(future, a.draws, generator), occupancy(future)
        for _ in range(a.warmup):
            guard.check()
            full()
        synchronize()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        functions = dict(encode=encode,
            imagine=lambda: model.imagine(belief, horizon_s=a.horizon),
            joint_sample=lambda: joint_samples(forecast, a.draws, generator),
            tracked_occupancy=lambda: tracked_occupancy(forecast),
            total_occupancy=lambda: occupancy(forecast)[1],
            full_forecast_readout=full)
        row = dict(batch_size=size, legal_tracks=len(ids), history_frames=len(history), preprocessing_parity=parity,
            stages={name: measure(fn, a.repetitions, synchronize, guard)
                    for name, fn in functions.items()})
        if device.type == 'cuda':
            row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        rows.append(row)
        save_json(out.parent / 'progress.json', dict(completed_batches=len(rows), target_batches=len(a.batch_sizes)))
    if checkpoint_hash != sha256(a.checkpoint):
        raise ValueError('Checkpoint changed during measurement')
    result = dict(schema_version=2, checkpoint=str(Path(a.checkpoint).resolve()),
        checkpoint_sha256=checkpoint_hash, legal_history_sha256=sha256(a.history),
        static_map_sha256=sha256(a.static_map) if a.static_map else None,
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        cpu_threads=torch.get_num_threads(), torch_version=torch.__version__,
        resource_condition=a.resource_condition, host_load=os.getloadavg(),
        horizon_s=a.horizon, joint_draws=a.draws, warmup=a.warmup, batches=rows,
        cpu_preparation=a.cpu_preparation,
        occupancy_readouts='Full timing includes tracked and total center occupancy; hybrid anonymous prediction is included in encode/imagine. Total stage includes tracked rasterization and union with available anonymous marginals.',
        scope='Synchronized model calls only; excludes disk IO, simulator, sensors and initial input transfer. CPU-preparation mode includes recomputing and transferring legal features on every encode. No planner/query benchmark. Batch entries repeat the same legal scene.',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=sha256(__file__))
    save_json(out, result)
    print(json.dumps(dict(output=str(out), batches=[dict(batch_size=r['batch_size'],
        full_p95_ms=r['stages']['full_forecast_readout']['p95_ms']) for r in rows])))


if __name__ == '__main__':
    main()
