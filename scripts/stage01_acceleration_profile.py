#!/usr/bin/env python
"""Small fixed native workload, with the frozen accepted producer as reference.

This is a component profile, not a campaign throughput benchmark. No historical
output is written and only preserved train scenes may be selected.
"""
import argparse
import cProfile
import importlib
import io
import json
import os
from pathlib import Path
import pstats
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[610201008, 610201009, 610201038, 610201039])
    p.add_argument('--backend', choices=('reference', 'fast'), default='reference')
    args = p.parse_args()
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MAGNUM_LOG='quiet',
                      HABITAT_SIM_LOG='quiet', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(args.source.resolve()))
    from eventwm_graph.config import read_json, write_json, file_digest
    from eventwm_graph.partial import campaign, collect, observer
    from eventwm_graph.stage1.storage import scene_split
    from eventwm_graph.partial.map import PartialGraph
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = read_json(ROOT / 'configs/stage01_partial_graph_production.json')
    cfg.update(output=str(args.output.resolve()), dataset=str((args.output / 'dataset').resolve()))
    backend = None
    if args.backend == 'fast':
        from eventwm_graph.partial.acceleration import configure, MODE
        cfg['acceleration'] = {'mode': MODE, 'persistent_scene': False}
        backend = configure(cfg)
    totals = {}
    def add(key, value=1):
        totals[key] = totals.get(key, 0) + value
    if backend is None:
        original = collect.GeometryObserver
        worker = campaign.worker_batch
    else:
        from eventwm_graph.partial.fast_observer import FastGeometryObserver
        from eventwm_graph.partial.acceleration import worker_batch as worker
        original = FastGeometryObserver
    class MeasuredObserver(original):
        def _support(self, key, height):
            cache_key = (self.cache_namespace, key[0], key[1], round(height/.1))
            add('support_calls'); add('support_hits' if cache_key in self.cache else 'support_misses')
            began = time.perf_counter()
            try:
                return super()._support(key, height)
            finally:
                add('support_seconds', time.perf_counter() - began)
        def clear_ray(self, origin, target, ignore=()):
            kind = 'human' if ignore else 'support'
            add(kind + '_ray_calls')
            began = time.perf_counter()
            result = super().clear_ray(origin, target, ignore)
            add(kind + '_ray_seconds', time.perf_counter() - began)
            add(kind + ('_clear' if result else '_blocked'))
            return result
        def observe(self, timestamp=None):
            n = self.performance['observations']; rays = self.performance['rays']
            began = time.perf_counter()
            result = super().observe(timestamp)
            add('observe_including_memo_seconds', time.perf_counter()-began)
            if self.performance['observations'] != n:
                add('observations'); add('rays', self.performance['rays']-rays)
                add('static_update_records', len(result['static_updates']))
            else:
                add('observation_timestamp_memo_hits')
            return result
    if backend is None:
        collect.GeometryObserver = MeasuredObserver
    else:
        from eventwm_graph.partial.acceleration import bind
        backend.collect_root = bind(backend.collect_root, GeometryObserver=MeasuredObserver)
    old_update = PartialGraph.update
    def update(self, *a, **kw):
        began = time.perf_counter()
        try:
            return old_update(self, *a, **kw)
        finally:
            add('all_graph_update_seconds', time.perf_counter()-began); add('all_graph_update_calls')
    PartialGraph.update = update
    db = sqlite3.connect('file:' + str(ROOT / 'output/stage01_partial_graph_query_20261020/scaling_curriculum_benchmark/campaign.sqlite3') + '?mode=ro', uri=True)
    jobs = []
    for seed in args.seeds:
        scene, manifest = db.execute('SELECT scene,manifest FROM jobs WHERE seed=?', (seed,)).fetchone()
        if scene_split(scene) != 'train':
            raise ValueError('Profile refuses held-out scenes')
        jobs.append(dict(seed=seed, scene=scene, manifest=manifest))
    profiler = cProfile.Profile(); began = time.perf_counter(); profiler.enable()
    results = []
    for j in jobs:
        results.extend(worker([j], cfg))
    profiler.disable(); elapsed = time.perf_counter()-began
    profiler.dump_stats(str(args.output / 'cpu.prof'))
    s = io.StringIO(); pstats.Stats(profiler, stream=s).strip_dirs().sort_stats('cumtime').print_stats(65)
    (args.output / 'cpu.txt').write_text(s.getvalue())
    stats = pstats.Stats(profiler)
    native = [{'file': key[0], 'function': key[2], 'calls': val[1], 'self_seconds': val[2], 'cumulative_seconds': val[3]}
              for key, val in stats.stats.items() if any(k in key[2] for k in ('cast_ray', 'snap_point', 'try_step', 'closest_obstacle'))]
    totals['rays_per_observation'] = totals.get('rays', 0) / max(1, totals.get('observations', 0))
    import resource
    report = dict(schema='stage01_focused_profile_1', source=str(args.source.resolve()),
                  backend=args.backend, elapsed_seconds=elapsed, jobs=jobs, results=results,
                  counters=totals, native=native, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  note='cProfile instrumented single-process component workload; not 28-worker throughput')
    write_json(args.output / 'profile.json', report)
    print(json.dumps({k: report[k] for k in ('elapsed_seconds', 'counters', 'native', 'peak_rss_kib')}, indent=2))
    if any(r['status'] == 'error' for r in results):
        raise SystemExit('Unexpected profile worker error; inspect profile.json')


if __name__ == '__main__':
    main()
