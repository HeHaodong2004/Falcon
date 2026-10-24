#!/usr/bin/env python
"""Fixed-workload end-to-end benchmark, always exactly 28 collector workers.

Includes cold worker imports, scene loading, rejected attempts, native work,
complete validation, SHA/fsync publication, sampling receipts, SQLite WAL/FULL
acknowledgement and final drain. Never starts a service or training.
"""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
_COUNTERS = None


def instrument_native():
    """Read-only counters, applied equally to both benchmark producers."""
    global _COUNTERS
    if _COUNTERS is not None:
        return
    import habitat_sim  # Installed native import order.
    import falcon.additional_action as actions
    from falcon.rearrange_sim_v2 import RearrangeSim_v2
    _COUNTERS = dict(native_robot_actions=0, simulator_frames=0, simulator_world_calls=0,
                     simulated_seconds=0., fixed_timestep_s=None)
    def action_wrapper(original):
        def step(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            _COUNTERS['native_robot_actions'] += 1
            return result
        return step
    for cls in (actions.DiscreteMoveForwardAction, actions.DiscreteTurnLeftAction, actions.DiscreteTurnRightAction):
        cls.step = action_wrapper(cls.step)
    original = RearrangeSim_v2.step_world
    def world(self, dt=-1):
        before = float(self.get_world_time())
        result = original(self, dt)
        elapsed = float(self.get_world_time())-before
        fixed = float(self.get_physics_time_step())
        _COUNTERS['simulator_world_calls'] += 1
        _COUNTERS['simulated_seconds'] += elapsed
        _COUNTERS['simulator_frames'] += int(round(elapsed/fixed))
        _COUNTERS['fixed_timestep_s'] = fixed
        return result
    RearrangeSim_v2.step_world = world


def worker(source, backend, jobs, configuration):
    sys.path.insert(0, str(source))
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MAGNUM_LOG='quiet',
                      HABITAT_SIM_LOG='quiet', PYTHONDONTWRITEBYTECODE='1')
    instrument_native()
    before = dict(_COUNTERS)
    start = time.perf_counter()
    if backend == 'reference':
        from eventwm_graph.partial.campaign import worker_batch
    else:
        from eventwm_graph.partial.acceleration import worker_batch
    results = worker_batch(jobs, configuration)
    counters = {k: _COUNTERS[k]-before[k] for k in before if k != 'fixed_timestep_s'}
    counters['fixed_timestep_s'] = _COUNTERS['fixed_timestep_s']
    return dict(results=results, counters=counters, wall_seconds=time.perf_counter()-start,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, pid=os.getpid())


def make_plan(path, attempts):
    sys.path.insert(0, str(ROOT))
    from eventwm_graph.config import read_json, write_json, file_digest
    from eventwm_graph.stage1.storage import scene_split
    production = ROOT/'output/stage01_partial_graph_query_20261020/production'
    configuration = read_json(ROOT/'configs/stage01_partial_graph_production.json')
    db = sqlite3.connect('file:'+str(production/'campaign.sqlite3')+'?mode=ro', uri=True)
    # Predeclare by seed/scene only, without consulting attempt outcomes. Both
    # members of a two-seed production scene batch are retained together.
    rows = db.execute('SELECT seed,scene,manifest FROM jobs ORDER BY seed').fetchall()
    eligible = [dict(seed=s, scene=c, manifest=m) for s, c, m in rows if scene_split(c) == 'train']
    if attempts % 2 or attempts > len(eligible):
        raise ValueError('Plan requires an even available attempt count')
    jobs = eligible[:attempts]
    for a, b in zip(jobs[::2], jobs[1::2]):
        if a['scene'] != b['scene'] or a['seed']+1 != b['seed']:
            raise ValueError('Production two-seed batch identity changed')
    plan = dict(schema='stage01_fixed_train_attempt_plan_1', workers=28, jobs=jobs,
                configuration=configuration,
                selection='First N journal attempts in seed order conditional only on preserved train scene membership; no outcome-based selection',
                source_configuration_sha256=file_digest(ROOT/'configs/stage01_partial_graph_production.json'),
                represented_scenes=len({j['scene'] for j in jobs}))
    if Path(path).exists():
        raise ValueError('Refusing to overwrite a predeclared workload')
    write_json(path, plan)
    print(json.dumps(dict(plan=str(Path(path).resolve()), attempts=len(jobs), scenes=plan['represented_scenes'], workers=28)))


def benchmark(args):
    started = time.perf_counter()
    if args.source is None:
        raise ValueError('Explicit source directory required for both producers')
    source = args.source.resolve()
    if args.backend == 'fast' and source != ROOT.resolve():
        raise ValueError('fast_benchmark_driver_source_mismatch: run the driver inside the requested source bundle')
    sys.path.insert(0, str(ROOT))
    from eventwm_graph.config import digest, file_digest, read_json, write_json
    from eventwm_graph.partial.campaign import database, statistics
    from eventwm_graph.partial.collect import verify_gate
    plan = read_json(args.plan)
    if plan.get('workers') != 28:
        raise ValueError('This benchmark fixes the production worker count at 28')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = dict(plan['configuration'])
    cfg.update(workers=28, output=str(args.output.resolve()), dataset=str((args.output/'dataset').resolve()))
    if args.backend == 'fast':
        from eventwm_graph.partial.acceleration import MODE
        cfg['acceleration'] = dict(mode=MODE, persistent_scene=False)
    verify_gate(cfg['stage0_gate']); verify_gate(cfg['stage1_gate'], 'stage1')
    write_json(args.output/'configuration.json', cfg)
    if args.backend == 'reference':
        frozen = read_json(source/'manifest.json')
        for name, sha in frozen['frozen_sources'].items():
            if file_digest(source/name) != sha:
                raise ValueError('Frozen reference source changed: '+name)
    else:
        from eventwm_graph.partial.acceleration import source_identity
        frozen = dict(implementation=source_identity())
    write_json(args.output/'source.json', dict(source=str(source), identity=frozen,
        benchmark_script_sha256=file_digest(__file__), plan_sha256=file_digest(args.plan)))
    jobs = plan['jobs']; groups = [jobs[i:i+2] for i in range(0, len(jobs), 2)]
    db = database(args.output/'campaign.sqlite3')
    db.executemany('INSERT INTO jobs VALUES(?,?,?,?,NULL)', [(j['seed'], j['scene'], j['manifest'], 'queued') for j in jobs])
    db.execute('INSERT INTO metadata VALUES(?,?)', ('configuration_id', digest(cfg)))
    db.execute('INSERT INTO metadata VALUES(?,?)', ('plan_sha256', file_digest(args.plan)))
    db.commit()
    initial_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
    samples, batches, counters, results = [], [], {}, []
    stopping, failed, next_group = False, False, 0
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    pool_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=28, mp_context=multiprocessing.get_context('spawn')) as pool:
        inflight = {}
        last_sample = 0.
        while inflight or (next_group < len(groups) and not stopping):
            if (args.output/'STOP').exists():
                stopping = True
            while not stopping and len(inflight) < 28 and next_group < len(groups):
                group = groups[next_group]; next_group += 1
                db.executemany("UPDATE jobs SET state='running' WHERE seed=?", [(j['seed'],) for j in group]); db.commit()
                future = pool.submit(worker, str(source), args.backend, group, cfg)
                inflight[future] = group
            done, _ = concurrent.futures.wait(inflight, timeout=.25, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                group = inflight.pop(future)
                try:
                    batch = future.result()
                except Exception as error:
                    batch = dict(results=[dict(seed=group[0]['seed'], status='error', error=repr(error))], counters={})
                batches.append(batch)
                for k, v in batch['counters'].items():
                    if k == 'fixed_timestep_s':
                        counters[k] = v
                    else:
                        counters[k] = counters.get(k, 0)+v
                for r in batch['results']:
                    db.execute('UPDATE jobs SET state=?,result=? WHERE seed=?', (r['status'], json.dumps(r, allow_nan=False), r['seed']))
                    results.append(r)
                    if r['status'] == 'error':
                        failed = stopping = True
                db.commit()
            now = time.perf_counter()
            if now-last_sample >= 1.:
                children = list(pool._processes)
                rss = 0; child_cpu = 0.
                for pid in [os.getpid(), *children]:
                    try:
                        rss += int(Path('/proc', str(pid), 'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
                        if pid != os.getpid():
                            child_cpu += (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
                    except FileNotFoundError:
                        pass
                samples.append(dict(elapsed_s=now-started, completed_attempts=len(results),
                    validated_outcomes=sum(r.get('outcomes', 0) for r in results if r['status'] == 'sealed'),
                    sealed_roots=sum(r['status'] == 'sealed' for r in results),
                    rss_bytes=rss, live_processes=len(children)+1, live_child_cpu_seconds=child_cpu))
                last_sample = now
                if len(samples) % 10 == 0:
                    print(json.dumps(samples[-1]), flush=True)
    # The pool is joined; all active work has drained. All counted records were
    # acknowledged only after validation, durable publication and receipt fsync.
    stats = statistics(db)
    integrity = db.execute('PRAGMA integrity_check').fetchall()
    db.close()
    elapsed = time.perf_counter()-started
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = child_usage.ru_utime+child_usage.ru_stime-initial_cpu.ru_utime-initial_cpu.ru_stime
    published = list((args.output/'dataset/roots').glob('*/*/manifest.json'))
    published = [p for p in published if not p.parent.name.startswith('.')]
    complete = not failed and len(results) == len(jobs) and stats['attempts'].get('running', 0) == 0 and len(published) == stats['roots']
    if integrity != [('ok',)]:
        complete = False
    report = dict(schema='stage01_full_pipeline_fixed28_benchmark_1', status='PASS' if complete else 'INCOMPLETE_OR_ERROR',
        backend=args.backend, workers=28, plan_sha256=file_digest(args.plan), elapsed_seconds=elapsed,
        pool_including_startup_and_drain_seconds=time.perf_counter()-pool_start,
        validated_outcomes_per_second=stats['outcomes']/elapsed,
        independently_sealed_roots_per_second=stats['roots']/elapsed,
        outcomes_per_root=stats['outcomes']/max(1, stats['roots']), statistics=stats,
        planned_attempts=len(jobs), planned_scene_coverage=plan['represented_scenes'], results=results,
        native_counters=counters,
        native_actions_per_second=counters.get('native_robot_actions', 0)/elapsed,
        actual_simulator_frames_per_second=counters.get('simulator_frames', 0)/elapsed,
        simulator_frame_definition='Actual fixed physics substeps from returned world-clock advances/get_physics_time_step, including rejected prefixes, scene initialization and shared human futures',
        child_cpu_seconds=child_cpu, average_worker_cpu_cores=child_cpu/elapsed,
        peak_aggregate_rss_bytes=max((s['rss_bytes'] for s in samples), default=0), resource_samples=samples,
        batches=batches, sqlite_integrity='ok' if integrity == [('ok',)] else str(integrity),
        throughput_scope='Cold worker startup through final validation/publication/SQLite acknowledgement and complete pool drain; fixed attempt workload')
    write_json(args.output/'benchmark.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'backend', 'elapsed_seconds', 'validated_outcomes_per_second',
        'independently_sealed_roots_per_second', 'native_actions_per_second', 'actual_simulator_frames_per_second',
        'average_worker_cpu_cores', 'peak_aggregate_rss_bytes')}, indent=2))
    if not complete:
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan'); plan.add_argument('--output', type=Path, required=True)
    plan.add_argument('--attempts', type=int, default=224)
    run = sub.add_parser('run'); run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--backend', choices=('reference', 'fast'), required=True)
    run.add_argument('--source', type=Path, required=True); run.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'plan':
        make_plan(args.output, args.attempts)
    else:
        benchmark(args)
