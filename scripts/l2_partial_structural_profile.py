"""One primed full-call cProfile per unchanged 6/12-person certificate fixture.

Synthetic systems attribution only. It loads frozen research checkpoints and
reads collector status, but opens no collection root payloads. Run only in an
authorized one-CPU research slot. Runtime modules are never changed or patched.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import signal
import subprocess
import time

from l2_partial_structural_benchmark import (
    REPO, _sources, audit_preserved_baseline, cache_state, collector_metadata,
    convert_incoming, json_default, make_models, output_path, preserved_fixture,
    validate_fixture)
import torch
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=45.)
    args = parser.parse_args()
    output = output_path(args.output)
    prefix = output.with_suffix('')
    artifacts = {n: (Path(str(prefix) + '_' + str(n) + '.prof'),
                     Path(str(prefix) + '_' + str(n) + '.txt')) for n in (6, 12)}
    if any(path.exists() for pair in artifacts.values() for path in pair):
        raise FileExistsError('profile output already exists')
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    source = _sources()
    before = collector_metadata()
    provenance = audit_preserved_baseline()
    started = time.perf_counter()
    records, failure = [], None

    def expired(signum, frame):
        raise TimeoutError('bounded final synthetic profile deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    try:
        with threadpool_limits(limits=1):
            models, load_ms = make_models()
            model = models['structural_bound']
            for n in (6, 12):
                previous, row, queries = preserved_fixture(models['old_reference'], n, True)
                b, _ = convert_incoming(previous, model, row, models['old_reference'])
                fingerprint = b.fingerprint()
                model.online_call(b, row, queries, horizon=3., num_worlds=16, seed=73)
                warm = cache_state(model)
                profile = cProfile.Profile()
                wall = time.perf_counter(); cpu = time.process_time()
                profile.enable()
                try:
                    posterior, batch = model.online_call(b, row, queries,
                        horizon=3., num_worlds=16, seed=73)
                finally:
                    profile.disable()
                cpu_ms = (time.process_time() - cpu) * 1000.
                wall_ms = (time.perf_counter() - wall) * 1000.
                checks = validate_fixture(posterior, batch, row, n)
                checks.pop('observation_evidence', None)
                if b.fingerprint() != fingerprint:
                    raise AssertionError('profile changed the incoming belief')
                if checks['resolved_hypotheses'] != 288:
                    raise AssertionError('certificate fixture lost components')
                prof_path, text_path = artifacts[n]
                profile.dump_stats(str(prof_path))
                text = io.StringIO()
                pstats.Stats(profile, stream=text).strip_dirs().sort_stats('cumulative').print_stats(45)
                text_path.write_text(text.getvalue())
                records.append({'synthetic_tracked': n, 'certificate': True, 'cache_mode': 'warm',
                    'wall_ms_instrumented': wall_ms, 'cpu_ms_instrumented': cpu_ms,
                    'timings_ms_instrumented': batch.timings_ms,
                    'profile_path': str(prof_path.relative_to(REPO)),
                    'text_path': str(text_path.relative_to(REPO)),
                    'cache_before': warm, 'cache_after': cache_state(model),
                    'checks': checks, 'top': text.getvalue()})
    except Exception as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
    after = _sources()
    report = {'schema': 'structural_dense_certificate_profile_2', 'synthetic': True,
        'source_sha256': source, 'source_sha256_after': after,
        'sources_unchanged': source == after,
        'source_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
            check=True, capture_output=True, text=True).stdout.strip(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'provenance': provenance, 'collector_before': before, 'collector_after': collector_metadata(),
        'records': records, 'failure': failure, 'elapsed_seconds': time.perf_counter() - started,
        'qualifications': ['One cProfile-instrumented warm call per synthetic case; no release tail-latency claim.',
            'Unchanged preserved fixture: 24 distinct candidates,16 worlds,3s horizon,4 refined endpoints.',
            'All 288 posterior certificate components and population count support remain present.',
            'No future, holdout, fresh or other collection root payload is read.']}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False, default=json_default)
    print(json.dumps({'output': str(output), 'failure': failure,
        'sources_unchanged': report['sources_unchanged'],
        'records': [{k: row[k] for k in ('synthetic_tracked', 'wall_ms_instrumented',
                    'cpu_ms_instrumented', 'checks')} for row in records]}))
    if failure:
        raise RuntimeError(failure)
    if not report['sources_unchanged']:
        raise RuntimeError('runtime source changed during profile')


if __name__ == '__main__':
    main()
