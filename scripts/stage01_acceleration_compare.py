#!/usr/bin/env python
"""Exact scientific-record differential for two fixed-workload result files."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eventwm_graph.config import file_digest, read_json, write_json


def compare(reference, candidate, output):
    old, new = read_json(reference), read_json(candidate)
    a = {r['seed']: r for r in old['results']}
    b = {r['seed']: r for r in new['results']}
    errors, rows = [], []
    if set(a) != set(b):
        errors.append('attempt_seed_set')
    for seed in sorted(set(a) & set(b)):
        x, y = a[seed], b[seed]
        checks = {'attempt_status': x['status'] == y['status']}
        if x['status'] == y['status'] == 'rejected':
            checks['rejection_reason'] = x['error'] == y['error']
        elif x['status'] == y['status'] == 'sealed':
            px, py = Path(x['folder']), Path(y['folder'])
            checks['outcomes'] = x['outcomes'] == y['outcomes']
            for member in ('legal_root.json.gz', 'queries.json.gz', 'future_observations.json.gz',
                           'privileged_step_labels.json.gz', 'outcomes.json.gz', 'privileged_initialization.json.gz'):
                with gzip.open(px/member, 'rt') as stream:
                    vx = json.load(stream)
                with gzip.open(py/member, 'rt') as stream:
                    vy = json.load(stream)
                if member == 'legal_root.json.gz':
                    vx.pop('version'); vy.pop('version')
                elif member == 'privileged_initialization.json.gz':
                    vx['identity'].pop('version'); vy['identity'].pop('version')
                checks[member] = vx == vy
            with np.load(px/'privileged_future.npz', allow_pickle=False) as vx, np.load(py/'privileged_future.npz', allow_pickle=False) as vy:
                checks['shared_native_future_arrays'] = set(vx.files) == set(vy.files) and all(np.array_equal(vx[k], vy[k]) for k in vx.files)
            if 'root_prior' in x and 'root_prior' in y:
                vx, vy = dict(x['root_prior']), dict(y['root_prior'])
                vx.pop('root_id'); vy.pop('root_id')
                checks['sampling_receipt'] = vx == vy
        else:
            checks['no_worker_errors'] = False
        for key, passed in checks.items():
            if not passed:
                errors.append(str(seed)+':'+key)
        rows.append(dict(seed=seed, checks=checks, passed=all(checks.values())))
    result = dict(schema='stage01_exact_scientific_differential_1', status='PASS' if not errors else 'FAIL',
                  reference=dict(path=str(Path(reference).resolve()), sha256=file_digest(reference)),
                  candidate=dict(path=str(Path(candidate).resolve()), sha256=file_digest(candidate)),
                  attempts=len(a), sealed_roots=sum(r['status'] == 'sealed' for r in a.values()),
                  validated_outcomes=sum(r.get('outcomes', 0) for r in a.values()),
                  permitted_metadata_differences=['producer version maps and derived immutable root/future identities'],
                  numerical_tolerance='none: exact Python records and native future arrays', results=rows, errors=errors)
    write_json(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True); p.add_argument('--candidate', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    r = compare(args.reference, args.candidate, args.output)
    print(json.dumps({k: r[k] for k in ('status', 'attempts', 'sealed_roots', 'validated_outcomes', 'errors')}, indent=2))
    if r['status'] != 'PASS':
        raise SystemExit(1)
