#!/usr/bin/env python3
"""Read-only verification and loader smoke checks for the production archive."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / 'archive/eventwm_v4_300k'


def read(path):
    return json.loads(Path(path).read_bytes())


def db(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    connection.execute('PRAGMA query_only=ON')
    connection.execute('BEGIN')
    return connection


def verify():
    manifest = read(ARCHIVE / 'production_manifest.json')
    release = read(REPO / 'configs/eventwm_pipeline_release.json')
    for name, expected in release['implementation_hashes'].items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected, name
    operations = ARCHIVE / 'operations'
    ops = read(operations / 'configs/eventwm_300k_operations.json')
    for name, expected in ops['hashes'].items():
        assert hashlib.sha256((operations / name).read_bytes()).hexdigest() == expected, name
    epoch_files = 0
    for epoch in manifest['source_epochs']:
        source = read(ARCHIVE / epoch['hash_manifest'])
        paths = sorted(source['implementation_hashes'])
        request = ''.join(epoch['source_commit'] + ':' + p + '\n' for p in paths).encode()
        result = subprocess.run(['git', '-C', str(REPO), 'cat-file', '--batch'],
                                input=request, stdout=subprocess.PIPE, check=True)
        stream = io.BytesIO(result.stdout)
        for name in paths:
            header = stream.readline().decode().split()
            assert len(header) == 3 and header[1] == 'blob', (epoch['source_commit'], name)
            raw = stream.read(int(header[2])); assert stream.read(1) == b'\n'
            assert hashlib.sha256(raw).hexdigest() == source['implementation_hashes'][name], name
        epoch_files += len(paths)
    for name, evidence in manifest['recorded_artifacts'].items():
        assert hashlib.sha256((ARCHIVE / 'records' / name).read_bytes()).hexdigest() == evidence['sha256'], name
    assert sum(e['queries'] for e in manifest['source_epochs']) == manifest['retained_queries']
    assert sum(e['roots'] for e in manifest['source_epochs']) == manifest['roots']
    assert sum(manifest['sr_distribution'].values()) == manifest['retained_queries']
    assert read(ARCHIVE / 'records/successor_audit.json')['queries'] == manifest['retained_queries']
    return dict(passed=True, final_source_files=len(release['implementation_hashes']),
                operational_files=len(ops['hashes']), source_epochs=len(manifest['source_epochs']),
                epoch_file_hashes=epoch_files, retained_queries=manifest['retained_queries'])


def inspect(index):
    with db(index) as connection:
        return dict(queries=connection.execute('SELECT COUNT(*) FROM examples').fetchone()[0],
                    roots=connection.execute('SELECT COUNT(*) FROM roots').fetchone()[0],
                    scenes=connection.execute('SELECT COUNT(*) FROM scenes').fetchone()[0],
                    sr={str(k):n for k,n in connection.execute('SELECT sr,COUNT(*) FROM examples GROUP BY sr')},
                    roles={k:n for k,n in connection.execute('SELECT role,SUM(queries) FROM roots GROUP BY role')})


def load_smoke(index, supervision, limit):
    sys.path.insert(0, str(REPO / 'scripts'))
    import collector.core as core
    selected = []
    with db(index) as connection:
        for role in ('train', 'natural_reference', 'unseen_evaluation'):
            for sr in (0, 1):
                selected.extend(connection.execute('SELECT r.dataset,r.path,r.root_id,e.query_id,e.sr '
                    'FROM examples e JOIN roots r USING(root_id) WHERE r.role=? AND e.sr=? LIMIT ?',
                    (role, sr, max(1, limit // 6))).fetchall())
    selected = selected[:limit]
    assert selected, 'No retained queries in index'
    queries = horizons = 0
    for dataset, path, root_id, query_id, sr in selected:
        root = Path(dataset) / path
        opened = []
        original = core.read
        def observed(path):
            opened.append(Path(path).relative_to(root).as_posix())
            return original(path)
        core.read = observed
        try:
            inputs = core.load_wm_input(root, query_id)
        finally:
            core.read = original
        assert set(opened) == {'legal_observations/history.json', 'legal_observations/image_storage.json', 'queries/queries.json'}
        assert set(inputs) == {'history', 'history_images', 'query'}
        assert inputs['query']['query_id'] == query_id and inputs['query']['schema_version'] == 3
        assert len(inputs['query']['executable_controls']) * inputs['query']['control_dt'] == inputs['query']['planned_duration']
        labels = {r['query_id']:r for r in read(root / 'labels/labels.json')}
        label = labels[query_id]
        assert label['sr_valid'] and label['sr_label'] == sr and label['social_outcome_valid']
        assert not label['static_feasibility_violation']
        if sr == 0:
            assert label['native_robot_human_contact']
        else:
            assert label['query_completed']
        if supervision:
            with gzip.open(Path(supervision) / (root_id + '.jsonl.gz'), 'rt') as stream:
                successor = next(json.loads(line) for line in stream if json.loads(line)['query_id'] == query_id)
            assert successor['schema'] == 'eventwm-successor-horizons-v1'
            assert successor['supervision_only'] and successor['sr_label'] == sr
            assert [h['horizon_seconds'] for h in successor['horizons']] == [.25, .5, 1., 2., 3.]
            horizons += len(successor['horizons'])
        queries += 1
    return dict(passed=True, loaded_queries=queries, loaded_horizon_targets=horizons,
                roles=['train', 'natural_reference', 'unseen_evaluation'], sr_classes=[0, 1],
                legal_loader_opens_only_history_query_image_metadata=True, raw_data_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'inspect', 'load-smoke'])
    parser.add_argument('--index', type=Path)
    parser.add_argument('--supervision', type=Path)
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()
    if args.action != 'verify' and not args.index:
        parser.error('--index is required for dataset checks')
    if not 1 <= args.limit <= 60:
        parser.error('--limit must be between 1 and 60')
    result = verify() if args.action == 'verify' else inspect(args.index) if args.action == 'inspect' else load_smoke(args.index, args.supervision, args.limit)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
