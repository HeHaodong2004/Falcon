#!/usr/bin/env python3
"""Place future primary-filesystem roots outside the growing run manifest.

The frozen collector still writes the canonical dataset paths through directory
links. Its existing protocol guard checks a small, explicit storage marker.
Existing roots are never moved, replaced, or adopted from unknown storage.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROTOCOL = 'eventwm-social-outcome-v3'


def publish_json(path, value):
    try:
        with path.open('x') as stream:
            stream.write(json.dumps(value, indent=2) + '\n')
    except FileExistsError:
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f'Existing storage metadata differs: {path}')


def configure(dataset, primary, commit):
    dataset = dataset.resolve()
    if primary.is_symlink():
        raise RuntimeError('Primary storage must be a real directory')
    primary = primary.absolute()
    if dataset == primary or dataset in primary.parents or primary in dataset.parents:
        raise RuntimeError('Primary storage must be outside the canonical dataset')
    if primary.exists() and any(primary.iterdir()) and not (primary/'manifest.json').exists():
        raise RuntimeError('Refuse to use unidentified nonempty storage')
    primary.mkdir(mode=0o700, parents=True, exist_ok=True)
    if primary.stat().st_uid != os.getuid() or primary.stat().st_dev != dataset.stat().st_dev:
        raise RuntimeError('Primary storage owner/filesystem mismatch')
    marker = dict(protocol_version=PROTOCOL, kind='eventwm-storage-partition',
                  canonical_dataset=str(dataset), scientific_commit=commit,
                  placement='Future sequence modulo 3 != 0; independent of outcomes')
    publish_json(primary/'manifest.json', marker)
    (primary/'roots').mkdir(exist_ok=True)
    alias = dataset/'storage_primary'
    if alias.is_symlink():
        if alias.resolve() != primary.resolve():
            raise RuntimeError('Existing primary alias differs')
    elif alias.exists():
        raise RuntimeError('Refuse to replace an existing primary alias')
    else:
        alias.symlink_to(primary, target_is_directory=True)
    layout = dict(version=2, dataset=str(dataset), primary=str(primary),
                  previous_layout='storage_layout.json', alias='storage_primary',
                  primary_sequences='sequence modulo 3 != 0',
                  secondary_sequences='sequence modulo 3 == 0, unchanged existing overflow placement',
                  scientific_commit=commit, minimum_free_bytes=222000000000,
                  preservation='Existing directories and links are never moved or overwritten',
                  archival='Follow roots/* directory links; do not duplicate top-level storage aliases')
    publish_json(dataset/'storage_layout_primary.json', layout)
    return primary


def reserve_root(dataset, primary, root_id):
    if not root_id.startswith('v3m_') or not root_id.replace('_', '').isalnum():
        raise ValueError('Invalid root directory identifier')
    logical = dataset/'roots'/root_id
    physical = primary/'roots'/root_id
    if logical.is_symlink():
        if logical.resolve() != physical.resolve():
            raise RuntimeError(f'Unexpected root link: {logical}')
        return False
    if logical.exists():
        if not logical.is_dir():
            raise RuntimeError(f'Existing root is not a directory: {logical}')
        return False
    if physical.is_symlink() or (physical.exists() and any(physical.iterdir())):
        raise RuntimeError(f'Refuse to adopt unidentified root data: {physical}')
    physical.mkdir(exist_ok=True)
    relative = Path('../storage_primary/roots')/root_id
    try:
        logical.symlink_to(relative, target_is_directory=True)
    except FileExistsError:
        # The live collector may have created its ordinary directory meanwhile.
        if logical.is_symlink() or not logical.is_dir():
            raise RuntimeError(f'Unexpected concurrent root creation: {logical}')
        return False
    if logical.resolve(strict=True) != physical.resolve(strict=True):
        raise RuntimeError('Root storage link resolution mismatch')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--worker-checkout', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    sys.path.insert(0, str(args.worker_checkout.resolve()/'scripts'))
    from eventwm_mass_collection import schedule_spec
    sys.path.insert(0, str(args.evidence.resolve()))
    from stop_dataset_runner import stop_dataset
    manifest = json.loads((dataset/'manifest.json').read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.worker_checkout, text=True).strip()
    if manifest['protocol_version'] != PROTOCOL or manifest['schema_version'] != 4 or manifest['frozen_commit'] != commit:
        raise RuntimeError('Dataset/worker contract mismatch')
    primary = configure(dataset, args.primary, commit)
    source = Path(__file__).resolve()
    identity = dict(time=time.time(), helper=str(source), helper_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    helper_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source.parents[1], text=True).strip(),
                    scientific_commit=commit, dataset=str(dataset), primary=str(primary),
                    stop_helper_sha256=hashlib.sha256((args.evidence/'stop_dataset_runner.py').read_bytes()).hexdigest())
    events = args.evidence/'primary_storage_placement_events.jsonl'
    with (primary/'.placement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (dataset/'primary_storage_provenance.jsonl').open('a') as stream:
            stream.write(json.dumps(identity) + '\n')
        next_reserved = None
        while True:
            manifest = json.loads((dataset/'manifest.json').read_text())
            if manifest['state'] not in ('running', 'prepared_not_started', 'draining_user_stop', 'stopped_by_user'):
                break
            free = shutil.disk_usage(primary).free
            if free < 222000000000:
                event = dict(time=time.time(), action='drain_primary_disk_guard', free_bytes=free)
                with events.open('a') as stream:
                    stream.write(json.dumps(event) + '\n')
                stop_dataset(dataset)
                print(json.dumps(event), flush=True)
                return 2
            start = manifest['next_sequence']
            if next_reserved is None or next_reserved-start < 128:
                first = start if next_reserved is None else next_reserved
                end = start + 512
                created = []
                for sequence in range(first, end):
                    if sequence % 3 == 0:
                        continue
                    root = schedule_spec(sequence, manifest['scenes'], manifest['master_seed'])
                    if reserve_root(dataset, primary, root['root_id']):
                        created.append(dict(sequence=sequence, root_id=root['root_id']))
                next_reserved = end
                event = dict(time=time.time(), action='reserve_future_primary_roots', next_sequence=start,
                             reserved_through=end, created=created, free_bytes=free)
                with events.open('a') as stream:
                    stream.write(json.dumps(event) + '\n')
                print(json.dumps(dict(event, created=len(created))), flush=True)
            if args.once:
                break
            time.sleep(20)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
