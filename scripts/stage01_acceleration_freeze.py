#!/usr/bin/env python
"""Freeze a committed accelerated candidate without installing or starting it."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eventwm_graph.config import digest, file_digest, read_json, write_json
from eventwm_graph.partial.acceleration import source_identity
from eventwm_graph.partial.native_batch import DEFAULT_BUILD, load_backend


def _verify_payload(destination, expected, manifest_exists):
    members = list(destination.rglob('*'))
    if destination.is_symlink() or any(path.is_symlink() for path in members):
        raise ValueError('Frozen candidate contains symlinks')
    actual = {str(path.relative_to(destination)) for path in members if path.is_file()}
    inventory = set(expected) | ({'manifest.json'} if manifest_exists else set())
    if actual != inventory:
        raise ValueError('Frozen candidate inventory changed')
    if any(file_digest(destination/name) != sha for name, sha in expected.items()):
        raise ValueError('Frozen candidate payload changed')


def _verify_existing(destination, expected_manifest, current_commit):
    existing = read_json(destination/'manifest.json')
    if (set(existing) != set(expected_manifest) | {'repository_commit'} or
            any(existing.get(name) != value for name, value in expected_manifest.items()) or
            existing.get('automatic_training') is not False or existing.get('service_started') is not False):
        raise ValueError('Frozen candidate manifest changed')
    commit = existing['repository_commit']
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Frozen candidate repository commit is invalid')
    # A later documentation-only commit may have the same source identity.
    # Preserve the original receipt and verify its own source commitment.
    if commit != current_commit:
        for name, sha in expected_manifest['original_sources'].items():
            committed = subprocess.run(['git', 'show', commit+':'+name], cwd=ROOT, capture_output=True)
            if committed.returncode or digest_bytes(committed.stdout) != sha:
                raise ValueError('Frozen candidate repository commitment changed: '+name)
    _verify_payload(destination, expected_manifest['frozen_sources'], manifest_exists=True)


def freeze(output):
    import os
    versions = source_identity()
    tracked = subprocess.check_output(['git', 'ls-files', 'eventwm_graph'], cwd=ROOT, text=True).splitlines()
    paths = [ROOT/p for p in tracked if Path(p).suffix in ('.py', '.cpp')]
    for name in ('acceleration.py', 'fast_observer.py', 'fast_graph.py', 'fast_validation.py', 'native_batch.cpp'):
        if ROOT/'eventwm_graph/partial'/name not in paths:
            raise ValueError('Commit accelerated sources before freezing')
    paths += list((ROOT/'scripts').glob('stage01_*.py'))
    paths += [ROOT/'scripts/build_stage01_native_batch.py', ROOT/'scripts/collector/observation.py',
              ROOT/'scripts/collector/__init__.py', ROOT/'configs/stage0_scenes.json']
    paths = sorted(set(p for p in paths if p.exists() and '__pycache__' not in p.parts))
    sources = {str(p.relative_to(ROOT)): file_digest(p) for p in paths}
    for path in paths:
        name = str(path.relative_to(ROOT))
        committed = subprocess.run(['git', 'show', 'HEAD:'+name], cwd=ROOT, capture_output=True)
        if committed.returncode or digest_bytes(committed.stdout) != sources[name]:
            raise ValueError('Freeze requires committed source: '+name)
    _, _, native = load_backend()
    build = Path(os.environ.get('EVENTWM_NATIVE_BATCH_BUILD', str(DEFAULT_BUILD))).resolve()
    if read_json(build/'build.json') != native:
        raise ValueError('Native build metadata changed after validation')
    if Path(native['extension']).name != native['extension']:
        raise ValueError('Native extension name must be a filename')
    identity = digest(dict(sources=sources, native_extension=native['extension_sha256']))
    destination = Path(output).resolve()/identity
    old = 'REPO = Path(__file__).resolve().parents[1]'
    config_text = (ROOT/'eventwm_graph/config.py').read_text()
    if config_text.count(old) != 1:
        raise ValueError('Unknown asset-root binding')
    shim = ('REPO = Path('+repr(str(ROOT))+')\nimport sys as _bundle_sys\n'
            'if str(REPO) not in _bundle_sys.path: _bundle_sys.path.append(str(REPO))')
    config_text = config_text.replace(old, shim)
    expected = dict(sources)
    expected['eventwm_graph/config.py'] = digest_bytes(config_text.encode())
    expected.update({'native_batch/'+name: file_digest(build/name)
                     for name in ('build.json', native['extension'])})
    if expected['native_batch/'+native['extension']] != native['extension_sha256']:
        raise ValueError('Native extension changed after validation')
    native_dir = destination/'native_batch'
    current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = dict(source_id=identity, implementation=versions,
        original_sources=sources, repository_assets=str(ROOT),
        external_sources={p: file_digest(ROOT/p) for p in ('falcon/additional_action.py', 'falcon/additional_metric.py')},
        frozen_sources=expected, native_build_environment=str(native_dir),
        automatic_training=False, service_started=False)
    if destination.exists():
        _verify_existing(destination, manifest, current_commit)
        return destination
    for path in paths:
        target = destination/path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (destination/'eventwm_graph/config.py').write_text(config_text)
    native_dir.mkdir()
    for name in ('build.json', native['extension']):
        shutil.copyfile(build/name, native_dir/name)
    _verify_payload(destination, expected, manifest_exists=False)
    write_json(destination/'manifest.json', dict(manifest, repository_commit=current_commit))
    for path in destination.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    return destination


def digest_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    print(freeze(p.parse_args().output))
