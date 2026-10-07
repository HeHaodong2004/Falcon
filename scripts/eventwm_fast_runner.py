#!/usr/bin/env python3
"""Speed up atomic orchestration checkpoints; execute the frozen collector unchanged.

The worker checkout retains its clean commit, source hashes, simulation, audits,
JSON readers and data schema. This separately versioned launcher overrides only
mass.save and records its source/dependency identity in orchestration_history.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def native_finite(value):
    """Reject nonfinite numbers; use stdlib for types needing its exact behavior."""
    kind = type(value)
    if kind is float:
        if not math.isfinite(value):
            raise ValueError('Out of range float values are not JSON compliant')
        return True
    if kind in (str, int, bool, type(None)):
        return True
    if kind in (list, tuple):
        return all(native_finite(item) for item in value)
    if kind is dict:
        return all(type(key) is str and native_finite(item) for key, item in value.items())
    return False


def encode_metadata(value, backend):
    if native_finite(value):
        try:
            return backend.dumps(value) + b'\n'
        except TypeError:
            pass  # large integers, escaped lone surrogates, etc. keep stdlib behavior
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write_metadata(path, value, backend):
    payload = encode_metadata(value, backend)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)


def install_overrides(mass, backend, identity):
    original_init = mass.init

    def traced_init(args):
        manifest = original_init(args)
        manifest.setdefault('orchestration_history', []).append(dict(identity, started=time.time()))
        return manifest

    def fast_save(output, manifest):
        write_metadata(output / 'manifest.json', manifest, backend)
        write_metadata(output / 'aggregate.json', mass.aggregate(manifest), backend)

    mass.init = traced_init
    mass.save = fast_save


def main():
    launcher = argparse.ArgumentParser(add_help=False)
    launcher.add_argument('--worker-checkout', type=Path, required=True)
    launcher.add_argument('--json-vendor', type=Path, required=True)
    options, rest = launcher.parse_known_args()
    worker = options.worker_checkout.resolve()
    vendor = options.json_vendor.resolve()
    sys.path.insert(0, str(vendor))
    import orjson
    if orjson.__version__ != '3.10.18' or vendor not in Path(orjson.__file__).resolve().parents:
        raise RuntimeError('Use the isolated, pinned JSON backend')
    sys.path.insert(0, str(worker / 'scripts'))
    import eventwm_mass_collection as mass
    if mass.REPO.resolve() != worker:
        raise RuntimeError('Worker checkout import mismatch')
    args = mass.parser().parse_args(rest)
    args.gpus = [int(value) for value in args.gpus.split(',')]
    if args.action != 'run':
        raise ValueError('This launcher only supports run/resume')
    source = Path(__file__).resolve()
    launcher_repo = source.parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=launcher_repo, text=True).strip():
        raise RuntimeError('Launcher checkout must be clean')
    package_files = sorted(path for path in (vendor / 'orjson').rglob('*') if path.is_file() and '__pycache__' not in path.parts)
    identity = dict(launcher_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=launcher_repo, text=True).strip(),
                    launcher_path=str(source), launcher_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    worker_checkout=str(worker), worker_commit=mass.git_head(), checkpoint_encoding='JSON, identical decoded values',
                    backend_version=orjson.__version__, backend_directory=str(vendor),
                    backend_sha256={str(path.relative_to(vendor)): hashlib.sha256(path.read_bytes()).hexdigest() for path in package_files},
                    overrides=['mass.save: atomic manifest/aggregate JSON encoding only'])
    install_overrides(mass, orjson, identity)
    return mass.run(args)


if __name__ == '__main__':
    raise SystemExit(main())
