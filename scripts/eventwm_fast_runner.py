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

SUPPORTED_TERMINAL_CACHE_COMMIT = 'dd1400e0121a8353d79be12d18f98c15bb17e43f'
NONRETRYABLE_FAILURES = frozenset({
    'requested_legal_stratum_unavailable', 'insufficient_static_feasible_queries',
    'prefix_terminated_before_root', 'episode_preparation_unavailable',
    'no_retained_valid_queries', 'collection_failure_unresolved',
})


def _immutable(*args, **kwargs):
    raise TypeError('Validated terminal metadata is immutable')


class _FrozenDict(dict):
    __slots__ = ()

    def __new__(cls, value):
        if type(value) is not dict or not native_finite(value):
            raise TypeError('Only validated native dictionaries can be cached')
        result = dict.__new__(cls)
        dict.update(result, ((key, _freeze(item)) for key, item in value.items()))
        return result

    def __init__(self, value):
        pass  # Reinitialization must not provide a mutation route.

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable


class _FrozenList(list):
    __slots__ = ()

    def __new__(cls, value):
        if type(value) not in (list, tuple) or not native_finite(value):
            raise TypeError('Only validated native sequences can be cached')
        result = list.__new__(cls)
        list.extend(result, (_freeze(item) for item in value))
        return result

    def __init__(self, value):
        pass

    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _immutable


def _freeze(value):
    if type(value) is dict:
        return _FrozenDict(value)
    if type(value) in (list, tuple):
        return _FrozenList(value)
    return value


def freeze_terminal_roots(manifest):
    """Cache only roots the pinned coordinator never mutates or retries again.

    Completed roots and the listed nonretryable rejections are terminal in the
    supported source commit. Running/pending roots and timeout retries retain
    their original objects. A restart loads ordinary JSON objects afresh.
    """
    for index, root in enumerate(manifest['roots']):
        if type(root) is _FrozenDict:
            continue
        terminal = root['state'] == 'completed' or (
            root['state'] == 'failed' and root.get('attempts') and
            root['attempts'][-1].get('reason') in NONRETRYABLE_FAILURES)
        if terminal and native_finite(root):
            manifest['roots'][index] = _freeze(root)


def native_finite(value):
    """Reject nonfinite numbers; use stdlib for types needing its exact behavior."""
    pending = [value]
    native = True
    while pending:
        item = pending.pop()
        kind = type(item)
        if kind is _FrozenDict or kind is _FrozenList:
            continue  # Constructors validate every value; ordinary mutation raises.
        if kind is str or kind is int or kind is bool or item is None:
            continue
        if kind is float:
            if not math.isfinite(item):
                raise ValueError('Out of range float values are not JSON compliant')
        elif kind is list or kind is tuple:
            pending.extend(item)
        elif kind is dict:
            if any(type(key) is not str for key in item):
                native = False
            pending.extend(item.values())
        else:
            native = False
    return native


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
        if identity.get('worker_commit') == SUPPORTED_TERMINAL_CACHE_COMMIT:
            freeze_terminal_roots(manifest)
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
                    terminal_metadata_cache='immutable-closed-roots-v1' if mass.git_head() == SUPPORTED_TERMINAL_CACHE_COMMIT else None,
                    overrides=['mass.save: atomic manifest/aggregate JSON encoding only'])
    install_overrides(mass, orjson, identity)
    return mass.run(args)


if __name__ == '__main__':
    raise SystemExit(main())
