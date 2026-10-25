"""Read-only quiet-host check for this diagnostic mission's timing helpers."""
import os
from pathlib import Path
import hashlib
import json


def accepted_primary_identity():
    """Verify frozen release identities without executing historical helpers."""
    from .common import ROOT, OUT, digest
    primary_path = OUT.parent / 'stage2_segment_20261024/benchmarks/release_single_v1/protocol.json'
    primary = json.loads(primary_path.read_text())
    manifest = json.loads((Path(primary['source']['path']) / 'manifest.json').read_text())
    identity = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest()
    if identity != primary['source']['identity'] or manifest['identity'] != identity:
        raise ValueError('accepted primary source manifest identity mismatch')
    for relative, expected in manifest['files'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('accepted primary source changed: ' + relative)
    checkpoint = Path(primary['checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if digest(checkpoint) != primary['checkpoint_sha256']:
        raise ValueError('accepted primary checkpoint changed')
    if digest(OUT.parent / 'stage2_segment_20261024/benchmark_inputs/manifest.json') != primary['input_manifest_sha256']:
        raise ValueError('accepted primary input manifest changed')
    return dict(source=primary['source'], checkpoint_sha256=primary['checkpoint_sha256'],
                primary_protocol_sha256=digest(primary_path))


def assert_quiet(priority_released, proc=Path('/proc'), own_pid=None):
    if not priority_released:
        raise RuntimeError('parent must release Priority C/D before diagnostic timings')
    own_pid = os.getpid() if own_pid is None else own_pid
    benchmarks = {'eventwm_graph.stage2_segment.benchmark', 'eventwm_graph.stage2_quality.benchmark',
                  'eventwm_graph.stage2.benchmark'}
    for path in proc.glob('[0-9]*/cmdline'):
        if path.parent.name == str(own_pid):
            continue
        try:
            args = path.read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        if any(arg in benchmarks or arg.startswith('scripts.stage2_diagnostic.')
               or ((arg.startswith('scripts.stage2_segment_') or Path(arg).name.startswith('stage2_segment_'))
                   and any(word in arg for word in ('benchmark', 'pilot', 'supplement_tail')))
               or Path(arg).name == 'sustained_service.py'
               or arg.endswith('.sustained_service') for arg in args if arg):
            raise RuntimeError('competing sustained-service/benchmark/diagnostic process: ' + path.parent.name)
