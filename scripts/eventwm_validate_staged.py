"""Unchanged scientific persistence/labels/audits with a durable completion receipt."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from collector.branch_storage import write_branches
from collector.core import read
import eventwm_collect as collector


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate(out):
    out = Path(out)
    rid = read(out/'roots/legal_root.json')['root_id']
    timings = {}
    started = time.perf_counter()
    ids = [*[f'b{i:03}' for i in range(33)], 'reverse_1', 'reverse_0', 'truncated_wait']
    records = [read(out/'audits/workers'/wid/'branch.json') for wid in ids]
    timings['load_records'] = time.perf_counter()-started
    started = time.perf_counter()
    write_branches(out, records, compressed=True)
    timings['persistence'] = time.perf_counter()-started
    args = SimpleNamespace(output=out, protocol='v3', storage_profile='production')
    started = time.perf_counter()
    collector.labels(args)
    timings['labels'] = time.perf_counter()-started
    started = time.perf_counter()
    collector.validate(args)
    timings['validation'] = time.perf_counter()-started
    started = time.perf_counter()
    for folder in ('branch_truth', 'labels', 'audits', 'diagnostics'):
        for file in (out/folder).rglob('*'):
            if file.is_file():
                with file.open('rb') as stream:
                    os.fsync(stream.fileno())
    timings['seal'] = time.perf_counter()-started
    durable_json(out/'VALIDATED.json', dict(root_id=rid, timings=timings,
                                          validator='unchanged-eventwm-v3-audits'))
    labels = read(out/'labels/labels.json')
    return dict(root_id=rid, output=str(out), timings=timings,
                retained=sum(l['branch_id'].startswith('b') and l['sr_valid'] for l in labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection-output', required=True, type=Path)
    args = parser.parse_args()
    allowed = args.collection_output.resolve()
    print('__READY__'+json.dumps(dict(pid=os.getpid())), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        out = Path(req['output']).resolve()
        started = time.perf_counter()
        try:
            if allowed not in out.parents:
                raise ValueError('Validation output outside the collection directory')
            answer = dict(result=validate(out), error=None, seconds=time.perf_counter()-started)
        except Exception as exc:
            traceback.print_exc()
            answer = dict(result=None, error=dict(type=type(exc).__name__, message=str(exc)),
                          seconds=time.perf_counter()-started)
            durable_json(out/'VALIDATION_FAILED.json', answer['error'])
        print('__RESULT__'+json.dumps(answer), flush=True)
        if answer['error']:
            return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
