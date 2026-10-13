"""Lossless branch truth storage; old plain JSONL remains readable."""
import gzip
import io
import json
from pathlib import Path

PLAIN = 'branches.jsonl'
COMPRESSED = 'branches.jsonl.gz'


def branch_path(output):
    directory = Path(output) / 'branch_truth'
    paths = [p for p in (directory / PLAIN, directory / COMPRESSED) if p.exists()]
    if not paths:
        raise FileNotFoundError(f'Canonical branch stream missing: {directory}')
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one canonical branch stream, found {len(paths)}: {directory}')
    return paths[0]


def iter_branches(output):
    path = branch_path(output)
    opener = gzip.open if path.name == COMPRESSED else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def write_branches(output, records, compressed=False):
    directory = Path(output) / 'branch_truth'
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in (PLAIN, COMPRESSED)):
        raise ValueError('Refuse to replace existing canonical branch truth')
    path = directory / (COMPRESSED if compressed else PLAIN)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as raw:
        if compressed:
            # Fixed header metadata; every original JSON byte survives decompression.
            packed = gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0)
            stream = io.TextIOWrapper(packed, encoding='utf-8', newline='\n')
        else:
            stream = io.TextIOWrapper(raw, encoding='utf-8', newline='\n')
        with stream:
            for row in records:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    temporary.replace(path)
    return path
