"""Verify an explicitly supplied external release checkpoint without loading it."""
import argparse
import hashlib
import json
from pathlib import Path


MANIFEST = Path(__file__).with_name('artifact_manifest.json')


def verify_checkpoint(path, role='primary'):
    expected = json.loads(MANIFEST.read_text())['models'][role]
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if path.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
        raise ValueError('Checkpoint does not match the frozen role: ' + role)
    return dict(role=role, path=str(path), sha256=digest.hexdigest(), verified=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--role', choices=list(json.loads(MANIFEST.read_text())['models']), default='primary')
    args = parser.parse_args()
    print(json.dumps(verify_checkpoint(args.checkpoint, args.role), indent=2))


if __name__ == '__main__':
    main()
