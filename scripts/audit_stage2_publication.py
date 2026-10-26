"""Read-only audit of every Git blob reachable from a publication ref.

Does not inspect working-tree data or print credential contents. The size and
binary checks are publication hygiene, not scientific evaluation criteria.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess


def audit(repo, ref):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])

    objects = {}
    for line in git('rev-list', '--objects', ref).decode().splitlines():
        oid, _, path = line.partition(' ')
        objects.setdefault(oid, path)
    headers = subprocess.check_output(
        ['git', '-C', str(repo), 'cat-file',
         '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
        input=('\n'.join(objects) + '\n').encode())
    blobs = []
    for line in headers.decode().splitlines():
        oid, kind, size = line.split()
        if oid in objects and kind == 'blob':
            blobs.append((oid, int(size), objects[oid]))
    patterns = {
        'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'github_classic': rb'gh[pousr]_[A-Za-z0-9]{30,}',
        'github_fine_grained': rb'github_pat_[A-Za-z0-9_]{40,}',
        'aws_access_key': rb'AKIA[0-9A-Z]{16}',
        'openai_key': rb'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{30,}',
        'slack_token': rb'xox[baprs]-[0-9A-Za-z-]{20,}',
    }
    compiled = {name: re.compile(value) for name, value in patterns.items()}
    bad_assets, binary, findings = [], [], []
    extensions = ('.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.npz',
                  '.npy', '.pkl', '.pickle', '.zip', '.tar', '.gz', '.so', '.whl')
    proc = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for oid, size, path in blobs:
            proc.stdin.write((oid + '\n').encode())
            proc.stdin.flush()
            header = proc.stdout.readline().decode().split()
            assert header[0] == oid and int(header[2]) == size
            data = proc.stdout.read(size)
            assert len(data) == size and proc.stdout.read(1) == b'\n'
            if b'\x00' in data:
                binary.append(path)
            if path.lower().endswith(extensions):
                bad_assets.append(path)
            for name, regex in compiled.items():
                if regex.search(data):
                    findings.append({'path': path, 'pattern': name, 'object': oid})
    finally:
        proc.stdin.close()
        proc.stdout.close()
        assert proc.wait() == 0
    oversized = [dict(object=oid, bytes=size, path=path)
                 for oid, size, path in blobs if size > 10 * 1024 ** 2]
    return dict(
        scope='ALL objects reachable from ref, including inherited history',
        ref=ref, head=git('rev-parse', ref).decode().strip(),
        objects=len(objects), blobs=len(blobs),
        total_blob_bytes=sum(size for _, size, _ in blobs),
        largest_20=[dict(object=oid, bytes=size, path=path)
                    for oid, size, path in sorted(blobs, key=lambda x: -x[1])[:20]],
        over_10MiB=oversized, asset_patterns=bad_assets,
        binary_nul_paths=binary, credential_pattern_findings=findings,
        credential_scan_limit='Pattern scan is not proof that arbitrary secrets are absent.',
        passed=not (oversized or bad_assets or binary or findings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--ref', default='HEAD')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = audit(Path(args.repo), args.ref)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in
                      ('head', 'objects', 'blobs', 'total_blob_bytes', 'passed')}))
    raise SystemExit(0 if report['passed'] else 1)
