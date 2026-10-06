"""Retention policy only: never affects simulation, legal observations or labels."""
import hashlib
from collector.core import read, write


def compact_workers(output):
    """Remove raw worker duplicates only after proving canonical content equality.

    No failed/incomplete root calls this function. Root images, structured worker
    diagnostics, configs, fingerprints and execution logs remain available.
    """
    from collector.branch_storage import iter_branches, branch_path
    canonical=list(iter_branches(output))
    labels=read(output/'labels/labels.json')
    assert {b['branch_id'] for b in canonical}=={b['branch_id'] for b in labels}
    pending=[]
    for b in canonical:
        path=output/'audits/workers'/b['branch_id']/'branch.json'
        if read(path)!=b:
            raise RuntimeError(f'Canonical/worker mismatch; refusing compaction: {path}')
        pending.append((path,dict(path=str(path.relative_to(output)),bytes=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    # Write the removal inventory atomically before deleting any duplicate.
    report=dict(profile='production',canonical_branch_path=str(branch_path(output).relative_to(output)),
                canonical_branch_encoding='jsonl-gzip-v1' if branch_path(output).suffix=='.gz' else 'jsonl-v1',root_images_retained=(output/'legal_observations/history_images.npz').exists(),branch_images_retained=False,
                removed_worker_records=[row for _,row in pending],
                removed_bytes=sum(row['bytes'] for _,row in pending))
    write(output/'audits/storage.json',report)
    for path,_ in pending: path.unlink()
    return report


def write_compact_json(path, value):
    """Atomic JSON metadata; whitespace only differs from core.write."""
    import json
    from pathlib import Path
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    temporary.replace(path)


def write_prefix_truth(output, rows, compressed=False):
    """Preserve all prefix evidence, including rejected-root evidence, losslessly."""
    import gzip
    import json
    from pathlib import Path
    output=Path(output)
    if not compressed:
        return write(output/'roots/prefix_truth.json',rows)
    path=output/'roots/prefix_truth.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    payload=(json.dumps(rows,indent=2,allow_nan=False)+'\n').encode()
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as packed:
            packed.write(payload)
    temporary.replace(path)
