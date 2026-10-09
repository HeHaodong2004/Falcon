"""Lossless evidence storage, restricted to this release's new staging roots."""
import gzip
import hashlib
import lzma
import os
from pathlib import Path
import shutil
from collector.core import read
from eventwm_validate_staged import durable_json

ARCHIVE=Path('/home/haodong/workspace/projects/eventwm-falcon/datasets/raw/eventwm_v4_300k_archive')

def capacity(output):
    a=shutil.disk_usage(output).free
    b=shutil.disk_usage(ARCHIVE.parent).free
    return a+b if os.stat(output).st_dev!=os.stat(ARCHIVE.parent).st_dev else a

def compress_prefix(output, rows, compressed=False):
    # Called only by the additive worker wrapper. Identical decompressed bytes.
    if not compressed:
        from collector.core import write
        return write(Path(output)/'roots/prefix_truth.json',rows)
    import json
    path=Path(output)/'roots/prefix_truth.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.gz.tmp')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0) as stream:
            stream.write((json.dumps(rows,indent=2,allow_nan=False)+'\n').encode())
    temporary.replace(path)

def compact_validated(path,known_encoding=None):
    """Existing exact canonical-equality proof, then stronger gzip encoding."""
    marker=Path(path)/'audits/encoding.json'
    if marker.exists():
        receipt=read(marker)
        if receipt.get('compressed_sha256'):
            actual=hashlib.sha256()
            with (Path(path)/'branch_truth/branches.jsonl.gz').open('rb') as stream:
                while data:=stream.read(1024*1024):actual.update(data)
            if actual.hexdigest()!=receipt['compressed_sha256']:raise RuntimeError('Packed canonical file changed')
            return
        expected=receipt['decoded_sha256']
        actual=hashlib.sha256()
        with gzip.open(Path(path)/'branch_truth/branches.jsonl.gz','rb') as stream:
            while data:=stream.read(1024*1024):actual.update(data)
        if actual.hexdigest()!=expected:raise RuntimeError('Packed canonical data changed')
        return
    from collector.storage import compact_workers
    if known_encoding is not None:
        p=Path(path)/'branch_truth/branches.jsonl.gz'
        before=hashlib.sha256()
        with p.open('rb') as stream:
            while data:=stream.read(1024*1024):before.update(data)
        if before.hexdigest()!=known_encoding['compressed_sha256']:
            raise RuntimeError('Canonical file changed before compaction')
    compact_workers(path)
    if known_encoding is not None:
        after=hashlib.sha256()
        with p.open('rb') as stream:
            while data:=stream.read(1024*1024):after.update(data)
        if before.digest()!=after.digest():raise RuntimeError('Canonical file changed during equality proof')
        durable_json(marker,known_encoding)
        return
    p=Path(path)/'branch_truth/branches.jsonl.gz'
    temporary=p.with_suffix('.gz.repack')
    # Streaming avoids holding a second full branch stream in publisher RAM.
    before=hashlib.sha256()
    with gzip.open(p,'rb') as source,temporary.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0) as target:
            while data:=source.read(1024*1024):
                before.update(data);target.write(data)
    after=hashlib.sha256()
    with gzip.open(temporary,'rb') as stream:
        while data:=stream.read(1024*1024):after.update(data)
    if before.digest()!=after.digest():raise RuntimeError('Canonical branch recompression mismatch')
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    temporary.replace(p)
    durable_json(Path(path)/'audits/encoding.json',dict(branch_encoding='jsonl-gzip-v1',gzip_level=6,decoded_sha256=before.hexdigest()))

def pack_rejected(output, root):
    """Keep rejected prefix evidence losslessly in explicit XZ storage.

    Accepted-root canonical training paths remain gzip. Rejected roots never
    enter training. Packing consumes no privileged outcome for admission.
    """
    path=Path(output)/root['attempts'][-1]['path']
    p=path/'roots/prefix_truth.json.gz'
    receipt=path/'audits/rejected_storage.json'
    if p.exists():
        target=p.with_suffix('.xz');temporary=target.with_suffix('.xz.tmp')
        before=hashlib.sha256()
        with gzip.open(p,'rb') as source,lzma.open(temporary,'wb',preset=1) as packed:
            while data:=source.read(1024*1024):before.update(data);packed.write(data)
        after=hashlib.sha256()
        with lzma.open(temporary,'rb') as source:
            while data:=source.read(1024*1024):after.update(data)
        if before.digest()!=after.digest():raise RuntimeError('Rejected prefix compression mismatch')
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        temporary.replace(target)
        durable_json(receipt,dict(original='roots/prefix_truth.json.gz',path='roots/prefix_truth.json.xz',decoded_sha256=before.hexdigest(),encoding='json-xz-v1',retained_all_evidence=True))
        p.unlink()
    spill_root(output,root)
    return dict(packed=True)

def spill_root(output, root):
    path=Path(output)/root['attempts'][-1]['path']
    if path.is_symlink() or (path/'COMPLETE.json').exists():return
    if os.stat(path).st_dev==os.stat(ARCHIVE.parent).st_dev:return
    home=shutil.disk_usage(ARCHIVE.parent).free;native=shutil.disk_usage(output).free
    if home<100e9 or home<native*.45:return
    dest=ARCHIVE/root['attempts'][-1]['path'];temporary=dest.with_name(dest.name+'.copying')
    receipt=Path(output)/'spill_receipts'/f"{root['root_id']}_{root['attempts'][-1]['number']}.json"
    if dest.exists():
        if not receipt.exists():raise RuntimeError('Unexpected archive path: '+str(dest))
    else:
        if temporary.exists():shutil.rmtree(temporary)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(path,temporary)
        for f in path.rglob('*'):
            if not f.is_file():continue
            target=temporary/f.relative_to(path)
            if hashlib.sha256(f.read_bytes()).digest()!=hashlib.sha256(target.read_bytes()).digest():raise RuntimeError('Spill hash mismatch')
            with target.open('rb') as stream:os.fsync(stream.fileno())
        temporary.rename(dest)
        durable_json(receipt,dict(source=str(path),destination=str(dest),state='verified_copy'))
    # Rename staging first; receipt enables recovery across either interruption.
    original=path.with_name(path.name+'.spilling')
    path.rename(original)
    path.symlink_to(dest,target_is_directory=True)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    shutil.rmtree(original)
    durable_json(receipt,dict(source=str(path),destination=str(dest),state='linked'))

def recover_spills(output):
    for receipt in (Path(output)/'spill_receipts').glob('*.json'):
        row=read(receipt);path=Path(row['source']);dest=Path(row['destination']);original=path.with_name(path.name+'.spilling')
        if not path.exists() and original.exists() and dest.exists():path.symlink_to(dest,target_is_directory=True)
        if path.is_symlink() and original.exists():shutil.rmtree(original)
