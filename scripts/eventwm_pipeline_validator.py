"""Frozen audits with per-root decoded-record caching and proved compaction.

No audit/label code changes. All persisted canonical records must still pass the
original raw-worker equality proof before a VALIDATED receipt is published.
"""
from pathlib import Path
import sys
import hashlib
import json
import gzip
import time
from collector import storage as duplicate_storage
from types import SimpleNamespace
import eventwm_collect as collector
import eventwm_validate_staged as staged
from collector import branch_storage
from collector.core import read
from eventwm_pipeline_io import install_indexed_reads
from eventwm_300k_storage import compact_validated

output=Path(sys.argv[sys.argv.index('--collection-output')+1]).resolve()
install_indexed_reads(collector,SimpleNamespace(read=read),{},output)
original_write=branch_storage.write_branches
original_branches=collector.branches
original_validate=staged.validate
original_receipt=staged.durable_json
cached={};receipts={};raw_records={};encodings={}
original_staged_read=staged.read
original_duplicate_read=duplicate_storage.read

def read_captured(path):
    p=Path(path)
    if p.name=="branch.json" and p.parent.parent.name=="workers":
        raw=p.read_bytes();value=json.loads(raw)
        raw_records[p.resolve()]=(value,hashlib.sha256(raw).digest())
        return value
    return original_staged_read(path)

def read_duplicate(path):
    p=Path(path).resolve()
    if p in raw_records:
        value,expected=raw_records[p]
        if hashlib.sha256(p.read_bytes()).digest()!=expected:
            raise RuntimeError("Raw native record changed during validation: "+str(p))
        return value
    return original_duplicate_read(path)

def write_cached(out,records,compressed=False):
    # Keep the original JSON serializer and newline behavior. Observe exact
    # bytes entering gzip, changing only its lossless compression level.
    packed=[]
    class CanonicalGzip(gzip.GzipFile):
        def __init__(self,*args,**kwargs):
            kwargs['compresslevel']=6
            super().__init__(*args,**kwargs)
            self.decoded_digest=hashlib.sha256();packed.append(self)
        def write(self,data):
            self.decoded_digest.update(data)
            return super().write(data)
    original_gzip=branch_storage.gzip
    branch_storage.gzip=SimpleNamespace(GzipFile=CanonicalGzip,open=gzip.open)
    try:answer=original_write(out,records,compressed=compressed)
    finally:branch_storage.gzip=original_gzip
    if compressed:
        assert len(packed)==1
        digest=hashlib.sha256()
        with answer.open('rb') as stream:
            while data:=stream.read(1024*1024):digest.update(data)
        encodings[Path(out).resolve()]=dict(branch_encoding='jsonl-gzip-v1',gzip_level=6,
            decoded_sha256=packed[0].decoded_digest.hexdigest(),compressed_sha256=digest.hexdigest(),
            proof='Original serializer observed byte-for-byte; unchanged canonical/raw equality gate')
    cached.clear();cached[Path(out).resolve()]=records
    return answer

def branches(out):
    records=cached.get(Path(out).resolve())
    return records if records is not None else original_branches(out)

def receipt(path,value):
    if Path(path).name=='VALIDATED.json':receipts[Path(path)]=value
    else:original_receipt(path,value)

def validate(out):
    try:
        answer=original_validate(out)
        # Before publishing validation: compare canonical persisted data against
        # every raw native branch and retain the original removal/hash inventory.
        duplicate_storage.read=read_duplicate
        began=time.perf_counter()
        compact_validated(Path(out),known_encoding=encodings.get(Path(out).resolve()))
        for value in receipts.values():value['timings']['compact_and_pack']=time.perf_counter()-began
        for path,value in receipts.items():original_receipt(path,value)
        return answer
    finally:
        duplicate_storage.read=original_duplicate_read
        cached.clear();receipts.clear();raw_records.clear();encodings.clear()

staged.read=read_captured
staged.write_branches=write_cached
collector.branches=branches
staged.durable_json=receipt
staged.validate=validate
if __name__=='__main__':raise SystemExit(staged.main())
