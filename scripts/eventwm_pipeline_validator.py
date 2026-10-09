"""Frozen audits with per-root decoded-record caching and proved compaction.

No audit/label code changes. All persisted canonical records must still pass the
original raw-worker equality proof before a VALIDATED receipt is published.
"""
from pathlib import Path
import sys
import hashlib
import json
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
cached={};receipts={};raw_records={}
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
    answer=original_write(out,records,compressed=compressed)
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
        compact_validated(Path(out))
        for path,value in receipts.items():original_receipt(path,value)
        return answer
    finally:
        duplicate_storage.read=original_duplicate_read
        cached.clear();receipts.clear();raw_records.clear()

staged.read=read_captured
staged.write_branches=write_cached
collector.branches=branches
staged.durable_json=receipt
staged.validate=validate
if __name__=='__main__':raise SystemExit(staged.main())
