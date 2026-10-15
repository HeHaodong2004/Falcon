"""Bounded 500k continuation using the audited freezer and revised sharing policy.

The check action reads counters and completion receipts only. The freeze action
preserves the original data audit, parent identities, and immutable cutoff. It
changes only the process-local scheduling guard, never production collection.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from . import resource_policy
from .resource_policy import AdaptiveGuard,VERSION,extraction_recovery

PROJECT=Path('/home/haodong/workspace/projects/eventwm-falcon')
CAMPAIGN=PROJECT/'output/eventwm_forecast_performance_v2_20261015'
INDEX=PROJECT/'output/eventwm_v4_only_index/verified.sqlite3'
PARENT=PROJECT/'output/eventwm_350k_belief_snapshot_20261014'
MINIMUM_QUERIES=495000


def eligibility(index,priority_runs):
    with sqlite3.connect(Path(index).resolve().as_uri()+'?mode=ro',uri=True) as database:
        database.execute('PRAGMA query_only=ON')
        roots,queries=database.execute('SELECT count(*),coalesce(sum(queries),0) FROM roots').fetchone()
    pending=[]
    for directory in priority_runs:
        receipt=Path(directory)/'result.json'
        if not receipt.exists() or read(receipt).get('status')!='training_complete':pending.append(str(directory))
    return dict(timestamp=time.time(),verified_index=str(Path(index).resolve()),verified_roots=roots,
        verified_queries=queries,minimum_queries=MINIMUM_QUERIES,priority_training_pending=pending,
        eligible_to_attempt_freeze=queries>=MINIMUM_QUERIES and not pending,
        actual_audit_still_required=True,production_actions=[])


def freeze(run,parent,index,priority_runs):
    run=Path(run).resolve();parent=Path(parent).resolve()
    if CAMPAIGN not in run.parents or run==parent or parent in run.parents:
        raise ValueError('Use a new nested campaign output, separate from the parent snapshot')
    if (run/'result.json').exists():raise FileExistsError('Completed scaling freeze is immutable')
    gate=eligibility(index,priority_runs)
    if not gate['eligible_to_attempt_freeze']:
        raise RuntimeError('Scaling gate closed: '+json.dumps(gate,sort_keys=True))
    extraction_recovery(CAMPAIGN/'cache_full/guard.json')
    run.mkdir(parents=True,exist_ok=True)
    import eventwm_belief_freeze as audited
    identity=dict(parent_snapshot_sha256=sha256(parent/'snapshot.json'),index=str(Path(index).resolve()),
        source_sha256=sha256(__file__),resource_policy=VERSION,minimum_queries=MINIMUM_QUERIES,
        resource_policy_sha256=sha256(resource_policy.__file__),audited_source_sha256=sha256(audited.__file__),
        priority_runs=[str(Path(p).resolve()) for p in priority_runs])
    if (run/'identity.json').exists() and read(run/'identity.json')!=identity:
        raise ValueError('Scaling freeze continuation identity changed')
    save_json(run/'identity.json',identity);save_json(run/'eligibility.json',gate)
    class FreezeGuard(AdaptiveGuard):
        def check(self,force=False):
            if (run/'STOP_RESEARCH').exists():raise RuntimeError('Research stopped: explicit scaling freeze stop')
            extraction_recovery(CAMPAIGN/'cache_full/guard.json')
            return super().check(force)
    original=audited.full.WMGuard;audited.full.WMGuard=FreezeGuard
    frozen=run/'frozen'
    try:
        if not (frozen/'snapshot.json').exists():
            audited.freeze(SimpleNamespace(output=frozen,parent=parent,index=Path(index)))
        elif read(frozen/'status.json').get('stage')!='frozen':
            raise ValueError('Snapshot lacks the completed audited freeze receipt')
    finally:audited.full.WMGuard=original
    snapshot_identity=read(frozen/'snapshot_identity.json');audit=read(frozen/'audit.json')
    if (sha256(frozen/'snapshot.json')!=snapshot_identity['sha256'] or
        snapshot_identity['parent_snapshot_sha256']!=identity['parent_snapshot_sha256']):
        raise ValueError('Audited snapshot identity changed')
    if audit['retained_queries']<MINIMUM_QUERIES:
        raise ValueError('Audit retained fewer than 495000 queries; preserve evidence, do not train a 500k claim')
    receipt=dict(status='scaling_snapshot_complete',snapshot_identity=snapshot_identity,
        validated_queries=audit['retained_queries'],validated_roots=audit['retained_roots'],
        frozen_directory=str(frozen),audited_freeze_source_sha256=sha256(audited.__file__),
        identity=identity,production_actions=[],training_complete=False)
    save_json(run/'result.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('check','freeze'))
    parser.add_argument('--index',type=Path,default=INDEX);parser.add_argument('--parent',type=Path,default=PARENT)
    parser.add_argument('--run',type=Path);parser.add_argument('--priority-run',type=Path,action='append')
    args=parser.parse_args()
    priority=args.priority_run or [CAMPAIGN/f'full_self_s{s}' for s in (20261014,20261015)]
    if args.action=='freeze' and args.run is None:parser.error('--run is required for freeze')
    result=eligibility(args.index,priority) if args.action=='check' else freeze(args.run,args.parent,args.index,priority)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
