#!/usr/bin/env python3
"""Finish the explicit four-root smoke after a normal stop at its first root.

Interrupt the next root after its first branch, then resume to completion. This
requires an existing bounded manifest, GPU access, and sufficient retry budget.
"""
import argparse
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from collector.core import read,write
from eventwm_collect import branches


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); output=args.output.absolute(); m=read(output/'manifest.json')
    completed={r['root_id']:copy.deepcopy(r) for r in m['roots'] if r['state']=='completed'}
    assert completed and len(completed)<len(m['roots'])
    target=next(r for r in m['roots'] if r['state']=='pending')
    cmd=[sys.executable,str(REPO/'scripts/eventwm_scale.py'),'run','--output',str(output),'--resume']
    with (output/'resume_interrupt.log').open('w') as log:
        child=subprocess.Popen(cmd+['--stop-after-roots','1'],stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            records=list((output/'roots'/target['root_id']).glob('attempt_*/audits/workers/b000/branch.json'))
            if records: break
            if child.poll() is not None: raise RuntimeError('Runner exited before interruption point')
            time.sleep(.5)
        else:
            child.terminate(); child.wait(timeout=30); raise RuntimeError('Interruption fixture timed out')
        child.send_signal(signal.SIGTERM); rc=child.wait(timeout=30)
    stopped=read(output/'manifest.json'); interrupted=next(r for r in stopped['roots'] if r['root_id']==target['root_id'])
    assert rc!=0 and interrupted['state']=='failed'
    partial=output/interrupted['attempts'][-1]['path']
    assert not (partial/'COMPLETE.json').exists()
    assert records[0].exists()
    with (output/'resume_finish.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    final=read(output/'manifest.json')
    assert all(r['state']=='completed' for r in final['roots'])
    for r in final['roots']:
        if r['root_id'] in completed: assert r==completed[r['root_id']]
        bs=branches(output/r['path']); assert len({b['branch_id'] for b in bs})==len(bs)
        assert all(b['root_id']==r['root_id'] for b in bs)
    retry=next(r for r in final['roots'] if r['root_id']==target['root_id'])
    assert len(retry['attempts'])==2 and retry['attempts'][0]['state']=='failed'
    assert partial!=output/retry['path'] and records[0].exists()
    report=dict(status='PASS',normal_stop_completed_roots=len(completed),completed_roots_unchanged=True,
        interrupted_root=target['root_id'],partial_attempt_preserved=str(partial),incomplete_retry_separate=True,
        completed_roots=len(final['roots']),duplicate_published_branch_ids_within_roots=0,
        primary_branches=sum(r['stats']['primary_branches'] for r in final['roots']),
        grouped_branches=sum(r['stats']['branches'] for r in final['roots']))
    write(output/'resume_audit.json',report); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
