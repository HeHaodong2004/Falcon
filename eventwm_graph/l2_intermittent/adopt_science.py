"""Export only completed immutable scientific results; never controls jobs."""
import json
from pathlib import Path
import time
from .common import BASE,OUT,sha,dump
from .package_release import copy_once


def run():
    destination=OUT/'scientific_adoption.json'
    if destination.exists():raise ValueError('preserve completed adoption')
    handoff=OUT/'primary_agent_notes/SCIENCE_HANDOFF_COMPLETE.json'
    if not handoff.is_file():raise ValueError('wait for closed scientific handoff and postprocessing')
    json.loads(handoff.read_text())
    owned=OUT/'primary_agent_notes/corrected_science_runtime'
    if sha(owned/'manifest.json')!='8bc21f50efb8d8d955658a120b81ec49c0cc27ce3e7bf8fb4f4cd3c305a3ea8d':
        raise ValueError('incorrect executed source seal')
    seal=json.loads((owned/'manifest.json').read_text())
    for name,value in seal['sealed_sources'].items():
        if sha(owned/name)!=value:raise ValueError('executed source changed: '+name)
    source=owned/'output/l2_intermittent_wm_20261022';evaluation=source/'evaluation'
    guard=json.loads((evaluation/'merge_execution_guard.json').read_text())
    if not all(guard[k] for k in ('sources_stable','inputs_stable','orchestration_stable')):
        raise ValueError('unstable or incomplete native science')
    if guard['complete_ordered_roots']!=2110 or guard['scenes']!=108:raise ValueError('incomplete development population')
    for name,value in guard['output_sha256'].items():
        if sha(evaluation/name)!=value:raise ValueError('merged scientific output changed: '+name)
    gap_guard=json.loads((source/'gap_execution_guard.json').read_text())
    if gap_guard['error'] or not gap_guard['sources_stable'] or not gap_guard['inputs_stable']:
        raise ValueError('unstable strict gap audit')
    acquisition_path=OUT/'primary_agent_notes/strict_gap_acquisition.json'
    acquisition=json.loads(acquisition_path.read_text())
    expected_gap_inputs={str(path.relative_to(BASE)) for path in
        (source/'data_and_gaps_strict.json',source/'gap_execution_guard.json',
         owned/'manifest.json',owned/'eventwm_graph/l2_intermittent/strict_gap_audit.py')}
    if set(acquisition['files'])!=expected_gap_inputs or not acquisition['sources_stable'] or not acquisition['inputs_stable']:
        raise ValueError('strict gap acquisition inventory or stability differs')
    if any(sha(BASE/name)!=value for name,value in acquisition['files'].items()):
        raise ValueError('strict gap acquired bytes changed')
    gaps=json.loads((source/'data_and_gaps_strict.json').read_text())
    if len(gaps['per_root_counts'])!=12045 or len({r['root_id'] for r in gaps['per_root_counts']})!=12045:
        raise ValueError('strict gap audit is incomplete')
    copied={}
    for path in sorted(evaluation.rglob('*')):
        if path.is_file():
            target=OUT/'evaluation'/path.relative_to(evaluation);copy_once(path,target)
            copied[str(target.relative_to(OUT))]=sha(path)
    for name in ('data_and_gaps_strict.json','gap_execution_guard.json'):
        copy_once(source/name,OUT/name);copied[name]=sha(source/name)
    diagnostics=OUT/'sealed_runtime/output/l2_intermittent_wm_20261022'
    for operation,name in (('old_physical','old_physical_reproduction.json'),('physical','physical.json'),('optical','optical_validation.json')):
        check=json.loads((diagnostics/(operation+'_execution_guard.json')).read_text())
        if check['error'] or not check['sources_stable'] or not check['inputs_stable']:raise ValueError('unstable diagnostic: '+operation)
        for filename in (name,operation+'_execution_guard.json'):
            copy_once(diagnostics/filename,OUT/filename);copied[filename]=sha(diagnostics/filename)
    if any(sha(OUT/name)!=value for name,value in copied.items()):raise ValueError('adoption byte mismatch')
    result=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_manifest_sha256=sha(owned/'manifest.json'),merge_guard_sha256=sha(evaluation/'merge_execution_guard.json'),
        handoff_sha256=sha(handoff),
        strict_gap_acquisition_sha256=sha(acquisition_path),
        gap_guard_sha256=sha(source/'gap_execution_guard.json'),files=copied,
        scope='Byte-exact export of completed guarded2110-root native results,12045-root strict gap audit and sealed diagnostics. No new model samples, target changes, job signals or production actions.')
    dump(destination,result);print(json.dumps(dict(adopted_files=len(copied),roots=2110,source=str(evaluation))))


if __name__=='__main__':run()
