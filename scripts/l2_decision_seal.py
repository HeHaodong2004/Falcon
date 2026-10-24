"""Archive decision-revision reports and evidence without overwriting releases."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
ACCELERATION=REPO.parent/'eventwm-falcon-acceleration-v2'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args,cwd=REPO):
    return subprocess.run(['git','--no-optional-locks',*args],cwd=cwd,check=True,capture_output=True,text=True).stdout.strip()


def write_new(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True);parser.add_argument('--rendered',required=True);parser.add_argument('--destination',required=True)
    a=parser.parse_args();src=(REPO/a.inputs).resolve();rendered=(REPO/a.rendered).resolve();dest=(REPO/a.destination).resolve()
    if (not src.is_relative_to(REPO/'output/l2_decision_ready_20261021') or
            not dest.is_relative_to(REPO/'docs/l2/decision_ready_20261021') or not rendered.is_relative_to(src)):
        raise ValueError('isolated decision output/report destinations required')
    artifacts=dest/'artifacts';raw=artifacts/'raw'
    if (artifacts/'evidence_manifest.json').exists():raise FileExistsError('sealed evidence exists')
    primary=json.loads((src/'benchmark_release_v1.json').read_text())
    batches=json.loads((src/'benchmark_batches_v1.json').read_text())
    validation=json.loads((src/'validation_release_v3.json').read_text())
    development=json.loads((src/'development_v1.json').read_text())
    current={p.name:sha(p) for p in sorted((REPO/'eventwm_graph/l2_partial').glob('*.py'))}
    if (current!=primary['sources_before'] or current!=batches['sources_before'] or
            current!=development['source_hashes'] or current!=validation['source_after']):
        raise ValueError('final source differs from measured runtime')
    if (primary['failure'] or batches['failure'] or validation['failure'] or development['failure'] or
            primary['source_changed_during_measurement'] or batches['source_changed_during_measurement'] or
            not validation['sources_unchanged'] or not development['sources_unchanged']):
        raise ValueError('unsealed/incomplete final measurements')
    protocol=REPO/'docs/l2/decision_ready_20261021/EXPERIMENT_PROTOCOL.md'
    if sha(protocol)!=development['protocol_sha256']:raise ValueError('frozen decision rule changed')
    committed=subprocess.run(['git','show','a443eb47:docs/l2/decision_ready_20261021/EXPERIMENT_PROTOCOL.md'],cwd=REPO,check=True,capture_output=True).stdout
    if hashlib.sha256(committed).hexdigest()!=sha(protocol):raise ValueError('protocol is not the pre-outcome committed version')
    old_reports='docs/l2/partial_graph_wm_structural_20261021'
    if git('diff','--name-only','d66e6691','--',old_reports):raise ValueError('historical reports changed')
    accel_status=git('status','--porcelain',cwd=ACCELERATION)
    if accel_status:raise ValueError('acceleration project no longer clean; inspect read-only changes')
    checkpoints={}
    for path in sorted((REPO/old_reports/'artifacts/checkpoints').iterdir()):
        if path.is_file():checkpoints[str(path.relative_to(REPO))]=sha(path)
    status_path=ACCELERATION/'output/stage01_acceleration_v2_20261021/collection_user_authorized_5s_20261021_1/status.json'
    status=json.loads(status_path.read_text())
    provenance={'schema':'l2_decision_release_provenance_1','timestamp_unix':time.time(),
        'start_release':'d66e669127d1e59c7ec4effb2f45a45fa4205c9b','protocol_commit':'a443eb47',
        'runtime_commit':'386ea59a','git_head_before_seal':git('rev-parse','HEAD'),
        'research_branch':git('branch','--show-current'),'git_status_before_seal':git('status','--porcelain').splitlines(),
        'runtime_sha256':current,'protocol_sha256':sha(protocol),'unchanged_checkpoints':checkpoints,
        'acceleration_project':{'revision':git('rev-parse','HEAD',cwd=ACCELERATION),'git_status':accel_status,
            'mode':'read-only throughout research','collector_status_path':str(status_path),
            'collector_status_sha256':sha(status_path),'collector_status':{k:status.get(k) for k in (
                'status','status_reason','active_worker_count','supervisor_pid','outcomes','roots','validated_outcomes_per_second_recent')}},
        'new_tests':39,'baseline_tests':322,'final_tests':validation['total_tests'],
        'same_law_comparisons':len(primary['same_law_parity'])+len(batches['same_law_parity']),
        'decision':'DECISION INTERFACE REVISE; SYSTEMS REVISE independently',
        'training_launched':False,'production_changes':False,'preserved_reports_changed':False,
        'numerical_resource_policy':'one CPU thread per job,nice19,idle I/O,max two jobs',
        'seal_script_sha256':sha(__file__)}
    write_new(src/'release_provenance.json',json.dumps(provenance,indent=2,allow_nan=False).encode())
    # Rendered reports are new; never replace existing report/evidence files.
    for path in sorted(rendered.rglob('*')):
        if path.is_file():write_new(dest/path.relative_to(rendered),path.read_bytes())
    for name in ('EXPERIMENT_PROTOCOL.md','L2_TO_L25_CONTRACT.md'):
        original=REPO/'docs/l2/decision_ready_20261021'/name
        if not (dest/name).exists():write_new(dest/name,original.read_bytes())
    entries=[]
    for path in sorted(src.iterdir()):
        if not path.is_file():continue
        original=path.read_bytes();binary=path.suffix=='.prof';encoded=original if binary else gzip.compress(original,compresslevel=9,mtime=0)
        target=raw/(path.name if binary else path.name+'.gz')
        write_new(target,encoded)
        entries.append({'original':str(path.relative_to(REPO)),'bundled':str(target.relative_to(REPO)),
            'sha256_original':hashlib.sha256(original).hexdigest(),'sha256_bundled':hashlib.sha256(encoded).hexdigest(),
            'bytes_original':len(original),'bytes_bundled':len(encoded),'encoding':'binary' if binary else 'gzip'})
    manifest={'schema':'l2_decision_evidence_manifest_1','entries':entries,'runtime_sha256':current,
        'rendered_source':str(rendered.relative_to(REPO)),'render_script_sha256':sha(REPO/'scripts/l2_decision_report.py'),
        'frozen_assets':checkpoints,'historical_release':'d66e6691','prior_runtime':'ec2ad5b1da4437b2a0d9d747d7659323af07450a'}
    write_new(artifacts/'evidence_manifest.json',json.dumps(manifest,indent=2).encode())
    paths=set(dest.rglob('*'))
    for pattern in ('eventwm_graph/l2_partial/*.py','tests/l2_partial/*.py','tests/l2_partial_structural/*.py',
                    'tests/l2_decision_ready/*.py','scripts/l2_decision_*.py','scripts/l2_partial_structural_*.py'):
        paths.update(REPO.glob(pattern))
    paths.update(REPO/name for name in ('scripts/l2_partial_repair_systems_stress.py','scripts/l2_partial_scene_calibration.py'))
    lines=[sha(path)+'  '+str(path.relative_to(REPO)) for path in sorted(paths) if path.is_file() and path.name!='release_files.sha256']
    write_new(artifacts/'release_files.sha256',('\n'.join(lines)+'\n').encode())
    print(json.dumps({'bundled_files':len(entries),'checksum_entries':len(lines),'destination':str(dest),
        'decision':provenance['decision'],'collector':provenance['acceleration_project']['collector_status']}),flush=True)


if __name__=='__main__':main()
