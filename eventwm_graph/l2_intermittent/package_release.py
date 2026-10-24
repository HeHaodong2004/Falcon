"""Preserve this research campaign's evidence without changing historical data."""
import argparse
import collections
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from .common import BASE, OUT, OLD, SNAP, DOC, sha, dump


def copy_once(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError('refusing to replace preserved evidence: '+str(destination))
        return
    shutil.copyfile(source, destination)


def gzip_once(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with gzip.open(destination, 'rb') as previous:
            old = hashlib.sha256()
            for block in iter(lambda: previous.read(1024*1024), b''):
                old.update(block)
        if old.hexdigest() != sha(source):
            raise ValueError('refusing to replace compressed evidence: '+str(destination))
        return
    with source.open('rb') as incoming, destination.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as outgoing:
            shutil.copyfileobj(incoming, outgoing)


def input_paths():
    paths = [SNAP/'manifest.json', OLD/'train.npy', OLD/'dev.npy',
        OLD/'motion.json', OLD/'population_prior.json', OLD/'training/checkpoint.pt',
        DOC/'PROTOCOL.md', DOC/'EVALUATION_FREEZE.md',
        OUT/'motion.json', OUT/'train.npy', OUT/'dev.npy',
        OUT/'training/configuration.json', OUT/'training/training.json']
    paths += [OUT/'training'/name for name in ('checkpoint.pt','epoch0.pt','last.pt')]
    inherited = BASE/'docs/l2/graph_wm_120k_20261021/artifacts'
    paths += [inherited/'motion.json', inherited/'population_prior.json',
        inherited/'checkpoints/frozen_observation_fit.json',
        BASE/'docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints/native_events_v5_scene.json']
    paths += [p for name in ('l2_partial','l2_120k')
              for p in (BASE/'eventwm_graph'/name).glob('*.py')]
    paths += [BASE/'eventwm_graph/l2_intermittent'/name for name in
        ('__init__.py','common.py','model.py','dynamics.py','neural.py',
         'baselines.py','observations.py','evaluate.py')]
    return sorted(set(paths))


def seal():
    path = OUT/'scientific_inputs_seal.json'
    if path.exists():
        raise ValueError('preserve the original input seal')
    dump(path, dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        sources_and_inputs={str(p.relative_to(BASE)):sha(p) for p in input_paths()},
        scope='Supplemental complete input identity seal. Existing evaluator imports preceded this seal; '
              'their own archived source guards and earlier training/cache/checkpoint identities remain authoritative.'))


def verify():
    original=json.loads((OUT/'scientific_inputs_seal.json').read_text())
    rows={name:dict(expected=value,actual=sha(BASE/name))
          for name,value in original['sources_and_inputs'].items()}
    changed=[name for name,value in rows.items() if value['expected']!=value['actual']]
    # The user explicitly prioritized inference speed after the scientific run
    # was frozen. Preserve the old seal and require its original source bytes,
    # plus source-bound equivalence of the separately versioned implementation.
    allowed={'eventwm_graph/l2_intermittent/'+name for name in
             ('model.py','dynamics.py','neural.py','baselines.py','evaluate.py')}
    if set(changed)-allowed:raise ValueError('unapproved scientific input change: '+repr(set(changed)-allowed))
    archived={name:sha(OUT/'sealed_runtime'/name) for name in changed}
    if any(archived[name]!=rows[name]['expected'] for name in changed):
        raise ValueError('original scientific source was not preserved')
    speed=read_json(OUT/'final_speed_manifest.json')
    for name,value in speed['runtime_sources'].items():
        if sha(BASE/name)!=value:raise ValueError('optimized implementation changed: '+name)
    accounting=read_json(OUT/'offline_accounting_integration.json')
    if not accounting['passed'] or not accounting['source_stable'] or accounting['public_evaluator_sha256']!=sha(BASE/'eventwm_graph/l2_intermittent/evaluate.py'):
        raise ValueError('public offline scoring repair is not verified')
    evidence=[]
    for name in speed['equivalence_evidence']:
        proof=read_json(OUT/name)
        if not proof['passed'] or not proof['source_stable']:raise ValueError('speed equivalence did not pass')
        for source,value in proof['sources'].items():
            if source in speed['runtime_sources'] and speed['runtime_sources'][source]!=value:
                raise ValueError('equivalence is for different runtime source')
        evidence.append(dict(path=name,sha256=sha(OUT/name)))
    result=dict(seal_sha256=sha(OUT/'scientific_inputs_seal.json'),
        unchanged=not changed,changed=changed,inputs=rows,
        frozen_sources_preserved=archived,immutable_data_and_checkpoints_unchanged=True,
        authorized_speed_changes=sorted(set(changed)-{'eventwm_graph/l2_intermittent/evaluate.py'}),
        verified_offline_accounting_changes=['eventwm_graph/l2_intermittent/evaluate.py'],
        offline_accounting_evidence_sha256=sha(OUT/'offline_accounting_integration.json'),
        speed_manifest_sha256=sha(OUT/'final_speed_manifest.json'),
        equivalence_evidence=evidence,
        semantics='Original scientific seal retained. Separately versioned/equivalence-checked speed implementation and verified post-run PUBLIC offline accounting differ; original executed scoring sources, scientific fitting, baseline, checkpoints and data are unchanged.')
    path=OUT/'scientific_inputs_verification.json'
    if path.exists():
        raise ValueError('preserve the original verification')
    dump(path,result)
    return result


def read_json(path):return json.loads(Path(path).read_text())


def resources():
    samples=[json.loads(line) for line in (OUT/'resources.jsonl').read_text().splitlines()]
    collectors=[r['collector'] for r in samples if 'collector' in r]
    def unique(key):return sorted({str(c.get(key)) for c in collectors})
    first,last=collectors[0],collectors[-1]
    return dict(samples=len(samples),first_utc_epoch=samples[0]['time'],last_utc_epoch=samples[-1]['time'],
        collector_status_counts=dict(collections.Counter(c.get('collection_status') for c in collectors)),
        service_status_counts=dict(collections.Counter(c.get('service_status') for c in collectors)),
        workers=unique('active_worker_count'),expected_workers=unique('expected_worker_count'),
        automatic_training=unique('automatic_training'),restart_counts=unique('collector_restart_count'),
        configuration_identities=unique('configuration_identity'),
        outcomes_delta=last['outcomes']-first['outcomes'],roots_delta=last['roots']-first['roots'],
        first_collector=first,last_collector=last,
        scope='Read-only shared-host observations; status/throughput do not establish causal absence of slowdown.')


def package():
    root=DOC/'artifacts';root.mkdir(exist_ok=True)
    required=('data_and_gaps.json','data_and_gaps_strict.json','gap_context.json','physical.json','old_physical_reproduction.json',
        'optical_validation.json','association_reference.json','api_example_final.json',
        'api_example_optimized.json','scientific_inputs_seal.json','scientific_inputs_verification.json',
        'final_speed_manifest.json','offline_accounting_integration.json','final_validation.json',
        'scientific_adoption.json','api_example_packaged_checkpoint.json','reproduction_setup_import_api.json')
    for name in required:gzip_once(OUT/name,root/'raw'/(name+'.gz'))
    copy_once(OUT/'primary_agent_notes/strict_gap_acquisition.json',
              root/'verified_execution/orchestration/strict_gap_acquisition.json')
    for name in ('preparation.json','motion.json','motion_fit.json'):
        copy_once(OUT/name,root/name)
    for name in ('checkpoint.pt','epoch0.pt','last.pt'):
        copy_once(OUT/'training'/name,root/'checkpoints'/name)
    for source in sorted((OUT/'training').iterdir()):
        if source.is_file() and source.suffix!='.pt':copy_once(source,root/'training'/source.name)
    copy_once(OLD/'training/checkpoint.pt',root/'baseline/checkpoint_43187f24.pt')
    for name in ('motion.json','population_prior.json'):
        copy_once(OLD/name,root/'baseline'/name)
    copy_once(BASE/'docs/l2/graph_wm_120k_20261021/artifacts/checkpoints/frozen_observation_fit.json',
              root/'baseline/frozen_observation_fit.json')
    evaluation=OUT/'evaluation'
    if json.loads((evaluation/'configuration.json').read_text())['roots']!=2110:
        raise ValueError('full scientific evaluation is required')
    for source in sorted(evaluation.rglob('*')):
        if source.is_file():copy_once(source,root/'evaluation'/source.relative_to(evaluation))
    for name in ('reviewer_c_systems_final_v2','validation','reference_runtime'):
        for source in sorted((OUT/name).rglob('*')):
            if source.is_file() and '__pycache__' not in source.parts:
                copy_once(source,root/name/source.relative_to(OUT/name))
    speed=read_json(OUT/'final_speed_manifest.json')
    final_validation=read_json(OUT/'final_validation.json')
    supplemental=[final_validation['directory'],'reviewer_c_reassociation_speed008_v1']
    for name in speed['preserved_directories']+supplemental:
        for source in sorted((OUT/name).rglob('*')):
            if source.is_file() and not source.is_symlink() and '__pycache__' not in source.parts and source.suffix not in ('.pyc','.nbc','.nbi'):
                copy_once(source,root/'speed'/name/source.relative_to(OUT/name))
    for name in speed['equivalence_evidence']:
        gzip_once(OUT/name,root/'speed'/(name+'.gz'))
    for source in sorted(OUT.glob('speed_checks_*.json')):
        gzip_once(source,root/'speed/history'/(source.name+'.gz'))
    for folder in sorted(OUT.glob('evaluation_shard_*')):
        for source in [folder/'configuration.json',folder/'forecast_execution_guard.json']+sorted((folder/'sources').glob('*')):
            copy_once(source,root/'shards'/folder.name/source.relative_to(folder))
    sealed=OUT/'sealed_runtime';manifest=json.loads((sealed/'manifest.json').read_text())
    copy_once(sealed/'manifest.json',root/'sealed_execution/manifest.json')
    for name,value in manifest['sealed_sources'].items():
        if sha(sealed/name)!=value:raise ValueError('sealed executable source changed: '+name)
        copy_once(sealed/name,root/'sealed_execution/sources'/name)
    for name in ('old_physical','physical','optical'):
        copy_once(sealed/'output/l2_intermittent_wm_20261022'/(name+'_execution_guard.json'),
                  root/'sealed_execution'/(name+'_execution_guard.json'))
    owned=OUT/'primary_agent_notes/corrected_science_runtime'
    owned_seal=read_json(owned/'manifest.json')
    copy_once(owned/'manifest.json',root/'verified_execution/manifest.json')
    for name,value in owned_seal['sealed_sources'].items():
        if sha(owned/name)!=value:raise ValueError('corrected executed source changed: '+name)
        copy_once(owned/name,root/'verified_execution/sources'/name)
    for name in ('run_corrected_science.py','setup_corrected_science.py','annotation_accounting.py',
                 'acquire_completed_outputs.py','merge_corrected_native.py','native_annotation_validity.json',
                 'seal_birth_and_gap_guards.py','seal_final_annotation_rule.py','snapshot_payload_integrity.json',
                 'verify_snapshot_payloads.py','visible_masked_monte_carlo.json','monte_carlo_sensitivity.py',
                 'belief_emission_readout.py','INHERITED_SUPERVISION_LIMITATION.md',
                 'REOBSERVATION_DEFINITIONS.md','CLOCK_RENEWAL_AND_PROGRESS.md',
                 'strict_gap_acquisition.json','STRICT_GAPS_COMPLETED.md',
                 'SCIENCE_HANDOFF_COMPLETE.json','native_monte_carlo.json',
                 'visible_masked_reconciliation.json','reconcile_visible_masked.py'):
        if (OUT/'primary_agent_notes'/name).exists():
            copy_once(OUT/'primary_agent_notes'/name,root/'verified_execution/orchestration'/name)
    owned_out=owned/'output/l2_intermittent_wm_20261022'
    for folder in sorted(owned_out.glob('evaluation_shard_*')):
        for source in sorted(folder.rglob('*')):
            if source.is_file():copy_once(source,root/'verified_execution/shards'/folder.name/source.relative_to(folder))
    for source in sorted(owned_out.glob('*execution_guard.json')):
        copy_once(source,root/'verified_execution'/source.name)
    for source in sorted((OUT/'primary_agent_notes').glob('*.log')):
        gzip_once(source,root/'verified_execution/logs'/(source.name+'.gz'))
    for folder in sorted(list(OUT.glob('superseded_birth_selector_evaluation_shard_*'))+
                         list(OUT.glob('superseded_lifetime_selector_evaluation_shard_*'))):
        for source in sorted(folder.rglob('*')):
            if source.is_file():copy_once(source,root/'superseded'/folder.name/source.relative_to(folder))
    for source in sorted((OUT/'primary_visible_masked_frozen_1').rglob('*')):
        if source.is_file():copy_once(source,root/'auxiliary_visible_masked'/source.relative_to(OUT/'primary_visible_masked_frozen_1'))
    for name in ('baseline_physical_comparison.json','reproduce_old_physical.py'):
        copy_once(OUT/'primary_agent_notes'/name,root/'auxiliary_old_replay'/name)
    for name in ('reviewer_c_reassociation_latency_v2.json','reviewer_c_final_audit.json',
        'reviewer_c_final_checks_frozen_v3.json','reviewer_c_boundary_checks_frozen_v3.json',
        'reviewer_c_receipt_checks_final_v3.json','pre_final_association_reference.json',
        'evaluation_v0_stop.json','benchmark_pre_optimization_stop.json','queue_wait_correction.json',
        'first_anchor_partial_stop.json','lifetime_selector_partial_stop.json','research_idle_scheduler.json',
        'final_baseline_integrity.json'):
        if (OUT/name).exists():gzip_once(OUT/name,root/'raw'/(name+'.gz'))
    for source in sorted(OUT.glob('*.log')):
        gzip_once(source,root/'logs'/(source.name+'.gz'))
    gzip_once(OUT/'resources.jsonl',root/'raw/resources.jsonl.gz')
    dump(root/'collector_resources.json',resources())
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        cpu_count=os.cpu_count(),affinity=list(os.sched_getaffinity(0)),
        thread_environment={name:os.environ.get(name) for name in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG')},
        installed={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        cpu_inventory=Path('/proc/cpuinfo').read_text(),
        base_commit=subprocess.check_output(['git','rev-parse','43187f24'],cwd=BASE,text=True).strip())
    dump(root/'environment.json',environment)
    print(json.dumps(dict(packaged=str(root),files=sum(p.is_file() for p in root.rglob('*')))),flush=True)


def hashes():
    root=DOC/'artifacts'
    sources=sorted((BASE/'eventwm_graph/l2_intermittent').glob('*.py'))+sorted((BASE/'tests/l2_intermittent').glob('*.py'))
    sources += [DOC/'PROTOCOL.md',DOC/'EVALUATION_FREEZE.md']
    dump(root/'source_hashes.json',{str(p.relative_to(BASE)):sha(p) for p in sources})
    files={str(p.relative_to(BASE)):sha(p) for p in sorted(root.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'
           and p.name not in ('evidence_manifest.json','release_files.sha256')}
    dump(root/'evidence_manifest.json',dict(base_commit='43187f24',files=files,
        snapshot_sha256=sha(SNAP/'manifest.json'),selected_checkpoint_sha256=sha(OUT/'training/checkpoint.pt'),
        baseline_checkpoint_sha256=sha(OLD/'training/checkpoint.pt'),
        output_tree=str(OUT),meaning='Preserved research evidence, not a passed acceptance certificate.'))
    release=sorted(p for p in DOC.rglob('*') if p.is_file() and
        '__pycache__' not in p.parts and p.suffix!='.pyc' and
        p.name!='release_files.sha256')+sources[:len(sources)-2]
    (root/'release_files.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(BASE))+'\n' for p in release))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('seal','verify','package','hashes'))
    {'seal':seal,'verify':verify,'package':package,'hashes':hashes}[parser.parse_args().operation]()
