"""Publish a compact, checksum-verified repair evidence copy, never dataset payloads.

Run only after final reports/figures exist. Existing destinations are rejected.
Large feature caches remain local and are pinned by their copied manifests.
No fitting, simulation, collection query or protected evaluation reader runs.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(repo):
    repo=Path(repo).resolve()
    old=repo/'output/l2_partial_graph_wm_explore_20261021'
    new=repo/'output/l2_partial_graph_wm_repair_20261021'
    documentation=repo/'docs/l2/partial_graph_wm_repair_20261021'
    destination=documentation/'artifacts'
    copies={
        'checkpoints/graph_dynamics.pt':old/'learned_posdiff/checkpoint.pt',
        'checkpoints/population_prior.json':old/'train.npy.population.json',
        'checkpoints/native_legacy.json':old/'native_events_final.json',
        'checkpoints/observation_fit.json':new/'observation_fit.json',
        'checkpoints/native_events_v4_prefit.json':new/'native_events_v4_prefit.json',
        'checkpoints/native_events_v4.json':new/'native_events_v4.json',
        'checkpoints/native_events_v5_scene.json':new/'native_events_v5_scene.json',
        'dataset_manifest.json.gz':old/'snapshot/manifest.json',
        'fresh_snapshot_manifest.json.gz':new/'fresh_snapshot/manifest.json',
    }
    for name in ['population_evaluation.json','paired_comparisons.json',
                 'risk_scene_calibrated.json','observation_calibration_scene.json',
                 'risk_development.json','observation_calibration.json',
                 'probability_diagnostics_v5_final.json','probability_diagnostics_final.json',
                 'probability_diagnostics_v4_corrected.json','oracle_association_v4.json',
                 'latency_v5_active.json','latency_final_active.json','trajectory_diagnostic.json','source_audit.json',
                 'untouched_candidates.json','resource_active_collector_v1.json']:
        copies['raw/'+name+'.gz']=new/name
    for name in ['population.json','risk.json','events.npy.manifest.json']:
        copies['fresh/'+name+'.gz']=new/'fresh_evaluation'/name
    for name in ['environment.json','development_cohort.json','smoke_cohort10.json','oracle_cohort21.json',
                 'evaluation_variants.json','model_kwargs.json','event_checkpoints_v5.json',
                 'fresh_config_v5.json','fresh_design.json','native_events_v5_scene_interpretation.json',
                 'native_clock_diagnostic.json','reproduced_release_failures.json',
                 'latency_v5_context_before.json','latency_v5_context_after.json',
                 'trajectory_diagnostic_provenance.json',
                 'evaluation_report_correction.json','evaluation_report_first_draft.md',
                 'tests_final_release_summary.json','tests_final_release.log','trajectory_diagnostic_selftests.log',
                 'events_train_v4.npy.manifest.json','events_dev_v4.npy.manifest.json',
                 'observations_dev_v4.npy.manifest.json','observation_fit_train_labels.npz',
                 'observation_fit_dev_labels.npz',
                 'systems_v5_synthetic6_cold.json','systems_v5_synthetic6_cold.prof',
                 'systems_v5_synthetic6_cold.txt','synthetic_systems_stress_v2.json',
                 'synthetic_systems_stress_v2.json.jsonl','synthetic_systems_stress_v1.json',
                 'synthetic_systems_stress_v1.json.jsonl']:
        copies[name]=new/name
    # The probe and failed-intermediate logs are small research artifacts only.
    for path in sorted(new.glob('*.log')):
        copies.setdefault('logs/'+path.name,path)
    for pattern in ['systems_*stress*','systems_*empty*','systems_*dense*','SYSTEMS_VERTICAL_REVIEW_*.md']:
        for path in sorted(new.glob(pattern)):
            if path.is_file() and path.suffix in ['.json','.jsonl','.txt','.prof','.py','.md']:
                copies.setdefault('systems/'+path.name,path)
    for name in ['ARCHITECTURE_PROBABILITY.md','API.md','DATA_TRAINING.md','SCIENTIFIC_TESTS.md',
                 'RISK_AND_OBSERVATIONS.md','ROLLOUT_AND_PROBABILITY.md','FRESH_EVALUATION.md',
                 'LATENCY_REPORT.md','EVALUATION.md','FINAL_REPORT.md','REVIEW_LOG.md','REPRODUCE.md']:
        if not (documentation/name).is_file():raise FileNotFoundError(documentation/name)
    missing=[str(p) for p in copies.values() if not p.is_file()]
    if missing:raise FileNotFoundError(missing)
    population=json.loads((new/'population_evaluation.json').read_text())
    if population['roots']!=632 or population['scenes']!=106 or population['smoke_test_only']:
        raise ValueError('Complete fixed development cohort required')
    profile=json.loads((new/'latency_v5_active.json').read_text())
    if profile['source_changed_during_measurement']:raise ValueError('Timing source changed')
    figures=json.loads((new/'figures/manifest.json').read_text())
    if len([n for n in figures['figures'] if n.endswith('.pdf')])!=11:
        raise ValueError('All11 measured diagnostic figures required')
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir()
    records=[]
    for name,source in copies.items():
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        data=source.read_bytes();digest=hashlib.sha256(data).hexdigest()
        if name.endswith('.gz'):
            with target.open('xb') as stream:
                with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:
                    compressed.write(data)
            restored=gzip.decompress(target.read_bytes())
            if hashlib.sha256(restored).hexdigest()!=digest:raise ValueError('Compression mismatch')
        else:shutil.copyfile(source,target)
        records.append({'artifact':name,'source':str(source.relative_to(repo)),
            'source_bytes':len(data),'source_sha256':digest,'artifact_sha256':sha(target)})
    inference={'class':'eventwm_graph.l2_partial.repair_model:RepairedGraphBeliefWorldModel',
        'checkpoints':{'dynamics':'checkpoints/graph_dynamics.pt','population':'checkpoints/population_prior.json',
                       'observation':'checkpoints/observation_fit.json','events':'checkpoints/native_events_v5_scene.json'},
        'kwargs':{'maneuver_rate':2.,'reset_std':2.,'maneuver_retention':1.},
        'decision':'REVISE','latency_approved':False,
        'scope':'frozen exploratory research; source API construction documented in API.md'}
    (destination/'inference_config.json').write_text(json.dumps(inference,indent=2))
    for path in sorted((new/'figures').iterdir()):
        if path.is_file():
            target=documentation/'figures'/path.name;target.parent.mkdir(exist_ok=True)
            if target.exists():raise FileExistsError(target)
            shutil.copyfile(path,target)
    value={'starting_release':'eb1127639f381fa3224863ce6cac589720eb15ff',
           'frozen_model_revision':'40da7347efd2058c2befe0a294f6454ccbb0dad6',
           'packaging_parent_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
           'created_unix':time.time(),'package_script_sha256':sha(Path(__file__)),
           'records':records,'dataset_payloads_included':False,'large_feature_caches_included':False,
           'limitation':'Exact data-dependent replay requires the preserved licensed snapshots; source/checkpoint/configuration and compressed raw evaluation evidence are included.'}
    (destination/'evidence_manifest.json').write_text(json.dumps(value,indent=2))
    files=[p for p in sorted(documentation.rglob('*')) if p.is_file() and p.name!='release_artifacts.sha256']
    with (destination/'release_artifacts.sha256').open('x') as stream:
        for path in files:stream.write(sha(path)+'  '+str(path.relative_to(documentation))+'\n')
    print(json.dumps({'artifacts':len(records),'hashed_files':len(files),'output':str(destination)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    run(parser.parse_args().repo)
