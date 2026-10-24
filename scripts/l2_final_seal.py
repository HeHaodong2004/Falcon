"""Seal completed research evidence and explicit nonpromotion; no collector writes."""
import csv,datetime,json,os,subprocess,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from eventwm_graph.l2_final.common import OUT,DOC,sha,dump


def main():
    required=['FINAL_REPORT.md','DATA_SNAPSHOT.md','TRAINING_REPORT.md','FORECASTING_EVALUATION.md','BELIEF_REOBSERVATION_EVALUATION.md',
              'PROBABILITY_CORRECTNESS.md','LATENCY_REPORT.md','DOWNSTREAM_INTEGRATION.md','LIMITATIONS.md','REPRODUCTION.md',
              'reviews/REVIEW_A_FINAL.md','reviews/REVIEW_B_FINAL.md','reviews/REVIEW_C_FINAL.md','reviews/RESOLUTIONS.md']
    assert all((DOC/p).exists() for p in required)
    metrics=json.loads((OUT/'metrics.json').read_text());assert set(metrics['public_initialization'])=={'B','C','C_minus_B'}
    assert sha(OUT/'metrics.json')==sha(DOC/'artifacts/metrics.json')
    tables=DOC/'artifacts/tables';tables.mkdir(exist_ok=True)
    with (tables/'forecasting.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['variant','group','cohort','scope','age_bin','H','metric','mean','lo','hi','roots','scenes','sufficient']);writer.writeheader()
        for variant,record in metrics['variants'].items():
            for group in ('scopes','age_horizon'):
                for cell in record['forecasting'][group]:
                    for key,value in cell['metrics'].items():writer.writerow(dict(variant=variant,group=group,**{k:cell.get(k) for k in ('cohort','scope','age_bin','H')},metric=key,**value,sufficient=value['roots']>=30 and value['scenes']>=10))
    with (tables/'paired_forecasting.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['contrast','cohort','scope','metric','mean','lo','hi','roots','scenes']);writer.writeheader()
        for contrast,record in metrics['comparisons'].items():
            for cell in record['forecasting']['cells']:
                for key,value in cell['metrics'].items():writer.writerow(dict(contrast=contrast,cohort=cell['cohort'],scope=cell['scope'],metric=key,**value))
    snapshot=json.loads((OUT/'final_snapshot_manifest.json').read_text());training=json.loads((OUT/'training/training.json').read_text())
    decision=dict(schema='l2_final_scientific_release_decision_1',sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        verdict='COMPLETED_RESEARCH_CAMPAIGN_NOT_PROMOTED',B_promoted=False,C_promoted=False,production_merge=False,
        retained_reference=dict(variant='A',source='runtime/control_A',checkpoint='artifacts/checkpoints/A/checkpoint.pt',
            model_source_sha256=sha(DOC/'runtime/control_A/eventwm_graph/l2_intermittent/model.py'),checkpoint_sha256=sha(DOC/'artifacts/checkpoints/A/checkpoint.pt'),
            caution='Original source and weights together; old weights through repaired class produce B. Retention is not new L3/calibration certification.'),
        full_data_candidate=dict(variant='C',source='runtime/final',checkpoint='artifacts/checkpoints/C/checkpoint.pt',
            model_source_sha256=sha(DOC/'runtime/final/eventwm_graph/l2_intermittent/model.py'),checkpoint_sha256=sha(DOC/'artifacts/checkpoints/C/checkpoint.pt'),status='UNPROMOTED_RESEARCH_CANDIDATE'),
        readiness=dict(level1='BOUNDED_CORRECTNESS_CHECKS_PASS_NO_UNQUALIFIED_HISTORICAL_FILTER_CLEARANCE',level2='NOT_ACHIEVED',level3='NOT_ACHIEVED'),
        snapshot=snapshot,training=dict(completed_epochs=training['completed_epochs'],steps=training['optimizer_steps'],seconds=training['seconds'],motion_roots=training['train_roots'],motion_scenes=training['train_scenes'],selected_epoch=52),
        metrics_sha256=sha(DOC/'artifacts/metrics.json'),supplement_metrics_sha256=sha(DOC/'artifacts/reobservation_supplement_metrics.json'),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
        protected_payloads_opened=False,large_inputs_in_git=False,large_inputs_location='data/',collector_control_actions=[])
    dump(DOC/'RELEASE_DECISION.json',decision)
    files={}
    for parent,dirs,names in os.walk(DOC):
        dirs[:]=sorted(d for d in dirs if d!='__pycache__' and not (Path(parent)==DOC and d=='data'))
        for name in sorted(names):
            path=Path(parent)/name
            if path.suffix in ('.pyc','.nbc','.nbi','.tmp') or path==DOC/'INTEGRITY.json':continue
            assert not path.is_symlink(),path
            files[str(path.relative_to(DOC))]=sha(path)
    anchors={}
    for name in ['snapshot/manifest.json','supplement/manifest.json','reobservation_supplement/manifest.json',
                 'frozen_evaluation/dev.npy','frozen_evaluation/motion.json','frozen_evaluation/population_prior.json',
                 'frozen_evaluation/training/checkpoint.pt','frozen_evaluation/snapshot/manifest.json','training/dev.npy','training/motion.json','training/reference_epoch0.pt']:
        path=DOC/'data'/name;anchors[str(path.relative_to(DOC))]=sha(path)
    dump(DOC/'INTEGRITY.json',dict(schema='l2_final_local_release_integrity_1',algorithm='SHA-256',files=files,data_anchor_files=anchors,
        raw_payload_verification='reproduce.py verify --full checks eligible training and authorized dev/return files against immutable manifests',
        compact_shard_verification='reproduce.py verify checks every preparation shard hash',
        exclusions=['INTEGRITY.json self-reference','data payloads individually bound by data-anchor manifests and preparation manifest','Python/Numba caches']))
    print(json.dumps(dict(sealed_files=len(files),data_anchor_files=len(anchors),decision=decision['verdict'],integrity_sha256=sha(DOC/'INTEGRITY.json'))))


if __name__=='__main__':main()
