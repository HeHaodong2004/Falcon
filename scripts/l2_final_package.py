"""Curate a standalone local release, never link to live collector payloads."""
import argparse,gzip,json,os,shutil,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from eventwm_graph.l2_final.common import OUT,DOC,OLD,PRIOR,PREVIOUS,sha,dump


def copy(source,target):
    source=Path(source);target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and sha(source)==sha(target):return
    shutil.copyfile(source,target)


def tree(source,target):
    for p in sorted(Path(source).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.nbc','.nbi'):
            copy(p,Path(target)/p.relative_to(source))


def own_links(source,target):
    """Only immutable independent copies already owned by this campaign."""
    source=Path(source).resolve();target=Path(target)
    assert source.is_relative_to(OUT.resolve()) and not source.is_symlink()
    for p in sorted(source.rglob('*')):
        if not p.is_file():continue
        assert not p.is_symlink()
        q=target/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists():assert os.path.samefile(p,q)
        else:os.link(p,q)


ASSETS=('docs/l2/graph_wm_120k_20261021/artifacts/population_prior.json',
        'docs/l2/graph_wm_120k_20261021/artifacts/motion.json',
        'docs/l2/graph_wm_120k_20261021/artifacts/checkpoints',
        'docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints/native_events_v5_scene.json',
        'docs/l2/intermittent_wm_20261022/artifacts/speed_reference_34425709',
        'docs/l2/intermittent_wm_20261022/third_party',
        'docs/l2/intermittent_wm_20261022/PROTOCOL.md',
        'docs/l2/intermittent_wm_20261022/EVALUATION_FREEZE.md')


def runtime(source,target):
    tree(source/'eventwm_graph',target/'eventwm_graph')
    for name in ASSETS:
        p=BASE/name
        if p.is_dir():tree(p,target/name)
        elif p.exists():copy(p,target/name)
    copy(BASE/'docs/l2/intermittent_wm_20261022/artifacts/verified_execution/sources/docs/l2/intermittent_wm_20261022/ANNOTATION_CENSOR_RULE.md',
         target/'docs/l2/intermittent_wm_20261022/ANNOTATION_CENSOR_RULE.md')
    for name in ('l2_partial_structural_validation_kernel.py','l2_partial_structural_api_example.py'):
        copy(BASE/'scripts'/name,target/'scripts'/name)


def core():
    art=DOC/'artifacts';data=DOC/'data'
    runtime(BASE,DOC/'runtime/final')
    for arm,path in [('A',PREVIOUS/'evaluation_runtime_new'),('B',OUT/'evaluation_runtime_original_repaired'),('C',OUT/'evaluation_runtime_new'),
                     ('B_public',OUT/'evaluation_runtime_original_public_final'),('C_public',OUT/'evaluation_runtime_public_final')]:
        runtime(path,DOC/'runtime'/('control_'+arm))
    for name in ('configuration.json','training.json','learning.json','checkpoint.pt','epoch0.pt','last.pt'):
        copy(OUT/'training'/name,art/'training'/name)
    tree(OUT/'training/sources',art/'training/sources')
    for arm,path in [('A',PREVIOUS),('C',OUT)]:
        for name in ('checkpoint.pt','epoch0.pt','training.json'):
            copy(path/'training'/name,art/'checkpoints'/arm/name)
        copy(path/'motion.json',art/'checkpoints'/arm/'motion.json')
    copy(PRIOR/'training/checkpoint.pt',art/'checkpoints/historical_3212_roots.pt')
    for name in ('final_snapshot_manifest.json','preparation.json','preparation_contract.json','preparation_main.json','environment.json',
                 'native_census_main.json','native_census_supplement.json','cache_reuse_equivalence.json','inference_reuse_equivalence.json',
                 'factor_kernel_probe.json','preprocessing_efficiency_change.json','regressions.json','campaign_inventory_initial.json'):
        copy(OUT/name,art/'audit'/name)
    for name in ('snapshot','supplement'):
        own_links(OUT/name,data/name)
        with gzip.open(art/'audit'/(name+'_manifest.json.gz'),'wb') as f:f.write((OUT/name/'manifest.json').read_bytes())
    prep=json.loads((OUT/'preparation.json').read_text())
    for shard in prep['shards']:copy(OUT/shard['path'],data/'training'/shard['path'])
    copy(PRIOR/'dev.npy',data/'training/dev.npy');copy(PRIOR/'motion.json',data/'training/motion.json')
    copy(PREVIOUS/'training/epoch0.pt',data/'training/reference_epoch0.pt')
    for name in ('dev.npy','motion.json','population_prior.json','training/checkpoint.pt','snapshot/manifest.json'):
        copy(OLD/name,data/'frozen_evaluation'/name)
    manifest=json.loads((OLD/'snapshot/manifest.json').read_text())
    for row in manifest['rows']:
        if row['split']=='dev':tree(OLD/'snapshot/roots'/row['root_id'],data/'frozen_evaluation/snapshot/roots'/row['root_id'])
    returns=OUT/'reobservation_supplement'
    if (returns/'manifest.json').exists():
        own_links(returns/'roots',data/'reobservation_supplement/roots')
        copy(returns/'manifest.json',data/'reobservation_supplement/manifest.json')
    fixed=BASE/'docs/l2/intermittent_wm_20261022/artifacts/verified_execution'
    for name in ('annotation_accounting.py','native_annotation_validity.json'):
        copy(fixed/'orchestration'/name,art/'evaluation_tools'/name)
    copy(BASE/'docs/l2/intermittent_wm_20261022/reviews/reviewer_c_benchmark.py',art/'evaluation_tools/reviewer_c_benchmark.py')
    tree(BASE/'tests',DOC/'runtime/tests')
    copy(BASE/'scripts/l2_final_tests.py',DOC/'runtime/l2_final_tests.py')
    for name in ('l2_final_reobservation_score.py','l2_final_reobservation_extract.py','l2_final_reobservation_campaign.py'):
        copy(BASE/'scripts'/name,art/'evaluation_tools'/name)
    consumer=OUT/'reviewer_c_checks/final_C_restored_bootstrap/committed_consumers'
    tree(consumer,DOC/'runtime/committed_consumers')
    copy(BASE/'scripts/l2_final_reproduce.py',DOC/'reproduce.py')
    print('Core runtime, full fitting inputs and independent data copies packaged',flush=True)


def final():
    art=DOC/'artifacts'
    runtime(BASE,DOC/'runtime/final')
    copy(BASE/'scripts/l2_final_reproduce.py',DOC/'reproduce.py')
    assert (OUT/'evaluation_campaign_complete.json').exists() and (OUT/'public_final_complete.json').exists()
    for arm,path in [('A',PREVIOUS/'evaluation_arm_new/evaluation'),('B',OUT/'evaluation_arm_original_repaired/evaluation'),('C',OUT/'evaluation_arm_new/evaluation'),
                     ('B_public',OUT/'evaluation_arm_original_public_final/evaluation_shard_0000_0528'),('C_public',OUT/'evaluation_arm_public_final/evaluation_shard_0000_0528'),
                     ('B_rejected_prefix',OUT/'evaluation_arm_original_public/evaluation_shard_0000_0528'),('C_rejected_prefix',OUT/'evaluation_arm_public/evaluation_shard_0000_0528')]:
        tree(path,art/'evaluation'/arm)
    for arm,path in [('A',OUT/'benchmark_runtime_A/output/benchmark_A/complete'),('B',OUT/'benchmark_B_final/complete'),('C',OUT/'benchmark_C/complete')]:
        tree(path,art/'latency'/arm)
    for name in ('metrics.json','reobservation_supplement_metrics.json','evaluation_campaign_complete.json','public_final_complete.json'):
        copy(OUT/name,art/name)
    for name in ('regression_diagnostics.json','fresh_process_startup.json'):
        copy(OUT/name,art/'audit'/name)
    copy(BASE/'docs/l2/intermittent_wm_20261022/reviews/reviewer_c_benchmark.py',art/'evaluation_tools/reviewer_c_benchmark.py')
    for name in ('portable_smoke.log','portable_fit_v2.log','portable_evaluation_v4.log','portable_evaluation_A.log','portable_regressions_v3.log'):
        copy(OUT/'logs'/name,art/'reproduction_checks'/name)
    copy(OUT/'portable_regressions_v3/output/l2_final_800k_20261023/regressions.json',art/'reproduction_checks/portable_regressions.json')
    for arm in ('A','B','C'):tree(OUT/'reobservation_supplement'/('evaluation_'+arm),art/'reobservation_supplement'/arm)
    for name in ('reviewer_a_checks','reviewer_c_checks'):tree(OUT/name,art/name)
    for arm,name in [('B','original_repaired'),('C','new'),('B_public','original_public_final'),('C_public','public_final')]:
        path=OUT/('evaluation_arm_'+name)
        for filename in ('public_initialization.jsonl','physical.json','physical_integrity.json'):
            if (path/filename).exists():copy(path/filename,art/'evaluation'/arm/filename)
    audit=Path('/research-user/workspace/projects/eventwm-falcon/output/l2_history_audit_20261022')
    for p in audit.rglob('*.md'):copy(p,art/'prior_audit'/p.relative_to(audit))
    print('All measured results and review evidence packaged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['core','final']);a=p.parse_args()
    (core if a.stage=='core' else final)()
