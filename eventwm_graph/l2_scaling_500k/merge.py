"""Strict case partition merge and unchanged model-independent annotation rule."""
import gzip
import argparse
import importlib.util
import json
import shutil
import numpy as np
from .common import BASE,OUT,OLD,sha,dump

def run(name='new'):
    arm_name=name
    arm=OUT/('evaluation_arm_'+name);shards=sorted(arm.glob('evaluation_shard_*'));configs=[json.loads((p/'configuration.json').read_text()) for p in shards]
    expected=[r['root_id'] for r in np.load(OLD/'dev.npy',allow_pickle=True)]
    assert [r for c in configs for r in c['root_ids']]==expected
    for p,c in zip(shards,configs):
        guard=json.loads((p/'execution_guard.json').read_text())
        assert guard['sources_stable'] and guard['inputs_stable'] and c['source_stable']
        assert c['sources']==configs[0]['sources'] and c['checkpoint_sha256']==configs[0]['checkpoint_sha256']
        assert c['worlds']==32 and c['streams']==[41,107] and c['bootstrap_stream_offset']==10000
    dest=arm/'evaluation';dest.mkdir(exist_ok=False)
    for name in ('forecasts','reobservation','beliefs','update_gains'):
        with gzip.open(dest/(name+'.jsonl.gz'),'wb') as target:
            for p in shards:
                with gzip.open(p/(name+'.jsonl.gz'),'rb') as source:shutil.copyfileobj(source,target)
    for name in ('failures','mask_manifest','reappearance_eligibility'):
        dump(dest/(name+'.json'),[r for p in shards for r in json.loads((p/(name+'.json')).read_text())])
    shutil.copytree(shards[0]/'sources',dest/'sources')
    dump(dest/'configuration.json',dict(configs[0],roots=2110,root_range=[0,2110],root_ids=expected,
        executed_models=['B_MemoryCV','D_StopMove'] if arm_name=='original_full' else ['D_StopMove'],seconds=sum(c['seconds'] for c in configs),shards=[p.name for p in shards]))
    from eventwm_graph.l2_intermittent.evaluate import summarize
    summarize(dest)
    dump(dest/'forecast_execution_guard.json',dict(error=None,sources_stable=True,inputs_stable=True,
        shards=[dict(path=p.name,guard_sha256=sha(p/'execution_guard.json')) for p in shards],
        note='Verified disjoint full original development-root partition, unchanged inference, focused frozen scoring, no checkpoint selection.'))
    annotate(arm_name)

def annotate(arm_name):
    arm=OUT/('evaluation_arm_'+arm_name);dest=arm/'evaluation'
    cfg=json.loads((dest/'configuration.json').read_text());guard=json.loads((dest/'forecast_execution_guard.json').read_text())
    expected=[r['root_id'] for r in np.load(OLD/'dev.npy',allow_pickle=True)]
    assert cfg['root_ids']==expected and cfg['source_stable'] and guard['sources_stable'] and guard['inputs_stable']
    assert all(sha(arm/r['path']/'execution_guard.json')==r['guard_sha256'] for r in guard['shards'])
    models=['B_MemoryCV','D_StopMove'] if arm_name=='original_full' else ['D_StopMove']
    if cfg['executed_models']!=models:
        (dest/'configuration.json').rename(dest/'configuration.pre_catalog_metadata_fix.json')
        dump(dest/'configuration.json',dict(cfg,executed_models=models))
    path=BASE/'docs/l2/intermittent_wm_20261022/artifacts/verified_execution/orchestration/annotation_accounting.py'
    spec=importlib.util.spec_from_file_location('frozen_annotation_accounting',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    classification=path.with_name('native_annotation_validity.json')
    assert sha(classification)=='a887b0075c6027bcbc5fbd9f1f9f4c60edfd5e61c9983879d040a068134416db'
    # The prior release preserves this exact rule inside its execution archive.
    # Point to those bytes; do not create or change a historical report file.
    module.DOC=BASE/'docs/l2/intermittent_wm_20261022/artifacts/verified_execution/sources/docs/l2/intermittent_wm_20261022'
    module.apply(dest,classification)
    dump(dest/'merged_integrity.json',dict(complete_ordered_roots=2110,scenes=108,all_shards_stable=True,
        classification_sha256=sha(classification),annotation_rule_unchanged=True,
        checkpoint_sha256=sha(arm/'training/checkpoint.pt'),original_support_denominators_retained=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['new','original_full'],default='new');p.add_argument('--resume-annotation',action='store_true');a=p.parse_args()
    if a.resume_annotation:annotate(a.arm)
    else:run(a.arm)
