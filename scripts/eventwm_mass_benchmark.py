#!/usr/bin/env python3
"""Bounded root concurrency benchmark with retained identical workload recipes."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collector.core import read,write,digest
from eventwm_mass_collection import REPO,REPORT,hashes,git_head,COUNTS,BANNED_SUFFIXES
from collector.v3 import PROTOCOL
from collector.static_feasibility import CHECKER_VERSION

BASE=Path('/home/haodong/workspace/projects/eventwm-falcon')


def make_panel():
    # Reuse legal, validated recipes without selecting by class/outcome. First
    # completed pilot root for each requested human count and a non-Castle smoke.
    inv=read(REPORT/'scene_inventory.json')
    byid={s['physical_scene_id']:s for s in inv['scenes'] if s.get('compatible')}
    source=BASE/'datasets/raw/eventwm_social_outcome_v3_wm_train_pilot_01'
    manifest=read(source/'manifest.json');roots=[r for r in manifest['roots'] if r['state']=='completed']
    catalog=BASE/'evaluation/controlled50';cases={r['tag']:r for r in read(catalog/'cases.json')};scenes={r['name']:r for r in read(catalog/'scenes.json')}
    panel=[]
    for count in (0,1,2,4,6):
        root=next(r for r in roots if r['human_count']==count)
        recipe=read(source/root['path']/'roots/replay.json')
        case=cases[recipe['case']];ep=read(catalog/case['episode'])
        scene=copy.deepcopy(byid['other:skokloster-castle'])
        scene['scene']=dict(scenes['castle'],scene_id=scene['scene_id'],navmesh=str(catalog/scenes['castle']['navmesh']))
        scene.update(name='castle',scene_role='training')
        panel.append(dict(recipe=recipe,case=case,episode=ep,scene=scene,human_count=count,origin=str(source/root['path'])))
    fixture=read(BASE/'configs/eventwm_v2_noncastle_smoke_prefix.json')
    case=cases[fixture['case']];s=scenes[case['scene']];scene=copy.deepcopy(byid['HM3D:NBg5UqG3di3'])
    scene['scene']=dict(s,scene_id=scene['scene_id'],scene_dataset=scene['scene_dataset'],navmesh=str(catalog/s['navmesh']))
    scene.update(name=case['scene'],scene_role='training')
    template=copy.deepcopy(panel[0]['recipe'])
    template.update(case=case['tag'],seed=fixture['seed'],prefix_controls=fixture['prefix_controls'])
    template.pop('sampling',None);template.pop('sampling_profile',None)
    panel.insert(2,dict(recipe=template,case=case,episode=read(catalog/case['episode']),scene=scene,human_count=0,origin='validated noncastle prefix fixture'))
    write(REPORT/'benchmark_panel.json',panel)
    return panel


def high_panel(count):
    """First completed pilot recipes, without consulting outcomes, for one wave.

    C1 includes a slow HM3D fixture. Compare the high wave both to full C1 and
    its matched Castle portion; do not claim an all-scene throughput estimate.
    """
    base=make_panel();castle=next(e for e in base if e['scene']['name']=='castle')
    source=BASE/'datasets/raw/eventwm_social_outcome_v3_wm_train_pilot_01'
    roots=[r for r in read(source/'manifest.json')['roots'] if r['state']=='completed']
    # Round-robin human counts using original manifest order, no label selection.
    buckets={h:[r for r in roots if r['human_count']==h] for h in COUNTS}
    selected=[]
    while len(selected)<count:
        for h in COUNTS:
            if buckets[h] and len(selected)<count:selected.append(buckets[h].pop(0))
        if not any(buckets.values()) and len(selected)<count:raise ValueError('Not enough distinct validated roots')
    result=[]
    catalog=BASE/'evaluation/controlled50';cases={r['tag']:r for r in read(catalog/'cases.json')}
    for r in selected:
        recipe=read(source/r['path']/'roots/replay.json');case=cases[recipe['case']]
        result.append(dict(recipe=recipe,case=case,episode=read(catalog/case['episode']),scene=castle['scene'],
                           human_count=r['human_count'],origin=str(source/r['path'])))
    write(REPORT/'high_concurrency_panel.json',result)
    return result


def prepare_dataset(output,concurrency,panel):
    if output.exists():raise ValueError('New benchmark output required')
    output.mkdir(parents=True)
    scenes=[];roots=[];cases=[]
    for i,entry in enumerate(panel):
        recipe=copy.deepcopy(entry['recipe']);rid='bench_'+digest(dict(original=recipe['root_id'],panel_index=i))[:20]
        recipe['root_id']=rid;recipe['debug_static_reference']=False
        case=copy.deepcopy(entry['case']);case['tag']=f'benchmark_{i}';case['episode']=f'episodes/{rid}.json';recipe['case']=case['tag']
        write(output/'catalogue'/case['episode'],entry['episode']);cases.append(case)
        write(output/'recipes'/f'{rid}.json',recipe)
        scene=entry['scene']
        if not any(s['name']==scene['name'] for s in scenes):scenes.append(scene)
        roots.append(dict(root_id=rid,sequence=i,scene=scene['name'],physical_scene_id=scene['physical_scene_id'],family=scene['family'],
            scene_role='training',role='benchmark_only',split_group=rid,human_count=entry['human_count'],root_profile='benchmark',
            sampling_profile=recipe.get('sampling_profile','explicit'),requested_category=recipe.get('sampling',{}).get('target_category'),
            seed=recipe['seed'],recipe=recipe,state='pending',attempts=[]))
    write(output/'catalogue/cases.json',cases);write(output/'catalogue/scenes.json',[s['scene'] for s in scenes])
    write(output/'manifest.json',dict(schema_version=4,prefix_truth_encoding='json-gzip-v1',branch_truth_encoding='jsonl-gzip-v1',sr_contract_version='binary-execution-safety-v1',contact_serialization='sorted-exact-multiset-v1',protocol_version=PROTOCOL,protocol=PROTOCOL,checker=CHECKER_VERSION,
         storage_profile='structured_production',collector_storage_profile='production',frozen_commit=git_head(),source_hashes=hashes(),
         roots=roots,scenes=scenes,target_primary_queries=33*len(panel),max_completed_roots=len(panel),min_free_gb=150,
         state='prepared_not_started',sessions=[],next_sequence=len(panel),master_seed=990000,max_attempts=1,
         root_timeout_seconds=12000,collection_seconds=0.))


def summarize(output,concurrency):
    m=read(output/'manifest.json');a=read(output/'aggregate.json')
    panel_file=REPORT/('high_concurrency_panel.json' if concurrency>1 else 'benchmark_panel.json')
    panel=read(panel_file)
    samples=[json.loads(l) for l in (output/'resource_samples.jsonl').read_text().splitlines()]
    means=lambda name:sum(s.get(name,0) for s in samples)/max(1,len(samples))
    gpus={}
    for gpu in (0,1):
        gs=[g for s in samples for g in s['gpus'] if g['index']==gpu]
        gpus[str(gpu)]=dict(mean_utilization=sum(g['utilization'] for g in gs)/max(1,len(gs)),
                           max_vram_mib=max([g['memory_used_mib'] for g in gs] or [0]))
    failures=Counter(a['failed_reasons'])
    result=dict(concurrency=concurrency,output=str(output),completed_roots=a['completed_roots'],failed_roots=a['failed_roots'],
        primary_queries=a['primary_queries'],wall_seconds=m['collection_seconds'],root_seconds=sum(r['attempts'][-1].get('seconds',0) for r in m['roots'] if r['state']=='completed'),roots_hour=a['completed_roots']/m['collection_seconds']*3600,
        queries_hour=a['queries_per_hour'],failure_rate=a['failed_roots']/len(m['roots']),failures=dict(failures),
        static_violations=a['static_violations'],integrity_failures=failures.get('integrity_failure',0),
        cpu_percent=means('cpu_percent'),cpu_percent_peak=max([s.get('cpu_percent',0) for s in samples] or [0]),cpu_load1_peak=max([s['load'][0] for s in samples] or [0]),cpu_load1_mean=sum(s['load'][0] for s in samples)/max(1,len(samples)),
        ram_peak_gb=max([s['ram_used_gb'] for s in samples] or [0]),gpus=gpus,
        disk_write_mb_s=means('disk_write_mb_s'),io_wait_percent=means('io_wait_percent'),io_wait_peak_percent=max([s.get('io_wait_percent',0) for s in samples] or [0]),
        startup_stagger_seconds=m['sessions'][-1].get('startup_stagger_seconds',0),no_image_video_files=not any(p.is_file() and p.suffix.lower() in BANNED_SUFFIXES for p in output.rglob('*')),
        source_hashes=m['source_hashes'],all_completed_root_native_audits_pass=not bool(a['failed_roots']),
        pass_correctness=bool(a['completed_roots']==len(m['roots']) and not (a['failed_roots'] or a['static_violations'] or m['state'].startswith(('global','stopped_')))),
        sizes=[dict(root_id=r['root_id'],source_recipe=panel[r['sequence']]['origin'],scene=r['scene'],human_count=r['human_count'],category=r['stats']['legal_state']['category'],
                    bytes=r['stats']['bytes'],primary_queries=r['stats']['primary_queries'],seconds=r['attempts'][-1]['seconds']) for r in m['roots'] if r['state']=='completed'])
    write(REPORT/f'benchmark_c{concurrency}.json',result);print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--concurrency',type=int,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    panel=read(REPORT/'benchmark_panel.json') if (REPORT/'benchmark_panel.json').exists() else make_panel()
    if a.concurrency not in (1,16,24):raise SystemExit('Intermediate benchmark sweep disabled')
    workload=panel[:4] if a.concurrency==1 else high_panel(a.concurrency)
    out=REPO/f'output/eventwm_social_outcome_v3_benchmark_c{a.concurrency}'
    prepare_dataset(out,a.concurrency,workload)
    if not a.prepare_only:
        cmd=[sys.executable,'-u',str(REPO/'scripts/eventwm_mass_collection.py'),'run','--output',str(out),
             '--benchmark','--stop-after-attempts',str(len(workload)),'--target',str(33*len(workload)),
             '--max-roots',str(len(workload)),'--concurrency',str(a.concurrency),'--gpus','1' if a.concurrency==1 else '0,1']
        subprocess.run(cmd,cwd=REPO,check=False)
        result=summarize(out,a.concurrency)
        if not result['pass_correctness']:sys.exit(1)
