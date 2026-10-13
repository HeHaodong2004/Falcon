#!/usr/bin/env python3
"""Local scene inventory and isolated preparation using the validated generator."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from collector.core import read,write

REPO=Path(__file__).resolve().parents[1]
REPORT=REPO/'reports/eventwm_mass_collection'
CACHE=REPO/'output/eventwm_mass_assets'

def physical_id(path):
    name=Path(path).name
    return name.replace('.basis.glb','').replace('.glb','').replace('.json.gz','')

def inventory():
    REPORT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    base=(REPO/'data').resolve(); assets={}
    for family,folder,pattern in [('MP3D','scene_datasets/mp3d','*.glb'),('HM3D','scene_datasets/hm3d','*.glb'),
                                  ('other','scene_datasets/habitat-test-scenes','*.glb')]:
        for mesh in sorted((base/folder).rglob(pattern)):
            identity=family+':'+physical_id(mesh)
            row=assets.setdefault(identity,dict(physical_scene_id=identity,name=family.lower()+'_'+physical_id(mesh),family=family,
                scene_id=str(mesh.resolve()),scene_dataset=str((base/'scene_datasets/hm3d/hm3d_basis.scene_dataset_config.json').resolve()) if family=='HM3D' else 'default',
                aliases=[],episode_files=[],present=True))
            row['aliases'].append(str(mesh.resolve()))
            row['source_navmesh']=str(mesh.with_suffix('.navmesh').resolve())
            house=mesh.with_suffix('.house')
            if house.exists():
                row['semantic_rooms']=[dict(id=l.split()[1],raw=l) for l in house.read_text().splitlines() if l.startswith('R ')]
                row['semantic_source']=str(house)
    episode_files=[];refs=set()
    for f in sorted((base/'datasets').rglob('*.json.gz')):
        # Per-scene Habitat content files have one physical scene. Read the first
        # complete scene reference; inventory records this inference explicitly.
        with gzip.open(f,'rt') as stream:prefix=stream.read(262144)
        found=re.findall(r'"scene_id"\s*:\s*"([^"]+)"',prefix)
        family='MP3D' if 'mp3d' in str(f) else 'HM3D' if 'hm3d' in str(f) else 'other'
        keys={family+':'+physical_id(x) for x in found}
        record=dict(path=str(f.resolve()),bytes=f.stat().st_size,scene_references=sorted(keys),
                    reference_method='content file first 256 KiB; all references therein')
        episode_files.append(record);refs.update(keys)
        for key in keys:
            if key in assets:assets[key]['episode_files'].append(str(f.resolve()))
            else:assets[key]=dict(physical_scene_id=key,name=family.lower()+'_'+key.split(':',1)[1],family=family,
                 scene_id=next(x for x in found if physical_id(x)==key.split(':',1)[1]),scene_dataset='default',
                 aliases=[],episode_files=[str(f.resolve())],present=False,exclusion='referenced physical mesh missing locally')
    # Existing controlled catalog references expose unavailable families too.
    original=Path(os.environ.get('EVENTWM_VALIDATED_REPO',str(REPO.parents[1])))
    controlled=original/'evaluation/controlled50/scenes.json'
    if controlled.exists():
        for s in read(controlled):
            family='ReplicaCAD' if 'ReplicaCAD' in s['family'] else 'HM3D' if 'HM3D' in s['family'] else 'MP3D' if 'MP3D' in s['family'] else 'other'
            key=family+':'+physical_id(s['scene_id'])
            refs.add(key)
            if key not in assets:
                mesh=original/s['scene_id'];assets[key]=dict(physical_scene_id=key,name=s['name'],family=family,
                    scene_id=str(mesh.resolve()),scene_dataset=str((original/s['scene_dataset']).resolve()) if s['scene_dataset']!='default' else 'default',
                    aliases=[],episode_files=[],present=mesh.is_file(),exclusion=None if mesh.is_file() else 'controlled scene asset missing locally')
    configs=sorted(str(p.resolve()) for p in base.rglob('*.scene_dataset_config.json'))
    scenes=sorted(assets.values(),key=lambda r:r['physical_scene_id'])
    result=dict(scan_roots=[str(base)],additional_asset_search='find /home/haodong /mnt /media /data (cache/env/output roots pruned); sole scene_datasets under Falcon/data',
        episode_files=episode_files,scene_config_files=configs,controlled_catalog=str(controlled),
        episode_config_files=len(episode_files)+len(configs)+(1 if controlled.exists() else 0),
        unique_episode_referenced_scenes=len(refs),unique_physical_scenes=len(scenes),
        local_scene_assets_present=sum(r['present'] for r in scenes),scenes=scenes)
    write(REPORT/'scene_inventory.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('scenes','episode_files','scene_config_files')}),flush=True)
    return result

def prepare_one(scene):
    name=scene['name'];dest=CACHE/name;meta=dest/'preparation_result.json'
    if meta.exists():return read(meta)
    dest.mkdir(parents=True,exist_ok=True)
    if not scene['present']:
        result=dict(physical_scene_id=scene['physical_scene_id'],compatible=False,reason=scene.get('exclusion'))
    else:
        manifest=dest/'scene.json';write(manifest,[{k:scene[k] for k in ('name','scene_id','scene_dataset','family')}])
        cmd=[sys.executable,str(REPO/'scripts/controlled_matrix.py'),'generate','--scene-manifest',str(manifest),
             '--output',str(dest/'prepared'),'--seed-base','310000']
        started=time.monotonic()
        try:
            with (dest/'prepare.log').open('w') as log:
                process=subprocess.run(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=180,
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet'))
            if process.returncode:raise RuntimeError((dest/'prepare.log').read_text(errors='replace')[-2400:])
            config=read(dest/'prepared/scenes.json')[0];config['navmesh']=str(dest/'prepared'/config['navmesh'])
            import hashlib
            result=dict(physical_scene_id=scene['physical_scene_id'],compatible=True,reason=None,scene=config,
                        navmesh_sha256=hashlib.sha256(Path(config['navmesh']).read_bytes()).hexdigest(),
                        prepared_cases=str(dest/'prepared/cases.json'),seconds=time.monotonic()-started,
                        compatibility_evidence='native physics scene load, canonical Spot navmesh recomputation, connected robot/human episode generation; full v3 rollout smoke is separate')
        except Exception as e:result=dict(physical_scene_id=scene['physical_scene_id'],compatible=False,reason=str(e),seconds=time.monotonic()-started)
    write(meta,result);return result

def prepare(inventory_path,workers):
    inv=read(inventory_path);results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(prepare_one,s):s for s in inv['scenes']}
        for f in as_completed(futures):
            results.append(f.result())
            if len(results)%25==0 or len(results)==len(futures):
                write(REPORT/'scene_preparation_progress.json',dict(done=len(results),total=len(futures),compatible=sum(r['compatible'] for r in results)))
                print(len(results),'/',len(futures),'compatible',sum(r['compatible'] for r in results),flush=True)
    by_id={r['physical_scene_id']:r for r in results}
    for s in inv['scenes']:s.update(by_id[s['physical_scene_id']])
    compatible=[s for s in inv['scenes'] if s['compatible']]
    # Stratified stable hash partition; approximately 12.5% per family.
    for family in sorted({s['family'] for s in compatible}):
        members=sorted([s for s in compatible if s['family']==family],key=lambda s:hashlib.sha256(s['physical_scene_id'].encode()).hexdigest())
        n=round(len(members)*.125) if len(members)>=8 else 0
        for i,s in enumerate(members):s['scene_role']='unseen_evaluation' if i<n else 'training'
    inv['collector_compatible_scenes']=len(compatible)
    inv['families']=dict(Counter(s['family'] for s in compatible))
    inv['semantic_rooms_available']=sum(len(s.get('semantic_rooms',[])) for s in compatible)
    write(REPORT/'scene_inventory.json',inv)
    print('DONE',inv['collector_compatible_scenes'],inv['families'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['inventory','prepare'])
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.action=='inventory':inventory()
    else:prepare(REPORT/'scene_inventory.json',a.workers)
