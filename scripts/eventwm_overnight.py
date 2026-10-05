#!/usr/bin/env python3
"""Fast launch preflight and read-only status; all collection stays in ScaleRunner."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
OUTPUT=REPO/'datasets/raw/eventwm_prod_v1'
PYTHON=Path('/home/haodong/miniforge3/envs/eventwm-falcon/bin/python')
TAG='eventwm-scale-pilot-v1'
sys.path.insert(0,str(REPO/'scripts'))
from collector.core import read
from eventwm_scale import SOURCE,plan,provenance,verify_root


def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def matrix_args():
    scenes=read(SOURCE/'scenes.json'); cases=read(SOURCE/'cases.json'); counts=[0,1,2,4,6]
    names=[s['name'] for s in scenes]
    missing=[]
    if len(names)!=len(set(names)): raise RuntimeError('Duplicate scene names')
    for scene in scenes:
        for count in counts:
            matches=[c for c in cases if c['scene']==scene['name'] and c['humans']==count]
            if len(matches)!=1: raise RuntimeError(f'Missing/duplicate configuration: {scene["name"]} h{count}')
        asset=(SOURCE/scene['local_scene']) if 'local_scene' in scene else REPO/scene['scene_id']
        for path in (asset,SOURCE/scene['navmesh']):
            if not path.exists(): missing.append(str(path))
    roots=len(names)*len(counts)
    return argparse.Namespace(scenes=names,human_counts=counts,roots_per_config=1,max_total_roots=roots,
        max_total_branches=roots*36*2,max_attempts=2,seed=8301,concurrency=1,min_free_gb=150.,
        profiles=['wait','seek_visible','seek_move','seek_lost'],storage_profile='production',missing_assets=missing)


def nearest_existing(path):
    while not path.exists(): path=path.parent
    return path


def preflight():
    if Path(git('rev-parse','--show-toplevel')).resolve()!=REPO: raise RuntimeError('Wrong repository root')
    head=git('rev-parse','HEAD'); tag=git('rev-parse',TAG+'^{commit}')
    if head!=tag: raise RuntimeError(f'HEAD must exactly equal {TAG}: {head} != {tag}')
    if git('status','--porcelain','--untracked-files=normal'):
        raise RuntimeError('Repository has uncommitted files; keep the tagged collection source fixed')
    if Path(sys.prefix).resolve()!=PYTHON.parent.parent.resolve(): raise RuntimeError(f'Wrong Python environment: {sys.prefix}')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','1')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()<1: raise RuntimeError('No visible CUDA GPU')
    args=matrix_args(); expected=plan(args)
    free=shutil.disk_usage(nearest_existing(OUTPUT)).free
    if free<150_000_000_000: raise RuntimeError(f'Free disk below 150 GB: {free/1e9:.2f} GB')
    if (OUTPUT/'manifest.json').exists():
        current=read(OUTPUT/'manifest.json')
        for field in ('schema_version','storage_profile','selection','bounds','provenance'):
            if current[field]!=expected[field]: raise RuntimeError(f'Production manifest mismatch: {field}')
        fields=('root_id','scene','human_count','episode','episode_id','config','seed','episode_sha256','navmesh_sha256')
        if [[r[k] for k in fields] for r in current['roots']] != [[r[k] for k in fields] for r in expected['roots']]:
            raise RuntimeError('Production root/config matrix mismatch')
        for root in current['roots']:
            if root['state']=='completed': verify_root(OUTPUT,root,deep=False)
    elif OUTPUT.exists():
        unexpected=[p.name for p in OUTPUT.iterdir() if p.name not in ('logs','.lock')]
        if unexpected: raise RuntimeError(f'Output has data without a manifest: {unexpected}')
    summary=dict(repository=str(REPO),commit=head,tag=TAG,python=sys.executable,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],gpu=torch.cuda.get_device_name(0),
        output=str(OUTPUT),free_disk_gb=round(free/1e9,2),storage_profile='production',
        missing_assets_will_fail_in_isolated_attempts=args.missing_assets,
        scenes=args.scenes,number_of_scenes=len(args.scenes),human_counts=args.human_counts,
        intended_roots=args.max_total_roots,approximate_primary_queries=33*args.max_total_roots,
        branches_with_audits=36*args.max_total_roots,absolute_branch_reservation_limit=args.max_total_branches,
        max_attempts_per_root=2,concurrency=1,min_free_disk_gb=150)
    print(json.dumps(summary,indent=2),flush=True)
    cmd=[str(PYTHON),'-u',str(REPO/'scripts/eventwm_scale.py'),'run','--output',str(OUTPUT),
        '--scenes',*args.scenes,'--human-counts',*map(str,args.human_counts),'--roots-per-config','1',
        '--max-total-roots',str(args.max_total_roots),'--max-total-branches',str(args.max_total_branches),
        '--max-attempts','2','--seed','8301','--concurrency','1','--root-timeout','1800',
        '--storage-profile','production','--min-free-gb','150','--resume']
    return cmd


def status():
    print('Production output:',OUTPUT)
    manifest=OUTPUT/'manifest.json'
    if manifest.exists():
        m=read(manifest); counts=Counter(r['state'] for r in m['roots'])
        runner=m.get('runner',{}); running=False
        if runner.get('host')==socket.gethostname():
            proc=Path('/proc')/str(runner.get('pid',-1))/'cmdline'
            try: running=b'eventwm_scale.py' in proc.read_bytes()
            except OSError: pass
        print('Run status:', 'RUNNING' if running else 'NOT RUNNING / not discoverable',runner)
        attempted=[r for r in m['roots'] if r['attempts']]
        latest=max(attempted,key=lambda r:r['attempts'][-1]['started']) if attempted else None
        if latest: print('Current/latest root:',latest['root_id'],latest['scene'],'humans=',latest['human_count'],latest['state'])
        print('Completed roots:',counts['completed'],'Pending roots:',counts['pending'],
              'Failed roots:',counts['failed'],'Running/incomplete:',counts['running'])
        print('Retry attempts:',sum(max(0,len(r['attempts'])-1) for r in m['roots']))
        print('Failed attempts:',sum(a['state']=='failed' for r in m['roots'] for a in r['attempts']))
        print('Completed primary query branches:',sum(r.get('stats',{}).get('primary_branches',0) for r in m['roots'] if r['state']=='completed'))
        print('Completed branches including audits:',sum(r.get('stats',{}).get('branches',0) for r in m['roots'] if r['state']=='completed'))
        if (OUTPUT/'aggregate.json').exists():
            a=read(OUTPUT/'aggregate.json')
            for k in ('roots_with_visible_human_history','roots_with_visible_tracks_at_root','roots_with_lost_tracks_at_root'):
                print(k+':',a[k])
    else:
        print('Run status: NOT STARTED (no manifest)')
        print('Completed roots: 0; Pending intended roots:',matrix_args().max_total_roots,'; Failed roots: 0; Retry attempts: 0; Completed query branches: 0')
    if OUTPUT.exists():
        size=int(subprocess.check_output(['du','-sb',str(OUTPUT.resolve())],text=True).split()[0])
    else: size=0
    print(f'Dataset size: {size/1e9:.3f} GB ({size} bytes)')
    print(f'Free disk: {shutil.disk_usage(nearest_existing(OUTPUT)).free/1e9:.2f} GB')
    logs=list((OUTPUT/'logs').glob('*.log')) if (OUTPUT/'logs').exists() else []
    if logs:
        log=max(logs,key=lambda p:p.stat().st_mtime); print('Latest log:',log)
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-16384)); tail=f.read().decode(errors='replace').splitlines()[-20:]
        print('\n'.join(tail))
    else: print('Latest log: none')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['launch','preflight','status']);args=p.parse_args()
    if args.command=='status': status(); return
    cmd=preflight()
    if args.command=='preflight':
        print('PREFLIGHT PASS; no output created and no collection started',flush=True); return
    # Shell captures this process and the exec'ed runner in the same persistent log.
    os.chdir(REPO); os.execv(str(PYTHON),cmd)


if __name__=='__main__': main()
