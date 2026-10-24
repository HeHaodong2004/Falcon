#!/usr/bin/env python3
"""Hard-bounded root-sampling diagnostic; calls the canonical one-root collector.

Two isolated root processes only reduce test runtime. No branch distribution,
production/resume/retry scheduler, label balancing, or changing job table exists.
"""
import argparse
from dataclasses import asdict
import hashlib
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from collector.core import digest,read,write
from collector.v3 import PROTOCOL,guard_output
from collector.v2 import QueryConfig
from collector.static_feasibility import CHECKER_VERSION,StaticConfig
from collector.natural_sampling import natural_selection
from eventwm_scale import provenance,seals
from eventwm_v3 import summarize

REPO=Path(__file__).resolve().parents[1]


def job_table(config,comparison=False):
    cases=read(REPO/'evaluation/controlled50/cases.json')
    fixture=read(REPO/'configs/eventwm_v2_noncastle_smoke_prefix.json')
    specifications=[]
    for repeat in range(config['natural_repeats_castle']):
        for count in config['human_counts_castle']:
            specifications.append(('castle',count,repeat))
    specifications += [('hm3d_NBg5UqG3di3',0,i) for i in range(config['noncastle_zero_human_roots'])]
    if len(specifications)>120:raise ValueError('This diagnostic is bounded to 120 requested natural roots')
    jobs=[]
    for index,(scene,count,repeat) in enumerate(specifications):
        if comparison and (scene!='castle' or repeat>=config['comparison_repeats'] or count not in config['comparison_human_counts']):continue
        profiles=config['comparison_profiles'] if comparison else ['natural']
        for profile in profiles:
            case=next(c for c in cases if c['scene']==scene and c['humans']==count)
            seed=config['master_seed']+index if scene=='castle' else fixture['seed']
            prelude=[] if scene=='castle' else fixture['prefix_controls']
            sampling=dict(profile=profile,sampler_seed=config['master_seed']+10000+index,
                window_start_step=len(prelude)+config['window_start_after_prelude'],
                window_end_step=len(prelude)+config['window_end_after_prelude'],
                prefix_movement_queries=config['prefix_movement_queries'],prelude_controls=prelude)
            recipe=dict(protocol_version=PROTOCOL,case=case['tag'],seed=seed,sampling=sampling,
                sampling_profile=profile,sampling_profile_family='natural' if profile=='natural' else 'coverage',
                mechanism='fresh_process_seeded_episode_plus_exact_prefix',query_config=asdict(QueryConfig(movement_queries=config['movement_queries'])),
                static_config=asdict(StaticConfig()),debug_static_reference=False,sampling_audit=True)
            if prelude:recipe['prefix_fixture_sha256']=digest(fixture)
            rid='v3n_'+digest(recipe)[:16];recipe['root_id']=rid
            jobs.append(dict(root_id=rid,scene=scene,human_count=count,root_profile=profile,
                sampling_profile=profile,sampling_profile_family=recipe['sampling_profile_family'],
                matched_episode_index=index,repeat=repeat,seed=seed,recipe=recipe,
                selected_root=natural_selection(seed,sampling) if profile=='natural' else None))
    if comparison and len(jobs)>24:raise ValueError('Comparison limited to 24 roots')
    return jobs


def run(args, jobs=None, config=None):
    cfg=config if config is not None else read(args.config)
    jobs=job_table(cfg,args.comparison) if jobs is None else jobs
    if args.max_roots is not None:
        if args.max_roots<=0:raise ValueError('max-roots must be positive')
        jobs=jobs[:args.max_roots]
    if not jobs:raise ValueError('Empty bounded workload')
    out=guard_output(args.output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty diagnostic dataset; no append or retries')
    out.mkdir(parents=True);write(out/'jobs.json',jobs)
    manifest=dict(schema_version=3,protocol_version=PROTOCOL,checker_version=CHECKER_VERSION,
        purpose=cfg.get('purpose','coverage_comparison' if args.comparison else 'natural_sampling_pilot'),configuration=cfg,
        query_config=asdict(QueryConfig(movement_queries=cfg['movement_queries'])),
        provenance=provenance(),roots=[],state='running',visualization=False,job_limit=args.max_roots)
    manifest['provenance']['source_sha256'][str(Path(__file__).relative_to(REPO))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(out/'manifest.json',manifest)
    active={};next_job=0;start=time.monotonic();last=0;stop=False
    def request_stop(signum, frame):
        nonlocal stop
        stop=True;manifest['state']='stopped_by_user'
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    slots=int(cfg['max_root_processes'])
    if slots not in (1,2):raise ValueError('Only 1–2 isolated processes in this bounded test')
    while active or (next_job<len(jobs) and not stop):
        if sum(r['state']=='completed' for r in manifest['roots']) >= cfg.get('max_completed_roots',len(jobs)):
            stop=True
            if manifest['state']=='running':manifest['state']='completed_root_budget_reached'
        for slot in range(slots):
            if slot in active or next_job==len(jobs) or stop:continue
            job=jobs[next_job];next_job+=1
            root={k:v for k,v in job.items() if k!='recipe'}
            root.update(state='running',path='roots/'+job['root_id'],gpu=cfg['gpu_devices'][slot%len(cfg['gpu_devices'])])
            manifest['roots'].append(root);path=out/root['path']
            recipe=out/'recipes'/f"{root['root_id']}.json";write(recipe,job['recipe'])
            logpath=out/'logs'/f"{root['root_id']}.log";logpath.parent.mkdir(exist_ok=True)
            log=logpath.open('w');env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(root['gpu'])
            cmd=[sys.executable,'-u',str(REPO/'scripts/eventwm_collect.py'),'collect','--output',str(path),
                '--recipe',str(recipe),'--protocol','v3','--movement-queries',str(cfg['movement_queries']),
                '--max-branches',str(4*cfg['movement_queries']+4),'--storage-profile','production','--no-render']
            proc=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=env)
            root['pid']=proc.pid;active[slot]=(proc,log,root,time.monotonic())
            write(out/'manifest.json',manifest)
        for slot,(proc,log,root,t0) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();root.update(returncode=code,seconds=time.monotonic()-t0);path=out/root['path']
            if code:
                root['state']='failed';root['error']=(out/'logs'/f"{root['root_id']}.log").read_text(errors='replace')[-2400:]
                if (path/'audits/STATIC_FEASIBILITY_VIOLATION.json').exists():
                    root['state']='quarantined';stop=True;manifest['state']='stopped_static_violation'
            else:
                root.update(state='completed',seals=seals(path))
                write(path/'COMPLETE.json',dict(root_id=root['root_id'],seal_digest=digest(root['seals'])))
            del active[slot];write(out/'manifest.json',manifest)
            print(root['state'].upper(),root['root_id'],root['scene'],root['human_count'],root['root_profile'],round(root['seconds'],1),flush=True)
        if time.monotonic()-last>30:
            last=time.monotonic()
            status=dict(elapsed_seconds=last-start,completed=sum(r['state']=='completed' for r in manifest['roots']),
                failed=sum(r['state']=='failed' for r in manifest['roots']),requested=len(jobs),
                running=[dict(root_id=r['root_id'],pid=r['pid'],gpu=r['gpu']) for _,_,r,_ in active.values()])
            write(out/'status.json',status);print('STATUS',status,flush=True)
        if active:time.sleep(1)
    if manifest['state']=='running':manifest['state']='finished_bounded_diagnostic'
    write(out/'manifest.json',manifest)
    result=summarize(out,manifest);print(result,flush=True)
    # Root feasibility/prefix failures remain in the denominator; never replace
    # based on social outcomes or quietly call the complete-case set unbiased.
    return 1 if manifest['state']=='stopped_static_violation' else 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=REPO/'configs/eventwm_v3_natural_pilot.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--comparison',action='store_true')
    p.add_argument('--max-roots',type=int)
    args=p.parse_args();sys.exit(run(args))

if __name__=='__main__':main()
