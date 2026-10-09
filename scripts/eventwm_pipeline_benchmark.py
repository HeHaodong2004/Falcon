#!/usr/bin/env python3
"""Native bounded A/B: ORIGINAL coordinator vs candidate, fresh identical stream.

Both use C36, original science and unique output directories. No production
launch is hidden here. PASS requires exact input/job/label agreement, >=100
retained queries, and the requested measured end-to-end speedup.
"""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from collections import Counter

from collector.core import read, load_wm_input
from eventwm_validate_staged import durable_json
import eventwm_combined_runner as old
import eventwm_mass_collection as mass
import eventwm_pipeline_runner as new


def workload_summary(output):
    m=read(output/'manifest.json')
    roots=[]
    totals=Counter()
    for root in m['roots']:
        if not root['attempts'] or 'path' not in root['attempts'][-1]: continue
        path=output/root['attempts'][-1]['path']
        for file in (path/'diagnostic_jobs').glob('*/result.json'):
            result=read(file)
            totals['native_job_wall_seconds']+=result['wall_seconds']
            totals['native_job_cpu_seconds']+=result['cpu_seconds']
            if file.parent.name=='root':
                row=dict(root_id=root['root_id'],wall_seconds=result['wall_seconds'],cpu_seconds=result['cpu_seconds'],
                         prefix_inclusive=result['metrics'].get('prefix_inclusive',0.),
                         query_generation=result['metrics'].get('query_generation',0.))
                roots.append(row)
    return dict(completed_roots=sum(r['state']=='completed' for r in m['roots']),proposals=len(m['roots']),
                failures=dict(Counter(r['attempts'][-1].get('reason') for r in m['roots'] if r['state']=='failed')),
                totals=dict(totals),slowest_root_jobs=sorted(roots,key=lambda r:r['wall_seconds'],reverse=True)[:8],
                timer_note='Worker wall sums are slot seconds, not elapsed time; nested phase timers overlap.')


def compare(a,b):
    ma,mb=read(a/'manifest.json'),read(b/'manifest.json')
    ra,rb=ma['roots'],mb['roots']
    errors=[]
    if [r['root_id'] for r in ra] != [r['root_id'] for r in rb]: errors.append('proposal order/IDs')
    if len(ra)!=len(rb): errors.append('proposal counts')
    jobs=0
    public_inputs=0
    for x,y in zip(ra,rb):
        rid=x['root_id']
        for key in ('recipe','episode_sha256','start_region','state','query_ids'):
            if x.get(key)!=y.get(key): errors.append(rid+': '+key)
        if [z.get('reason') for z in x['attempts']] != [z.get('reason') for z in y['attempts']]:
            errors.append(rid+': attempt/retry history')
        if x['state']=='failed':
            if x['attempts'][-1].get('reason') != y['attempts'][-1].get('reason'):
                errors.append(rid+': failure reason')
        if 'path' not in x['attempts'][-1] or 'path' not in y['attempts'][-1]: continue
        pa,pb=a/x['attempts'][-1]['path'],b/y['attempts'][-1]['path']
        ja={p.parent.name:p for p in (pa/'diagnostic_jobs').glob('*/result.json')}
        jb={p.parent.name:p for p in (pb/'diagnostic_jobs').glob('*/result.json')}
        if set(ja)!=set(jb): errors.append(rid+': job IDs')
        for wid in sorted(set(ja)&set(jb)):
            va,vb=read(ja[wid]),read(jb[wid])
            for key in ('scientific_hashes','label','error','full_native_trace_scientific_sha256'):
                if va.get(key)!=vb.get(key): errors.append(rid+'/'+wid+': '+key)
            ea,eb=pa/'audits/workers'/wid/'episode.json.gz',pb/'audits/workers'/wid/'episode.json.gz'
            if ea.exists()!=eb.exists(): errors.append(rid+'/'+wid+': episode presence')
            elif ea.exists():
                with gzip.open(ea,'rt') as fa,gzip.open(eb,'rt') as fb:
                    if json.load(fa)!=json.load(fb): errors.append(rid+'/'+wid+': decoded episode')
            jobs+=1
        for rel in ('branch_truth/branches.jsonl.gz','roots/prefix_truth.json.gz'):
            fa,fb=pa/rel,pb/rel
            if fa.exists()!=fb.exists():errors.append(rid+': '+rel+' presence')
            elif fa.exists():
                with gzip.open(fa,'rb') as sa,gzip.open(fb,'rb') as sb:
                    while True:
                        ca,cb=sa.read(1024*1024),sb.read(1024*1024)
                        if ca!=cb:errors.append(rid+': '+rel+' decoded bytes');break
                        if not ca:break
        for rel in ('labels/labels.json','queries/queries.json','legal_observations/history.json'):
            if (pa/rel).exists()!=(pb/rel).exists(): errors.append(rid+': '+rel+' presence')
            elif (pa/rel).exists() and read(pa/rel)!=read(pb/rel): errors.append(rid+': '+rel)
        for qid in set(x.get('query_ids',[])) & set(y.get('query_ids',[])):
            if load_wm_input(pa,qid) != load_wm_input(pb,qid): errors.append(rid+': public WM input '+qid)
            public_inputs+=1
    aa,ab=mass.aggregate(ma),mass.aggregate(mb)
    if aa['primary_queries']!=ab['primary_queries']: errors.append('retained count')
    if aa['static_violations'] or ab['static_violations']: errors.append('static violations')
    sa,sb=ma['sessions'][-1]['seconds'],mb['sessions'][-1]['seconds']
    return dict(scientific_matches=not errors,errors=errors,proposals=len(ra),jobs_compared=jobs,public_inputs_compared=public_inputs,
                retained=ab['primary_queries'],baseline_seconds=sa,candidate_seconds=sb,
                baseline_qph=aa['primary_queries']/sa*3600,candidate_qph=ab['primary_queries']/sb*3600,
                speedup=sa/sb,baseline_state=ma['state'],candidate_state=mb['state'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--proposals',type=int,default=256)
    p.add_argument('--minimum-speedup',type=float,default=1.5)
    p.add_argument('--stage',choices=['all','baseline','candidate'],default='all')
    args=p.parse_args()
    if args.proposals<1: p.error('proposals must be positive')
    output=args.output.resolve()
    release=new.preflight()
    if args.stage=='baseline':
        new.reject_pipeline_collectors()
        original_schedule=mass.schedule_spec
        def schedule(index,scenes,master_seed):
            result=original_schedule(index,scenes,master_seed)
            if index>=args.proposals: raise RuntimeError('Bounded baseline overshot')
            if index==args.proposals-1:
                # Existing documented drain handler; last proposal finishes its
                # normal preparation and all reserved native jobs drain.
                os.kill(os.getpid(),signal.SIGTERM)
            return result
        mass.schedule_spec=schedule
        baseline_release=copy.deepcopy(read(old.RELEASE))
        baseline_release['configuration']['output']=str(output)
        # Imported ORIGINAL function; only output location and bounded stopping
        # are adapted. Same code/hashes/physics as the published original.
        return old.run(SimpleNamespace(action='run',output=output),baseline_release)
    if args.stage=='candidate':
        return new.run(SimpleNamespace(output=output,target=30000,concurrency=36,preparers=4,
                                      validators=2,prefetch=8,backlog=72,max_proposals=args.proposals,fixture=None,master_seed=410000),release)
    if output.exists(): raise RuntimeError('Use a fresh benchmark output directory')
    output.mkdir(parents=True)
    for name in ('baseline','candidate'):
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--stage',name,'--output',str(output/name),
                 '--proposals',str(args.proposals)]
        began=time.perf_counter()
        with (output/(name+'.log')).open('w') as log:
            print('RUNNING',name, 'log:', log.name,flush=True)
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=new.REPO)
        if result.returncode:
            durable_json(output/'VERDICT.json',dict(passed=False,failed_stage=name,returncode=result.returncode))
            return result.returncode
        print('FINISHED',name,round(time.perf_counter()-began,3),'seconds',flush=True)
    verdict=compare(output/'baseline',output/'candidate')
    verdict['workload']={name:workload_summary(output/name) for name in ('baseline','candidate')}
    verdict.update(minimum_speedup=args.minimum_speedup,minimum_retained=100)
    verdict['passed']=(verdict['scientific_matches'] and verdict['retained']>=100 and
                       verdict['speedup']>=args.minimum_speedup and verdict['candidate_state']=='bounded_finished')
    verdict['scope']='Measured native paired stream; not a sustained-production guarantee'
    durable_json(output/'VERDICT.json',verdict)
    print(json.dumps(verdict,indent=2),flush=True)
    return 0 if verdict['passed'] else 3


if __name__=='__main__': raise SystemExit(main())
