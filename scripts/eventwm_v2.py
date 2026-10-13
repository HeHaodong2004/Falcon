#!/usr/bin/env python3
"""Bounded v2 human-event collection; intentionally no scale/resume mode."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from collector.core import read, write, digest
from collector.v2 import PROTOCOL, QueryConfig, guard_output
from collector.static_feasibility import CHECKER_VERSION, StaticConfig
from eventwm_scale import seals, provenance

REPO=Path(__file__).resolve().parents[1]


def run(args):
    output=guard_output(args.output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new empty v2 output; no append/resume')
    # Explicit small matrix; the historical profiles/seeds give legal visible/lost coverage.
    specs=[('castle',0,'wait',8301),('castle',1,'seek_visible',8401),
           ('castle',2,'seek_lost',8502),('hm3d_NBg5UqG3di3',0,'seek_visible',13301)]
    if args.first_root_only:specs=specs[:1]
    cases=read(REPO/'evaluation/controlled50/cases.json')
    config=QueryConfig(movement_queries=args.movement_queries,max_attempts=args.candidate_attempts)
    static=StaticConfig(safety_margin=args.static_margin)
    manifest=dict(schema_version=2,protocol_version=PROTOCOL,checker_version=CHECKER_VERSION,
        purpose='controlled_smoke_only',query_config=asdict(config),static_config=asdict(static),
        provenance=provenance(),roots=[],state='running')
    manifest['provenance'].setdefault('source_sha256',{})[str(Path(__file__).relative_to(REPO))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=True);write(output/'manifest.json',manifest)
    for scene,count,profile,seed in specs:
        case=next(c for c in cases if c['scene']==scene and c['humans']==count)
        recipe=dict(protocol_version=PROTOCOL,case=case['tag'],seed=seed,sampling=dict(profile=profile,wait_steps=20),
            mechanism='fresh_process_seeded_episode_plus_exact_prefix',query_config=asdict(config),static_config=asdict(static))
        if scene=='hm3d_NBg5UqG3di3':
            fixture=read(REPO/'configs/eventwm_v2_noncastle_smoke_prefix.json')
            assert fixture['case']==case['tag'] and fixture['seed']==seed
            recipe['prefix_controls']=fixture['prefix_controls']
            recipe['prefix_fixture_sha256']=digest(fixture)
            profile='explicit_static_open_space'
        rid='v2_'+digest(recipe)[:16];recipe['root_id']=rid
        path=output/'roots'/rid;recipefile=output/'recipes'/f'{rid}.json';write(recipefile,recipe)
        root=dict(root_id=rid,scene=scene,human_count=count,root_profile=profile,path=str(path.relative_to(output)),state='running')
        manifest['roots'].append(root);write(output/'manifest.json',manifest)
        log=output/'logs'/f'{rid}.log';log.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,'-u',str(REPO/'scripts/eventwm_collect.py'),'collect','--output',str(path),
             '--recipe',str(recipefile),'--protocol','v2','--movement-queries',str(args.movement_queries),
             '--max-branches',str(args.movement_queries*4+4),'--storage-profile','production','--no-render']
        start=time.time();print('START',rid,scene,count,profile,flush=True)
        with log.open('w') as f:
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
        root['seconds']=time.time()-start
        root['returncode']=result.returncode
        gen=path/'audits/generation_v2.json'
        if gen.exists():root['generation']=read(gen)
        if result.returncode:
            root['state']='failed';root['error']=log.read_text(errors='replace')[-4000:]
            if (path/'audits/STATIC_FEASIBILITY_VIOLATION.json').exists():
                root['state']='quarantined_static_violation';manifest['state']='stopped_static_violation'
            else:manifest['state']='running'
        else:
            root['state']='completed'
            root['seals']=seals(path)
            write(path/'COMPLETE.json',dict(root_id=rid,seal_digest=digest(root['seals'])))
        write(output/'manifest.json',manifest)
        print(root['state'].upper(),rid,round(root['seconds'],2),'s',flush=True)
        if manifest['state']=='stopped_static_violation':break
    if manifest['state']=='running':manifest['state']='finished_smoke'
    write(output/'manifest.json',manifest)
    report=summarize(output,manifest);write(output/'smoke_summary.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['status']=='PASS' else 1


def summarize(output,manifest):
    rows=[];attempts=[];rejections=0;proposals=0;movement=0;waits=0;violations=0
    for r in manifest['roots']:
        gen=r.get('generation',{});proposals+=gen.get('attempts',0)
        rejections+=sum(x['rejection_reason']!='near_duplicate_endpoint' for x in gen.get('candidate_rejections',[]))
        movement+=gen.get('accepted_movement_queries',0);waits+=gen.get('wait_count',0)
        if r['state']=='completed':attempts.append(gen['attempts'])
        p=output/r['path'];labs=p/'labels/labels.json'
        recovered=[]
        if not labs.exists() and (p/'queries/queries.json').exists():
            from collector.v2 import human_event_labels
            saved_queries={q['query']['query_id']:q['query'] for q in read(p/'queries/queries.json')}
            for worker in sorted(p.glob('audits/workers/b*/branch.json')):
                b=read(worker);recovered.append(human_event_labels(b,saved_queries[b['query_id']],r['human_count']))
        if labs.exists() or recovered:
            qs={q['query']['query_id']:q['query'] for q in read(p/'queries/queries.json')}
            history=read(p/'legal_observations/history.json');last=history[-1]['human_tracks']
            visible=sum(t['visible_now'] for t in last);lost=sum(not t['visible_now'] for t in last)
            seen=any(t['visible_now'] for h in history for t in h['human_tracks'])
            for l in (read(labs) if labs.exists() else recovered):
                if not l['branch_id'].startswith('b'):continue
                q=qs[l['query_id']]
                rows.append(dict(root_id=r['root_id'],root_state=r['state'],scene=r['scene'],human_count=r['human_count'],root_profile=r['root_profile'],
                    legal_history_reference=str(Path(r['path'])/'legal_observations/history.json'),
                    visible_tracks_at_root=visible,lost_tracks_at_root=lost,visible_human_history=seen,
                    visibility_history_category='visible_and_lost' if visible and lost else 'visible' if visible else 'lost' if lost else 'observed_no_current' if seen else 'never_observed',
                    query_reference=str(Path(r['path'])/'queries/queries.json'),query_sha256=digest(q),
                    spatial_path_id=q['spatial_path_id'],wait_duration=q['wait_duration'],speed_scale=q['speed_scale'],
                    static_feasible=q['static_precheck']['static_feasible'],static_precheck=q['static_precheck'],**l))
        quarantine=p/'labels/quarantined_branch.json'
        if quarantine.exists() and not any(x['root_id']==r['root_id'] and x['branch_id']==read(quarantine).get('branch_id') for x in rows):
            violations+=int(read(quarantine)['static_feasibility_violation'])
    violations+=sum(r['static_feasibility_violation'] for r in rows)
    write(output/'primary_index.json',dict(protocol_version=PROTOCOL,samples=rows))
    bycondition={}
    for scene,count in sorted({(r['scene'],r['human_count']) for r in manifest['roots']}):
        rs=[r for r in rows if r['scene']==scene and r['human_count']==count]
        bycondition[f'{scene}/h{count}']=dict(branches=len(rs),positive=sum(r['human_event_label']==1 for r in rs),
              negative=sum(r['human_event_label']==0 for r in rs),invalid=sum(not r['human_event_valid'] for r in rs))
    passed=(len(manifest['roots'])>0 and all(r['state']=='completed' for r in manifest['roots']) and
            violations==0 and all(r['human_event_valid'] for r in rows))
    return dict(status='PASS' if passed else 'FAIL',protocol_version=PROTOCOL,checker_version=CHECKER_VERSION,
        roots_attempted=len(manifest['roots']),roots_completed=sum(r['state']=='completed' for r in manifest['roots']),
        candidate_proposals=proposals,static_candidate_rejections=rejections,accepted_movement_queries=movement,
        requested_movement_queries_per_root=manifest['query_config']['movement_queries'],
        attempts_to_n_mean=sum(attempts)/len(attempts) if attempts else None,attempts_to_n_max=max(attempts) if attempts else None,
        movement_per_root={r['root_id']:r.get('generation',{}).get('accepted_movement_queries',0) for r in manifest['roots']},
        wait_count=waits,primary_branches=len(rows),human_event_positives=sum(r['human_event_label']==1 for r in rows),
        human_event_negatives=sum(r['human_event_label']==0 for r in rows),invalid_branches=sum(not r['human_event_valid'] for r in rows)+violations-sum(r['static_feasibility_violation'] for r in rows),
        unexpected_static_collisions=violations,termination_reasons=dict(Counter(r['termination_reason'] for r in rows)),
        scene_human_breakdown=bycondition)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=REPO/'datasets/raw/eventwm_human_event_v2_smoke')
    p.add_argument('--movement-queries',type=int,default=8)
    p.add_argument('--candidate-attempts',type=int,default=128)
    p.add_argument('--static-margin',type=float,default=.03)
    p.add_argument('--first-root-only',action='store_true')
    args=p.parse_args();sys.exit(run(args))

if __name__=='__main__':main()
