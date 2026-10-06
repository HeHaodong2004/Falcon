#!/usr/bin/env python3
"""Bounded social-outcome v3 smoke; no production, resume or training mode."""
import argparse
import hashlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import time
from collector.core import read, write, digest, load_wm_input
from collector.v2 import QueryConfig
from collector.v3 import PROTOCOL, SOCIAL_METRIC, guard_output, social_outcome_labels
from collector.static_feasibility import CHECKER_VERSION, StaticConfig
from eventwm_scale import seals, provenance, verify_root
from eventwm_collect import branches

REPO = Path(__file__).resolve().parents[1]


def summarize(output, manifest):
    samples, audit_count = [], 0
    for root in manifest['roots']:
        if root['state'] != 'completed':
            continue
        verify_root(output, root, deep=True)
        path = output/root['path']
        queries = {q['query']['query_id']:q['query'] for q in read(path/'queries/queries.json')}
        saved = read(path/'labels/labels.json')
        derived = [social_outcome_labels(b, queries[b['query_id']], root['human_count']) for b in branches(path)]
        if saved != derived:
            raise RuntimeError('Offline label mismatch')
        history = read(path/'legal_observations/history.json')
        tracks = history[-1]['human_tracks']
        for label in saved:
            if not label['branch_id'].startswith('b'):
                audit_count += 1
                continue
            load_wm_input(path, label['query_id'])
            samples.append(dict(root_id=root['root_id'], scene=root['scene'],human_count=root['human_count'],
                root_profile=root['root_profile'],root_path=root['path'],
                visible_tracks_at_root=sum(t['visible_now'] for t in tracks),
                lost_tracks_at_root=sum(not t['visible_now'] for t in tracks),
                visible_human_history=any(t['visible_now'] for h in history for t in h['human_tracks']), **label))
    if len({r['query_id'] for r in samples}) != len(samples):
        raise RuntimeError('Duplicate primary query IDs')
    violations=sum((output/r['path']/'audits/STATIC_FEASIBILITY_VIOLATION.json').exists() for r in manifest['roots'])
    violations+=sum(r['static_feasibility_violation'] for r in samples)
    invalid=sum(not r['social_outcome_valid'] for r in samples)
    write(output/'primary_index.json',dict(protocol_version=PROTOCOL,samples=samples))
    report=dict(protocol_version=PROTOCOL, checker_version=CHECKER_VERSION,
        status='PASS' if samples and all(r['state']=='completed' for r in manifest['roots']) and not (violations or invalid) else 'FAIL',
        roots_attempted=len(manifest['roots']),roots_completed=sum(r['state']=='completed' for r in manifest['roots']),
        primary_branches=len(samples),audit_branches=audit_count,
        class_counts=dict(Counter(r['social_outcome_name'] for r in samples)),invalid=invalid,
        unexpected_static_collisions=violations,
        by_root={r['root_id']:dict(Counter(s['social_outcome_name'] for s in samples if s['root_id']==r['root_id'])) for r in manifest['roots']})
    write(output/'smoke_summary.json',report)
    return report


def run(args):
    if not 1 <= args.debug_visual_roots <= 4 or not 1 <= args.debug_visual_branches <= 6:
        raise ValueError('Debug limits: 1–4 roots and 1–6 branches')
    output=guard_output(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('New empty v3 output required; never append/overwrite')
    config=QueryConfig(movement_queries=args.movement_queries)
    cases=read(REPO/'evaluation/controlled50/cases.json')
    matrix=[('castle',0,'wait',8301),('castle',1,'seek_visible',8401),
            ('castle',2,'seek_lost',8502),('hm3d_NBg5UqG3di3',0,'seek_visible',13301)]
    if args.first_root_only: matrix=matrix[:1]
    manifest=dict(schema_version=3,protocol_version=PROTOCOL,checker_version=CHECKER_VERSION,
        purpose='bounded_smoke_only',social_metric_version=SOCIAL_METRIC,provenance=provenance(),query_config=asdict(config),
        static_config=asdict(StaticConfig()),debug_visuals=bool(args.debug_visuals),roots=[],state='running')
    manifest['provenance'].setdefault('source_sha256',{}).update({str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__),REPO/'falcon/additional_metric.py')})
    output.mkdir(parents=True,exist_ok=True)
    write(output/'manifest.json',manifest)
    for scene,count,profile,seed in matrix:
        case=next(c for c in cases if c['scene']==scene and c['humans']==count)
        recipe=dict(protocol_version=PROTOCOL,case=case['tag'],seed=seed,
            sampling=dict(profile=profile,wait_steps=20),query_config=asdict(config),static_config=asdict(StaticConfig()),
            mechanism='fresh_process_seeded_episode_plus_exact_prefix')
        if scene=='hm3d_NBg5UqG3di3':
            fixture=read(REPO/'configs/eventwm_v2_noncastle_smoke_prefix.json')
            assert (case['tag'],seed)==(fixture['case'],fixture['seed'])
            recipe.update(prefix_controls=fixture['prefix_controls'],prefix_fixture_sha256=digest(fixture))
            profile='explicit_static_open_space'
        rid='v3_'+digest(recipe)[:16];recipe['root_id']=rid
        # Debug retention does not change the physical root/query identity or RNG.
        recipe['debug_static_reference']=bool(args.debug_visuals and len(manifest['roots']) < args.debug_visual_roots)
        root=dict(root_id=rid,scene=scene,human_count=count,root_profile=profile,path=f'roots/{rid}',state='running')
        manifest['roots'].append(root);write(output/'manifest.json',manifest)
        recipefile=output/'recipes'/f'{rid}.json';write(recipefile,recipe)
        path=output/root['path'];log=output/'logs'/f'{rid}.log';log.parent.mkdir(exist_ok=True)
        cmd=[sys.executable,'-u',str(REPO/'scripts/eventwm_collect.py'),'collect','--output',str(path),
             '--recipe',str(recipefile),'--protocol','v3','--movement-queries',str(args.movement_queries),
             '--max-branches',str(4*args.movement_queries+4),'--storage-profile','production','--no-render']
        start=time.monotonic();print('START',rid,scene,count,profile,flush=True)
        with log.open('w') as f: result=subprocess.run(cmd,cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
        root.update(seconds=time.monotonic()-start,returncode=result.returncode)
        if result.returncode:
            root['state']='quarantined' if (path/'audits/STATIC_FEASIBILITY_VIOLATION.json').exists() else 'failed'
            root['error']=log.read_text(errors='replace')[-3000:]
            manifest['state']='stopped_failure'
        else:
            root.update(state='completed',seals=seals(path))
            write(path/'COMPLETE.json',dict(root_id=rid,seal_digest=digest(root['seals'])))
        write(output/'manifest.json',manifest);print(root['state'],rid,round(root['seconds'],2),flush=True)
        if result.returncode: break
    if manifest['state']=='running':manifest['state']='finished_smoke'
    write(output/'manifest.json',manifest)
    report=summarize(output,manifest);print(report,flush=True)
    if args.debug_visuals:
        subprocess.run([sys.executable,str(REPO/'scripts/eventwm_v3_visualize.py'),'--dataset',str(output),
            '--output',str(output/'debug_visuals'),'--max-roots',str(args.debug_visual_roots),
            '--max-branches',str(args.debug_visual_branches)],check=True,cwd=REPO)
    return 0 if report['status']=='PASS' else 1


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=REPO/'datasets/raw/eventwm_social_outcome_v3_smoke_01')
    p.add_argument('--movement-queries',type=int,default=8)
    p.add_argument('--first-root-only',action='store_true')
    p.add_argument('--debug-visuals',action='store_true')
    p.add_argument('--debug-visual-roots',type=int,default=4)
    p.add_argument('--debug-visual-branches',type=int,default=6)
    return p

if __name__=='__main__':sys.exit(run(parser().parse_args()))
