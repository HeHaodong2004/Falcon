#!/usr/bin/env python3
"""Bounded legally targeted root pilot, reusing canonical collection/replay."""
import argparse
from dataclasses import asdict
from pathlib import Path
import random
from collector.core import read,digest
from collector.v3 import PROTOCOL
from collector.v2 import QueryConfig
from collector.static_feasibility import StaticConfig
from collector.wm_sampling import SAMPLER_VERSION
from eventwm_v3_sampling_pilot import run,REPO


def allocation(percent,n,rng):
    counts={k:int(n*v/100) for k,v in percent.items()}
    order=sorted(percent,key=lambda k:(-(n*percent[k]/100-counts[k]),k))
    for k in order[:n-sum(counts.values())]:counts[k]+=1
    values=[k for k,c in counts.items() for _ in range(c)];rng.shuffle(values)
    return values


def job_table(cfg):
    if not 1<=cfg['attempts']<=80 or not 1<=cfg['max_completed_roots']<=50:
        raise ValueError('Bounded pilot only: at most 80 attempts / 50 completed roots')
    rng=random.Random(cfg['master_seed']);categories=allocation(cfg['root_target_percent'],cfg['attempts'],rng)
    visible=iter(allocation(cfg['visible_target_percent'],sum(c in ('visible','visible_lost') for c in categories),rng))
    lost=iter(allocation(cfg['lost_target_percent'],sum(c in ('lost','visible_lost') for c in categories),rng))
    cases=read(REPO/'evaluation/controlled50/cases.json');jobs=[];counts={}
    for i,category in enumerate(categories):
        j=counts.get(category,0);counts[category]=j+1
        allowed=[c for c in cfg['human_counts'] if c>=2] if category=='visible_lost' else cfg['human_counts']
        count=0 if category=='zero' else allowed[j%len(allowed)]
        case=next(c for c in cases if c['scene']=='castle' and c['humans']==count)
        sampling=dict(profile='wm_train',sampler_version=SAMPLER_VERSION,human_count=count,
            target_category=category,target_visible_bucket=next(visible) if category in ('visible','visible_lost') else None,
            target_lost_bucket=next(lost) if category in ('lost','visible_lost') else None,
            min_prefix_steps=rng.randint(20,60) if category in ('zero','unseen') else 10,
            max_prefix_steps=cfg['max_prefix_steps'],scan_angular_velocity=6. if i%2 else -6.,
            visible_boundaries_m=cfg['visible_boundaries_m'],lost_boundaries_s=cfg['lost_boundaries_s'],
            allow_visible_inside_1m=cfg['allow_visible_inside_1m'])
        recipe=dict(protocol_version=PROTOCOL,case=case['tag'],seed=cfg['master_seed']+i,
            sampling=sampling,sampling_profile='wm_train',sampling_profile_family='targeted',
            mechanism='fresh_process_seeded_episode_plus_exact_prefix',
            query_config=asdict(QueryConfig(movement_queries=cfg['movement_queries'])),
            static_config=asdict(StaticConfig()),debug_static_reference=False,sampling_audit=True)
        rid='v3w_'+digest(recipe)[:16];recipe['root_id']=rid
        jobs.append(dict(root_id=rid,scene='castle',human_count=count,root_profile='wm_train',
            sampling_profile='wm_train',sampling_profile_family='targeted',target_category=category,
            seed=recipe['seed'],recipe=recipe))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=REPO/'configs/eventwm_wm_train_pilot.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-roots',type=int)
    args=p.parse_args();args.comparison=False;cfg=read(args.config)
    raise SystemExit(run(args,jobs=job_table(cfg),config=cfg))

if __name__=='__main__':main()
