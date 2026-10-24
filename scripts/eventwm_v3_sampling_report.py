#!/usr/bin/env python3
"""Read-only sampling distribution audit; never feeds decisions to the collector."""
import argparse
from collections import Counter
import json
import statistics
from pathlib import Path
import sys
from collector.core import read,write,load_wm_input
from collector.v3 import social_outcome_labels
from collector.sampling_audit import root_diagnostics
from eventwm_scale import verify_root
from eventwm_collect import branches

NAMES=('SAFE_SUCCESS','SOCIAL_VIOLATION_SUCCESS','HUMAN_COLLISION')


def outcomes(rows):
    counts={n:sum(r['social_outcome_name']==n and r['social_outcome_valid'] for r in rows) for n in NAMES}
    counts['INVALID']=sum(not r['social_outcome_valid'] for r in rows)
    return dict(count=len(rows),counts=counts,rates={k:v/len(rows) if rows else None for k,v in counts.items()})


def profile_statistics(report):
    """Root-weighted diagnostics and branch-weighted outcomes, kept separate."""
    result={}
    for profile in sorted({r['root_profile'] for r in report['roots']}):
        attempts=[r for r in report['roots'] if r['root_profile']==profile and r['human_count']>0]
        roots=[r for r in attempts if r['state']=='completed']
        distances=[r['diagnostics']['nearest_social_distance_at_root'] for r in roots]
        distances=[d for d in distances if d is not None]
        rows=[s for s in report['samples'] if s['sampling_profile']==profile and s['human_count']>0]
        n=len(roots)
        result[profile]=dict(attempted_human_roots=len(attempts),completed_human_roots=n,
            mean_psc_distance=statistics.mean(distances) if distances else None,
            median_psc_distance=statistics.median(distances) if distances else None,
            already_under_1m=sum(r['diagnostics']['already_in_social_violation_at_root'] for r in roots),
            visible_roots=sum(r['diagnostics']['visible_tracks']>0 for r in roots),
            lost_roots=sum(r['diagnostics']['lost_tracks']>0 for r in roots),
            branch_outcomes=outcomes(rows))
    return result


def matched_comparison(natural,comparison):
    """Compare only completed root pairs with the same scene/count/episode seed.

    This controls the episode specification, not root pose or surviving-prefix
    selection. Never use this offline table to select collection jobs.
    """
    key=lambda r:(r['scene'],r['human_count'],r['seed'])
    result={}
    for profile in sorted({r['root_profile'] for r in comparison['roots']}):
        coverage=[r for r in comparison['roots'] if r['root_profile']==profile and r['state']=='completed']
        common={key(r) for r in coverage}&{key(r) for r in natural['roots'] if r['state']=='completed'}
        entries={}
        for name,report in [('natural',natural),(profile,comparison)]:
            roots=[r for r in report['roots'] if r['root_profile']==name and r['state']=='completed' and key(r) in common]
            ids={r['root_id'] for r in roots}
            entries[name]=profile_statistics(dict(roots=roots,samples=[s for s in report['samples'] if s['root_id'] in ids])).get(name)
        result[profile]=dict(completed_pairs=len(common),profiles=entries)
    return result


def dataset_report(dataset):
    dataset=Path(dataset);manifest=read(dataset/'manifest.json');roots=[];samples=[];partial=[]
    for root in manifest['roots']:
        path=dataset/root['path'];dpath=path/'audits/root_sampling.json'
        diag=read(dpath) if dpath.exists() else None
        if diag is None and (path/'roots/privileged_root.json').exists():
            recipe=read(path/'roots/replay.json');hist=read(path/'legal_observations/history.json')
            diag=root_diagnostics(recipe,read(path/'roots/privileged_root.json')['state'],hist,len(recipe.get('prefix_controls',[])))
        info={k:v for k,v in root.items() if k not in ('seals','error','recipe')}
        info.update(diagnostics=diag)
        if root['state']!='completed':
            fail=path/'audits/prefix_failure.json'
            info['prefix_failure']=read(fail) if fail.exists() else None
            worker_log=path/'audits/workers/root/run.log'
            error=root.get('error','')+(worker_log.read_text(errors='replace')[-4000:] if worker_log.exists() else '')
            error+=''.join(p.read_text(errors='replace')[-1200:] for p in path.glob('audits/workers/b*/run.log'))
            info['failure_reason']='static_feasibility_violation' if (path/'audits/STATIC_FEASIBILITY_VIOLATION.json').exists() else 'prefix_human_contact' if info['prefix_failure'] and info['prefix_failure']['native_human_contact'] else 'prefix_termination' if info['prefix_failure'] else 'wm_train_stratum_unavailable' if 'wm_train_stratum_unavailable' in error else 'root_replay_mismatch' if 'Root replay mismatch' in error else 'insufficient_static_feasible_queries' if 'insufficient_static_feasible_queries' in error else 'other_collection_failure'
        roots.append(info)
        labelpath=path/'labels/labels.json'
        if root['state']=='completed':verify_root(dataset,root,deep=True)
        if labelpath.exists():labels=read(labelpath)
        elif (path/'queries/queries.json').exists():
            qs={r['query']['query_id']:r['query'] for r in read(path/'queries/queries.json')}
            labels=[social_outcome_labels(read(p),qs[read(p)['query_id']],root['human_count']) for p in sorted(path.glob('audits/workers/b*/branch.json'))]
        else:continue
        qs={r['query']['query_id']:r['query'] for r in read(path/'queries/queries.json')}
        if root['state']=='completed':
            assert labels==[social_outcome_labels(b,qs[b['query_id']],root['human_count']) for b in branches(path)]
        for label in labels:
            if not label['branch_id'].startswith('b'):continue
            q=qs[label['query_id']]
            if root['state']=='completed' and diag is not None:
                assert diag['already_in_social_violation_at_root']==(label['first_social_violation_time']==0), 'Root/social event-time mismatch'
            row=dict(root_id=root['root_id'],scene=root['scene'],human_count=root['human_count'],sampling_profile=root['root_profile'],
                root_state=root['state'],direction=q['direction'],wait_duration=q['wait_duration'],speed_scale=q['speed_scale'],
                history=diag,**label)
            if root['state']=='completed':
                load_wm_input(path,label['query_id']);samples.append(row)
            else:partial.append(row)
    done=[r for r in roots if r['state']=='completed'];human=[r for r in done if r['human_count']>0]
    def by(key):return dict(Counter(str(r[key]) for r in done))
    bins={'<1.0':0,'[1.0,1.5)':0,'[1.5,2.0)':0,'[2.0,3.0)':0,'>=3.0':0,'unknown':0}
    for r in human:
        d=r['diagnostics']['nearest_social_distance_at_root']
        bins['unknown' if d is None else '<1.0' if d<1 else '[1.0,1.5)' if d<1.5 else '[1.5,2.0)' if d<2 else '[2.0,3.0)' if d<3 else '>=3.0']+=1
    groupings=dict(overall=outcomes(samples),human_present=outcomes([r for r in samples if r['human_count']>0]),
        by_human_count={str(c):outcomes([r for r in samples if r['human_count']==c]) for c in (0,1,2,4,6)},
        by_scene={s:outcomes([r for r in samples if r['scene']==s]) for s in sorted({r['scene'] for r in roots})},
        by_scene_human={f"{s}/h{c}":outcomes([r for r in samples if (r['scene'],r['human_count'])==(s,c)]) for s,c in sorted({(r['scene'],r['human_count']) for r in roots})},
        by_profile={p:outcomes([r for r in samples if r['sampling_profile']==p]) for p in sorted({r['root_profile'] for r in roots})},
        by_initial_social_state={str(flag):outcomes([r for r in samples if r['human_count']>0 and r['history']['already_in_social_violation_at_root']==flag]) for flag in (False,True)},
        by_history={k:outcomes([r for r in samples if predicate(r['history'])]) for k,predicate in {
            'visible_any':lambda d:d['visible_tracks']>0,'lost_any':lambda d:d['lost_tracks']>0,
            'unseen_no_current_track':lambda d:d['visible_tracks']==d['lost_tracks']==0}.items()},
        by_query={k:outcomes([r for r in samples if predicate(r)]) for k,predicate in {
            'movement':lambda r:r['direction']!='WAIT','WAIT':lambda r:r['direction']=='WAIT',
            'immediate_normal':lambda r:r['direction']!='WAIT' and r['wait_duration']==0 and r['speed_scale']==1,
            'half_speed':lambda r:r['direction']!='WAIT' and r['speed_scale']==.5,
            'wait_0.5':lambda r:r['direction']!='WAIT' and r['wait_duration']==.5,
            'wait_1.0':lambda r:r['direction']!='WAIT' and r['wait_duration']==1.}.items()})
    root_signal=Counter();human_root_signal=Counter();class_sets=Counter()
    for root in done:
        labels={r['social_outcome_class'] for r in samples if r['root_id']==root['root_id']}
        root_signal['with_safe']+=0 in labels;root_signal['with_social']+=1 in labels;root_signal['with_collision']+=2 in labels
        root_signal['mixed_classes']+=len(labels)>1
        class_sets[','.join(map(str,sorted(labels,key=lambda x:-1 if x is None else x)))]+=1
        if root['human_count']:
            for name,present in [('with_safe',0 in labels),('with_social',1 in labels),('with_collision',2 in labels),('mixed_classes',len(labels)>1)]:human_root_signal[name]+=present
    return dict(dataset=str(dataset),requested=len(read(dataset/'jobs.json')) if (dataset/'jobs.json').exists() else len(roots),attempted=len(roots),completed=len(done),
        failed=len(roots)-len(done),failures=dict(Counter(r['failure_reason'] for r in roots if r['state']!='completed')),
        attempts_by_scene_human=dict(Counter(f"{r['scene']}/h{r['human_count']}" for r in roots)),
        failures_by_scene_human=dict(Counter(f"{r['scene']}/h{r['human_count']}" for r in roots if r['state']!='completed')),
        roots_by_scene=by('scene'),roots_by_human_count=by('human_count'),roots_by_profile=by('root_profile'),human_present_roots=len(human),
        visibility=dict(visible=sum(r['diagnostics']['visible_tracks']>0 for r in done),lost=sum(r['diagnostics']['lost_tracks']>0 for r in done),
            unseen=sum(r['diagnostics']['visible_tracks']==r['diagnostics']['lost_tracks']==0 for r in done),
            human_present_visible=sum(r['diagnostics']['visible_tracks']>0 for r in human),
            human_present_lost=sum(r['diagnostics']['lost_tracks']>0 for r in human),
            human_present_unseen=sum(r['diagnostics']['visible_tracks']==r['diagnostics']['lost_tracks']==0 for r in human),
            human_present_never_observed=sum(not r['diagnostics']['any_human_observed_in_history'] for r in human)),
        roots_already_under_1m=sum(r['diagnostics']['already_in_social_violation_at_root'] for r in human),root_psc_distance_bins=bins,
        root_learning_signal=dict(root_signal),human_present_root_learning_signal=dict(human_root_signal),root_class_sets=dict(class_sets),branch_outcomes=groupings,excluded_partial_branch_outcomes=outcomes(partial),
        unexpected_static_collisions=sum((dataset/r['path']/'audits/STATIC_FEASIBILITY_VIOLATION.json').exists() for r in roots),
        roots=roots,samples=samples)



def merge_reports(a,b):
    """Offline union of separately declared blocks; keep every attempted root."""
    def counts(x,y):return {k:x.get(k,0)+y.get(k,0) for k in set(x)|set(y)}
    def outcome(x,y):
        n=x['count']+y['count'];c=counts(x['counts'],y['counts'])
        return dict(count=n,counts=c,rates={k:v/n if n else None for k,v in c.items()})
    r=dict(dataset=[a['dataset'],b['dataset']],roots=a['roots']+b['roots'],samples=a['samples']+b['samples'])
    assert len({x['root_id'] for x in r['roots']})==len(r['roots']), 'Overlapping pilot root IDs'
    for key in ('requested','attempted','completed','failed','human_present_roots','roots_already_under_1m','unexpected_static_collisions'):
        r[key]=a[key]+b[key]
    for key in ('failures','attempts_by_scene_human','failures_by_scene_human','roots_by_scene','roots_by_human_count','roots_by_profile','visibility','root_psc_distance_bins',
                'root_learning_signal','human_present_root_learning_signal','root_class_sets'):
        r[key]=counts(a[key],b[key])
    r['excluded_partial_branch_outcomes']=outcome(a['excluded_partial_branch_outcomes'],b['excluded_partial_branch_outcomes'])
    r['branch_outcomes']={}
    for key,x in a['branch_outcomes'].items():
        y=b['branch_outcomes'][key]
        r['branch_outcomes'][key]=outcome(x,y) if 'count' in x else {k:outcome(x.get(k,outcomes([])),y.get(k,outcomes([]))) for k in set(x)|set(y)}
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--comparison',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--supplement',type=Path);a=p.parse_args()
    report=dict(natural=dataset_report(a.dataset))
    if a.supplement:
        report['natural_initial']=report['natural'];report['natural_supplement']=dataset_report(a.supplement)
        report['natural']=merge_reports(report['natural_initial'],report['natural_supplement'])
    if a.comparison:report['comparison']=dataset_report(a.comparison)
    report['natural']['profile_statistics']=profile_statistics(report['natural'])
    if a.comparison:
        report['comparison']['profile_statistics']=profile_statistics(report['comparison'])
        report['comparison']['matched_natural_comparison']=matched_comparison(report['natural'],report['comparison'])
    write(a.output,report)
    for key,r in report.items():print(key,r['completed'],r['branch_outcomes']['human_present'],r['failures'])

if __name__=='__main__':main()
