#!/usr/bin/env python3
"""Offline reference-only training sampler; never changes collection or labels."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import sys
from collector.core import read,write,load_wm_input,digest
from collector.v3 import PROTOCOL,SocialOutcome
from collector.static_feasibility import CHECKER_VERSION
from collector.wm_sampling import legal_root_state,matches
from eventwm_v3_sampling_report import dataset_report

NAMES=[v.name for v in SocialOutcome]


def verify_job_declaration(persisted,generated,manifest):
    """Require a declared deterministic workload and matching scheduled roots."""
    limit=manifest.get('job_limit')
    if limit is not None:
        if not isinstance(limit,int) or limit<=0:raise ValueError('Invalid declared job limit')
        generated=generated[:limit]
    if digest(persisted)!=digest(generated):
        raise ValueError('Persisted job declaration differs from deterministic configuration')
    jobs={j['root_id']:j for j in persisted}
    if len(jobs)!=len(persisted):raise ValueError('Duplicate declared root IDs')
    for root in manifest['roots']:
        job=jobs.get(root['root_id'])
        if job is None:raise ValueError('Undeclared collected root')
        for key,value in job.items():
            if key!='recipe' and digest(root.get(key))!=digest(value):
                raise ValueError('Root metadata differs from declared job: '+key)


def balance_weights(rows,class_target=(.3,.3,.4),mixed_target=.45,binary_mixed_target=.40):
    """Exact feasible class/mixed margins; equal root/group mass inside each cell.

    All saved valid rows remain referenced. Weights specify replacement sampling,
    not independent additional data or a physically duplicated balanced dataset.
    """
    cells=defaultdict(list)
    for i,r in enumerate(rows):cells[(r['social_outcome_class'],r['mixed_outcome_group'])].append(i)
    present={k[0] for k in cells};total=sum(class_target[k] for k in present)
    targets={k:class_target[k]/total for k in present} if total else {}
    lower=sum(v for k,v in targets.items() if (k,False) not in cells)
    upper=sum(v for k,v in targets.items() if (k,True) in cells)
    achieved=min(upper,max(lower,mixed_target))
    flexible=sum(v for k,v in targets.items() if (k,True) in cells and (k,False) in cells)
    flexible_fraction=(achieved-lower)/flexible if flexible else 0.
    cell_masses={}
    for k,mass in targets.items():
        fraction=1. if (k,False) not in cells else 0. if (k,True) not in cells else flexible_fraction
        for mixed,cell_mass in [(True,mass*fraction),(False,mass*(1-fraction))]:
            if (k,mixed) in cells:cell_masses[(k,mixed)]=cell_mass
    # Preserve class/three-class margins while favoring actual collision contrasts.
    subcells={(cell,b):[i for i in ids if bool(rows[i].get('mixed_binary_collision_group',False))==b]
              for cell,ids in cells.items() for b in (False,True)}
    binary_lower=sum(m for cell,m in cell_masses.items() if not subcells[(cell,False)])
    binary_upper=sum(m for cell,m in cell_masses.items() if subcells[(cell,True)])
    binary_achieved=min(binary_upper,max(binary_lower,binary_mixed_target))
    binary_flexible=sum(m for cell,m in cell_masses.items() if subcells[(cell,True)] and subcells[(cell,False)])
    binary_fraction=(binary_achieved-binary_lower)/binary_flexible if binary_flexible else 0.
    weights=[0.]*len(rows)
    for cell,mass in cell_masses.items():
        fraction=1. if not subcells[(cell,False)] else 0. if not subcells[(cell,True)] else binary_fraction
        for binary,cell_mass in [(True,mass*fraction),(False,mass*(1-fraction))]:
            indices=subcells[(cell,binary)]
            if not indices:continue
            roots=defaultdict(lambda:defaultdict(list))
            for i in indices:roots[rows[i]['root_id']][rows[i]['group_id']].append(i)
            for groups in roots.values():
                for items in groups.values():
                    for i in items:weights[i]=cell_mass/len(roots)/len(groups)/len(items)
    return weights,dict(requested_class_mass={NAMES[k]:v for k,v in enumerate(class_target)},
        achieved_class_mass={NAMES[k]:v for k,v in targets.items()},missing_classes=[NAMES[k] for k in range(3) if k not in present],
        requested_mixed_mass=mixed_target,feasible_mixed_interval=[lower,upper],achieved_mixed_mass=achieved,
        requested_binary_mixed_mass=binary_mixed_target,feasible_binary_mixed_interval=[binary_lower,binary_upper],
        achieved_binary_mixed_mass=binary_achieved,
        method='class/three-class-mixed/binary-mixed cell -> equal root -> equal spatial group -> equal variants; replacement sampling')


def index_dataset(dataset,role):
    dataset=Path(dataset);manifest=read(dataset/'manifest.json')
    assert manifest['protocol_version']==PROTOCOL and manifest['checker_version']==CHECKER_VERSION
    expected='wm_train_targeted_pilot' if role=='train' else 'natural_sampling_pilot'
    if manifest['purpose']!=expected:raise ValueError('Training/holdout purpose mismatch; no natural roots in training')
    root_ids=[r['root_id'] for r in manifest['roots']]
    if len(root_ids)!=len(set(root_ids)):raise ValueError('Duplicate root IDs in manifest')
    profile='wm_train' if role=='train' else 'natural'
    if any(r['root_profile']!=profile for r in manifest['roots']):
        raise ValueError('Mixed sampling profiles in dataset role')
    if role=='train':
        from eventwm_wm_train_pilot import job_table
    else:
        from eventwm_v3_sampling_pilot import job_table
    verify_job_declaration(read(dataset/'jobs.json'),job_table(manifest['configuration']),manifest)
    audit=dataset_report(dataset);rows=[];groups=defaultdict(list)
    if audit['unexpected_static_collisions']:
        raise ValueError('Dataset has a static contract violation; quarantine, do not silently filter it')
    for root in audit['roots']:
        if root['state']!='completed':continue
        path=dataset/root['path'];recipe=read(path/'roots/replay.json')
        history=read(path/'legal_observations/history.json')
        legal=legal_root_state(history,root['human_count'],recipe['sampling'])
        if role=='train':
            if not matches(legal,recipe['sampling']):raise ValueError('Completed root does not satisfy declared legal stratum')
            if legal!=root['diagnostics']['legal_selection_state']:raise ValueError('Legal sampling audit mismatch')
        qs={x['query']['query_id']:x['query'] for x in read(path/'queries/queries.json')}
        for label in read(path/'labels/labels.json'):
            if not label['branch_id'].startswith('b'):continue
            q=qs[label['query_id']]
            if label['static_feasibility_violation']:raise ValueError('Static contract failure cannot be indexed')
            if not label['social_outcome_valid']:continue
            load_wm_input(path,label['query_id'])
            assert q['root_id']==root['root_id'] and label['social_outcome_name']==NAMES[label['social_outcome_class']]
            if root['human_count']==0 and label['social_outcome_class']!=0:raise ValueError('Zero-human label contradiction')
            gid=root['root_id']+'/'+q['spatial_path_id']
            row=dict(dataset=str(dataset),root_path=root['path'],root_id=root['root_id'],query_id=label['query_id'],
                branch_id=label['branch_id'],group_id=gid,spatial_path_id=q['spatial_path_id'],
                scene=root['scene'],human_count=root['human_count'],sampling_profile=root['root_profile'],
                legal_root_state=legal,wait_duration=q['wait_duration'],speed_scale=q['speed_scale'],
                spatial_candidate_index=q['direction'],
                query_type='WAIT' if q['direction']=='WAIT' else 'movement',
                social_outcome_class=label['social_outcome_class'],social_outcome_name=label['social_outcome_name'],
                binary_human_collision_target=int(label['social_outcome_class']==2),
                hard_negative=label['social_outcome_class']==1,split=role,split_group=root['root_id'],
                model_input_loader='collector.core.load_wm_input',history_reference=root['path']+'/legal_observations/history.json')
            rows.append(row);groups[gid].append(row)
    group_report=[]
    for gid,members in groups.items():
        classes={r['social_outcome_class'] for r in members};binary={r['binary_human_collision_target'] for r in members}
        variants={(r['wait_duration'],r['speed_scale']) for r in members}
        mixed=len(classes)>1 and len(variants)>1;bmixed=len(binary)>1 and len(variants)>1
        for r in members:r.update(mixed_outcome_group=mixed,mixed_binary_collision_group=bmixed)
        group_report.append(dict(group_id=gid,root_id=members[0]['root_id'],query_type=members[0]['query_type'],
            spatial_candidate_index=members[0]['spatial_candidate_index'],
            variants=len(variants),classes=sorted(classes),mixed_outcome=mixed,mixed_binary_collision=bmixed))
    if role=='train':weights,sampler=balance_weights(rows)
    else:weights=[1/len(rows)]*len(rows) if rows else [];sampler=dict(method='unbalanced complete primary census; no class oversampling')
    for r,w in zip(rows,weights):r['sampling_probability']=w
    if len({r['query_id'] for r in rows})!=len(rows):raise ValueError('Duplicate primary query IDs')
    root_mass=Counter()
    for r in rows:root_mass[r['root_id']]+=r['sampling_probability']
    info=dict(dataset=str(dataset),role=role,protocol_version=PROTOCOL,checker_version=CHECKER_VERSION,
        training_view_version='eventwm-physical-human-collision-index-v1',
        target_definition='social_outcome_class == HUMAN_COLLISION (2); native physical human contact; class 1 retained as hard negative',
        primary_samples=len(rows),unique_roots=len(root_mass),class_counts=dict(Counter(r['social_outcome_name'] for r in rows)),
        branches_per_root=dict(Counter(r['root_id'] for r in rows)),
        counterfactual_groups=sum(g['variants']>=2 for g in group_report),
        mixed_outcome_groups=sum(g['mixed_outcome'] for g in group_report),
        mixed_binary_collision_groups=sum(g['mixed_binary_collision'] for g in group_report),
        counterfactual_groups_per_root=dict(Counter(g['root_id'] for g in group_report if g['variants']>=2)),
        raw_mixed_sample_fraction=sum(r['mixed_outcome_group'] for r in rows)/len(rows) if rows else None,
        sampled_mixed_binary_collision_mass=sum(r['sampling_probability'] for r in rows if r['mixed_binary_collision_group']),
        root_sampling_mass=dict(root_mass),sampler=sampler,groups=group_report,rows=rows)
    info['max_root_probability']=max(root_mass.values(),default=0.)
    info['effective_root_count']=1/sum(v*v for v in root_mass.values()) if root_mass else 0.
    return info,audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--natural-holdout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();train,ta=index_dataset(a.training,'train');holdout,ha=index_dataset(a.natural_holdout,'natural_holdout')
    if set(train['branches_per_root'])&set(holdout['branches_per_root']):raise ValueError('Root leakage between training and natural holdout')
    write(a.output/'training_index.json',train);write(a.output/'natural_holdout_index.json',holdout)
    write(a.output/'sampling_audit.json',dict(targeted=ta,natural=ha))
    print({k:train[k] for k in ('primary_samples','unique_roots','class_counts','mixed_outcome_groups','mixed_binary_collision_groups','sampler')})

if __name__=='__main__':main()
