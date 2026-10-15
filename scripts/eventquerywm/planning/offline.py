"""Selection first, retrospective outcome scoring second. Root-cluster statistics."""
from collections import defaultdict
import numpy as np
from ..sr_metrics import ranking_metrics, metrics


def interval(values, replicates=2000, seed=20261010):
    v=np.asarray(values,dtype=float)
    if not len(v):return dict(roots=0,mean=None,ci95=None)
    rng=np.random.RandomState(seed)
    means=v[rng.randint(0,len(v),size=(replicates,len(v)))].mean(1)
    return dict(roots=len(v),mean=float(v.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                replicates=replicates,seed=seed)


def banks(metadata, movement, matched):
    result=defaultdict(list)
    for i,m in enumerate(metadata):result[m['group_id'] if matched else m['root_id']].append(i)
    if matched:
        result={k:[i for i in ix if movement[i]>.1 and movement[i]>=.95*max(movement[ix])]
                for k,ix in result.items()}
    return {k:ix for k,ix in result.items() if len(ix)>=2}


def choose(bank, score, duration, query_ids):
    """No outcome argument. Static eligibility must have been verified upstream."""
    score,duration=np.asarray(score),np.asarray(duration)
    if not np.isfinite(score).all():raise ValueError('Nonfinite decision score')
    return {k:min(ix,key=lambda i:(score[i],duration[i],query_ids[i])) for k,ix in bank.items()}


def realized_cost(y,duration,lambda_c,lambda_s):
    return duration+lambda_c*(y==2)+lambda_s*(y==1)


def posthoc_oracle(bank,y,duration,query_ids,lambda_c,lambda_s):
    return choose(bank,realized_cost(y,duration,lambda_c,lambda_s),duration,query_ids)


def root_summary(metadata, selected, y, duration, movement, bank, lambda_c=10.,lambda_s=5.):
    # Nominal time plus observed outcome penalties is a saved-query diagnostic,
    # not realized time-to-goal or an episode completion-time estimate.
    cost=realized_cost(y,duration,lambda_c,lambda_s)
    balanced_cost=realized_cost(y,duration,10.,5.)
    byroot=defaultdict(lambda:defaultdict(list))
    for k,i in selected.items():
        r=metadata[i]['root_id'];best=min(cost[bank[k]])
        values=dict(collision=float(y[i]==2),social=float(y[i]==1),safe=float(y[i]==0),
                    duration=float(duration[i]),movement=float(movement[i]),regret=float(cost[i]-best),
                    realized_nominal_cost=float(cost[i]),wait=float(movement[i]<=.1))
        values['balanced_regret']=float(balanced_cost[i]-min(balanced_cost[bank[k]]))
        for name,v in values.items():byroot[r][name].append(v)
    roots={r:{k:float(np.mean(v)) for k,v in values.items()} for r,values in byroot.items()}
    result={k:interval([row[k] for row in roots.values()]) for k in next(iter(roots.values()))} if roots else {}
    return dict(groups=len(selected),roots=len(roots),root_metrics=result,root_values=roots,
                selected_query_ids={k:metadata[i]['query_id'] for k,i in selected.items()})


def paired_selection(a,b):
    aa,bb=a['root_values'],b['root_values']
    if aa.keys()!=bb.keys():raise ValueError('Unmatched selected roots')
    keys=list(next(iter(aa.values())))
    if a.get('regret_preference')!=b.get('regret_preference'):
        keys=[k for k in keys if k not in ('regret','realized_nominal_cost')]
    return {k:interval([aa[r][k]-bb[r][k] for r in aa]) for k in keys}


def ranking(metadata,y,score,bank,good,bad):
    byroot=defaultdict(list);groups=pairs=0;total=0.
    for ix in bank.values():
        a=np.asarray([i for i in ix if good(y[i])],dtype=int)
        b=np.asarray([i for i in ix if bad(y[i])],dtype=int)
        if not len(a) or not len(b):continue
        wins=(score[a,None]>score[b][None]).astype(float)+.5*(score[a,None]==score[b][None])
        v=wins.ravel().tolist();byroot[metadata[ix[0]]['root_id']].extend(v)
        total+=sum(v);pairs+=len(v);groups+=1
    roots={r:float(np.mean(v)) for r,v in byroot.items()}
    return dict(pairs=pairs,informative_groups=groups,informative_roots=len(roots),
                pair_accuracy=total/pairs if pairs else None,root_mean=interval(list(roots.values())),root_values=roots)


def paired_ranking(a,b):
    aa,bb=a['root_values'],b['root_values']
    if aa.keys()!=bb.keys():raise ValueError('Unmatched informative roots')
    return interval([aa[r]-bb[r] for r in aa])


def dispersion(metadata,score,group_key):
    bygroup=defaultdict(list)
    for i,m in enumerate(metadata):bygroup[m[group_key]].append(i)
    means=np.array([score[ix].mean() for ix in bygroup.values()])
    within=float(np.mean([np.var(score[ix]) for ix in bygroup.values()]))
    between=float(means.var());centered=np.zeros(len(score))
    for ix in bygroup.values():centered[ix]=score[ix]-score[ix].mean()
    return dict(between_variance=between,within_variance=within,
        between_to_within=between/within if within else None,
        mean_within_range=float(np.mean([np.ptp(score[ix]) for ix in bygroup.values()]))),centered


def prediction_diagnostics(metadata,y,physical,social,safe,bank):
    result={}
    tests=[('physical',lambda a:a!=2,lambda a:a==2),
           ('social',lambda a:a==0,lambda a:a==1),
           ('safe',lambda a:a==0,lambda a:a!=0)]
    allbanks=banks(metadata,np.ones(len(y)),False)
    for name,risk in physical.items():
        item=dict(collision_ap=ranking_metrics((y==2).astype(int),risk)[0],
            within_root=ranking(metadata,y,-risk,allbanks,tests[0][1],tests[0][2]),
            within_path=ranking(metadata,y,-risk,bank,tests[0][1],tests[0][2]))
        for key in ('root_id','group_id'):
            d,c=dispersion(metadata,risk,key)
            d['centered_collision_ap']=ranking_metrics((y==2).astype(int),c)[0]
            item[key+'_dispersion']=d
        if name!='cv_exposure':item['probability_metrics']=metrics((y!=2).astype(int),1-risk)
        else:item['probability_metrics']=None
        result[name]=item
    for name,scores in (('hierarchy_social',social),('hierarchy_safe',safe)):
        good,bad=tests[1 if name.endswith('social') else 2][1:]
        result[name]=dict(within_root=ranking(metadata,y,scores,allbanks,good,bad),
                          within_path=ranking(metadata,y,scores,bank,good,bad))
    for name,risk in physical.items():
        result[name]['social_proxy_within_path']=ranking(metadata,y,-risk,bank,tests[1][1],tests[1][2])
        result[name]['safe_proxy_within_path']=ranking(metadata,y,-risk,bank,tests[2][1],tests[2][2])
        result[name]['proxy_semantics']='Physical score used only for ordering; not a social or safe-success probability'
    return result


def controllability(metadata,y,matched_bank,targets):
    roots=banks(metadata,np.ones(len(y)),False)
    def census(bank):
        counters=defaultdict(int);informative=defaultdict(set)
        for k,ix in bank.items():
            classes=set(y[ix].tolist());root=metadata[ix[0]]['root_id']
            flags=dict(avoidable_collision=2 in classes and bool(classes&{0,1}),
                avoidable_social_success=0 in classes and 1 in classes,
                no_successful_candidate=classes=={2},no_clean_candidate=0 not in classes,
                mixed_outcomes=len(classes)>1,all_human_collision=classes=={2})
            for name,value in flags.items():
                counters[name]+=int(value)
                if value:informative[name].add(root)
        return dict(groups=len(bank),counts=dict(counters),root_counts={k:len(v) for k,v in informative.items()})
    transitions={};contrast_roots={};contrast_groups={}
    for kind,variants in [('waiting',('wait_0.5','wait_1.0')),('speed',('half_speed',))]:
        matrix=np.zeros((3,3),dtype=int);changed=set();groups=set()
        for k,ix in matched_bank.items():
            immediate=[i for i in ix if metadata[i]['variant']=='immediate_normal']
            if len(immediate)!=1:raise ValueError('Expected one immediate query per path')
            a=immediate[0]
            for b in ix:
                if metadata[b]['variant'] in variants:
                    matrix[y[a],y[b]]+=1
                    if y[a]!=y[b]:changed.add(metadata[a]['root_id']);groups.add(k)
        transitions[kind]=matrix.tolist();contrast_roots[kind]=len(changed);contrast_groups[kind]=len(groups)
    # Standalone WAIT has zero motion and no same-path moving counterpart.
    wait={}
    for i,m in enumerate(metadata):
        if m['variant']=='WAIT':wait[m['root_id']]=int(y[i])
    class1=y==1;path_safe=np.zeros(len(y),dtype=bool);root_safe=np.zeros(len(y),dtype=bool)
    for ix in matched_bank.values():path_safe[ix]=np.any(y[ix]==0)
    for ix in roots.values():root_safe[ix]=np.any(y[ix]==0)
    times=np.asarray(targets['social_violation_time_s']);valid=np.asarray(targets['social_violation_time_valid'],dtype=bool)
    return dict(root_banks=census(roots),same_path=census(matched_bank),
        transition_axes='rows immediate outcome, columns timing/speed outcome; exclusive classes 0/1/2',
        transitions=transitions,changed_roots=contrast_roots,changed_path_groups=contrast_groups,
        standalone_wait_class_counts=np.bincount(list(wait.values()),minlength=3).tolist(),
        class1_queries=int(class1.sum()),class1_with_safe_same_path=int((class1&path_safe).sum()),
        class1_with_safe_same_root=int((class1&root_safe).sum()),
        class1_root_time_psc=int((class1&valid&(times<=1e-5)).sum()),
        note='Retrospective saved-bank controllability; hidden PSC and labels never used for selection. No-clean differs from no noncollision success.')


def error_decomposition(metadata,y,bank,selected):
    counts=defaultdict(int);roots=defaultdict(set)
    for k,i in selected.items():
        classes=set(y[bank[k]].tolist());r=metadata[i]['root_id']
        flags=dict(selected_collision=y[i]==2,avoidable_selected_collision=y[i]==2 and bool(classes&{0,1}),
                   collision_bank_limited=classes=={2},selected_social=y[i]==1,
                   avoidable_selected_social=y[i]==1 and 0 in classes,
                   selected_nonclean_no_clean_alternative=y[i]!=0 and 0 not in classes)
        for name,flag in flags.items():
            counts[name]+=int(flag)
            if flag:roots[name].add(r)
    return dict(counts=dict(counts),roots={k:len(v) for k,v in roots.items()})
