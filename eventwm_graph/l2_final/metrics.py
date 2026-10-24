"""Frozen proper scores, paired scene uncertainty, and complete support ledgers."""
import collections
import gzip
import json
from pathlib import Path
from .common import OUT,PREVIOUS,sha,dump
from eventwm_graph.l2_scaling_500k.compare import rows,ci,pair,practical_age,case_key,FIELDS as BASE_FIELDS
FIELDS=BASE_FIELDS+('endpoint_error','radius50','radius90','direction_error')


FOLDERS=dict(A=PREVIOUS/'evaluation_arm_new/evaluation',B=OUT/'evaluation_arm_original_repaired/evaluation',C=OUT/'evaluation_arm_new/evaluation')
RECOVERY=('tracked_es_gain','path_es_gain','future_path_es_gain','coverage50_change','coverage90_change','association_accuracy','false_split','false_merge','duplicate','missing_antecedent_mass')


def selected(record,cohort='native',scope='all',horizon=None):
    if record['cohort']!=cohort:return False
    if horizon is not None and record['H']!=horizon:return False
    if scope=='visible':return record.get('age') is not None and record['age']<=1e-7
    if scope=='practical':return practical_age(record) and record['H']<=2.
    return True


def forecasting(records):
    records=[r for r in records if r['status']!='joint_supported'];scopes=[];cells=[]
    for cohort in ('native','masked'):
        for scope in ('all','visible','practical'):
            part=[r for r in records if selected(r,cohort,scope)]
            scopes.append(dict(cohort=cohort,scope=scope,status=dict(collections.Counter(r['status'] for r in part)),metrics={k:ci(part,k) for k in FIELDS}))
        for age in sorted({r['age_bin'] for r in records if r['cohort']==cohort}):
            for horizon in (.5,1.,2.,3.):
                part=[r for r in records if r['cohort']==cohort and r['age_bin']==age and r['H']==horizon]
                cells.append(dict(cohort=cohort,age_bin=age,H=horizon,status=dict(collections.Counter(r['status'] for r in part)),metrics={k:ci(part,k) for k in FIELDS}))
    return dict(rows=len(records),roots=len({r['root_id'] for r in records}),scenes=len({r['scene_id'] for r in records}),
                status=dict(collections.Counter(r['status'] for r in records)),scopes=scopes,age_horizon=cells)


def recovery(records):
    cells=[]
    for scope in ('all','practical'):
        for horizon in (.5,1.,2.,3.):
            part=[r for r in records if r['H']==horizon and r.get('gap') is not None and (scope=='all' or .75-1e-8<=r['gap']<=1.25+1e-8)]
            cells.append(dict(scope=scope,H=horizon,status=dict(collections.Counter(r['prior_status']+'/'+r['posterior_status'] for r in part)),
                              metrics={k:ci(part,k) for k in RECOVERY}))
    return dict(rows=len(records),roots=len({r['root_id'] for r in records}),scenes=len({r['scene_id'] for r in records}),cells=cells)


def belief(records):
    result=dict(states=len(records),roots=len({r['root_id'] for r in records}),scenes=len({r['scene_id'] for r in records}))
    predicates=dict(zero_true_count_support=lambda r:not r['true_count_support'],duplicate_ancestry=lambda r:r['expected_duplicate_ancestries']>0,
                    missing_preferred_ancestry=lambda r:r['expected_missing_preferred_ancestries']>0)
    for name,predicate in predicates.items():
        part=[r for r in records if predicate(r)]
        result[name]=dict(dependent_states=len(part),roots=len({r['root_id'] for r in part}),scenes=len({r['scene_id'] for r in part}))
        event_by_root={}
        for r in records:
            key=(r['scene_id'],r['root_id']);event_by_root[key]=event_by_root.get(key,False) or predicate(r)
        result[name]['any_state_root_rate']=ci([dict(scene_id=s,root_id=root,event=float(event)) for (s,root),event in event_by_root.items()],'event')
    result['max_arrived_coordinate_error']=max((r['max_arrived_coordinate_error'] or 0. for r in records),default=None)
    result['true_count_nll']='INFINITE: at least one zero-support truth; no probability floor' if any(not r['true_count_support'] for r in records) else 'all represented true counts positive'
    return result


def contrast(left,right,kind,fields):
    pairs,ledger=pair(left,right,kind,fields);cells=[]
    if kind=='forecasts':
        for cohort in ('native','masked'):
            for scope in ('all','visible','practical'):
                part=[r for r in pairs if selected(r,cohort,scope)]
                cells.append(dict(cohort=cohort,scope=scope,metrics={k:ci(part,k) for k in fields},
                                  common_left={k:ci(part,'old_'+k) for k in fields},common_right={k:ci(part,'new_'+k) for k in fields}))
    else:
        for scope in ('all','practical'):
            for horizon in (.5,1.,2.,3.):
                part=[r for r in pairs if r['H']==horizon and r.get('gap') is not None and (scope=='all' or .75-1e-8<=r['gap']<=1.25+1e-8)]
                cells.append(dict(scope=scope,H=horizon,metrics={k:ci(part,k) for k in fields}))
    return dict(case_ledger=ledger,cells=cells)


def three_way(data):
    maps={name:{case_key(r,'forecasts'):r for r in values['raw_forecasts']} for name,values in data.items()}
    common=set.intersection(*(set(value) for value in maps.values()));cells=[]
    for cohort in ('native','masked'):
        for scope in ('all','visible','practical'):
            records={name:[] for name in maps};fields=('path_es','tracked_es','coverage50','coverage90','endpoint_error','radius50','radius90')
            for key in common:
                if not selected(maps['A'][key],cohort,scope):continue
                # Each reported field uses exactly the same cases in all arms.
                available={field for field in fields if all(maps[name][key].get(field) is not None for name in maps)}
                for name in maps:
                    row=maps[name][key];records[name].append({k:row[k] for k in ('root_id','scene_id')}|{field:row[field] for field in available})
            cells.append(dict(cohort=cohort,scope=scope,variants={name:{field:ci(part,field) for field in fields} for name,part in records.items()}))
    return dict(case_key_intersection=len(common),cells=cells,scope='fieldwise common supported physical cases in all three variants; all-arm failure ledgers reported separately')


def public_summary(controlled,final=False):
    result={}
    suffix='_final' if final else ''
    for variant,name in (('B','original_public'+suffix),('C','public'+suffix)):
        folder=OUT/('evaluation_arm_'+name)/'evaluation_shard_0000_0528'
        if not (folder/'execution_guard.json').exists():continue
        forecasts=rows(folder,'forecasts');rec=rows(folder,'reobservation');beliefs=rows(folder,'beliefs')
        # Preserve the model-independent frozen annotation validity rule.
        from .common import BASE
        import importlib.util
        path=BASE/'docs/l2/intermittent_wm_20261022/artifacts/verified_execution/orchestration/annotation_accounting.py'
        classification=path.with_name('native_annotation_validity.json')
        marker=folder/'public_annotation_applied.json'
        if not marker.exists():
            guard=json.loads((folder/'execution_guard.json').read_text())
            assert guard['sources_stable'] and guard['inputs_stable']
            dump(folder/'forecast_execution_guard.json',dict(error=None,sources_stable=True,inputs_stable=True,
                 wrapper_guard_sha256=sha(folder/'execution_guard.json'),scope='source-isolated public initializer; unchanged frozen scoring functions'))
            spec=importlib.util.spec_from_file_location('public_annotation_accounting',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.DOC=BASE/'docs/l2/intermittent_wm_20261022/artifacts/verified_execution/sources/docs/l2/intermittent_wm_20261022'
            module.apply(folder,classification)
            dump(marker,dict(rule_sha256=sha(path),classification_sha256=sha(classification)))
            forecasts=rows(folder,'forecasts');rec=rows(folder,'reobservation');beliefs=rows(folder,'beliefs')
        init_path=folder.parent/'public_initialization.jsonl'
        initializations=[json.loads(line) for line in init_path.read_text().splitlines()]
        result[variant]=dict(forecasting=forecasting(forecasts),reobservation=recovery(rec),belief=belief(beliefs),
                             initialization_attempts=len(initializations),initialization_roots=len({r['root_id'] for r in initializations}),
                             initialization_failures=[r for r in initializations if r['status']=='INITIALIZATION_FAILED'],
                             fallback_roots=len({r['root_id'] for r in initializations if r.get('fallback')}),
                             configuration_sha256=sha(folder/'configuration.json'),frozen_annotation_rule=True)
        root_ids={r['root_id'] for r in forecasts}
        result[variant]['public_minus_common_bootstrap']=dict(
            forecasting=contrast([r for r in controlled[variant]['raw_forecasts'] if r['root_id'] in root_ids],
                                 [r for r in forecasts if r['status']!='joint_supported'],'forecasts',FIELDS),
            reobservation=contrast([r for r in controlled[variant]['raw_reobservation'] if r['root_id'] in root_ids],rec,'reobservation',RECOVERY))
    if len(result)==2:
        a=OUT/('evaluation_arm_original_public'+suffix)/'evaluation_shard_0000_0528';b=OUT/('evaluation_arm_public'+suffix)/'evaluation_shard_0000_0528'
        result['C_minus_B']=dict(forecasting=contrast([r for r in rows(a,'forecasts') if r['status']!='joint_supported'],[r for r in rows(b,'forecasts') if r['status']!='joint_supported'],'forecasts',FIELDS),
                                 reobservation=contrast(rows(a,'reobservation'),rows(b,'reobservation'),'reobservation',RECOVERY))
    return result


def run():
    data={};inputs={}
    for name,folder in FOLDERS.items():
        forecasts=rows(folder,'forecasts');rec=rows(folder,'reobservation');beliefs=rows(folder,'beliefs')
        data[name]=dict(raw_forecasts=[r for r in forecasts if r['status']!='joint_supported'],raw_reobservation=rec,
                        forecasting=forecasting(forecasts),reobservation=recovery(rec),belief=belief(beliefs))
        inputs[name]={kind:sha(folder/(kind+'.jsonl.gz')) for kind in ('forecasts','reobservation','beliefs')}
    comparisons={}
    for left,right in (('A','B'),('A','C'),('B','C')):
        comparisons[right+'_minus_'+left]=dict(forecasting=contrast(data[left]['raw_forecasts'],data[right]['raw_forecasts'],'forecasts',FIELDS),
                                              reobservation=contrast(data[left]['raw_reobservation'],data[right]['raw_reobservation'],'reobservation',RECOVERY))
    common_three=three_way(data)
    rejected_public=public_summary(data)
    public=public_summary(data,final=True)
    for record in data.values():record.pop('raw_forecasts');record.pop('raw_reobservation')
    result=dict(variants=data,comparisons=comparisons,public_initialization=public,
                rejected_causal_prefix_experiment=rejected_public,common_three_variants=common_three,inputs=inputs,
                uncertainty='1,000 paired scene bootstrap draws seed73; average dependent states/streams/targets within root, then equal roots within scene; cells below30roots/10scenes inconclusive',
                interpretation='development reuse, not protected confirmation; C-v-B is the full-data training campaign effect including required label hygiene, not data size alone')
    dump(OUT/'metrics.json',result)


if __name__=='__main__':run()
