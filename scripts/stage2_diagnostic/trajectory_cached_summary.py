"""Scene/Root weighted neutral and sensitivity reports from frozen B caches."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from .common import OUT,EDGES,digest,save
from .statistics import quantiles


def hierarchy(records,getter):
    by=defaultdict(list)
    for r in records:
        value=getter(r)
        if value is not None:by[r['scene_id']].append(np.atleast_1d(np.asarray(value,np.float64)))
    if not by:return None
    names=sorted(by);values=np.array([np.mean(by[s],axis=0) for s in names])
    draw=np.random.default_rng(202610252).integers(0,len(names),size=(2000,len(names)))
    return dict(roots=sum(map(len,by.values())),scenes=len(by),mean=values.mean(0).tolist(),
                ci95=np.quantile(values[draw].mean(1),[.025,.975],axis=0).T.tolist())


def main(name='primary_v1'):
    root=OUT/'trajectory'/name;records=[json.loads(p.read_text()) for p in sorted((root/'roots').glob('*.json'))]
    assert len(records)==json.loads((root/'complete.json').read_text())['roots']
    neutral=dict(label='MOTION_ABLATION_OOD_NO_RECORDED_COUNTERFACTUAL_GT',weighting='RootmeanwithinScene thenequalScene; pairbranches remaincorrelated',
        confidence='2000scene-cluster draws seed202610252; variabilityofmeasuredsensitivity, NOT neutraloutcome/calibration uncertainty',
        counts_order=['revelation','FREE','BLOCKED'],statistics={})
    for domain in ('U','I'):
        for branch in ('a','b'):
            key='vs_actual_'+branch+'_'+domain
            neutral['statistics']['TV_vs_'+branch+'_'+domain]=hierarchy(records,lambda r,k=key:r['neutral'][k]['mean_total_variation'] if r['neutral'][k] else None)
        neutral['statistics']['TV_branch_mean_'+domain]=hierarchy(records,lambda r,d=domain: .5*(r['neutral']['vs_actual_a_'+d]['mean_total_variation']+r['neutral']['vs_actual_b_'+d]['mean_total_variation']) if r['neutral']['vs_actual_a_'+d] and r['neutral']['vs_actual_b_'+d] else None)
        neutral['statistics']['neutral_expected_counts_'+domain]=hierarchy(records,lambda r,d=domain:r['neutral']['vs_actual_a_'+d]['expected_counts_a'] if r['neutral']['vs_actual_a_'+d] else None)
    neutral['statistics']['neutral_fraction_outside_own_natural_domain']=hierarchy(records,lambda r:r['neutral']['union_cells_outside_neutral_natural_domain']/r['union_candidates'])
    neutral['source_summary_sha256']=digest(root/'summary.json');neutral['script_sha256']=digest(__file__)
    neutral_path=root/'neutral_scene_weighted.json'
    if neutral_path.exists():
        previous=json.loads(neutral_path.read_text())
        assert {k:v for k,v in previous.items() if k!='script_sha256'}=={k:v for k,v in neutral.items() if k!='script_sha256'}
    else:save(neutral_path,neutral)
    modes=('production_U','production_I','direct_fixed_union_U','direct_fixed_union_I')
    report=dict(weighting='HierarchicalRoot/Scene means; onepairperRoot',counts_order=['revelation','FREE','BLOCKED'],probability_order=['UNKNOWN','FREE','BLOCKED'],modes={},
        hard_overlap_empty_union_convention=0,quantile_method='HierarchicallycellnormalizedCDF of absoluteF32revelationprobability differences; common highresolutionbins with explicitintervals. These are NOT averages of perRootquantiles.')
    for mode in modes:
        def field(r,key):
            value=r['modes'][mode]
            return value['sensitivity'][key] if value['cells'] else None
        result={key:hierarchy(records,lambda r,k=key:field(r,k)) for key in ('mean_absolute_probability_change','mean_total_variation','mean_absolute_revelation_change','hard_labeled_set_overlap','soft_labeled_minmax_overlap','soft_revelation_minmax_overlap')}
        for key in ('expected_counts_a','expected_counts_b','true_counts_a','true_counts_b'):
            result[key]=hierarchy(records,lambda r,k=key:r['modes'][mode].get(k) if r['modes'][mode]['cells'] else None)
        result['expected_counts_b_minus_a']=hierarchy(records,lambda r:np.asarray(r['modes'][mode]['expected_counts_b'])-r['modes'][mode]['expected_counts_a'])
        result['absolute_expected_count_difference']=hierarchy(records,lambda r:abs(np.asarray(r['modes'][mode]['expected_counts_b'])-r['modes'][mode]['expected_counts_a']))
        result['both_hard_predictions_empty_roots']=int(sum(np.asarray(r['modes'][mode]['own_confusion'])[:,1:].sum()==0 for r in records))
        report['modes'][mode]=result
    report['candidate_jaccard']=hierarchy(records,lambda r:r['candidate_jaccard'])
    report['production_minus_direct_gain_same_U']=hierarchy(records,lambda r:np.asarray(r['modes']['production_U']['gain'])-r['modes']['direct_fixed_union_U']['gain'])
    report['production_minus_direct_gain_same_I']=hierarchy(records,lambda r:np.asarray(r['modes']['production_I']['gain'])-r['modes']['direct_fixed_union_I']['gain'])
    report['proper_gain_domain_decomposition']={}
    for prefix in ('production','direct_fixed_union'):
        def contribution(r):
            whole=r['modes'][prefix+'_U'];inside=r['modes'][prefix+'_I'];outside=r['modes'][prefix+'_exclusive']
            a=np.asarray(inside['gain'])*inside['cells']/whole['cells'] if inside['cells'] else np.zeros(2)
            b=np.asarray(outside['gain'])*outside['cells']/whole['cells'] if outside['cells'] else np.zeros(2)
            np.testing.assert_allclose(a+b,whole['gain'],atol=1e-12,rtol=0)
            return np.r_[a,b]
        report['proper_gain_domain_decomposition'][prefix]=dict(columns=['I_logloss_gain_contribution','I_Brier_gain_contribution','exclusive_logloss_gain_contribution','exclusive_Brier_gain_contribution'],
            statistics=hierarchy(records,contribution),note='Contributions use eachRootdomainfraction BEFORE Root/Sceneaveraging; supports differ from standaloneexclusive-onlycohort.')
    scene_roots=Counter(r['scene_id'] for r in records);scene_count=len(scene_roots)
    hist={mode:np.zeros(len(EDGES)-1,np.float64) for mode in modes};means=Counter()
    for i,r in enumerate(records):
        path=root/'roots'/(r['root_id']+'.npz');assert digest(path)==r['cache_sha256']
        with np.load(path,allow_pickle=False) as a:
            mask=a['support_a']&a['support_b']
            for mode in modes:
                prefix='production' if mode.startswith('production') else 'direct'
                pa=a[prefix+'_a'];pb=a[prefix+'_b']
                qdiff=abs((pa[:,1]+pa[:,2]).astype(np.float64)-(pb[:,1]+pb[:,2]).astype(np.float64))
                if mode.endswith('_I'):qdiff=qdiff[mask]
                if not len(qdiff):continue
                root_weight=1/(scene_count*scene_roots[r['scene_id']])
                indices=np.clip(np.searchsorted(EDGES,qdiff,side='right')-1,0,len(EDGES)-2)
                hist[mode]+=np.bincount(indices,minlength=len(EDGES)-1)*(root_weight/len(qdiff))
                means[mode]+=float(qdiff.mean())*root_weight
        if i%250==0:print(json.dumps(dict(cached_roots=i+1,total=len(records))),flush=True)
    for mode in modes:
        assert abs(hist[mode].sum()-1)<1e-10
        np.testing.assert_allclose(means[mode],report['modes'][mode]['mean_absolute_revelation_change']['mean'][0],atol=1e-12,rtol=0)
        report['modes'][mode]['absolute_revelation_change_weighted_quantiles']=quantiles(hist[mode])
        report['modes'][mode]['exact_weighted_mean_from_cache']=means[mode]
    report['source_summary_sha256']=digest(root/'summary.json');report['script_sha256']=digest(__file__)
    save(root/'sensitivity_scene_weighted.json',report)
    np.savez_compressed(root/'absolute_revelation_change_histograms.npz',edges=EDGES,**hist)
    fields=[]
    for r in records:
        fields.append(dict(root_id=r['root_id'],scene_id=r['scene_id'],scene_split=r['pair']['diagnostic_scene_split'],
            duration=r['duration'],poses=r['poses'],start_node=r['pair']['start_node'],end_a=r['pair']['end_a'],end_b=r['pair']['end_b'],
            graph_sha256=r['graph_sha256'],natural_candidate_counts=r['natural_candidate_counts'],union_candidates=r['union_candidates'],intersection_candidates=r['intersection_candidates'],
            true_counts_a=r['true_counts_a'],true_counts_b=r['true_counts_b'],
            modes={mode:{key:r['modes'][mode][key] for key in ('true_counts_a','true_counts_b','expected_counts_a','expected_counts_b')} for mode in modes},
            source_record_sha256=digest(root/'roots'/(r['root_id']+'.json'))))
    save(root/'decision_count_table.json',dict(records=fields,counts_order=['revelation','FREE','BLOCKED'],
        scope='Recordedpairedcountdescriptives; notQueryrankingtraining or navigationutility',script_sha256=digest(__file__)))
    print(json.dumps(dict(status='CACHED_ONLY_WEIGHTED_REPORTS_COMPLETE',roots=len(records),scenes=scene_count)),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--name',default='primary_v1');main(p.parse_args().name)
