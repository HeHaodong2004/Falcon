"""Derive paired scene-clustered comparisons from saved repair predictions.

This does not fit/select models or reopen any dataset payload. The metric
intersection and missing-denominator counts are always retained in the output.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eventwm_graph.l2_partial.repair_evaluation import scene_ci


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(rows,left,right,key,*,horizon=None):
    selected=[r for r in rows if horizon is None or r.get('horizon')==horizon]
    index={(r['root_id'],r['variant']):r for r in selected}
    roots={r['root_id'] for r in selected};result=[];missing=0
    for root in sorted(roots):
        a=index.get((root,left));b=index.get((root,right))
        if a is None or b is None or a.get(key) is None or b.get(key) is None:
            missing+=1;continue
        result.append({'root_id':root,'scene_id':a['scene_id'],'delta':a[key]-b[key]})
    return {'left':left,'right':right,'metric':key,'horizon':horizon,'left_minus_right':scene_ci(result,'delta'),
            'excluded_roots_missing_either_score':missing,'root_rows':result}


def run(directory):
    directory=Path(directory).resolve()
    source=directory/'population_evaluation.json';evaluation=json.loads(source.read_text())
    populations=evaluation['population_rows'];forecasts=evaluation['forecast_rows']
    pairs=[('full','legacy'),('association_only','legacy'),('full','association_only'),
           ('full','full_no_maneuver'),('full','full_kinematic'),('full','full_unfitted_optics')]
    result={'population_pairs':[], 'forecast_pairs':[], 'localized_rates':{},
            'source_sha256':{source.name:sha(source),'summary_script':sha(Path(__file__))},
            'scope':'paired root differences followed by equal-scene means and1000 scene bootstrap replicates; development only'}
    for left,right in pairs:
        for key in ['count_absolute_error','gospa','false_localized','missed_localized','sampled_population_gospa','unresolved_mass']:
            result['population_pairs'].append(paired(populations,left,right,key))
        for horizon in [.5,1.,2.,3.]:
            for key in ['ancestry_energy_score','gospa','ancestry_coverage90','sampled_population_gospa']:
                result['forecast_pairs'].append(paired(forecasts,left,right,key,horizon=horizon))
    for name in evaluation['variants']:
        rows=[]
        for r in populations:
            if r['variant']!=name:continue
            predicted=r.get('conditional_localized_count');truth=r['true_count']
            rows.append({'root_id':r['root_id'],'scene_id':r['scene_id'],
                'false_fraction':r['false_localized']/predicted if predicted and r['false_localized'] is not None else None,
                'missed_fraction':r['missed_localized']/truth if truth and r['missed_localized'] is not None else None})
        result['localized_rates'][name]={'false_track_fraction':scene_ci(rows,'false_fraction'),
            'missed_human_fraction':scene_ci(rows,'missed_fraction'),
            'zero_predicted_denominator_roots':sum(r['false_fraction'] is None for r in rows),
            'zero_truth_denominator_roots':sum(r['missed_fraction'] is None for r in rows),
            'definition':'per-root expected localized false count / expected localized count; expected missed count / true population; zero denominators undefined; all roots remain in count and GOSPA tables'}
    risk_path=directory/'risk_scene_calibrated.json'
    risk=json.loads(risk_path.read_text());result['source_sha256'][risk_path.name]=sha(risk_path)
    result['risk_pairs']=[]
    for left,right in [('calibrated','legacy'),('calibrated','prefit'),('calibrated','brier_v4')]:
        la={(r['root_id'],r['query_id']):r for r in risk['models'][left]['query_rows']}
        rb={(r['root_id'],r['query_id']):r for r in risk['models'][right]['query_rows']}
        if set(la)!=set(rb):raise ValueError('risk comparison candidate inventories differ')
        for event in ['collision','psc','untracked_appearance_proxy']:
            rows=[{'root_id':a['root_id'],'scene_id':a['scene_id'],
                    'delta':a['brier_'+event]-rb[key]['brier_'+event]}
                  for key,a in la.items() if a['endpoint_eligible'] and rb[key]['endpoint_eligible']]
            result['risk_pairs'].append({'left':left,'right':right,'event':event,
                                         'brier_left_minus_right':scene_ci(rows,'delta')})
    decisions=risk['models']['calibrated']['decision']['root_rows']
    result['decision_pairs']=[]
    for left,right in [('native_expected_utility','depth0_planned_progress'),
                       ('residual_upper_risk','depth0_planned_progress')]:
        la={r['root_id']:r for r in decisions if r['rule']==left}
        rb={r['root_id']:r for r in decisions if r['rule']==right}
        if set(la)!=set(rb):raise ValueError('decision rule root inventories differ')
        for key in ['actual_utility','regret','selected_collision','selected_psc']:
            rows=[{'root_id':root,'scene_id':a['scene_id'],'delta':a[key]-rb[root][key]}
                  for root,a in la.items()]
            result['decision_pairs'].append({'left':left,'right':right,'metric':key,
                                             'left_minus_right':scene_ci(rows,'delta')})
    fresh_path=directory/'fresh_evaluation/population.json'
    if fresh_path.exists():
        fresh=json.loads(fresh_path.read_text());result['fresh_summary']=fresh['summary']
        result['source_sha256'][str(fresh_path.relative_to(directory))]=sha(fresh_path)
        result['fresh_status']='INCONCLUSIVE: only15 roots/11 scenes; no selection or fitting'
    destination=directory/'paired_comparisons.json'
    with destination.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'output':str(destination),'population_pairs':len(result['population_pairs']),
                      'forecast_pairs':len(result['forecast_pairs'])},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');run(parser.parse_args().directory)
