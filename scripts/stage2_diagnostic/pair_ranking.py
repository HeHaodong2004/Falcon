"""Matched same-decision/horizon count-order diagnostic, never a ranking model."""
from collections import defaultdict
import gzip
import json
import numpy as np
from .common import OUT,save

def main():
    protocol=json.loads((OUT/'protocol.json').read_text());validation=set(protocol['diagnostic_validation_scenes'])
    csummary=json.loads((OUT/'visibility/full_v1/summary.json').read_text());rates=np.array(csummary['calibration']['category_rates'])
    soft=json.loads((OUT/'visibility/soft_paired_v1/branch_expected_counts.json').read_text())
    soft_by_root={r['root_id']:r for r in soft['records']}
    records=[];maximum_count_difference=0.
    for path in sorted((OUT/'trajectory/primary_v1/roots').glob('*.json')):
        b=json.loads(path.read_text());pair=b['pair']
        with gzip.open(OUT/'probability/roots'/(b['root_id']+'.json.gz'),'rt') as f:a=json.load(f)
        with gzip.open(OUT/'visibility/full_v1/roots'/(b['root_id']+'.json.gz'),'rt') as f:c=json.load(f)
        aw={(r['start_node'],r['end_node']):r for r in a['records']};cw={(r['start_node'],r['end_node']):r for r in c['records']}
        sides=[(aw[(pair['start_node'],pair[end])],cw[(pair['start_node'],pair[end])]) for end in ('end_a','end_b')]
        true=np.array([r[0]['true_counts'] for r in sides]);pred=np.array([r[0]['expected_counts'] for r in sides])
        np.testing.assert_array_equal(true,[b['true_counts_a'],b['true_counts_b']])
        bp=np.array([b['modes']['production_E']['expected_counts_a'],b['modes']['production_E']['expected_counts_b']])
        maximum_count_difference=max(maximum_count_difference,float(abs(pred-bp).max()))
        baseline={
            'stage2':pred[:,0],
            'stage2_fixed_union':np.array([b['modes']['direct_fixed_union_U']['expected_counts_a'][0],b['modes']['direct_fixed_union_U']['expected_counts_b'][0]]),
            'soft_geometry_fixed_union':np.array(soft_by_root[b['root_id']]['geometry_expected_counts']),
            'calibrated_geometry':np.array([np.dot(np.asarray(r[1]['category_label_counts']).sum(1),rates) for r in sides]),
            'strict_frontier_count':np.array([r[1]['raw_opportunity_counts'][0] for r in sides]),
            'frustum_opportunity_count':np.array([r[1]['raw_opportunity_counts'][1] for r in sides]),
            'candidate_domain_area':np.array([r[0]['candidates'] for r in sides]),
            'duration_only_constant':np.zeros(2)}
        records.append(dict(root_id=b['root_id'],scene_id=b['scene_id'],duration=b['duration'],
            start_node=pair['start_node'],end_a=pair['end_a'],end_b=pair['end_b'],
            true_counts=true.tolist(),expected_counts=pred.tolist(),baseline_revelation_counts={k:v.tolist() for k,v in baseline.items()}))
    if len(records)!=1705:raise ValueError('complete paired cohort required')
    output=dict(pairs=len(records),maximum_A_vs_B_production_expected_count_difference=maximum_count_difference,
        method='One pose-selected existing same-start/exact-timestamps pairperRoot; countorder only, no Query-ranking model fit/selection; actualties excluded, predictedties score0.5',
        limitations=['Pose-diverse selectedpairs are not allQueries; overlap is not summed as independentinformation.',protocol['split_caveat']],subsets={})
    for subset,rs in [('all',records),('diagnostic_validation38',[r for r in records if r['scene_id'] in validation])]:
        output['subsets'][subset]={}
        for outcome,index in [('revelation',0),('free',1),('blocked',2)]:
            target={}
            for margin in (0,10,50):
                eligible=[r for r in rs if abs(r['true_counts'][0][index]-r['true_counts'][1][index])>margin]
                if not eligible:continue
                names=list(eligible[0]['baseline_revelation_counts']) if index==0 else ['stage2']
                values=defaultdict(list)
                for r in eligible:
                    truth=np.sign(r['true_counts'][0][index]-r['true_counts'][1][index]);row=[]
                    for name in names:
                        p=r['baseline_revelation_counts'][name] if index==0 else np.array(r['expected_counts'])[:,index]
                        difference=p[0]-p[1];direction=0 if abs(difference)<=1e-9 else np.sign(difference)
                        row.append(.5 if direction==0 else float(direction==truth))
                    values[r['scene_id']].append(row)
                array=np.array([np.mean(v,axis=0) for _,v in sorted(values.items())]);mean=array.mean(0)
                rng=np.random.default_rng(2525);draws=np.array([array[rng.integers(0,len(array),len(array))].mean(0) for _ in range(2000)])
                target[f'actual_difference_gt_{margin}']=dict(pairs=len(eligible),scenes=len(values),
                    exact_actual_ties_excluded=sum(r['true_counts'][0][index]==r['true_counts'][1][index] for r in rs),
                    accuracy={name:dict(value=float(mean[j]),ci95=np.quantile(draws[:,j],[.025,.975]).tolist(),
                        stage2_minus_baseline_ci95=np.quantile(draws[:,0]-draws[:,j],[.025,.975]).tolist()) for j,name in enumerate(names)})
            output['subsets'][subset][outcome]=target
    save(OUT/'decision_relevance/pairwise_counts.json',output)
    save(OUT/'decision_relevance/pairwise_count_records.json',records)
    print(json.dumps(output),flush=True)

if __name__=='__main__':main()
