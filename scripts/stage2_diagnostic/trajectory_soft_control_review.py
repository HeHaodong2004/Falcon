"""Independent small-formula and cached-aggregate review of soft geometry control."""
from __future__ import annotations
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/stage2_diagnostic_20261025'
SRC=ROOT/'scripts/stage2_diagnostic/soft_visibility_paired.py'
SOFT=BASE/'visibility/soft_paired_v1'
DEST=BASE/'trajectory/soft_geometry_independent_review.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def main():
    protocol=read(SOFT/'protocol.json');summary=read(SOFT/'summary.json')
    assert sha(SRC)==protocol['source_sha256']
    assert sha(SOFT/'protocol.json')==summary['protocol_sha256']
    # Execute only target-free scorer/bin/PAV pure functions: no inference import.
    tree=ast.parse(SRC.read_text());names={'opportunity_score','score_bins','fit_pav'}
    pure=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
    ns={'np':np,'SCORE_EDGES':np.linspace(0.,6.,33)};exec(compile(pure,str(SRC),'exec'),ns)
    score=ns['opportunity_score'];pav=ns['fit_pav'];bins=ns['score_bins']
    graph={'resolution':1.,'timestamp':1.};sensor={'horizontal_fov_deg':90.,'max_range_m':10.}
    path=[[1.,.5,0.,.5,0.],[2.,.5,0.,.5,0.]];keys=np.asarray([[0,0],[1,0],[-1,0],[0,-1]],np.int64)
    np.testing.assert_allclose(score(graph,path,keys,sensor),[0.,.9,0.,0.],atol=1e-15,rtol=0)
    # Initial pose does not itself invent an observation; dt scales saved future step.
    longpath=np.asarray(path);longpath[-1,0]=3.
    np.testing.assert_allclose(score(graph,longpath,keys,sensor),[0.,1.8,0.,0.],atol=1e-15,rtol=0)
    np.testing.assert_array_equal(bins([0.,np.nextafter(0.,1.),.1875,.18750001,6.,7.]),[0,1,1,2,32,32])
    rng=np.random.default_rng(25271)
    for _ in range(200):
        c=rng.integers(0,20,(6,2)).astype(float);c[rng.integers(0,6)]=0
        support=np.flatnonzero(c.sum(1)>0);mass=c.sum(1);independent=[]
        for pos,i in enumerate(support):
            independent.append(max(min(c[support[a:b+1],1].sum()/mass[support[a:b+1]].sum() for b in range(pos,len(support))) for a in range(pos+1)))
        expected=np.interp(np.arange(6),support,independent)
        np.testing.assert_allclose(pav(c),expected,atol=1e-14,rtol=0)
    records=[read(p) for p in sorted((SOFT/'roots').glob('*.json'))]
    assert len(records)==1705
    scenes=defaultdict(list)
    for r in records:
        assert r['protocol_sha256']==sha(SOFT/'protocol.json')
        assert r['out_of_union_truth_counts']==[0,0]
        scenes[r['scene_id']].append(r)
    selected=protocol['selection_scenes'];validation=protocol['validation_scenes']
    assert len(selected)==70 and len(validation)==38 and set(selected).isdisjoint(validation)
    assert sum(len(scenes[s]) for s in selected)==1136 and sum(len(scenes[s]) for s in validation)==569
    # Independent analytic min-max isotonic fit to equal-scene/Root/pair/cell mass.
    fit=np.mean([np.mean([np.asarray(r['counts']).sum(0)/(2*r['candidates']) for r in scenes[s]],axis=0) for s in selected],axis=0)
    supported=np.flatnonzero(fit.sum(1)>0);mass=fit.sum(1);support_rates=[]
    for pos,i in enumerate(supported):
        support_rates.append(max(min(fit[supported[a:b+1],1].sum()/mass[supported[a:b+1]].sum() for b in range(pos,len(supported))) for a in range(pos+1)))
    rates=np.interp(np.arange(33),supported,support_rates)
    np.testing.assert_allclose(rates,summary['rates'],atol=1e-14,rtol=0)
    np.testing.assert_allclose(mass,summary['selection_bin_mass'],atol=1e-14,rtol=0)
    def proper(hist):
        n=hist.sum();return [-np.sum(hist[:,0]*np.log(np.clip(1-rates,1e-9,1))+hist[:,1]*np.log(np.clip(rates,1e-9,1)))/n,
                            np.sum(hist[:,0]*rates**2+hist[:,1]*(1-rates)**2)/n]
    values=[]
    for s in validation:
        roots=[]
        for r in scenes[s]:
            own=np.mean([proper(h) for h in np.asarray(r['counts'])],axis=0)
            crossed=np.mean([proper(h) for h in np.asarray(r['crossed_counts'])],axis=0)
            expected=np.asarray(r['counts']).sum(2)@rates
            errors=expected-r['true_counts']
            geom=np.r_[own,np.mean(abs(errors)),np.mean(errors),crossed[0]-own[0]]
            roots.append(np.stack([np.r_[r['stage2'][name]['own_scores'],r['stage2'][name]['count_mae'],r['stage2'][name]['count_bias'],r['stage2'][name]['logloss_gain']] for name in ('direct','production')]+[geom]))
        values.append(np.mean(roots,axis=0))
    values=np.asarray(values);mean=values.mean(0);fields=('binary_logloss','binary_brier','count_mae','count_bias','own_cross_logloss_gain')
    maxdiff=0.
    for row,name in zip(mean,('stage2_direct','stage2_production','calibrated_geometry')):
        for field,value in zip(fields,row):
            delta=abs(value-summary['metrics'][name][field]);assert delta<=1e-12;maxdiff=max(maxdiff,delta)
    draw=np.random.default_rng(2527).integers(0,38,(2000,38));diff=values[draw].mean(1)[:,2]-values[draw].mean(1)[:,0]
    for j,f in enumerate(fields):np.testing.assert_allclose(np.quantile(diff[:,j],[.025,.975]),summary['paired_scene_geometry_minus_direct_ci95'][f],atol=1e-12,rtol=0)
    assert summary['score_overflow_count']==0 and summary['stage2_binary_score_outside_unit_interval']==0
    result=dict(status='PASS_WITH_DECLARED_SCOPE',source_sha256=sha(SRC),protocol_sha256=sha(SOFT/'protocol.json'),summary_sha256=sha(SOFT/'summary.json'),
                review_source_sha256=sha(__file__),source_archive_sha256=sha(SOFT/'source_snapshot.json') if (SOFT/'source_snapshot.json').exists() else None,
                scorer_target_free=True,sensor_and_current_graph_hash_checks=True,
                toy_checks=['forward/behind/zero-range bearing convention','actual saved-step dt scaling','right-closed bins and overflow','200 randomized weighted PAV comparisons to independent analytic max-min solution'],
                independently_reaggregated_pairs=1705,selection_pairs=1136,validation_pairs=569,selection_scenes=70,validation_scenes=38,
                calibration_rates_max_absolute_difference=float(np.max(abs(rates-summary['rates']))),metric_max_absolute_difference=maxdiff,
                all_validation_paired_scene_ci95_reproduced=True,weighted_fit='branch mean; per-cell normalized per-Root; Root mean within scene; equal 70 selection scenes',
                conclusion='Declared fixed soft-opportunity control supports a Stage2 advantage on the paired 38-scene diagnostic cohort. Proper scores/counts and calibration rates independently reproduced from cached root records.',
                limitations=['Calibrator uses binary revelation targets on 70 scenes; no labels enter the raw geometric scorer, but calibrated baseline is not globally label-free.',
                '38 scenes were scored historically in Stage2 development; diagnostic partition is not an untouched protected confirmation set.',
                '2D yaw/FOV/distance opportunity omits occlusion and camera articulation; failure of this control does not exclude stronger geometric baselines.',
                'Direct fixed-U includes extrapolated rows outside each natural candidate support. Production-padded-U also reported and agrees closely.',
                'Control predicts revelation only; no FREE/BLOCKED class, height, or full future graph claim.',
                'Bootstrap conditions on fixed 70-scene PAV calibration; calibration-estimation uncertainty is not resampled.',
                'High-resolution raw-score rank is binned with bounds; exact tied PAV rank is a separate score. No raw-score overflow observed.',
                'This review reaggregates cached bin counts and stage metrics, not all raw score caches or source legal maps. Earlier independent causal replay/B checks cover source-target integrity.'])
    with open(DEST,'x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
