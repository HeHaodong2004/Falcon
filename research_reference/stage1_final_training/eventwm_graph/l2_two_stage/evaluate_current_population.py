"""All-anchor CURRENT population diagnostics, separate from future forecasting.

No future-support filter, target clipping, identity-based exclusions, or hidden
predictor inputs. Finite-bank reference scores and unbiased underlying-law
scores are distinct. Resource caps remain explicit INCONCLUSIVE records.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import digamma,polygamma
from scipy.stats import poisson
from .association import AssociationGraph
from .common import OUT,DOC,dump,sha,seed_for
from .conditioned import HistoryConditionedWorldModel,draw_conditioned_current,pad_current,source_coordinates,PopulationSamplingFailure
from .conditioned_training import make_batch
from .count import sample_conditioned
from .data import load_records
from .evaluate import summarize,paired,occupancy
from .future_population_targets import attach_targets,current_measurement_overlay
from .history_proposal import augment_numpy
from .last_event_metadata import attach_metadata
from .population import set_nll
from .positive_history import attach_histories
from .train import root_validation

VERSION='all_current_population_evaluation_2'
FEATURE_FIELDS=('keys','x','graph','global_graph','origin','velocity','age','global_features','robotxy',
                'timestamp','minimum_count','positive_history','source_last_event_indices')
KEYS=('current_set_nll','current_count_nll','count_rps_finite_bank','count_rps_law_unbiased',
      'occupancy_brier_finite_bank','occupancy_brier_law_unbiased')


def count_rps(counts,truth):
    values=np.asarray(counts,np.float64);k=len(values)
    if k<2:raise ValueError('RPS requires at least two iid worlds')
    first=np.abs(values-int(truth)).mean();pair=np.abs(values[:,None]-values[None]).sum()
    return float(first-pair/(2*k*k)),float(first-pair/(2*k*(k-1)))


def occupancy_scores(indicators,truth):
    bank=np.asarray(indicators,np.float64);target=np.asarray(truth,np.float64);k=len(bank)
    if k<2 or not np.isin(bank,(0.,1.)).all():raise ValueError('binary occupancy and at least two iid worlds required')
    hits=bank.sum(0);mean=hits/k
    finite=(mean-target)**2
    unbiased=hits*(hits-1)/(k*(k-1))-2*target*mean+target
    return float(finite.mean()),float(unbiased.mean())


def occupied(positions,robotxy):
    p=np.asarray(positions,np.float64).reshape(-1,2)
    return occupancy(p[None],np.ones((1,len(p)),bool),robotxy)


def world_indicators(positions,present,robotxy):
    return np.asarray([occupancy(positions[i:i+1],present[i:i+1],robotxy) for i in range(len(positions))])


def poisson_tail_log(rate,minimum):
    return 0. if int(minimum)==0 else float(poisson.logsf(int(minimum)-1,float(rate)))


def residual_log_probability(count,visible,minimum,rate):
    u=int(count)-int(visible);lower=max(0,int(minimum)-int(visible))
    if u<lower:return -math.inf
    if rate==0:return 0. if u==lower else -math.inf
    return float(poisson.logpmf(u,rate)-poisson_tail_log(rate,lower))


def fit_residual_rate(groups):
    """Exact Root-weighted truncated-Poisson MLE; no development fitting."""
    totals=collections.Counter();target=0.;weight=0.;extra=0.
    for unseen,minimum,mass in groups:
        unseen=int(unseen);minimum=int(minimum);mass=float(mass)
        if unseen<minimum:raise ValueError('native count below legal historical minimum')
        totals[minimum]+=mass;target+=mass*unseen;weight+=mass;extra+=mass*(unseen-minimum)
    if not weight:raise ValueError('empty fitting corpus')
    if extra==0:return 0.
    def derivative(lograte):
        rate=math.exp(lograte);expectation=0.
        for lower,mass in totals.items():
            expected=rate if not lower else rate*math.exp(poisson_tail_log(rate,lower-1)-poisson_tail_log(rate,lower))
            expectation+=mass*expected
        return target-expectation
    lower=-25.;upper=math.log(max(1.,target/weight+max(totals)+1.))
    while derivative(lower)<=0:
        lower-=5.
        if lower<-100:raise RuntimeError('residual Poisson MLE numeric boundary unresolved')
    while derivative(upper)>=0:upper+=1.
    return float(math.exp(brentq(derivative,lower,upper,xtol=1e-11)))


def residual_occupancy(rate,minimum_unseen,point_mass):
    """Poisson conditional-count PGF gives a coherent residual grid prior."""
    pi=np.asarray(point_mass,np.float64);lower=int(minimum_unseen)
    if rate==0:
        return np.zeros_like(pi) if lower==0 else -np.expm1(lower*np.log1p(-pi))
    logpgf=-rate*pi
    if lower:
        logpgf=logpgf+poisson.logsf(lower-1,rate*(1-pi))-poisson_tail_log(rate,lower)
    answer=-np.expm1(logpgf)
    if not np.isfinite(answer).all() or np.any(answer<0) or np.any(answer>1):raise RuntimeError('invalid conditional Poisson occupancy probability')
    return answer


def negative_binomial_count_nll(draw_counts,truth,successes=32,max_worlds=100000,chunk=32):
    """Fixed-success log probability estimator, using an independent stream."""
    observed=0;hits=0;generated=0
    while hits<successes:
        if generated>=max_worlds:
            return None,dict(status='INCONCLUSIVE_COUNT_CAP',conditional_worlds_examined=observed,conditional_worlds_generated=generated,successes=hits,required_successes=successes)
        values=np.asarray(draw_counts(min(chunk,max_worlds-generated)),int);generated+=len(values)
        if not len(values):raise RuntimeError('count sampler returned no worlds')
        for value in values:
            observed+=1;hits+=int(value==truth)
            if hits==successes:break
    return float(digamma(observed)-digamma(successes)),dict(status='OK',conditional_worlds_examined=observed,conditional_worlds_generated=generated,successes=hits,required_successes=successes,
        method='one fixed-success NB count-event estimate among iid already-conditioned populations; no separately estimated logZ subtraction',
        number_of_independent_log_estimators=1,variance_upper_bound=float(polygamma(1,successes)))


def actual_current(record):
    h=record['positive_history'];start=int(h.positive_offsets[-2]);observed=h.positive_xyz[start:,[0,2]]
    full,unseen=current_measurement_overlay(record['population_current_positions'],observed)
    return observed,full,unseen


def current_log_normalizer(successes,stats):
    """Original proposal logZ from exact proposal-change and rejection factors.

    The inference-only Palm proposal reports its deterministic density-ratio
    multiplier. Ordinary rejection has multiplier one (log multiplier zero).
    Only the acceptance log probability needs a fixed-success MC estimate.
    """
    waiting=stats['proposals_examined_through_last_accept']
    if not isinstance(successes,(int,np.integer)) or successes<2:
        raise ValueError('current normalizer needs at least two fixed successes')
    if not isinstance(waiting,(int,np.integer)) or waiting<successes:
        raise ValueError('invalid fixed-success rejection waiting count')
    multiplier=float(stats.get('log_normalizer_multiplier',0.))
    if not math.isfinite(multiplier):raise ValueError('nonfinite exact normalizer multiplier')
    acceptance=float(digamma(successes)-digamma(waiting))
    return float(multiplier+acceptance)


def fit_priors(split,history_cache,role):
    all_records=load_records(split)
    records=[r for r in all_records if role=='all_refit' or not root_validation(r['root_id'])]
    del all_records
    attach_histories(records,history_cache);attach_targets(records,'train_campaign_20261023')
    counts=collections.Counter();scenes=set();groups=[];rate_sum=0.;mass=0.;grid=np.zeros(145);marks=np.zeros(145)
    previous={};roots=set()
    for record in records:
        rid=record['root_id'];h=record['positive_history']
        graph=previous[rid].append(h) if rid in previous else AssociationGraph.from_history(h);previous[rid]=graph
        observed,full,unseen=actual_current(record);minimum=max(int(record['minimum_count']),int(graph.minimum_current_count))
        if not bool(graph.feasible(full[None],np.ones((1,len(full)),bool))[0]):raise ValueError('fitting current target has zero legal-history support')
        weight=float(record['weight']);rate_sum+=weight*len(full);mass+=weight
        # Original unconditional grid reference retains native current positions.
        grid+=weight*occupied(record['population_current_positions'],record['robotxy'])
        relative=unseen-record['robotxy'];index=np.floor((relative+12)/2).astype(int)
        inside=(index>=0).all(-1)&(index<12).all(-1);bins=index[inside,0]*12+index[inside,1]
        marks+=weight*np.bincount(np.r_[bins,np.full(int((~inside).sum()),144)],minlength=145)
        groups.append((len(unseen),max(0,minimum-len(observed)),weight))
        roots.add(rid);scenes.add(record['scene_id']);counts['anchors']+=1
    # A zero-residual fitting sample defines an empty residual population.
    # The mark law is then immaterial; retain an explicit neutral full support.
    point_mass=marks/marks.sum() if marks.sum()>0 else np.full(145,1/145)
    result=dict(schema='current_population_train_only_priors_1',fitting_split=split,fitting_role=role,
        fitting_manifest_sha256=sha(OUT/'prepared'/split/'manifest.json'),history_manifest_sha256=sha(OUT/'positive_history'/history_cache/'manifest.json'),
        roots=len(roots),scenes=len(scenes),anchors=counts['anchors'],total_root_weight=mass,
        unconditional_poisson_rate=rate_sum/mass,unconditional_grid_occupancy=(grid/mass).tolist(),
        residual_conditional_poisson_rate=fit_residual_rate(groups),residual_point_mark_probability=point_mass.tolist(),
        residual_grid_law='known measured cells=1; other cells use exact PGF of Poisson(rate) conditional on U>=max(0,legal_minimum-visible), iid train-only residual point marks; outside bin included',
        root_weighting='one total weight per fitting Root across all original anchors; no development labels')
    return result,roots,scenes


def baseline_rows(record,graph,priors,worlds):
    observed,truth,_=actual_current(record);minimum=max(int(record['minimum_count']),int(graph.minimum_current_count));count=len(truth)
    target=occupied(truth,record['robotxy']);known=occupied(observed,record['robotxy'])>0
    output={}
    for name,strong in (('constant_unconditional',False),('constant_detection_overlay_residual',True)):
        rng=np.random.default_rng(seed_for(record['root_id'],811 if strong else 809,record['anchor_index']))
        if strong:
            rate=float(priors['residual_conditional_poisson_rate']);lower=max(0,minimum-len(observed))
            unseen=np.full(worlds,lower,int) if rate==0 else sample_conditioned(np.empty(0),np.empty(0),rate,lower,worlds,rng)[1]
            counts=unseen+len(observed)
            predicted=residual_occupancy(rate,lower,priors['residual_point_mark_probability']);predicted[known]=1.
            log_probability=residual_log_probability(count,len(observed),minimum,rate)
        else:
            rate=float(priors['unconditional_poisson_rate']);counts=rng.poisson(rate,worlds)
            predicted=np.asarray(priors['unconditional_grid_occupancy']);log_probability=float(poisson.logpmf(count,rate))
        finite,rps=count_rps(counts,count);brier=float(np.mean((predicted-target)**2))
        output[name]=dict(current_count_nll=-log_probability if np.isfinite(log_probability) else None,
            current_count_nll_extended='+infinity' if not np.isfinite(log_probability) else None,
            count_rps_finite_bank=finite,count_rps_law_unbiased=rps,occupancy_brier_finite_bank=brier,occupancy_brier_law_unbiased=brier,
            status='OK' if np.isfinite(log_probability) else 'ZERO_BASELINE_COUNT_SUPPORT')
    return output


def model_row(model,record,graph,worlds,max_count_worlds,max_raw_proposals):
    features={k:record[k] for k in FEATURE_FIELDS};history=record['positive_history']
    observed,truth,_=actual_current(record);minimum=max(int(record['minimum_count']),int(graph.minimum_current_count))
    if not bool(graph.feasible(truth[None],np.ones((1,len(truth)),bool))[0]):
        return dict(status='FAIL_ZERO_NATIVE_TARGET_SUPPORT',current_set_nll=None,current_set_nll_extended='+infinity')
    parameters={key:np.asarray(value,np.float64) for key,value in model.parameters(features).items()}
    parameters=augment_numpy(parameters,features,graph,model.history_fraction)
    unique=model.network.compulsory_unique_episode
    rng=np.random.default_rng(seed_for(record['root_id'],801,record['anchor_index']))
    try:
        bank,_,stats=draw_conditioned_current(features,parameters,history,graph,worlds,rng,unique,max_raw_proposals)
    except PopulationSamplingFailure as error:return dict(status='INCONCLUSIVE_DENSITY_SAMPLING_CAP',error=str(error))
    positions,present=pad_current(bank,len(features['keys']));counts=present.sum(-1)
    batch=make_batch([record],device='cpu');batch['age']=torch.as_tensor(np.asarray(record['age'],np.float64)[None]) if len(record['keys']) else torch.zeros((1,1),dtype=torch.float64)
    last,_=source_coordinates(features,history);padded=np.zeros((batch['mask'].shape[1],2));padded[:len(last)]=last
    batch.update(source_last_positions=torch.as_tensor(padded)[None],minimum_count=torch.as_tensor([minimum]),origin=batch['origin'].double())
    tensor_parameters={key:torch.as_tensor(value)[None] for key,value in parameters.items()}
    with torch.no_grad():q_nll=float(set_nll(tensor_parameters,batch,True,unique,3.)[0])
    if not math.isfinite(q_nll):return dict(status='FAIL_ZERO_OR_NONFINITE_CURRENT_SET_DENSITY',current_set_nll=None,current_set_nll_extended='+infinity')
    correction=current_log_normalizer(worlds,stats)
    indicators=world_indicators(positions,present,record['robotxy']);target=occupied(truth,record['robotxy'])
    finite_brier,u_brier=occupancy_scores(indicators,target);finite_rps,u_rps=count_rps(counts,len(truth))
    count_rng=np.random.default_rng(seed_for(record['root_id'],803,record['anchor_index']))
    def draw_counts(number):
        rows,_,_=draw_conditioned_current(features,parameters,history,graph,number,count_rng,unique,max_raw_proposals)
        return np.asarray([row.count for row in rows],int)
    try:count_nll,count_stats=negative_binomial_count_nll(draw_counts,len(truth),worlds,max_count_worlds)
    except PopulationSamplingFailure as error:count_nll=None;count_stats=dict(status='INCONCLUSIVE_COUNT_SAMPLING_CAP',error=str(error))
    return dict(status=count_stats['status'],current_set_nll=q_nll+correction,current_set_nll_unnormalized_q=q_nll,
        current_normalizer_log_estimate=correction,
        current_normalizer_log_multiplier=float(stats.get('log_normalizer_multiplier',0.)),current_count_nll=count_nll,
        count_rps_finite_bank=finite_rps,count_rps_law_unbiased=u_rps,occupancy_brier_finite_bank=finite_brier,occupancy_brier_law_unbiased=u_brier,
        current_count_mean=float(counts.mean()),true_current_count=len(truth),visible_current_count=len(observed),legal_minimum=minimum,
        density_sampling=dict(stats),count_sampling=count_stats)


def nll_mc_uncertainty(rows,successes=32):
    """Conditional MC bound, separate from between-scene sampling intervals."""
    by_scene=collections.defaultdict(collections.Counter)
    for row in rows:by_scene[row['scene_id']][row['root_id']]+=1
    scenes=len(by_scene)
    squared_weight_sum=sum(1./(scenes*scenes*len(roots)**2*anchors)
        for roots in by_scene.values() for anchors in roots.values()) if scenes else 0.
    variance_bound=float(polygamma(1,successes))*squared_weight_sum
    return dict(successes=successes,number_of_independent_log_estimators=1,
        applicable_methods='current set NLL: exact deterministic proposal multiplier plus one rejection log estimate; direct conditional count-event NLL: one NB estimate. Does not describe the separate q_N+log a_N-logZ two-stream API estimator.',
        per_anchor_variance_upper_bound=float(polygamma(1,successes)),
        aggregate_variance_upper_bound=variance_bound,aggregate_standard_error_upper_bound=math.sqrt(variance_bound),
        squared_aggregate_anchor_weights=squared_weight_sum,
        weighting='equal scenes; equal Roots within each scene; equal original cohort anchors within each Root',
        meaning='conditional Monte Carlo variance bound for the fixed-success log probability estimator only; separate from scene-bootstrap uncertainty')


def safe_summary(rows,key,monte_carlo=False):
    scored=[r for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    missing=len(rows)-len(scored);answer=summarize(scored,key)
    zero_support=sum(r.get(key+'_extended')=='+infinity' for r in rows)
    answer.update(total_anchors=len(rows),scored_anchors=len(scored),total_roots=len({r['root_id'] for r in rows}),missing_anchors=missing,
        status='FAIL_ZERO_SUPPORT' if zero_support else 'INCONCLUSIVE' if missing or not answer['sufficient'] else 'MEASURED',diagnostic_only=bool(missing),
        zero_support_anchors=zero_support)
    if monte_carlo and key in ('current_set_nll','current_count_nll') and rows:
        answer['monte_carlo_uncertainty_bound']=nll_mc_uncertainty(rows)
        answer['monte_carlo_uncertainty_bound']['applies_to_reported_complete_cohort_mean']=not bool(missing)
        if missing:answer['monte_carlo_uncertainty_bound']['warning']='Intended full-cohort weights only; observed scored-subset mean remains diagnostic and INCONCLUSIVE.'
    return answer


def run(name,checkpoint,split='dev_native_v2',history_cache='dev',event_cache='dev_native_v2',worlds=32,
        fitting_split=None,fitting_history_cache=None,max_count_worlds=100000,max_raw_proposals=1000000):
    if worlds!=32:raise ValueError('this predeclared current evaluation uses exactly32 iid accepted worlds/successes')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    checkpoint=Path(checkpoint);saved=torch.load(checkpoint,map_location='cpu')
    if 'conditioned_config' not in saved:raise ValueError('a trained conditioned checkpoint is required')
    cfg=saved['config'];fitting_split=fitting_split or cfg['split'];fitting_history_cache=fitting_history_cache or cfg['sidecar']
    if sha(OUT/'prepared'/fitting_split/'manifest.json')!=cfg['data_manifest_sha256']:raise ValueError('declared fitting corpus differs from checkpoint')
    role='all_refit' if cfg['full_fit'] else 'nonheld_fit';dest=OUT/'evaluation'/name;dest.mkdir(parents=True,exist_ok=False)
    protocol=dict(schema=VERSION,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),split=split,history_cache=history_cache,event_cache=event_cache,
        worlds=worlds,count_successes=worlds,max_count_worlds=max_count_worlds,max_raw_proposals=max_raw_proposals,
        density_estimator='q set NLL + exact sampler log_normalizer_multiplier + digamma(32)-digamma(proposals examined through32nd final acceptance); absent multiplier means ordinary rejection with log multiplier zero',
        proposal_multiplier='For a certified mandatory terminal residual rectangle, exact Palm factor log(lambda*pR*C_(d-1)/C_d); otherwise zero. Accepted population law and q density stay unchanged.',
        count_nll_estimator='independent fixed32 count-event successes among iid history-conditioned worlds; no clipping',
        monte_carlo_uncertainty='Both methods here use ONE fixed-success log estimator per score, variance <=trigamma(32); direct count-event sampling does not use the separate two-NB q_N+log a_N-logZ API. Complete aggregate SE uses actual equal-scene/equal-Root/equal-original-anchor squared weights, separate from scene bootstrap.',
        reference_scores='finite32bank Brier and integer RPS retain the frozen reference convention; unbiased underlying-law U-Brier/U-RPS labeled separately',
        occupancy_grid='same145bin robot-relative XZ convention:12x12 two-metre cells in[-12,12), plus outside bin',
        target='all native current people; exact measured Dirac atoms overlaid injectively at2e-4 native alignment tolerance, no atom/Root exclusions',
        status_policy='caps and missing scores make aggregates INCONCLUSIVE; zero target/density support is explicit FAIL, never discarded',
        fitting_split=fitting_split,fitting_role=role,source_sha256=sha(__file__))
    dump(dest/'protocol.json',protocol)
    priors,fit_roots,fit_scenes=fit_priors(fitting_split,fitting_history_cache,role);dump(dest/'constant_priors.json',priors)
    records=load_records(split)
    joins=dict(history=attach_histories(records,history_cache),events=attach_metadata(records,event_cache),future=attach_targets(records,split))
    if fit_roots&{r['root_id'] for r in records} or fit_scenes&{r['scene_id'] for r in records}:raise ValueError('evaluation overlaps fitting Root/scene boundary')
    dump(dest/'data_join_audit.json',joins);model=HistoryConditionedWorldModel(checkpoint,max_proposals=max_raw_proposals)
    strata={};ledger=DOC/'reviews/native_occlusion_development/anchors.jsonl.gz'
    if ledger.exists():
        with gzip.open(ledger,'rt') as stream:
            for line in stream:
                row=json.loads(line);strata[row['root_id'],row['anchor_index']]=row
    rows={key:[] for key in ('model','constant_unconditional','constant_detection_overlay_residual')};previous={};start=time.perf_counter()
    with (dest/'current_rows.jsonl').open('w') as stream:
        for i,record in enumerate(records):
            rid=record['root_id'];h=record['positive_history'];graph=previous[rid].append(h) if rid in previous else AssociationGraph.from_history(h);previous[rid]=graph
            raw_current=h.positive_offsets[-1]-h.positive_offsets[-2];s=strata.get((rid,record['anchor_index']),{})
            base=dict(root_id=rid,scene_id=record['scene_id'],anchor=int(record['anchor_index']),timestamp=float(record['timestamp']),
                valid_empty=bool(h.observation_valid[-1] and h.human_sensor_available[-1] and raw_current==0),
                native_occlusion=bool(s.get('has_native_occlusion',False)),native_visibility_cohort_supported=bool(s.get('cohort_supported',False)),
                future_supported=bool(np.any(record['population_future_valid'])))
            try:result=model_row(model,record,graph,worlds,max_count_worlds,max_raw_proposals)
            except Exception as error:result=dict(status='INCONCLUSIVE_MODEL_NUMERICAL_ERROR',error=repr(error))
            baselines=baseline_rows(record,graph,priors,worlds)
            for key,value in dict(model=result,**baselines).items():
                row=dict(base,**value);rows[key].append(row);stream.write(json.dumps(dict(variant=key,**row),allow_nan=False)+'\n')
            if (i+1)%200==0:print(json.dumps(dict(anchors=i+1,total=len(records),seconds=time.perf_counter()-start,model_failures=sum(r['status']!='OK' for r in rows['model']))),flush=True)
    summaries={};comparisons={}
    for cohort,predicate in (('all_current',lambda r:True),('valid_empty',lambda r:r['valid_empty']),('native_occlusion',lambda r:r['native_occlusion'])):
        selected={key:[r for r in value if predicate(r)] for key,value in rows.items()}
        summaries[cohort]={key:{metric:safe_summary(value,metric,monte_carlo=key=='model') for metric in KEYS if any(metric in r for r in value)} for key,value in selected.items()}
        comparisons[cohort]={}
        for baseline in ('constant_unconditional','constant_detection_overlay_residual'):
            comparisons[cohort][baseline]={}
            for metric in KEYS:
                if metric=='current_set_nll':continue
                if any(r.get(metric) is None or not math.isfinite(float(r[metric])) for r in selected['model']+selected[baseline]):
                    comparisons[cohort][baseline][metric]=dict(status='INCONCLUSIVE',reason='one or more original cohort anchors unscored; no dropped-anchor paired pass')
                else:
                    comparison=dict(status='MEASURED',**paired(selected['model'],selected[baseline],metric))
                    if metric=='current_count_nll':comparison['candidate_monte_carlo_uncertainty_bound']=nll_mc_uncertainty(selected['model'])
                    comparisons[cohort][baseline][metric]=comparison
    failures=[dict(root_id=r['root_id'],scene_id=r['scene_id'],anchor=r['anchor'],status=r['status']) for r in rows['model'] if r['status']!='OK']
    result=dict(schema=VERSION,protocol=protocol,joins=joins,roots=len({r['root_id'] for r in records}),scenes=len({r['scene_id'] for r in records}),anchors=len(records),
        all_current_anchors_retained=True,future_censored_anchors_retained=sum(not r['future_supported'] for r in rows['model']),
        summaries=summaries,paired_baseline_comparisons=comparisons,model_failures=failures,
        decision='FAIL' if any(r['status'].startswith('FAIL') for r in failures) else 'INCONCLUSIVE' if failures else 'CURRENT_POPULATION_METRICS_COMPLETE_NOT_STAGE1_ACCEPTANCE',seconds=time.perf_counter()-start)
    dump(dest/'summary.json',result);return result


def self_test():
    finite,unbiased=count_rps([0,1,2],1)
    assert abs(finite-2/9)<1e-12 and abs(unbiased)<1e-12
    finite,unbiased=occupancy_scores(np.array([[1.],[1.],[0.]]),np.array([1.]))
    assert abs(finite-1/9)<1e-12 and abs(unbiased)<1e-12
    assert abs(fit_residual_rate([(0,0,1),(1,0,1),(2,0,1),(3,0,1)])-1.5)<1e-8
    assert np.allclose(residual_occupancy(2.,0,[.2,.8]),1-np.exp(-2*np.array([.2,.8])))
    nll,stats=negative_binomial_count_nll(lambda n:np.full(n,2),2,32,32)
    assert nll==0 and stats['conditional_worlds_examined']==32
    nll,stats=negative_binomial_count_nll(lambda n:np.zeros(n,int),2,32,64)
    assert nll is None and stats['status']=='INCONCLUSIVE_COUNT_CAP'
    plain=dict(proposals_examined_through_last_accept=80)
    palm=dict(plain,log_normalizer_multiplier=math.log(2.5))
    assert current_log_normalizer(32,plain)==float(digamma(32)-digamma(80))
    assert abs(current_log_normalizer(32,palm)-current_log_normalizer(32,plain)-math.log(2.5))<1e-15
    weighted=[dict(scene_id='a',root_id='a1'),dict(scene_id='a',root_id='a1'),
              dict(scene_id='a',root_id='a2'),dict(scene_id='b',root_id='b1')]
    bound=nll_mc_uncertainty(weighted)
    assert abs(bound['squared_aggregate_anchor_weights']-(2*.125**2+.25**2+.5**2))<1e-15
    assert len(occupied(np.array([[0.,0.],[100.,100.]]),np.zeros(2)))==145
    return dict(status='PASS',scope='small scoring/MLE/stopping-rule regressions only; no model/data acceptance claim')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--name');parser.add_argument('--checkpoint')
    parser.add_argument('--split',default='dev_native_v2');parser.add_argument('--history-cache',default='dev');parser.add_argument('--event-cache',default='dev_native_v2')
    parser.add_argument('--fitting-split');parser.add_argument('--fitting-history-cache');parser.add_argument('--worlds',type=int,default=32)
    parser.add_argument('--max-count-worlds',type=int,default=100000);parser.add_argument('--max-raw-proposals',type=int,default=1000000);args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()))
    else:
        if not args.name or not args.checkpoint:parser.error('--name and --checkpoint required')
        values=vars(args);values.pop('self_test');run(**values)
