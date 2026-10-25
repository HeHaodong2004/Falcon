"""One frozen, inference-only validation on preselected protected scene Roots.

This never changes ordinary fitting/development loaders or fits any parameter.
Selection is metadata-only and precedes payload access. Only the committed
argmax and development-selected0.20 policies are evaluated, with frozen priors.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time
import numpy as np
from .common import ROOT,OUT,CHECKPOINT,CHECKPOINT_SHA,EDGES,digest,save,make_model,extended_rates
from .statistics import rank_metrics

DEST=OUT/'final_confirmation'
SELECTED=OUT/'final_confirmation_metadata/selected_manifest.json'
MODEL=None;CONFIG=None;IDENTITY=None
EXPECTED_SELECTION='f29a343e745005710a3a9e3e5a8fd4f7524b88e7fb6cce4065bac2f247691b46'

def source_identity():
    files=[Path(__file__),ROOT/'scripts/stage2_diagnostic/common.py',ROOT/'scripts/stage2_diagnostic/decision_overhead.py',ROOT/'scripts/stage2_diagnostic/soft_visibility_paired.py',ROOT/'eventwm_graph/stage2_segment/data.py',ROOT/'eventwm_graph/stage2_segment/evaluate.py']
    for folder in ('stage2','stage2_quality','stage2_segment'):
        files.extend((ROOT/'eventwm_graph'/folder).glob('*.py'))
    files.append(ROOT/'eventwm_graph/partial/map.py')
    files.extend(ROOT/'scripts/stage2_diagnostic'/name for name in ('statistics.py','decision_relevance.py','overhead_admission.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))}

def freeze():
    import subprocess
    if (DEST/'protocol.json').exists():raise FileExistsError('final confirmation already frozen')
    selected=json.loads(SELECTED.read_text());rows=selected['rows']
    scenes=defaultdict(list)
    for r in rows:scenes[r['scene_id']].append(r)
    if digest(SELECTED)!=EXPECTED_SELECTION or len(rows)!=222 or len(scenes)!=111 or any(len(v)!=2 for v in scenes.values()):raise ValueError('confirmation selection/count changed')
    if any(r['split']!='protected_final_confirmation' or r['source_split']!='train' or r['scene_hash_percent']<85 or not r['sealed'] or r['independent_futures']!=1 for r in rows):raise ValueError('invalid protected final-only tag')
    if selected['payloads_opened'] or any(r['payload_bytes_read'] for r in rows):raise ValueError('metadata selection was not pre-payload')
    a=json.loads((OUT/'probability/summary.json').read_text());counts=json.loads((OUT/'decision_relevance/counts.json').read_text());soft=json.loads((OUT/'visibility/soft_paired_v1/summary.json').read_text())
    assert set(a['diagnostic_scene_validation']['selected_rules'].values())=={'threshold_0.20'}
    value=dict(schema='stage2_final_protected_confirmation_v1',stage='FROZEN_BEFORE_SELECTED_LEGAL_PAYLOADS',
        user_authorization='Current mission A3: if untouched evaluation exists, evaluate frozen selected thresholds; no test-label configuration selection',
        development_release_commit='82eb2dab',current_source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        selected_manifest_sha256=digest(SELECTED),metadata_receipt_sha256=digest(SELECTED.parent/'receipt.json'),
        checkpoint_sha256=CHECKPOINT_SHA,roots=len(rows),scenes=len({r['scene_id'] for r in rows}),queries=sum(r['queries'] for r in rows),
        subset='Exactly two metadata-hash-selected sealed Roots per explicit OLDprotected_research_holdout TRAINscene; disjoint fitting/development/opened; all original mixed windows on those Roots',
        scope='Sampled final confirmation, NOT exhaustive protected pool. All model/configuration/threshold/calibration choices fixed before this protocol; no refitting or retuning after labels.',
        policies=['released_argmax','frozen_revelation_0.20'],threshold=.20,
        probability_contract='F32 classsum thenF64 comparison; FREE conditionalclass ties; unchanged probability arrays',
        weighting='CMrawcounts windowmeanwithinRoot thenRootmeanwithinScene thenScenemean; proper/hist perwindow-cellnormalized thenRoot/Scene; countmoments window/Root/Scene',
        metric_definitions='Unchanged initiallyUNKNOWN11m candidate domain plus every missedGTpositive as certainUNKNOWN; true legalreplay only; knowncorrections outside scope; actual fullfuturegraphs constructed and scored forbothrules',
        uncertainty='2000 paired whole-scene bootstrap draws seed2525; no test threshold search',
        prior=a['diagnostic_scene_validation']['metrics']['fitted_constant_probability_baseline']['probabilities'],
        duration_baselines=counts['fitted_baselines'],geometry_rates=soft['rates'],
        frozen_development_sources={str(p.relative_to(OUT)):digest(p) for p in [OUT/'probability/summary.json',OUT/'decision_relevance/counts.json',OUT/'visibility/soft_paired_v1/summary.json']},
        protected_boundary='Unchanged; selected Rootpayload access is final evaluation-only and logged separately; original metadata/data never modified',
        no_neural_training=True,no_privileged_geometry=True,no_simulator=True,no_background_process_changes=True,source=source_identity())
    save(DEST/'protocol.json',value);print(json.dumps({k:value[k] for k in ['roots','scenes','queries','selected_manifest_sha256']}),flush=True)

def initialize(device,config,identity):
    global MODEL,CONFIG,IDENTITY
    MODEL=make_model(device);CONFIG=config;IDENTITY=identity

def verify_config(config):
    if config['source']!=source_identity() or digest(SELECTED)!=EXPECTED_SELECTION or digest(SELECTED)!=config['selected_manifest_sha256']:raise ValueError('frozen confirmation source/selection changed')
    if digest(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('frozen checkpoint changed')
    for name,expected in config['frozen_development_sources'].items():
        if digest(OUT/name)!=expected:raise ValueError('development choice changed before/after finaltest')

def counts_moments(pred,truth):
    return np.stack([abs(pred-truth),pred-truth,pred,truth,pred*pred,truth*truth,pred*truth])

def score_fixed(ex,delta,y,missing):
    """Frozen score and actual graph checks, with NO threshold/PR sweep."""
    p=delta['probabilities'];pred=delta['states'];n=len(y);denom=n+len(missing)
    cm=np.bincount(3*y.astype(np.int64)+pred,minlength=9).reshape(3,3)
    for label,_ in missing:cm[label,0]+=1
    truep=p[np.arange(n),y];ce=-np.log(np.clip(truep,1e-9,1));br=np.square(p).sum(1)-2*truep+1
    scores=[float((ce.sum()-len(missing)*np.log(1e-9))/max(denom,1)),float((br.sum()+2*len(missing))/max(denom,1))]
    original={tuple(c['cell']):c for c in ex['graph']['cells']};future={tuple(c['cell']):c for c in delta['future_partial_graph']['cells']}
    if any(future.get(k)!=c for k,c in original.items()):raise ValueError('known input map cell changed')
    if len(future)!=len(original)+len(delta['static_updates']):raise ValueError('future graph cardinality mismatch')
    terminal=np.zeros((3,3),int);height=[0.,0]
    for key,truth in zip(delta['candidate_cells'],y):
        k=tuple(key);row=future.get(k);label=row['state'] if row else 0;terminal[truth,label]+=1
        if truth and label==truth:height[0]+=abs(row['height']-ex['target'][k][1]);height[1]+=1
    for label,_ in missing:terminal[label,0]+=1
    np.testing.assert_array_equal(terminal,cm)
    return dict(confusion=cm.tolist(),scores=scores,out_of_domain_positive=len(missing),future_graph_verified=True,height_error=height)

def root_worker(row):
    from eventwm_graph.stage2_segment.data import replay,read
    from .decision_overhead import construct_future,_case_digest
    from .soft_visibility_paired import opportunity_score,score_bins,binary_scores
    path=DEST/'roots'/(row['root_id']+'.json')
    if path.exists():raise FileExistsError('protected confirmation is single-pass; use a separately recorded recovery plan rather than silently rerunning')
    sensor=read(row,'legal_root.json.gz')['sensor'];records=[];h=np.zeros((3,len(EDGES)-1));geometry_h=np.zeros((33,2));geometry_ood_mass=0.;count_sums=np.zeros((3,7,3));tick=time.perf_counter();audit=None
    for ex in replay(row,'mixed'):
        if 'audit' in ex:audit=ex['audit'];continue
        before=_case_digest(ex['graph'],ex['trajectory'])
        delta=MODEL.predict(partial_graph=ex['graph'],trajectory=ex['trajectory'])
        prepared=MODEL.prepare_graph(ex['graph'])
        converted,_=construct_future(ex['graph'],delta,CONFIG['threshold'],getattr(prepared,'future_base',None))
        keys=delta['candidate_cells'];p=delta['probabilities'];target=ex['target']
        y=np.array([target.get(tuple(k),(0,0.))[0] for k in keys],np.uint8)
        present=set(map(tuple,keys));missing=[v for k,v in target.items() if k not in present];denom=len(y)+len(missing)
        original=score_fixed(ex,delta,y,missing);changed=score_fixed(ex,converted,y,missing)
        if before!=_case_digest(ex['graph'],ex['trajectory']):raise ValueError('confirmation inference mutated input')
        if original['out_of_domain_positive']!=len(missing):raise ValueError('domain score mismatch')
        r=(p[:,1]+p[:,2]).astype(np.float64);observed=y>0
        b=np.minimum(np.searchsorted(EDGES,r,side='right')-1,len(EDGES)-2);b=np.maximum(b,0)
        hist=np.bincount(y.astype(np.int64)*(len(EDGES)-1)+b,minlength=3*(len(EDGES)-1)).reshape(3,-1).astype(np.float64)
        for label,_ in missing:hist[label,0]+=1
        h+=hist/max(denom,1)
        binary=binary_scores(r,observed)*len(y) if len(y) else np.zeros(2)
        binary+=np.array([-np.log(1e-9),1.])*len(missing);binary/=max(denom,1)
        prior=np.array(CONFIG['prior']);tp=prior[y]
        prior_ce=-np.log(np.clip(tp,1e-9,1)).sum()-len(missing)*np.log(1e-9)
        prior_br=(1+np.square(prior).sum()-2*tp).sum()+2*len(missing)
        opportunity=opportunity_score(ex['graph'],ex['trajectory'],keys,sensor);category=score_bins(opportunity)
        pg=np.array(CONFIG['geometry_rates'])[category]
        geo=(binary_scores(pg,observed)*len(y) if len(y) else np.zeros(2))+np.array([-np.log(1e-9),1.])*len(missing);geo/=max(denom,1)
        gh=np.bincount(2*category.astype(np.int64)+observed,minlength=66).reshape(33,2).astype(float)
        # Out-of-domain targets have deterministicUNKNOWN for EVERY predictor.
        geometry_h+=gh/max(denom,1)
        geometry_ood_mass+=len(missing)/max(denom,1)
        true=np.array([len(target),sum(v[0]==1 for v in target.values()),sum(v[0]==2 for v in target.values())],float)
        expected=np.array([r.sum(),p[:,1].sum(dtype=np.float64),p[:,2].sum(dtype=np.float64)])
        duration=float(ex['trajectory'][-1,0]-ex['trajectory'][0,0]);band='duration:'+('<=0.5' if duration<=.5000001 else '0.5-1' if duration<=1.0000001 else '1-2' if duration<=2.0000001 else '>2')
        median=np.array(CONFIG['duration_baselines'][band]['median_counts'])
        count_sums[0]+=counts_moments(expected,true);count_sums[1]+=counts_moments(median,true)
        # R-only geometry count comparison; F/B occupancy is NOT assigned.
        gr=np.array([pg.sum(),0.,0.]);count_sums[2]+=counts_moments(gr,true)
        records.append(dict(start_node=ex['start_node'],end_node=ex['end_node'],duration=duration,
            confusions=[original['confusion'],changed['confusion']],scores=[*original['scores'],*binary],
            baseline_scores=[prior_ce/max(denom,1),prior_br/max(denom,1),*geo],
            true_counts=true.tolist(),expected_counts=expected.tolist(),candidates=len(y),out_of_domain=len(missing),
            known_terminal_differences=ex['known_terminal_differences'],
            full_future_graphs_verified=[original['future_graph_verified'],changed['future_graph_verified']],
            height_errors=[original['height_error'],changed['height_error']]))
    if audit is None or not records:raise ValueError('incomplete protected legal replay')
    n=len(records);path.parent.mkdir(parents=True,exist_ok=True)
    histfile=path.with_suffix('.npz');np.savez_compressed(histfile,revelation_histogram=h/n,geometry_histogram=geometry_h/n,geometry_certain_unknown_ood_positive_mass=geometry_ood_mass/n)
    save(path,dict(root_id=row['root_id'],scene_id=row['scene_id'],identity=IDENTITY,records=records,audit=audit,
        count_moments=(count_sums/n).tolist(),hist_sha256=digest(histfile),member_sha256=row['files'],seconds=time.perf_counter()-tick))
    return dict(root_id=row['root_id'],scene_id=row['scene_id'],segments=n,seconds=time.perf_counter()-tick)

def summarize():
    from .decision_relevance import rates as count_rates
    config=json.loads((DEST/'protocol.json').read_text());verify_config(config)
    complete=json.loads((DEST/'complete.json').read_text());rows=[json.loads(p.read_text()) for p in sorted((DEST/'roots').glob('*.json'))]
    if len(rows)!=config['roots']:raise ValueError('incomplete final confirmation')
    selected=json.loads(SELECTED.read_text())['rows'];expected={(r['root_id'],r['scene_id']) for r in selected}
    if {(r['root_id'],r['scene_id']) for r in rows}!=expected or any(r['identity']!=complete['identity'] for r in rows):raise ValueError('confirmation results do not match frozen identities')
    by=defaultdict(list)
    for r in rows:by[r['scene_id']].append(r)
    cms=[];scores=[];baselines=[];moments=[];hist=[];zeros=np.zeros((2,2),int);missing=0;known=0
    for scene,roots in sorted(by.items()):
        cms.append(np.mean([np.mean([v['confusions'] for v in r['records']],axis=0) for r in roots],axis=0))
        scores.append(np.mean([np.mean([v['scores'] for v in r['records']],axis=0) for r in roots],axis=0))
        baselines.append(np.mean([np.mean([v['baseline_scores'] for v in r['records']],axis=0) for r in roots],axis=0))
        moments.append(np.mean([r['count_moments'] for r in roots],axis=0));hs=[]
        for r in roots:
            file=DEST/'roots'/(r['root_id']+'.npz')
            if digest(file)!=r['hist_sha256']:raise ValueError('confirmation histogram changed')
            with np.load(file) as z:hs.append(z['revelation_histogram'])
            for v in r['records']:
                missing+=v['out_of_domain'];known+=v['known_terminal_differences']
                if not v['true_counts'][0]:
                    for j,c in enumerate(v['confusions']):
                        predicted=np.array(c)[:,1:].sum();zeros[j,0]+=predicted==0;zeros[j,1]+=predicted
        hist.append(np.mean(hs,axis=0))
    cms=np.array(cms);scores=np.array(scores);baselines=np.array(baselines);moments=np.array(moments);h=np.mean(hist,axis=0)
    avg=cms.mean(0);s=scores.mean(0);b=baselines.mean(0);m=moments.mean(0)
    result=dict(roots=len(rows),scenes=len(by),segments=sum(len(r['records']) for r in rows),
        policies={name:extended_rates(cm) for name,cm in zip(config['policies'],avg)},
        probability=dict(multiclass_logloss=float(s[0]),multiclass_brier=float(s[1]),binary_logloss=float(s[2]),binary_brier=float(s[3]),revelation_rank=rank_metrics(h[0],h[1:].sum(0))),
        frozen_baselines=dict(constant_prior_multiclass_logloss=float(b[0]),constant_prior_multiclass_brier=float(b[1]),geometry_binary_logloss=float(b[2]),geometry_binary_brier=float(b[3])),
        counts=dict(model=count_rates(m[0]),frozen_dev_duration_median=count_rates(m[1]),
            geometry_R_only=dict(revelation=count_rates(m[2])['revelation'])),
        out_of_domain_positives=missing,overlapping_window_known_terminal_differences=known,
        source_audit={k:sum(r['audit'].get(k,0) for r in rows) for k in set().union(*(r['audit'] for r in rows))},
        all_actual_future_graphs_verified=all(all(v['full_future_graphs_verified']) for r in rows for v in r['records']),
        zero_new={name:dict(exact_empty=int(z[0]),false_cells=int(z[1])) for name,z in zip(config['policies'],zeros)},
        protocol_sha256=digest(DEST/'protocol.json'),checkpoint_sha256=CHECKPOINT_SHA,
        scope=config['scope'],protected_boundary_preserved=True,no_test_refit_or_rule_selection=True)
    rng=np.random.default_rng(2525);draws=defaultdict(list)
    for _ in range(2000):
        index=rng.integers(0,len(by),len(by));c=cms[index].mean(0);rr=[extended_rates(x) for x in c]
        for key in ['label_aware_delta_iou','revelation_f1','terminal_label_agreement_on_true_revelations']:
            draws[key+'_difference'].append(rr[1][key]-rr[0][key])
        draws['model_minus_geometry_binary_logloss'].append(scores[index,2].mean()-baselines[index,2].mean())
        cmom=moments[index].mean(0);a=count_rates(cmom[0]);base=count_rates(cmom[1])
        draws['model_count_mae'].append(a['revelation']['mae']);draws['model_minus_duration_median_mae'].append(a['revelation']['mae']-base['revelation']['mae'])
    result['scene_ci95']={k:np.quantile(v,[.025,.975]).tolist() for k,v in draws.items()}
    save(DEST/'summary.json',result);print(json.dumps(result),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--summarize',action='store_true');parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if args.freeze:freeze();return
    if args.summarize:summarize();return
    if not args.run:raise ValueError('choose explicit phase')
    config=json.loads((DEST/'protocol.json').read_text())
    verify_config(config)
    if (DEST/'complete.json').exists() or (DEST/'exposure_receipt.json').exists():raise FileExistsError('final confirmation already completed or exposed; explicit separately frozen recovery plan required')
    rows=json.loads(SELECTED.read_text())['rows'];identity=dict(protocol_sha256=digest(DEST/'protocol.json'),source=config['source']);start=time.perf_counter();done=[]
    save(DEST/'exposure_receipt.json',dict(stage='FINAL_EVALUATION_PAYLOAD_ACCESS_START',roots=[r['root_id'] for r in rows],checkpoint_sha256=CHECKPOINT_SHA,protocol_sha256=identity['protocol_sha256'],no_fitting_or_retuning=True))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=('cuda:1',config,identity)) as pool:
        jobs={pool.submit(root_worker,row):row['root_id'] for row in rows}
        for future in as_completed(jobs):
            try:done.append(future.result())
            except BaseException as error:
                import traceback
                save(DEST/'failure_receipt.json',dict(root_id=jobs[future],identity=identity,error=repr(error),traceback=traceback.format_exc(),no_replacement_or_retuning=True));raise
            if len(done)%20==0 or len(done)==len(rows):print(json.dumps(dict(roots=len(done),total=len(rows),segments=sum(r['segments'] for r in done),seconds=time.perf_counter()-start)),flush=True)
    verify_config(config)
    if len(done)!=config['roots'] or len({r['scene_id'] for r in done})!=config['scenes']:raise ValueError('confirmation cohort incomplete')
    save(DEST/'complete.json',dict(roots=len(done),scenes=len({r['scene_id'] for r in done}),segments=sum(r['segments'] for r in done),seconds=time.perf_counter()-start,identity=identity,no_training=True))
    summarize()

if __name__=='__main__':main()
