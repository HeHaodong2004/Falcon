"""Recorded-pair execution and aggregation for the frozen trajectory audit."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
from .common import ROOT,OUT,CHECKPOINT_SHA,digest,save
from . import trajectory_metrics as metrics

DEST=OUT/'trajectory'
MODEL=None
RUN=None
IDENTITY=None


def graph_hash(graph):return hashlib.sha256(pickle.dumps(graph,protocol=5)).hexdigest()


def reconstruct(row,pair):
    from eventwm_graph.stage2_segment.data import load
    from eventwm_graph.partial.map import PartialGraph
    from eventwm_graph.stage2_quality.update import FastPartialGraph
    legal,queries,outcomes,nodes,children=load(row);root=legal['partial_graph']
    for name,expected in pair['source_member_sha256'].items():assert row['files'][name]==expected
    history=PartialGraph(root['resolution'],root['height'],root['clearance'])
    for ob in legal['history']:history.update(ob)
    initial=PartialGraph.restore(root)
    assert history.cells==initial.cells and history.timestamp==initial.timestamp and history.revision==initial.revision
    assert history.visited==initial.visited and np.array_equal(history.ego_support,initial.ego_support)
    prefix=[];node=pair['start_node']
    while node is not None:prefix.append(node);node=nodes[node]['parent']
    graph=FastPartialGraph.restore(root)
    for node in reversed(prefix):assert graph.update(nodes[node]['observation'])==nodes[node]['graph_delta']
    current=graph.snapshot();known={tuple(c['cell']):c for c in current['cells']};targets=[];corrections=[];terminals=[]
    for suffix,field in ((pair['node_ids_a'],'trajectory_a'),(pair['node_ids_b'],'trajectory_b')):
        end=FastPartialGraph.restore(current);parent=pair['start_node'];path=pair[field]
        assert path[0][0]==current['timestamp']
        for index,node in enumerate(suffix,1):
            assert nodes[node]['parent']==parent
            ob=nodes[node]['observation'];assert [ob['timestamp'],*ob['robot_pose']]==path[index]
            assert end.update(ob)==nodes[node]['graph_delta'];parent=node
        targets.append({k:(v['state'],v['height']) for k,v in end.cells.items() if k not in known})
        terminals.append(end.snapshot())
        corrections.append(sum(v['state']!=end.cells[k]['state'] or abs(v['height']-end.cells[k]['height'])>1e-8 for k,v in known.items()))
    pa=np.asarray(pair['trajectory_a'],np.float64);pb=np.asarray(pair['trajectory_b'],np.float64)
    np.testing.assert_array_equal(pa[:,0],pb[:,0]);assert pair['duration']==pa[-1,0]-pa[0,0]
    return current,targets,corrections,legal['sensor'],terminals


def initialize(device,run,identity):
    global MODEL,RUN,IDENTITY
    from .common import make_model
    MODEL=make_model(device);RUN=Path(run);IDENTITY=identity


def crop_feature_drift(natural,common,keys):
    from eventwm_graph.stage2_segment.fused import transform
    a,_=transform(natural,keys);b,_=transform(common,keys)
    change=abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(original_feature_columns=list(range(16,104)),
                max_absolute_change=change.max(0).tolist(),mean_absolute_change=change.mean(0).tolist(),
                exact_changed_rows=(a!=b).sum(0).tolist(),
                invariant_multiscale_25_exact=bool(np.array_equal(a[:,51:76],b[:,51:76])))


def evaluate_root(job):
    from .common import fixed_probabilities,confusion
    row,pair=job;record_path=RUN/'roots'/(row['root_id']+'.json')
    cache_path=RUN/'roots'/(row['root_id']+'.npz')
    if record_path.exists():
        saved=json.loads(record_path.read_text());assert saved['identity']==IDENTITY and digest(cache_path)==saved['cache_sha256']
        maps_path=RUN/'roots'/(row['root_id']+'.maps.json.gz');assert digest(maps_path)==saved['maps_cache_sha256']
        return dict(root_id=row['root_id'],resumed=True)
    tick=time.perf_counter();graph,targets,corrections,sensor,terminals=reconstruct(row,pair);before=graph_hash(graph)
    paths=[np.asarray(pair[k],np.float64) for k in ('trajectory_a','trajectory_b')]
    predictions=[MODEL.predict(partial_graph=graph,trajectory=p) for p in paths]
    natural_keys=[d['candidate_cells'] for d in predictions]
    keys=np.unique(np.concatenate(natural_keys),axis=0).astype(np.int32)
    if not len(keys):raise ValueError('empty paired candidate union')
    padded,supports=metrics.union_predictions(keys,natural_keys[0],predictions[0]['probabilities'],natural_keys[1],predictions[1]['probabilities'])
    prepared=MODEL.prepare_graph(graph)
    natural=[MODEL.feature_class(graph,p,prepared) for p in paths]
    for c,k in zip(natural,natural_keys):np.testing.assert_array_equal(c.candidates('both'),k)
    lo=np.minimum.reduce([natural[0].lo,natural[1].lo,keys.min(0)-16])
    hi=np.maximum.reduce([natural[0].lo+natural[0].shape,natural[1].lo+natural[1].shape,keys.max(0)+17])
    bounds=(lo,hi-lo)
    if int(np.prod(bounds[1]))>4000000:raise ValueError('shared diagnostic crop too large')
    for path in paths:
        index=np.floor(path[:,[1,3]]/float(graph['resolution'])).astype(np.int64)
        assert np.all(index>=lo) and np.all(index<hi)
    direct=[];direct_heights=[];contexts=[];crop=[]
    for path,ctx,own,delta in zip(paths,natural,natural_keys,predictions):
        p,h,common=fixed_probabilities(MODEL,graph,path,keys,bounds)
        control,_,_=fixed_probabilities(MODEL,graph,path,own,(ctx.lo,ctx.shape))
        control_common,_,_=fixed_probabilities(MODEL,graph,path,own,bounds)
        own_positions=np.searchsorted(metrics.key_codes(keys),metrics.key_codes(own))
        numerical=metrics.compare_probabilities(delta['probabilities'],control)
        assert numerical['max_probability_difference']<=2e-6 and numerical['hard_label_disagreements']==0
        block_noise=metrics.compare_probabilities(control_common,p[own_positions])
        if block_noise['max_probability_difference']>2e-6:raise ValueError('fixed-key batch numerical control exceeded tolerance')
        crop.append(dict(production_vs_fixed_natural=numerical,
            shared_crop_probability_effect_same_keys=metrics.compare_probabilities(control,control_common),
            common_crop_union_batch_shape_control=block_noise,
            feature_columns=crop_feature_drift(ctx,common,own)))
        direct.append(p);direct_heights.append(h);contexts.append(common)
    neutral=paths[0].copy();neutral[:,1:]=paths[0][0,1:]
    neutral_p,neutral_height,neutral_context=fixed_probabilities(MODEL,graph,neutral,keys,bounds)
    neutral_natural=MODEL.feature_class(graph,neutral,prepared).candidates('both')
    neutral_codes=metrics.key_codes(neutral_natural);at=np.searchsorted(neutral_codes,metrics.key_codes(keys));inside=at<len(neutral_codes)
    if len(neutral_codes):inside&=neutral_codes[np.minimum(at,len(neutral_codes)-1)]==metrics.key_codes(keys)
    for p in [*padded,*direct,neutral_p]:metrics.probability_check(p)
    ys,extra,tail,miss=metrics.project_targets(keys,*targets)
    unknown=np.zeros((len(extra),3),np.float32);unknown[:,0]=1
    y_e=[np.r_[y,t] for y,t in zip(ys,tail)]
    intersect=supports[0]&supports[1];exclusive=~intersect
    observed=(ys[0]>0)|(ys[1]>0);changed=ys[0]!=ys[1]
    modes={}
    for name,ps in (('production',padded),('direct_fixed_union',direct)):
        modes[name+'_E']=metrics.paired_scores(*y_e,*[np.r_[p,unknown] for p in ps])
        for label,mask in (('U',np.ones(len(keys),bool)),('I',intersect),('exclusive',exclusive),('true_revealed_union',observed),('true_label_changed',changed)):
            modes[name+'_'+label]=metrics.paired_scores(*[y[mask] for y in ys],*[p[mask] for p in ps])
    regions=metrics.spatial_regions(keys,*paths,float(graph['resolution']))
    spatial={name:metrics.spatial_shifts(regions,*ps,*ys) for name,ps in (('production',padded),('direct_fixed_union',direct))}
    for delta in predictions:
        future={tuple(c['cell']):c for c in delta['future_partial_graph']['cells']}
        for c in graph['cells']:assert future[tuple(c['cell'])]==c
    assert graph_hash(graph)==before
    record=dict(root_id=row['root_id'],scene_id=row['scene_id'],identity=IDENTITY,pair=pair,
        graph_sha256=before,checkpoint_sha256=CHECKPOINT_SHA,duration=pair['duration'],poses=pair['poses'],
        natural_candidate_counts=[len(k) for k in natural_keys],union_candidates=len(keys),intersection_candidates=int(intersect.sum()),
        candidate_jaccard=float(intersect.mean()),score_extension_cells=len(extra),out_of_union_truth_counts=miss,
        true_counts_a=metrics.counts(y_e[0]),true_counts_b=metrics.counts(y_e[1]),known_terminal_differences=corrections,
        modes=modes,crop_audit=crop,spatial=spatial,
        neutral=dict(label='MOTION_ABLATION_OOD_NO_RECORDED_COUNTERFACTUAL_GT',
            fixed_union_expected_counts=metrics.expected(neutral_p),union_cells_outside_neutral_natural_domain=int((~inside).sum()),
            vs_actual_a_U=metrics.sensitivity(neutral_p,direct[0]),vs_actual_b_U=metrics.sensitivity(neutral_p,direct[1]),
            vs_actual_a_I=metrics.sensitivity(neutral_p[intersect],direct[0][intersect]),vs_actual_b_I=metrics.sensitivity(neutral_p[intersect],direct[1][intersect])),
        seconds=time.perf_counter()-tick,source_root_files_verified=True,causal_replay_verified=True,input_immutable=True)
    np.savez_compressed(cache_path,keys=keys,y_a=ys[0],y_b=ys[1],extra_keys=extra,extra_y_a=tail[0],extra_y_b=tail[1],
        production_a=padded[0],production_b=padded[1],direct_a=direct[0],direct_b=direct[1],neutral=neutral_p,
        support_a=supports[0],support_b=supports[1],production_height_a=predictions[0]['heights'],production_height_b=predictions[1]['heights'],
        direct_height_a=direct_heights[0],direct_height_b=direct_heights[1],neutral_height=neutral_height)
    maps_path=RUN/'roots'/(row['root_id']+'.maps.json.gz')
    snapshots=[d['future_partial_graph'].to_snapshot() if hasattr(d['future_partial_graph'],'to_snapshot') else d['future_partial_graph'] for d in predictions]
    with gzip.open(maps_path,'wt') as stream:
        json.dump(dict(current_legal_graph=graph,actual_legal_terminal_a=terminals[0],actual_legal_terminal_b=terminals[1],
            hypothetical_production_terminal_a=snapshots[0],hypothetical_production_terminal_b=snapshots[1],
            input_graph_sha256=before,checkpoint_sha256=CHECKPOINT_SHA,
            comparison_scope='Only initiallyUNKNOWNcells enter task scores; knowncell corrections in actual terminal maps are audited outside prediction scope.',
            direct_fixed_union_maps='Reconstructible from currentG+keys/support/probability/height arrays and recorded terminalpose/time; flagged extended-domain diagnostic, not deployed observations.'),stream,allow_nan=False)
    record['maps_cache_sha256']=digest(maps_path);record['maps_cache_bytes']=maps_path.stat().st_size
    record['cache_sha256']=digest(cache_path);record['cache_bytes']=cache_path.stat().st_size
    save(record_path,record)
    return dict(root_id=row['root_id'],seconds=record['seconds'],cache_bytes=record['cache_bytes'])


def run(device='cuda:1',workers=4,name='primary_v1',limit=None):
    from eventwm_graph.stage2.data import source_rows
    if not (OUT/'probability/complete.json').exists():raise RuntimeError('PriorityA must complete and parent must coordinate GPU before B')
    runpath=DEST/name;runpath.mkdir(parents=True,exist_ok=True);(runpath/'roots').mkdir(exist_ok=True)
    complete=runpath/'complete.json'
    if complete.exists():raise FileExistsError('completed trajectory run exists')
    rows,_=source_rows('dev');by_id={r['root_id']:r for r in rows}
    pairs=json.loads((DEST/'availability_v1/pairs.json').read_text())
    if limit:pairs=pairs[:limit]
    identity=dict(protocol_sha256=digest(DEST/'protocol_v1.json'),global_protocol_sha256=digest(OUT/'protocol.json'),
        addendum_sha256=digest(DEST/'implementation_addendum_v1.json'),cache_addendum_sha256=digest(DEST/'cache_artifact_addendum_v1.json'),pair_manifest_sha256=digest(DEST/'availability_v1/pairs.json'),
        checkpoint_sha256=CHECKPOINT_SHA,device=device,
        sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),Path(__file__).with_name('trajectory_audit.py'),Path(__file__).with_name('trajectory_metrics.py'),Path(__file__).with_name('common.py')]})
    if (runpath/'identity.json').exists():assert json.loads((runpath/'identity.json').read_text())==identity
    else:save(runpath/'identity.json',identity)
    tick=time.perf_counter();done=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(device,runpath,identity)) as pool:
        futures=[pool.submit(evaluate_root,(by_id[p['root_id']],p)) for p in pairs]
        for future in as_completed(futures):
            done.append(future.result())
            if len(done)%25==0 or len(done)==len(pairs):
                progress=dict(roots=len(done),total=len(pairs),seconds=time.perf_counter()-tick)
                save(runpath/'progress.json',progress);print(json.dumps(progress),flush=True)
    save(complete,dict(roots=len(done),scenes=len({p['scene_id'] for p in pairs}),seconds=time.perf_counter()-tick,
        workers=workers,device=device,limited=bool(limit),checkpoint_sha256=CHECKPOINT_SHA,identity=identity))


def summarize(name='primary_v1'):
    from .common import extended_rates
    runpath=DEST/name;done=json.loads((runpath/'complete.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted((runpath/'roots').glob('*.json'))]
    assert len(records)==done['roots'];scenes=sorted({r['scene_id'] for r in records})
    output=dict(roots=len(records),scenes=len(scenes),checkpoint_sha256=CHECKPOINT_SHA,
        weighting='One pose-selected pairperRoot; RootmeanwithinScene thenScenemean. All cell/probability arrays aligned to declared shared domain.',
        score_precision='Float64 proper-score arithmetic applied to unchanged float32 probabilities;1e-9clip; summed3-coordinate Brier.',
        candidate_domains=dict(union_min=min(r['union_candidates'] for r in records),union_max=max(r['union_candidates'] for r in records),
            out_of_union_truth_counts=np.sum([r['out_of_union_truth_counts'] for r in records],axis=0).tolist()),
        modes={},spatial={},crop_audit={},neutral={},visual_selection={},limitations=[
        'Pairs are selected for pose diversity, not a representative sample of all47128windows; onepairperRoot and all pairedscenes disclosed.',
        'Common-crop network outputs on Uoutsideownnatural-domain are extrapolation; direct-I is the in-natural-support conditioning diagnostic.',
        'Neutral constant-pose motion has no recorded counterfactual target; only inputablation sensitivity is reported.',
        '70/38subsets are from historically used development, not untouched protected confirmation.'])
    groups=dict(all=records,threshold_selection70=[r for r in records if r['pair']['diagnostic_scene_split']=='threshold_selection70'],
                diagnostic_validation38=[r for r in records if r['pair']['diagnostic_scene_split']=='diagnostic_validation38'])
    rng=np.random.default_rng(202610252)
    for subset,rs in groups.items():
        ns=sorted({r['scene_id'] for r in rs});draws=rng.integers(0,len(ns),size=(2000,len(ns)))
        target=output['modes'].setdefault(subset,{})
        for mode in rs[0]['modes']:
            eligible=[r for r in rs if r['modes'][mode]['cells']]
            if not eligible:continue
            names=sorted({r['scene_id'] for r in eligible});by=defaultdict(list)
            for r in eligible:by[r['scene_id']].append(r['modes'][mode])
            own_cm=np.array([np.mean([v['own_confusion'] for v in by[s]],axis=0) for s in names]);cross_cm=np.array([np.mean([v['cross_confusion'] for v in by[s]],axis=0) for s in names])
            own=np.array([np.mean([v['own_scores'] for v in by[s]],axis=0) for s in names]);cross=np.array([np.mean([v['cross_scores'] for v in by[s]],axis=0) for s in names])
            localdraw=np.random.default_rng(202610252).integers(0,len(names),size=(2000,len(names)))
            proper_draw=(cross-own)[localdraw].mean(1);iou_draw=[]
            for indices in localdraw:
                a=extended_rates(own_cm[indices].mean(0));b=extended_rates(cross_cm[indices].mean(0));iou_draw.append(a['label_aware_delta_iou']-b['label_aware_delta_iou'])
            sens=np.array([np.mean([v['sensitivity']['mean_total_variation'] for v in by[s]],axis=0) for s in names])
            target[mode]=dict(roots=len(eligible),scenes=len(names),own=extended_rates(own_cm.mean(0)),cross=extended_rates(cross_cm.mean(0)),
                own_logloss=float(own.mean(0)[0]),own_brier=float(own.mean(0)[1]),cross_logloss=float(cross.mean(0)[0]),cross_brier=float(cross.mean(0)[1]),
                conditioning_logloss_gain=float((cross-own).mean(0)[0]),conditioning_brier_gain=float((cross-own).mean(0)[1]),
                mean_total_variation=float(sens.mean()),
                ci95=dict(logloss_gain=np.quantile(proper_draw[:,0],[.025,.975]).tolist(),brier_gain=np.quantile(proper_draw[:,1],[.025,.975]).tolist(),
                    own_minus_cross_iou=np.quantile(iou_draw,[.025,.975]).tolist(),mean_total_variation=np.quantile(sens[localdraw].mean(1),[.025,.975]).tolist()))
        output['spatial'][subset]={}
        for mode in ('production','direct_fixed_union'):
            output['spatial'][subset][mode]={}
            for region in rs[0]['spatial'][mode]:
                eligible=[r for r in rs if r['spatial'][mode][region]['per_cell_mean_expected_contrast'] is not None]
                by=defaultdict(list)
                for r in eligible:by[r['scene_id']].append(r['spatial'][mode][region])
                if not by:continue
                keys=('pair_mean_own_minus_cross_expected_count','pair_mean_own_minus_cross_true_count','per_cell_mean_expected_contrast','per_cell_mean_true_contrast')
                values=np.array([[np.mean([v[k] for v in by[s]]) for k in keys] for s in sorted(by)])
                draw=np.random.default_rng(202610252).integers(0,len(by),size=(2000,len(by)))
                output['spatial'][subset][mode][region]=dict(roots=len(eligible),scenes=len(by),mean=dict(zip(keys,values.mean(0).tolist())),ci95={k:np.quantile(values[draw,j].mean(1),[.025,.975]).tolist() for j,k in enumerate(keys)})
    for side in range(2):
        key='a' if side==0 else 'b';v=[r['crop_audit'][side] for r in records]
        output['crop_audit'][key]=dict(max_natural_fixed_probability_error=max(x['production_vs_fixed_natural']['max_probability_difference'] for x in v),
            max_common_union_batch_probability_error=max(x['common_crop_union_batch_shape_control']['max_probability_difference'] for x in v),
            max_map_crop_only_probability_effect=max(x['shared_crop_probability_effect_same_keys']['max_probability_difference'] for x in v),
            roots_with_map_crop_probability_changes_above_2e6=sum(x['shared_crop_probability_effect_same_keys']['max_probability_difference']>2e-6 for x in v),
            multiscale_25_exact_for_every_root=all(x['feature_columns']['invariant_multiscale_25_exact'] for x in v),
            original_feature_columns=list(range(16,104)),max_feature_change_across_roots=np.max([x['feature_columns']['max_absolute_change'] for x in v],axis=0).tolist())
    output['cohorts']={}
    cohort_groups=defaultdict(list)
    for r in records:
        duration=r['duration'];cohort_groups['duration:'+('<=.5' if duration<=.5000001 else '.5-1' if duration<=1.0000001 else '1-2' if duration<=2.0000001 else '>2')].append(r)
        cohort_groups['start:'+('root' if r['pair']['start_node'] is None else 'intermediate')].append(r)
        paths=[np.asarray(r['pair'][k]) for k in ('trajectory_a','trajectory_b')]
        movement=sum(float(np.linalg.norm(np.diff(p[:,[1,3]],axis=0),axis=1).sum()) for p in paths)
        rotation=sum(float(np.abs(np.arctan2(np.sin(np.diff(p[:,4])),np.cos(np.diff(p[:,4])))).sum()) for p in paths)
        cohort_groups['motion:'+('mixed' if movement>.01 and rotation>.01 else 'translation' if movement>.01 else 'rotation' if rotation>.01 else 'stationary')].append(r)
        cohort_groups['target:'+('different' if r['modes']['direct_fixed_union_true_label_changed']['cells'] else 'identical')].append(r)
    for group,rs in cohort_groups.items():
        output['cohorts'][group]={}
        for mode in ('production_E','direct_fixed_union_I'):
            by=defaultdict(list)
            for r in rs:
                if r['modes'][mode]['cells']:by[r['scene_id']].append(r['modes'][mode])
            if not by:continue
            values=np.array([np.mean([v['gain'] for v in vs],axis=0) for vs in by.values()])
            draw=np.random.default_rng(202610252).integers(0,len(by),size=(2000,len(by)))
            own=np.mean([np.mean([v['own_confusion'] for v in vs],axis=0) for vs in by.values()],axis=0)
            cross=np.mean([np.mean([v['cross_confusion'] for v in vs],axis=0) for vs in by.values()],axis=0)
            output['cohorts'][group][mode]=dict(roots=sum(len(v) for v in by.values()),scenes=len(by),gain=values.mean(0).tolist(),
                ci95=np.quantile(values[draw].mean(1),[.025,.975],axis=0).T.tolist(),own_iou=extended_rates(own)['label_aware_delta_iou'],cross_iou=extended_rates(cross)['label_aware_delta_iou'])
    output['neutral']=dict(label='OOD_INPUT_ABLATION_NO_COUNTERFACTUAL_GT',
        mean_total_variation_vs_a_U=float(np.mean([r['neutral']['vs_actual_a_U']['mean_total_variation'] for r in records])),
        mean_total_variation_vs_b_U=float(np.mean([r['neutral']['vs_actual_b_U']['mean_total_variation'] for r in records])),
        note='Descriptive Rootmean sensitivity only, no neutral accuracy/calibration score or trained-baseline claim.')
    eligible=[r for r in records if r['modes']['direct_fixed_union_I']['cells']]
    ordered=sorted(eligible,key=lambda r:(r['modes']['direct_fixed_union_I']['gain'][0],r['root_id']))
    fixed=sorted(records,key=lambda r:hashlib.sha256(('202610252:'+r['root_id']).encode()).hexdigest())[:6]
    output['visual_selection']=dict(fixed_hash=[r['root_id'] for r in fixed],lowest_observed_direct_I_gain=[r['root_id'] for r in ordered[:3]],highest_observed_direct_I_gain=[r['root_id'] for r in ordered[-3:]],
        extreme_selection_is_outcome_based=True,criterion='Direct-Icross-minus-ownclippedlogloss gain; extremesare explicitlynonrepresentative')
    save(runpath/'summary.json',output)
    save(runpath/'conditioning_gain_order.json',[dict(root_id=r['root_id'],scene_id=r['scene_id'],direct_I_logloss_gain=r['modes']['direct_fixed_union_I']['gain'][0]) for r in ordered])
    print(json.dumps(dict(roots=len(records),modes=output['modes']['all'],crop=output['crop_audit']),allow_nan=False),flush=True)
