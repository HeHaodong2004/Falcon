"""Frozen development decision diagnostic, no fitting or online outcome reads.

Online scores are published before hash-verified sibling outcomes are opened.
Only the original fixed development cohort is eligible. No future people tape,
future graph delta, protected confirmation or opened fresh cohort is accessed.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from dataclasses import replace
from l2_partial_structural_api_example import build_model
from l2_partial_structural_benchmark import selected_rows,SNAPSHOT,verify_legal_files,collector_metadata
from eventwm_graph.l2_partial.repair_profile import _read,_publish,_sources,_metadata,_percentiles,_sha
from eventwm_graph.l2_partial.planning import (
    packet_from_batch,compile_observed_queries,graph_decision_features,pessimistic_scores,
    CandidateStatus,
)
from eventwm_graph.l2_partial.training_contract import iter_versioned_root_shards

REPO=Path(__file__).resolve().parents[1]
PROTOCOL=REPO/'docs/l2/decision_ready_20261021/EXPERIMENT_PROTOCOL.md'


def numeric_summary(values):
    a=np.asarray(values,float)
    return {'n':len(a),'mean':float(a.mean()),'minimum':float(a.min()),'p05':float(np.percentile(a,5)),
            'p50':float(np.median(a)),'p95':float(np.percentile(a,95)),'maximum':float(a.max())} if len(a) else {'n':0}


def historical_score(query,observation,goal):
    poses=np.asarray(query['compiled']['planned_poses'],float)
    baseline=float(np.linalg.norm(poses[0,:3]-goal)-np.linalg.norm(poses[-1,:3]-goal)-.1*.048*(len(poses)-1))
    t=np.arange(len(poses))*.048;minimum=np.inf
    for track in observation['tracks']:
        if not track.get('valid',True):continue
        velocity=np.asarray(track.get('estimated_velocity') or [0.,0.,0.],float) if track.get('velocity_valid',False) else np.zeros(3)
        age=max(0.,observation['timestamp']-track['last_seen_time'])
        people=np.asarray(track['position'])+(age+t[:,None])*velocity
        minimum=min(minimum,float(np.linalg.norm(people[1:,[0,2]]-poses[1:,[0,2]],axis=1).min(initial=np.inf)))
    return baseline,baseline-5*bool(minimum<=.616)-bool(minimum<1.)


def record_history(model,legal):
    trace=[];factory=model.history_filter
    def make_filter():
        wm=factory();observe=wm.observe
        def arrival(b,row,*args,**kwargs):
            posterior=observe(b,row,*args,**kwargs)
            trace.append({'timestamp':posterior.timestamp,'unresolved_mass':posterior.unresolved_mass,
                          'hypotheses':len(posterior.hypotheses),'detections':posterior.evidence.get('last_detection_count'),
                          'accounting':posterior.evidence.get('mass_accounting',{})})
            return posterior
        wm.observe=arrival
        return wm
    model.history_filter=make_filter
    try:return model.initialize(legal['history'],legal['partial_graph']),trace
    finally:model.history_filter=factory


def online_records(model,rows,output):
    records=[];references=[]
    for i,row in enumerate(rows):
        verify_legal_files(row)
        folder=SNAPSHOT/'roots'/row['root_id'];legal=_read(folder/'legal_root.json.gz');saved=_read(folder/'queries.json.gz')
        began=time.perf_counter();belief,trace=record_history(model,legal)
        queries=compile_observed_queries(legal['partial_graph'],saved,belief.robot_pose.tolist())
        for before,after in zip(saved,queries):
            if before['compiled']['actions']!=after['compiled']['actions'] or not np.array_equal(before['compiled']['planned_poses'],after['compiled']['planned_poses']):
                raise AssertionError('saved proposal changed under observed-graph recompilation')
        prefix_ms=(time.perf_counter()-began)*1000.
        began=time.perf_counter();future=model.predict_shared(belief,6.,16,27)
        batch=model.evaluate_queries(belief,queries,future,budget_ms=1e12)
        packet=packet_from_batch(belief,queries,batch,pointgoal=legal['pointgoal'],stage0_snapshot=legal['partial_graph'])
        planning_ms=(time.perf_counter()-began)*1000.;features=graph_decision_features(packet);scores=pessimistic_scores(packet)
        candidates=[]
        for query,c,score in zip(queries,packet.candidates,scores):
            base,historical=historical_score(query,legal['history'][-1],np.asarray(legal['pointgoal']))
            provenance=query.get('provenance',{})
            if isinstance(provenance,list):provenance=provenance[0] if provenance else {}
            candidates.append({'query_id':c.query_id,'status':c.status.value,'supported':bool(c.status==CandidateStatus.COMPLETE),
                'source':provenance.get('source','unknown'),
                'baseline_score':base,'historical_score':historical,'l2_score':float(score) if np.isfinite(score) else None,
                'collision_interval':c.collision_risk_interval,'social_interval':c.social_risk_interval,
                'goal_progress_interval':c.goal_progress.mean_interval,'route_progress_interval':c.route_progress.mean_interval,
                'duration_interval':c.duration.mean_interval,'nominal_duration':c.geometry.planned_duration_s,
                'represented_mass':c.represented_mass,'unresolved_mass':c.unresolved_mass,
                'branches':[{'kind':b.kind,'lower_mass':b.lower_mass,'probability_interval':b.probability_interval,
                    'posterior_unresolved_mass':b.successor.unresolved_mass,'can_select_new_query':b.successor.can_select_new_query} for b in c.observation_branches]})
        record={'root_id':row['root_id'],'scene_id':row['scene_id'],'legal_hashes':row['files'],
            'actual_represented_mass':1-belief.unresolved_mass,'forecast_represented_mass':float(future.weights.sum()),
            'forecast_unresolved_mass':future.unresolved_mass,'history_mass_trace':trace,
            'graph_version':packet.graph_version,'stage0_revision':packet.stage0_context.stage0_revision,
            'global_nodes':len(packet.stage0_context.global_nodes),'global_edges':len(packet.stage0_context.global_edges),
            'candidate_feature_shape':features.candidate_features.shape,'candidate_feature_names':features.feature_names,
            'shared_future_reference':packet.shared_future.reference,'model_version':packet.model_version,
            'prefix_and_recompile_ms':prefix_ms,'forecast_evaluate_packet_ms':planning_ms,'timings_ms':dict(packet.timings_ms),
            'candidates':candidates}
        records.append(record);references.append((row,legal,queries))
        print(json.dumps({'online_roots':i+1,'total':len(rows),'represented_mass':record['forecast_represented_mass']}),flush=True)
    _publish(output,{'schema':'l2_decision_online_scores_1','protocol_sha256':_sha(PROTOCOL),'source_hashes':_sources(),
        'outcomes_opened':False,'rows':records})
    return records,references


def concordance(scores,truth):
    correct=0.;count=0
    for i in range(len(truth)):
        for j in range(i):
            if abs(truth[i]-truth[j])<=1e-9:continue
            count+=1;delta=(scores[i]-scores[j])*(truth[i]-truth[j])
            correct+=1. if delta>1e-12 else .5 if abs(delta)<=1e-12 else 0.
    return correct/count if count else None,count


def auc(scores,labels):
    positives=[s for s,y in zip(scores,labels) if y];negatives=[s for s,y in zip(scores,labels) if not y]
    if not positives or not negatives:return None
    return float(np.mean([float(a>b)+.5*float(a==b) for a in positives for b in negatives]))


def scene_root_interval(records,key):
    groups=defaultdict(list)
    for r in records:
        if r.get(key) is not None:groups[r['scene_id']].append(r[key])
    if not groups:return {'roots':0,'scenes':0}
    rng=np.random.default_rng(271);scenes=list(groups);values=[]
    for _ in range(1000):
        sample=[]
        for scene_index in rng.integers(len(scenes),size=len(scenes)):
            roots=groups[scenes[scene_index]]
            sample.extend(np.asarray(roots)[rng.integers(len(roots),size=len(roots))])
        values.append(np.mean(sample))
    return {'roots':sum(map(len,groups.values())),'scenes':len(groups),'root_mean':float(np.mean([x for group in groups.values() for x in group])),
            'scene_root_bootstrap_95':np.percentile(values,[2.5,97.5]).tolist(),'draws':1000,'seed':271}


def score_outcomes(online):
    result=[];all_candidates=[];outcome_hashes={};causes=Counter()
    for row in online:
        path=SNAPSHOT/'roots'/row['root_id']/'outcomes.json.gz'
        expected=row['legal_hashes']['outcomes.json.gz'];actual=_sha(path)
        if actual!=expected:raise ValueError('frozen development outcome checksum mismatch')
        outcome_hashes[row['root_id']]=actual
        outcomes={o['query_id']:o['summary'] for o in _read(path)}
        eligible=[];excluded=[]
        for c in row['candidates']:
            o=outcomes[c['query_id']];causes[o['termination']]+=1
            complete_followup=o['termination'] in ('query_complete','human_collision','query_timeout')
            if not c['supported'] or not complete_followup:
                excluded.append({'query_id':c['query_id'],'online_status':c['status'],'native_termination':o['termination']});continue
            truth=float(o['goal_euclidean_distance_change_m']-5*o['human_collision']-o['any_psc_violation']-.1*o['elapsed_s'])
            cc={**c,'realized_utility':truth,'native_contact':o['human_collision'],'native_psc':o['any_psc_violation'],
                'native_progress':o['goal_euclidean_distance_change_m'],'native_elapsed':o['elapsed_s'],
                'native_query_complete':o['query_complete'],'native_termination':o['termination'],
                'native_psc_fraction':o['psc']}
            eligible.append(cc);all_candidates.append(cc)
        r={'root_id':row['root_id'],'scene_id':row['scene_id'],'candidates':eligible,'excluded':excluded,
           'candidate_support':len(eligible),'complete_candidate_set':len(eligible)==len(row['candidates'])}
        if len(eligible)>=2:
            truth=[c['realized_utility'] for c in eligible];oracle=max(truth)
            for method in ('baseline','historical','l2'):
                scores=[c[method+'_score'] for c in eligible];picked=int(np.argmax(scores))
                r[method+'_pairwise'],r['informative_pairs']=concordance(scores,truth)
                r[method+'_regret_on_supported_set']=oracle-truth[picked]
                if r['complete_candidate_set']:r[method+'_regret']=oracle-truth[picked]
                r[method+'_choice']=eligible[picked]['query_id'];r[method+'_chosen_outcome']=eligible[picked]
            r['l2_changes_baseline']=r['l2_choice']!=r['baseline_choice']
            r['l2_changes_historical']=r['l2_choice']!=r['historical_choice']
            r['l2_contact_auc']=auc([c['collision_interval'][1] for c in eligible],[c['native_contact'] for c in eligible])
            r['l2_social_auc']=auc([c['social_interval'][1] for c in eligible],[c['native_psc'] for c in eligible])
            r['baseline_minus_l2_regret']=r['baseline_regret_on_supported_set']-r['l2_regret_on_supported_set']
            r['historical_minus_l2_regret']=r['historical_regret_on_supported_set']-r['l2_regret_on_supported_set']
        result.append(r)
    summary={key:scene_root_interval(result,key) for key in (
        'baseline_pairwise','historical_pairwise','l2_pairwise','baseline_regret','historical_regret','l2_regret',
        'baseline_minus_l2_regret','historical_minus_l2_regret','l2_contact_auc','l2_social_auc')}
    summary.update(roots=len(result),scenes=len({r['scene_id'] for r in result}),candidates=len(all_candidates),
        complete_candidate_set_roots=sum(r['complete_candidate_set'] for r in result),
        native_termination_counts=dict(causes),native_contact=sum(c['native_contact'] for c in all_candidates),
        native_psc=sum(c['native_psc'] for c in all_candidates),
        l2_changes_baseline=sum(r.get('l2_changes_baseline',False) for r in result),
        l2_changes_historical=sum(r.get('l2_changes_historical',False) for r in result))
    for method in ('baseline','historical','l2'):
        picked=[r[method+'_chosen_outcome'] for r in result if method+'_chosen_outcome' in r]
        summary[method+'_selected']={'native_contact':sum(c['native_contact'] for c in picked),'native_psc':sum(c['native_psc'] for c in picked),
            'progress':numeric_summary([c['native_progress'] for c in picked]),'duration':numeric_summary([c['native_elapsed'] for c in picked])}
    summary['duration_nominal_minus_arrived']=numeric_summary([c['nominal_duration']-c['native_elapsed'] for c in all_candidates])
    return {'summary':summary,'roots':result,'outcome_hashes':outcome_hashes,
            'offline_oracle':'max realized short-query reference utility among scored siblings; no navigation oracle'}


def uncertainty(online):
    cs=[c for r in online for c in r['candidates']]
    spans=[];ties=[]
    for r in online:
        c=[c for c in r['candidates'] if c['supported']]
        if len(c)<2:continue
        scores=np.asarray([x['l2_score'] for x in c]);risks=np.asarray([x['collision_interval'][1] for x in c])
        spans.append({'root_id':r['root_id'],'score_span':float(np.ptp(scores)),'contact_span':float(np.ptp(risks)),
                      'social_span':float(np.ptp([x['social_interval'][1] for x in c]))})
        ties.extend(abs(scores[i]-scores[j])<=1e-8 for i in range(len(scores)) for j in range(i))
    return {'represented_mass':numeric_summary([r['forecast_represented_mass'] for r in online]),
        'unresolved_mass':numeric_summary([r['forecast_unresolved_mass'] for r in online]),
        'contact_interval_width':numeric_summary([c['collision_interval'][1]-c['collision_interval'][0] for c in cs]),
        'social_interval_width':numeric_summary([c['social_interval'][1]-c['social_interval'][0] for c in cs]),
        'nearly_vacuous_candidates':sum(c['collision_interval'][1]-c['collision_interval'][0]>=.95 for c in cs),
        'nearly_vacuous_roots':sum(r['forecast_unresolved_mass']>=.95 for r in online),
        'unsupported_candidates':sum(not c['supported'] for c in cs),'timed_out_candidates':0,
        'pairwise_score_tie_frequency':float(np.mean(ties)) if ties else None,'within_root_spans':spans}


def sensitivity(references):
    records=[]
    from eventwm_graph.l2_partial.structural_model import StructuralGraphBeliefWorldModel
    base,_=build_model(True)
    for H in (6,12,24):
        model=StructuralGraphBeliefWorldModel(dynamics=base.dynamics,population_prior=base.population_prior,
            event_model=base.event_model,observation_law=base.observation_law,max_hypotheses=H,
            maneuver_rate=base.maneuver_rate,reset_std=base.reset_std,maneuver_retention=base.maneuver_retention)
        for row,legal,queries in references[:12]:
            b=model.initialize(legal['history'],legal['partial_graph'])
            for W in (8,16,32):
                begin=time.perf_counter();f=model.predict_shared(b,6.,W,27)
                batch=model.evaluate_queries(b,queries,f,budget_ms=1e12,refined_queries=0)
                p=packet_from_batch(b,queries,batch,pointgoal=legal['pointgoal']);scores=pessimistic_scores(p)
                records.append({'root_id':row['root_id'],'scene_id':row['scene_id'],'H':H,'W':W,
                    'actual_represented_mass':1-b.unresolved_mass,'forecast_represented_mass':float(f.weights.sum()),
                    'collision_width':float(f.unresolved_mass),'choice':queries[int(np.argmax(scores))]['query_id'],
                    'scoring_ms':(time.perf_counter()-begin)*1000.,'actual_hypotheses':len(b.hypotheses)})
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--roots',type=int,default=60)
    p.add_argument('--skip-sensitivity',action='store_true');a=p.parse_args()
    output=(REPO/a.output).resolve()
    if not output.is_relative_to(REPO/'output/l2_decision_ready_20261021') or output.exists():raise ValueError('new decision output required')
    output.parent.mkdir(parents=True,exist_ok=True)
    online_path=output.with_name(output.stem+'_online.json')
    sources=_sources();before=collector_metadata();begin=time.perf_counter();failure=None;online=[];evaluation=None;budget=[]
    try:
        model,assets=build_model(True);rows=selected_rows(a.roots)
        online,references=online_records(model,rows,online_path)
        # Separate phase: scoring cannot influence proposals, forecasts or scores.
        evaluation=score_outcomes(online)
        if not a.skip_sensitivity:budget=sensitivity(references)
        manifest=json.loads((SNAPSHOT/'manifest.json').read_text())
        smoke=list(iter_versioned_root_shards(manifest,split='dev'))[:2]
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};assets={};smoke=[]
    value={'schema':'l2_decision_development_1','protocol_sha256':_sha(PROTOCOL),'source_hashes':sources,
        'sources_unchanged':sources==_sources(),'script_sha256':_sha(__file__),'assets':assets,'metadata':_metadata(),
        'collector_before':before,'collector_after':collector_metadata(),'online_scores_path':str(online_path.relative_to(REPO)),
        'online_scores_sha256':_sha(online_path) if online_path.exists() else None,'evaluation':evaluation,
        'uncertainty':uncertainty(online) if online else None,'budget_sensitivity':budget,'shard_smoke':smoke,
        'elapsed_s':time.perf_counter()-begin,'failure':failure,'confirmation_claim':False,'statistical_fitting':False,
        'outcome_phase':'hash-verified original development outcomes opened only after online score publication',
        'hidden_future_or_future_graph_payloads_opened':False}
    _publish(output,json.loads(json.dumps(value,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist(),allow_nan=False)))
    print(json.dumps({'output':str(output),'failure':failure,'summary':evaluation['summary'] if evaluation else None}),flush=True)
    if failure or not value['sources_unchanged']:raise RuntimeError('diagnostic incomplete or source changed; preserve artifact')


if __name__=='__main__':main()
