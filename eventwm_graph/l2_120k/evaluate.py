"""Offline Layer 2 evaluation; no query ranking, actions, or navigation policy."""
import argparse
import collections
from dataclasses import fields
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from eventwm_graph.l2_partial.data import read
from eventwm_graph.l2_partial.state import ObservedGraph,Hypothesis
from eventwm_graph.l2_partial.structural_state import StructuralBeliefState
from eventwm_graph.l2_partial.repair_evaluation import gospa,scene_ci,DiagnosticHypothesis,association_ambiguous
from eventwm_graph.l2_partial.snapshot import sha
from .model import load_model
from .data import OUT,SNAP
from .metrics import population_energy,vector_energy,count_scores,count_pmf
from .observations import predict_observations

HORIZONS=(.5,1.,2.,3.)
OLD=Path('docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints/graph_dynamics.pt')
OPTICS='docs/l2/partial_graph_wm_structural_20261021/artifacts/checkpoints/observation_fit.json'


def truth_at(times,positions,t):
    if t<times[0]-1e-7 or t>times[-1]+1e-7:return None
    i=min(int(np.searchsorted(times,t)),len(times)-1)
    if i==0 or abs(times[i]-t)<1e-8:return positions[i]
    a=(t-times[i-1])/(times[i]-times[i-1]);return positions[i-1]*(1-a)+positions[i]*a


def restore(model,cached,graph,visible_only=False):
    hypotheses=[]
    for h in cached['filtered_hypotheses']:
        if visible_only:
            keep=np.isclose(h.last_seen,cached['timestamp'],rtol=0,atol=1e-7)
            removed=int((~keep).sum());pmf=np.pad(h.hidden_pmf,(removed,0))
            changes={k:getattr(h,k)[keep].copy() for k in ('mean','covariance','last_seen','slot_ids','height','birth_time','birth_position') if getattr(h,k) is not None}
            for key in ('maneuver_mean','maneuver_covariance','maneuver_probability'):
                changes[key]=None if getattr(h,key) is None else getattr(h,key)[keep].copy()
            h=h.fork(**changes,hidden_pmf=pmf)
        hypotheses.append(h.fork(intent=None,diffusion=None,mode=None))
    b=StructuralBeliefState(ObservedGraph.from_snapshot(graph),hypotheses,cached['timestamp'],cached['robot_pose'],
        model_version=model.model_version,evidence=dict(cached['filtered_evidence']))
    return model.prepare(b)


def source_maps(hypotheses,timestamp,truth0,aged=False):
    maps=[]
    for h in hypotheses:
        mapping={};available=list(range(len(truth0)))
        groups=[np.flatnonzero(np.isclose(h.last_seen,timestamp,rtol=0,atol=1e-7))]
        if aged:groups.append(np.flatnonzero(~np.isclose(h.last_seen,timestamp,rtol=0,atol=1e-7)))
        for gi,group in enumerate(groups):
            if not len(group) or not available:continue
            d=np.linalg.norm(h.mean[group,None,:2]-truth0[None,available],axis=-1);ii,jj=linear_sum_assignment(d)
            used=[]
            for i,j in zip(ii,jj):
                if d[i,j]<(1e-5 if gi==0 else 2.):mapping[int(h.slot_ids[group[i]])]=available[j];used.append(available[j])
            available=[j for j in available if j not in used]
        maps.append(mapping)
    return maps


def maps_for(future,truth0,aged=False):
    maps=source_maps(future.source.hypotheses,future.source.timestamp,truth0,aged)
    return [maps[i] for i in future.world_source_indices]


def carried_maps(future,origin_maps):
    # Only an arbitrary integer origin tag enters model state. Physical ancestry
    # remains an external OFFLINE scoring table, never an inference input.
    return [origin_maps[future.source.hypotheses[i].diagnostic_origin]
        for i in future.world_source_indices]


def visible_ids(legal,truth0):
    positions=[np.asarray(t['position'])[[0,2]] for t in legal['history'][-1]['tracks'] if t['visible_now'] and t.get('valid',True)]
    ids=[]
    for p in positions:
        if len(truth0):
            d=np.linalg.norm(truth0-p,axis=-1);j=int(d.argmin())
            if d[j]<1e-5:ids.append(j)
    return sorted(set(ids))


def forecast_score(future,index,truth,truth0,target_ids,ancestry=None):
    samples=[p[np.isfinite(p).all(-1)] for p in future.positions[:,index]]
    result=dict(population_es3=population_energy(samples,truth),population_score_definition='Gaussian counting-measure Hilbert ES, bandwidth3m',
        gospa=0.,false_population=0.,missed_population=0.,localization_squared=0.,cardinality_squared=0.)
    for p in samples:
        for k,v in gospa(p,truth).items():
            key={'false_localized':'false_population','missed_localized':'missed_population'}.get(k,k)
            result[key]=result.get(key,0)+v/len(samples)
    mappings=ancestry if ancestry is not None else maps_for(future,truth0)
    errors=[];es=[];coverage=[];vel_errors=[];radii=[];supported=0;all_joint=[]
    for identity in target_ids:
        draws=[];vel=[]
        for w,(h,mapping) in enumerate(zip(future.worlds,mappings)):
            slots=[i for i,s in enumerate(h.slot_ids) if mapping.get(int(s))==identity]
            if len(slots)==1:draws.append(future.positions[w,index,slots[0]]);vel.append(future.velocities[w,index,slots[0]])
        if len(draws)!=len(future.worlds):continue
        x=np.array(draws);supported+=1;es.append(vector_energy(x,truth[identity]));mean=x.mean(0)
        error=float(np.linalg.norm(mean-truth[identity]));radius=float(np.quantile(np.linalg.norm(x-mean,axis=-1),.9))
        errors.append(error);coverage.append(error<=radius);radii.append(radius);all_joint.append(x)
    result.update(tracked_es=float(np.mean(es)) if es else None,tracked_mean_error=float(np.mean(errors)) if errors else None,
        coverage90=float(np.mean(coverage)) if coverage else None,radius90=float(np.mean(radii)) if radii else None,
        tracked_targets=len(target_ids),tracked_supported=supported,tracked_missing=len(target_ids)-supported,
        tracked_position_nll=None,position_nll_status='NOT_COMPUTED: exact compound multi-step density not implemented')
    if supported and supported==len(target_ids):
        joint=np.stack(all_joint,axis=1);result['joint_tracked_es']=vector_energy(joint,np.asarray(truth)[target_ids])/math.sqrt(supported)
        if supported>=2:
            centered=joint-joint.mean(0);errors=joint.mean(0)-np.asarray(truth)[target_ids]
            predicted=[];observed=[]
            for i in range(supported):
                for j in range(i):
                    predicted.append(float(np.sum(centered[:,i]*centered[:,j])/(len(joint)-1)))
                    observed.append(float(errors[i]@errors[j]))
            result['cross_human_predicted_covariance_trace']=float(np.mean(predicted))
            result['cross_human_observed_residual_product']=float(np.mean(observed))
            result['cross_human_covariance_calibration_difference']=float(np.mean(predicted)-np.mean(observed))
    return result


def path_scores(future,truth_targets,ids,mappings):
    values=[];joint=[];matched=[]
    for identity in ids:
        draws=[]
        for w,(h,mapping) in enumerate(zip(future.worlds,mappings)):
            slots=[i for i,s in enumerate(h.slot_ids) if mapping.get(int(s))==identity]
            if len(slots)!=1:break
            draws.append(np.array([future.positions[w,idx,slots[0]] for idx,_ in truth_targets]))
        if len(draws)==len(future.worlds):
            x=np.array(draws);y=np.array([truth[identity] for _,truth in truth_targets]);joint.append(x);matched.append(identity)
            values.append(vector_energy(x,y)/math.sqrt(len(truth_targets)))
    result={'path_es':float(np.mean(values)) if values else None,'path_targets':len(values)}
    H=len(truth_targets);indices=[i for i,_ in truth_targets]
    samples=[]
    for w in range(len(future.worlds)):
        p=future.positions[w,indices].transpose(1,0,2)/math.sqrt(H)
        p=p[np.isfinite(p).all(axis=(1,2))];samples.append(p.reshape(len(p),H*2))
    truth_path=np.stack([t for _,t in truth_targets],axis=1)/math.sqrt(H)
    truth_path=truth_path.reshape(len(truth_path),H*2)
    population_path=population_energy(samples,truth_path,dimension=H*2,return_parts=True)
    result['population_path_es3']=population_path['score']
    result['joint_population_path_diversity']=population_path['mean_pairwise_distance']
    if len(values)==len(ids) and values:
        result['joint_path_es']=vector_energy(np.stack(joint,axis=1),np.stack([np.array([t[j] for _,t in truth_targets]) for j in matched]))/math.sqrt(len(truth_targets)*len(ids))
    return result


def model_set(checkpoint=None):
    population=json.loads((OUT/'population_prior.json').read_text());motion=json.loads((OUT/'motion.json').read_text())
    return dict(A_visible_cv=load_model(population=population,observation=OPTICS,motion={'maneuver_rate':0.,'reset_std':0.},kind='visible_cv'),
        B_memory_cv=load_model(population=population,observation=OPTICS,motion={'maneuver_rate':0.,'reset_std':0.},kind='memory_cv'),
        B_historical_cv=load_model(population=population,observation=OPTICS,motion={'maneuver_rate':0.,'reset_std':0.},kind='memory_cv',cv_diffusion=.64),
        C_frozen_graph=load_model(OLD,population,OPTICS,{'maneuver_rate':2.,'reset_std':2.,'maneuver_retention':1.}),
        C_matched_motion=load_model(OLD,population,OPTICS,motion),
        D_trained_graph=load_model(checkpoint or OUT/'training/checkpoint.pt',population,OPTICS,motion))


def run(limit=0,checkpoint=None,cache=None,tag=''):
    checkpoint=Path(checkpoint) if checkpoint else OUT/'training/checkpoint.pt'
    models=model_set(checkpoint);cache=np.load(cache or OUT/'dev.npy',allow_pickle=True).tolist()
    if limit:cache=cache[:limit]
    rows=[];paths=[];filtering=[];conditioning=[];observations=[];failures=[];started=time.perf_counter()
    destination=OUT/('evaluation_'+tag if tag else 'evaluation_smoke' if limit else 'evaluation');destination.mkdir(exist_ok=False)
    for ri,cached in enumerate(cache):
        rid=cached['root_id'];folder=SNAP/'roots'/rid;l=read(folder/'legal_root.json.gz');meta=dict(root_id=rid,scene_id=cached['scene_id'])
        futures={};beliefs={}
        try:
            # Predictions use arrived legal state only; offline truth is read below.
            for name,model in models.items():
                b=restore(model,cached,l['partial_graph'],name.startswith('A_'));beliefs[name]=b
                clock=np.unique(np.r_[0.,np.arange(1,63)*.048,HORIZONS,3.])
                futures[name]=model.predict_shared(b,3.,16,41,times=clock)
            with np.load(folder/'privileged_future.npz') as f:times=f['timestamps'].copy();truth=f['positions'][...,[0,2]].copy()
            ids=visible_ids(l,truth[0]);targets=[]
            common_root_maps=source_maps(cached['filtered_hypotheses'],cached['timestamp'],truth[0],aged=True)
            common_known=set(j for m in common_root_maps for j in m.values())
            for horizon in HORIZONS:
                target=truth_at(times,truth,cached['timestamp']+horizon)
                if target is None:continue
                for name,future in futures.items():
                    index=int(np.argmin(abs(future.times-(cached['timestamp']+horizon))))
                    score=forecast_score(future,index,target,truth[0],ids)
                    # Velocity truth is finite-difference native motion, not a
                    # simulator velocity channel or privileged identity online.
                    jj=min(max(1,int(np.searchsorted(times,cached['timestamp']+horizon))),len(times)-1)
                    true_velocity=(truth[jj]-truth[jj-1])/(times[jj]-times[jj-1])
                    mapping=maps_for(future,truth[0]);velocity_errors=[];velocity_coverage=[];speeds=[]
                    for w,h in enumerate(future.worlds):
                        speeds.extend(np.linalg.norm(future.velocities[w,index][np.isfinite(future.velocities[w,index]).all(-1)],axis=-1).tolist())
                    for identity in ids:
                        vv=[]
                        for w,(h,m) in enumerate(zip(future.worlds,mapping)):
                            ii=[i for i,s in enumerate(h.slot_ids) if m.get(int(s))==identity]
                            if len(ii)==1:vv.append(future.velocities[w,index,ii[0]])
                        if len(vv)==16:
                            v=np.array(vv);err=np.linalg.norm(v.mean(0)-true_velocity[identity]);rad=np.quantile(np.linalg.norm(v-v.mean(0),axis=-1),.9)
                            velocity_errors.append(err);velocity_coverage.append(err<=rad)
                    score.update(velocity_error=float(np.mean(velocity_errors)) if velocity_errors else None,
                        velocity_coverage90=float(np.mean(velocity_coverage)) if velocity_coverage else None,
                        predicted_speed_p95=float(np.quantile(speeds,.95)) if speeds else None,
                        predicted_speed_over3=float(np.mean(np.array(speeds)>3)) if speeds else None,
                        true_speed_over3=float(np.mean(np.linalg.norm(true_velocity,axis=-1)>3)) if len(true_velocity) else None,
                        true_speed_p95=float(np.quantile(np.linalg.norm(true_velocity,axis=-1),.95)) if len(true_velocity) else None)
                    aged_maps=maps_for(future,truth[0],aged=True)
                    aged_ids=sorted(common_known-set(ids))
                    if aged_ids:
                        aged_score=forecast_score(future,index,target,truth[0],aged_ids,ancestry=aged_maps)
                        score['aged_tracked_es']=aged_score['tracked_es'];score['aged_target_support']=aged_score['tracked_supported']
                    unseen=sorted(set(range(len(truth[0])))-common_known)
                    hidden_samples=[future.positions[w,index,h.n:][np.isfinite(future.positions[w,index,h.n:]).all(-1)] for w,h in enumerate(future.worlds)]
                    score['never_localized_es3']=population_energy(hidden_samples,target[unseen])
                    score['never_localized_targets']=len(unseen)
                    rows.append({**meta,'model':name,'horizon':horizon,**count_scores(beliefs[name],len(target)),**score})
                targets.append((int(np.argmin(abs(futures['D_trained_graph'].times-(cached['timestamp']+horizon)))),target))
                for name,future in futures.items():
                    paths.append({**meta,'model':name,'horizon':horizon,**path_scores(future,targets,ids,maps_for(future,truth[0]))})
            wm=models['D_trained_graph'];b=beliefs['D_trained_graph']
            diag=b.evidence.get('normalized_approximation',{});weights=np.array([h.weight for h in b.hypotheses])
            h=max(b.hypotheses,key=lambda h:h.weight);localized=gospa(h.mean[:,:2],truth[0])
            filtering.append({**meta,'phase':'root','timestamp':b.timestamp,**count_scores(b,len(truth[0])),
                'count_pmf':count_pmf(b).tolist(),
                'tracks':sum(x.weight*x.n for x in b.hypotheses),'ess':float(1/(weights@weights)),
                'retention_product':diag.get('retention_product',1.),'maximum_local_omission':diag.get('maximum_local_omission',0.),
                'residual_mass':b.unresolved_mass,'localized_ghosts':localized['false_localized'],
                'localized_missed':localized['missed_localized'],'history_length':cached['history_length'],
                'association_ambiguous':association_ambiguous(b),
                'expected_localized_ghosts':sum(h.weight*gospa(h.mean[:,:2],truth[0])['false_localized'] for h in b.hypotheses),
                'empty':not bool(ids),'dense':len(ids)>=3,'aged':bool(np.any(b.timestamp-h.last_seen>.2))})
            qs=read(folder/'queries.json.gz');outcomes=read(folder/'outcomes.json.gz');nodes=read(folder/'future_observations.json.gz')
            qi=max(range(len(qs)),key=lambda i:(len(qs[i]['compiled']['actions']),qs[i]['query_id']))
            outcome=outcomes[qi];node_ids=outcome['node_ids']
            # Sequential actual filtering on a metadata-fixed bounded cohort;
            # all development roots receive the root-history evaluation above.
            if ri<240:
                origin_maps=source_maps(b.hypotheses,b.timestamp,truth[0],aged=True)
                tagged=[DiagnosticHypothesis(**{f.name:getattr(h,f.name) for f in fields(Hypothesis)},diagnostic_origin=i) for i,h in enumerate(b.hypotheses)]
                state=b.fork(hypotheses=tagged);ever=set(j for m in origin_maps for j in m.values());last_visible=set(ids)
                for step,node_id in enumerate(node_ids[:30]):
                    ob=nodes[node_id]['observation'];t=float(ob['timestamp']);target_now=truth_at(times,truth,t)
                    if target_now is None:break
                    prior_n=sum(h.weight*h.n for h in state.hypotheses)
                    next_id=max((int(h.slot_ids.max()) for h in state.hypotheses if h.n),default=-1)+1
                    try:state=wm.observe(state,ob)
                    except Exception as e:
                        failures.append({**meta,'operation':'sequential_observe','error':repr(e),'step':step});break
                    w=np.array([h.weight for h in state.hypotheses]);diag=state.evidence.get('normalized_approximation',{})
                    exact_error=0.;false_birth=0.;false_recovery=0.;matched_support=0.;current_visible=set();det_ids=[]
                    detections=sorted([r for r in ob['tracks'] if r['visible_now'] and r.get('valid',True)],key=lambda r:tuple(r['position']))
                    for track in detections:
                        d=np.linalg.norm(target_now-np.asarray(track['position'])[[0,2]],axis=-1)
                        j=int(d.argmin()) if len(d) else -1;det_ids.append(j if j>=0 and d[j]<1e-5 else None)
                    current_visible={j for j in det_ids if j is not None}
                    traces=state.evidence.get('association_alternatives',[]) if state.evidence.get('last_sensor_timestamp')==t else []
                    for h,trace in zip(state.hypotheses,traces):
                        mapping=origin_maps[h.diagnostic_origin]
                        for slot,j in trace['matched_slots']:
                            if mapping.get(slot) is not None and det_ids[j] is not None:
                                matched_support+=h.weight
                                if mapping[slot]!=det_ids[j]:false_recovery+=h.weight
                        for slot in trace['new_slots']:
                            j=slot-next_id
                            if 0<=j<len(det_ids) and det_ids[j] is not None:
                                if det_ids[j] in ever:false_birth+=h.weight
                                mapping[slot]=det_ids[j]
                        for track in detections:
                            z=np.asarray(track['position'])[[0,2]]
                            exact_error=max(exact_error,float(np.linalg.norm(h.mean[:,:2]-z,axis=-1).min()))
                    filtering.append({**meta,'phase':'future_arrived','timestamp':t,**count_scores(state,len(target_now)),
                        'tracks':sum(h.weight*h.n for h in state.hypotheses),'ess':float(1/(w@w)),
                        'retention_product':diag.get('retention_product',1.),'maximum_local_omission':diag.get('maximum_local_omission',0.),
                        'residual_mass':state.unresolved_mass,'empty':not any(r['visible_now'] for r in ob['tracks']),
                        'dense':sum(r['visible_now'] for r in ob['tracks'])>=3,
                        'association_ambiguous':association_ambiguous(state),
                        'expected_localized_ghosts':sum(h.weight*gospa(h.mean[:,:2],target_now)['false_localized'] for h in state.hypotheses),
                        'exact_coordinate_error':exact_error,'false_birth_mass':false_birth,
                        'false_recovery_mass':false_recovery,'identity_diagnostic_supported_matches':matched_support,
                        'reappearance':bool((current_visible&ever)-last_visible),
                        'track_persistence_delta':sum(h.weight*h.n for h in state.hypotheses)-prior_n})
                    ever|=current_visible;last_visible=current_visible
            if node_ids:
                selected=min(9,len(node_ids)-1);ob=nodes[node_ids[selected]]['observation'];t=float(ob['timestamp'])
                origin_maps=source_maps(b.hypotheses,b.timestamp,truth[0])
                tagged=[DiagnosticHypothesis(**{f.name:getattr(h,f.name) for f in fields(Hypothesis)},diagnostic_origin=i) for i,h in enumerate(b.hypotheses)]
                tagged_b=b.fork(hypotheses=tagged)
                if tagged_b.fingerprint()!=b.fingerprint():raise AssertionError('diagnostic origin tag changed physical belief')
                arrived=wm.observe(tagged_b,ob);nohuman=wm.observe(tagged_b,{**ob,'human_sensor_available':False})
                # Both receive identical robot/static arrivals and share target T.
                for horizon in (2.,1.):
                    T=b.timestamp+horizon;target=truth_at(times,truth,T)
                    if target is None or t>=T:continue
                    scores=[]
                    for state in (nohuman,arrived):
                        f=wm.predict_shared(state,T-t,16,107,times=[0.,T-t])
                        now=truth_at(times,truth,t)
                        scores.append({**forecast_score(f,1,target,now,ids,ancestry=carried_maps(f,origin_maps)),**count_scores(state,len(target))})
                    old,new=scores
                    gains={k+'_gain':old[k]-new[k] if old.get(k) is not None and new.get(k) is not None else None for k in ('population_es3','tracked_es','count_nll','gospa','coverage90')}
                    gains['coverage90_change']=new['coverage90']-old['coverage90'] if old['coverage90'] is not None and new['coverage90'] is not None else None
                    gains.pop('coverage90_gain')
                    root_known=set(j for m in maps_for(futures['D_trained_graph'],truth[0],aged=True) for j in m.values())
                    detected_ids=[];now=truth_at(times,truth,t)
                    for track in ob['tracks']:
                        if track['visible_now'] and len(now):
                            distance=np.linalg.norm(now-np.asarray(track['position'])[[0,2]],axis=-1);j=int(distance.argmin())
                            if distance[j]<1e-5:detected_ids.append(j)
                    conditioning.append({**meta,'target_horizon':horizon,'observation_time':t-b.timestamp,
                        'positive':any(r['visible_now'] for r in ob['tracks']),
                        'sensor_available':ob.get('human_sensor_available',True) and ob.get('observation_valid',True),
                        'dense':sum(r['visible_now'] for r in ob['tracks'])>=3,
                        'aged':any(np.any(t-h.last_seen>.2) for h in b.hypotheses),
                        'reappearance':bool((set(detected_ids)&root_known)-set(ids)),
                        'first_localization':bool(set(detected_ids)-root_known),
                        'prior':old,'posterior':new,**gains})
            # Recorded endpoint labels, fixed predeclared query, all observed
            # endpoints up to3s. Evaluate at their ACTUAL absolute clocks.
            obs=[nodes[j]['observation'] for j in node_ids if nodes[j]['observation']['timestamp']<=b.timestamp+3.+1e-8]
            if obs:
                clock=np.unique(np.r_[0.,[r['timestamp']-b.timestamp for r in obs]])
                f=wm.predict_shared(b,float(clock[-1]),16,43,times=clock)
                poses=np.array([r['robot_pose'] for r in obs]);indices=np.searchsorted(f.times,np.array([r['timestamp'] for r in obs])-1e-8)
                pd=wm.observation_law.detection(f.positions[:,indices],poses,np.array([h.visibility_mode for h in f.worlds]))
                none=np.prod(1-pd,axis=-1);per=1-np.mean(none,axis=0);labels=np.array([any(t['visible_now'] and t.get('valid',True) for t in r['tracks']) for r in obs])
                avail=np.array([r.get('observation_valid',True) and r.get('human_sensor_available',True) for r in obs])
                for j in np.flatnonzero(avail):
                    p=float(per[j]);y=bool(labels[j]);lik=p if y else 1-p
                    observations.append({**meta,'kind':'recorded_pose_endpoint','time':obs[j]['timestamp']-b.timestamp,'probability':p,'observed':int(y),
                        'brier':(p-y)**2,'nll':-math.log(lik) if lik>0 else None,'infinite_nll':lik==0})
                none[:,~avail]=1.
                noall=np.prod(none,axis=1);pany=1-float(noall.mean());yany=bool(labels[avail].any())
                survival=np.cumprod(none,axis=1);before=np.c_[np.ones(len(f.worlds)),survival[:,:-1]]
                first_mass=np.mean(before*(1-none),axis=0)
                detected=np.flatnonzero(labels&avail)
                first_likelihood=float(first_mass[detected[0]]) if len(detected) else float(noall.mean())
                event_likelihood=pany if yany else float(noall.mean())
                observations.append({**meta,'kind':'recorded_pose_prefix_any','time':float(clock[-1]),'probability':pany,'observed':int(yany),
                    'none_probability':float(noall.mean()),
                    'brier':(pany-yany)**2,'nll':-math.log(event_likelihood) if event_likelihood>0 else None,
                    'infinite_nll':event_likelihood==0,
                    'first_detection_nll':-math.log(first_likelihood) if first_likelihood>0 else None,
                    'infinite_first_detection_nll':first_likelihood==0,
                    'first_distribution_normalization_error':abs(float(first_mass.sum()+noall.mean())-1.)})
                # Public forward law uses only legal compiled robot poses and
                # its declared nominal native schedule, not future recorded poses.
                planned=predict_observations(wm,b,qs[qi],futures['D_trained_graph'])
                if planned['successor_supported']:
                    pp=planned['endpoint_any_probability'];nn=min(len(pp),len(obs))
                    for j in range(nn):
                        if not avail[j]:continue
                        p=float(pp[j]);y=bool(labels[j]);lik=p if y else 1-p
                        observations.append({**meta,'kind':'planned_query_endpoint','time':(j+1)*.048,
                            'native_clock_error_s':obs[j]['timestamp']-b.timestamp-(j+1)*.048,
                            'probability':p,'observed':int(y),'brier':(p-y)**2,
                            'nll':-math.log(lik) if lik else None,'infinite_nll':lik==0})
                    if nn==len(pp) and avail[:nn].all():
                        p=planned['any_probability'];y=bool(labels[:nn].any());lik=p if y else planned['none_probability']
                        observations.append({**meta,'kind':'planned_query_complete_any','probability':p,'observed':int(y),
                            'none_probability':planned['none_probability'],
                            'brier':(p-y)**2,'nll':-math.log(lik) if lik else None,'infinite_nll':lik==0})
                    elif nn==len(pp):
                        observations.append({**meta,'kind':'planned_query_complete_unavailable',
                            'status':'complete-prefix score unsupported with unavailable endpoint; never negative human evidence'})
                else:observations.append({**meta,'kind':'planned_query_unsupported','status':planned['status']})
        except Exception as e:
            import traceback
            failures.append({**meta,'operation':'root_evaluation','error':repr(e),'traceback':traceback.format_exc()})
        if (ri+1)%50==0:
            print(json.dumps(dict(roots=ri+1,forecasts=len(rows),pairs=len(conditioning),failures=len(failures),seconds=time.perf_counter()-started)),flush=True)
            (destination/'progress.json').write_text(json.dumps(dict(completed=ri+1,failures=failures[-5:]),indent=2))
    raw=dict(forecasts=rows,paths=paths,filtering=filtering,conditioning=conditioning,observations=observations,failures=failures)
    for key,value in raw.items():(destination/(key+'.json')).write_text(json.dumps(value,indent=2))
    summary=summarize(raw)
    summary.update(seconds=time.perf_counter()-started,requested_roots=len(cache),failures=len(failures),
        checkpoint_sha256=sha(checkpoint),manifest_sha256=sha(SNAP/'manifest.json'))
    (destination/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary['primary_contrasts'],indent=2),flush=True)


def summarize(raw):
    forecasts=raw['forecasts'];paths=raw['paths'];filtering=raw['filtering'];pairs=raw['conditioning'];obs=raw['observations']
    names=sorted({r['model'] for r in forecasts});result={'forecasting':{},'paths':{},'conditioning':{},'primary_contrasts':{}}
    keys=('population_es3','gospa','count_mae','count_nll','tracked_es','tracked_mean_error','coverage90','radius90',
        'tracked_targets','tracked_supported','tracked_missing','joint_tracked_es','localization_squared','cardinality_squared',
        'false_population','missed_population','velocity_error','velocity_coverage90','predicted_speed_over3','true_speed_over3',
        'aged_tracked_es','never_localized_es3','cross_human_predicted_covariance_trace',
        'cross_human_observed_residual_product','cross_human_covariance_calibration_difference')
    for name in names:
        result['forecasting'][name]={str(h):{key:scene_ci([r for r in forecasts if r['model']==name and r['horizon']==h],key) for key in keys} for h in HORIZONS}
        for horizon in HORIZONS:
            subset=[r for r in forecasts if r['model']==name and r['horizon']==horizon]
            inf=sum(r.get('infinite_count_nll',False) for r in subset)
            if inf:result['forecasting'][name][str(horizon)]['count_nll']={'status':'INFINITE_WITH_ZERO_SUPPORT',
                'infinite_roots':inf,'finite_support_summary':result['forecasting'][name][str(horizon)]['count_nll']}
        result['paths'][name]={str(h):{key:scene_ci([r for r in paths if r['model']==name and r['horizon']==h],key) for key in ('path_es','joint_path_es','population_path_es3','joint_population_path_diversity')} for h in HORIZONS}
    table={(r['root_id'],r['horizon'],r['model']):r for r in forecasts}
    for baseline in ('A_visible_cv','B_memory_cv','B_historical_cv','C_frozen_graph','C_matched_motion'):
        contrasts=[]
        for r in forecasts:
            if r['model']!='D_trained_graph':continue
            other=table.get((r['root_id'],r['horizon'],baseline))
            if other:contrasts.append({**r,'difference':r['population_es3']-other['population_es3']})
        result['primary_contrasts']['D_minus_'+baseline]=scene_ci(contrasts,'difference')
    result['filtering']={key:scene_ci([r for r in filtering if r['phase']=='root'],key) for key in ('count_nll','count_mae','ess','retention_product','maximum_local_omission','localized_ghosts','localized_missed')}
    result['filtering']['infinite_count_nll_roots']=sum(r.get('infinite_count_nll',False) for r in filtering if r['phase']=='root')
    if result['filtering']['infinite_count_nll_roots']:
        result['filtering']['count_nll']={'status':'INFINITE_WITH_ZERO_SUPPORT; finite-only summary is conditional',
            'infinite_roots':result['filtering']['infinite_count_nll_roots'],
            'finite_support_summary':result['filtering']['count_nll']}
    result['filtering']['sequential_updates']=sum(r['phase']=='future_arrived' for r in filtering)
    result['filtering_strata']={}
    for phase in ('root','future_arrived'):
        result['filtering_strata'][phase]={}
        for label,predicate in [('all',lambda r:True),('no_detection',lambda r:r.get('empty',False)),
                ('dense',lambda r:r.get('dense',False)),('aged',lambda r:r.get('aged',False)),
                ('reappearance',lambda r:r.get('reappearance',False)),
                ('ambiguous',lambda r:r.get('association_ambiguous',False)),
                ('long_history',lambda r:r.get('history_length',0)>=30)]:
            subset=[r for r in filtering if r['phase']==phase and predicate(r)]
            result['filtering_strata'][phase][label]={key:scene_ci(subset,key) for key in (
                'count_mae','ess','retention_product','maximum_local_omission','expected_localized_ghosts',
                'false_birth_mass','false_recovery_mass','exact_coordinate_error','track_persistence_delta')}
            result['filtering_strata'][phase][label]['infinite_count_nll_updates']=sum(r.get('infinite_count_nll',False) for r in subset)
    for horizon in (2.,1.):
        result['conditioning'][str(horizon)]={}
        for name,predicate in [('all',lambda r:True),('positive',lambda r:r['positive']),('no_detection',lambda r:not r['positive']),('dense',lambda r:r['dense']),('aged',lambda r:r['aged']),('reappearance',lambda r:r['reappearance']),('first_localization',lambda r:r['first_localization'])]:
            subset=[r for r in pairs if r['target_horizon']==horizon and predicate(r)]
            result['conditioning'][str(horizon)][name]={k:scene_ci(subset,k) for k in ('population_es3_gain','tracked_es_gain','count_nll_gain','gospa_gain','coverage90_change')}
            result['conditioning'][str(horizon)][name]['count_nll_support']={
                'prior_infinite':sum(r['prior']['infinite_count_nll'] for r in subset),
                'posterior_infinite':sum(r['posterior']['infinite_count_nll'] for r in subset),
                'undefined_gain':sum(r['count_nll_gain'] is None for r in subset),
                'gain_interpretation':'finite-support paired contrast; undefined infinities excluded and counted explicitly'}
    endpoint=[r for r in obs if r.get('kind')=='planned_query_endpoint']
    result['observations']={kind:{k:scene_ci([r for r in obs if r.get('kind')==kind],k) for k in ('brier','nll','probability','observed','first_detection_nll')}
        for kind in ('recorded_pose_endpoint','recorded_pose_prefix_any','planned_query_endpoint','planned_query_complete_any')}
    for kind,summary in result['observations'].items():
        subset=[r for r in obs if r.get('kind')==kind]
        for key in ('nll','first_detection_nll'):
            unsupported=sum(key in r and r[key] is None for r in subset)
            summary[key+'_infinite_records']=unsupported
            if unsupported:summary[key]={'status':'INFINITE_OR_ZERO_SUPPORT','infinite_records':unsupported,
                'finite_support_summary':summary[key]}
    result['observations']['calibration']=[dict(lower=i/10,upper=(i+1)/10,n=len(part),
        predicted=float(np.mean([r['probability'] for r in part])) if part else None,
        observed=float(np.mean([r['observed'] for r in part])) if part else None)
        for i in range(10) for part in [[r for r in endpoint if int(min(9,r['probability']*10))==i]]]
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--checkpoint');p.add_argument('--cache');p.add_argument('--tag',default='');run(**vars(p.parse_args()))
