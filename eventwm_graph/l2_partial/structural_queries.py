"""Batched structural query path with explicit residual risk bounds."""
import time
import copy
import numpy as np
from dataclasses import replace
from .state import *
from .queries import NativeEventModel,_support,event_features_many
from .structural_kernels import event_features_many_fast
from .structural_budget import check_budget
from .structural_state import freeze_hypotheses, immutable_array

def query_fingerprint(query):
    value={k:query[k] for k in ('query_id','executor_version','waypoints','final_yaw')}
    value['compiled']={k:query['compiled'].get(k) for k in ('planned_poses','actions','executor_version','graph_revision')}
    return hashlib.blake2b(json.dumps(value,sort_keys=True,allow_nan=False,
        default=lambda v:np.asarray(v).tolist(),separators=(',',':')).encode(),digest_size=16).hexdigest()

def evaluate(model,belief,queries,future,executor,budget_ms,refined_queries,refined_query_indices=None):
    started=time.perf_counter();check_budget('queries:entry')
    if executor!=EXECUTOR_VERSION:raise ValueError('unknown executor version')
    if abs(future.source.timestamp-belief.timestamp)>1e-6 or future.source.graph.namespace!=belief.graph.namespace:raise ValueError('future/belief mismatch')
    if future.input_fingerprint!=belief.fingerprint():raise ValueError('shared future belongs to a different belief branch')
    if len(queries)>24:raise ValueError('candidate budget 24')
    event=model.event_model or NativeEventModel();out=[];geometry_ms=0.;branch_ms=0.
    hidden=np.array([h.expected_hidden for h in future.worlds]);world_weights=future.weights
    retained=float(world_weights.sum());u=future.unresolved_mass
    # Unknown physical outcomes retain the entire population support. One
    # immutable zero-weight template is shared safely across all candidates.
    selected=set(range(min(refined_queries,len(queries)))) if refined_query_indices is None else set(refined_query_indices)
    residual_templates=freeze_hypotheses(belief.hypotheses,weight=0.) if u and selected else ()
    prepared=[]
    for qi,query in enumerate(queries):
        check_budget('queries:prepare_candidate')
        if query.get('executor_version')!=executor:raise ValueError('query executor mismatch')
        cursor=belief.query_cursor or {};start_cursor=int(cursor.get('action_index',0)) if cursor.get('query_id')==query['query_id'] else 0
        allposes=np.asarray(query['compiled']['planned_poses'],float)
        if allposes.ndim!=2 or allposes.shape[1]!=4 or not len(allposes) or not np.isfinite(allposes).all():
            raise ValueError('finite compiled pose sequence required')
        actions=query['compiled'].get('actions')
        if actions is not None and (len(actions)!=len(allposes)-1 or any(a not in ('forward','left','right') for a in actions)):
            raise ValueError('motion query requires forward/left/right native primitives; no STOP')
        if (np.linalg.norm(allposes[start_cursor,:3]-belief.robot_pose[:3])>.03 or abs(np.arctan2(np.sin(allposes[start_cursor,3]-belief.robot_pose[3]),np.cos(allposes[start_cursor,3]-belief.robot_pose[3])))>.001) and not belief.terminal:raise ValueError('query must start at successor pose; recompile new routes upstream')
        remaining=len(allposes)-1-start_cursor
        n=min(remaining,int(np.floor((future.times[-1]-belief.timestamp+1e-8)/.048)),120-start_cursor)
        tg=time.perf_counter();supported=_support(model,belief.graph,query,start_cursor);geometry_ms+=(time.perf_counter()-tg)*1000
        if belief.terminal or not supported:n=0
        poses=allposes[start_cursor:start_cursor+n+1]
        times=np.arange(n+1)*.048
        # Native endpoints are never rounded to an earlier forecasting endpoint.
        indices=np.searchsorted(future.times,belief.timestamp+times-1e-8).clip(0,len(future.times)-1)
        if np.max(np.abs(future.times[indices]-(belief.timestamp+times)))>1e-6:raise ValueError('shared future must contain exact native action endpoints')
        prepared.append((query,start_cursor,allposes,remaining,n,supported,poses,times,indices))
    age=np.array([max(0.,belief.timestamp-h.last_seen.mean()) if h.n else 0. for h in future.worlds])
    clock_v2=getattr(model,'event_clock_version','')=='native_elapsed_age_v2'
    batched_hazards=None
    if prepared:
        maxn=max(p[4] for p in prepared);fulltimes=np.arange(maxn+1)*.048
        fullindices=np.searchsorted(future.times,belief.timestamp+fulltimes-1e-8).clip(0,len(future.times)-1)
        robots=np.stack([np.concatenate([p[6],np.repeat(p[6][-1:],maxn-p[4],axis=0)]) for p in prepared])
        event_age=age[:,None]+fulltimes[None,:] if clock_v2 else age[:,None]
        event_elapsed=fulltimes[None,:]+np.array([p[1] for p in prepared])[:,None]*.048 if clock_v2 else fulltimes
        check_budget('queries:event_features')
        feature_kernel=event_features_many_fast if model.optimized else event_features_many
        feats=feature_kernel(future.positions[:,fullindices],future.velocities[:,fullindices],robots,hidden,event_age,event_elapsed)
        batched_hazards=event.hazards(feats)[:,:,1:,:]
    for qi,(query,start_cursor,allposes,remaining,n,supported,poses,times,indices) in enumerate(prepared):
        check_budget('queries:candidate_risk')
        hz=batched_hazards[qi,:,:n]
        total=np.array([h.n for h in future.worlds])+hidden
        hz[total==0,:,:2]=0.;hz[hidden==0,:,2]=0.
        if n:
            hc,hp,ha=hz[...,0],hz[...,1],hz[...,2]
            survival=np.cumprod(1-hc,axis=1);before=np.c_[np.ones(len(hidden)),survival[:,:-1]]
            first=before*hc;pc=1-survival[:,-1]
            no_psc_before=np.c_[np.ones(len(hidden)),np.cumprod(1-hp,axis=1)[:,:-1]]
            first_psc=before*no_psc_before*hp;pp=first_psc.sum(1)
            hidden_appearance=1-np.prod(1-ha,axis=1)
            first_appearance=before*np.c_[np.ones(len(hidden)),np.cumprod(1-ha,axis=1)[:,:-1]]*ha
        else:
            survival=np.ones((len(hidden),1));first=np.empty((len(hidden),0));pc=np.zeros(len(hidden));pp=pc.copy();first_psc=first.copy();hidden_appearance=pc.copy();first_appearance=first.copy()
        collision_lower=float(world_weights@pc);psc_lower=float(world_weights@pp)
        # Omitted mass has no invented .005 contact/PSC law. Scalars are
        # explicitly conservative UPPER estimates; intervals retain the lower
        # represented-world contribution. These are conditional surrogate
        # bounds, not guarantees against native event-model statistical error.
        risk_u=u if n and supported and not belief.terminal else 0.
        collision=min(1.,collision_lower+risk_u);psc=min(1.,psc_lower+risk_u)
        collision_first_mass=world_weights@first
        psc_first_mass=world_weights@first_psc
        duration_weights=np.r_[collision_first_mass,max(0.,retained-collision_lower),u]
        duration_samples=np.r_[times[1:],times[-1],times[-1]]
        # Progress is distance traveled along the immutable plan, not navigation success.
        travel=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1))]
        progress=np.r_[travel[1:],travel[-1],0.]
        complete=n==remaining and supported and not belief.terminal
        endcursor={'query_id':query['query_id'],'action_index':start_cursor+n,'total_actions':len(allposes)-1}
        timeout=start_cursor+n>=120 and n<remaining
        termination={'human_collision':collision_lower,'query_complete':(retained-collision_lower) if complete else 0.,
                     'horizon_censored':(retained-collision_lower) if not complete and supported and not belief.terminal and not timeout else 0.,
                     'query_timeout':(retained-collision_lower) if timeout else 0.,
                     'route_invalidated':retained if not supported else 0.,'unresolved':u}
        if belief.terminal:termination={k:0. for k in termination};termination['absorbed']=1.
        elif not supported:termination={k:0. for k in termination};termination['route_invalidated']=1.
        elif not n:
            termination={k:0. for k in termination};termination['query_complete' if complete else 'horizon_censored']=1.
        branches=[];tb=time.perf_counter()
        check_budget('queries:successor_construction')
        if qi in selected:
            idx=int(indices[-1]);last=poses[-1]
            if belief.terminal:
                branches=[ObservationBranch(1.,{'kind':'terminal_absorbed'},belief)]
            elif not n:
                name='route_invalidated' if not supported else 'survived_no_detection'
                successor=belief.fork(query_cursor=copy.deepcopy(endcursor),robot_pose=last.copy(),evidence=copy.deepcopy(belief.evidence))
                branches=[ObservationBranch(1.,{'kind':name,'timestamp':belief.timestamp,
                    'probability_interval':[1.,1.],'no_executed_endpoints':True},successor)]
            else:
                if getattr(model,'observation_repair',False):
                    no_detection,no_before,no_last=model.observation_law.no_detection(future,indices,poses,return_parts=True)
                    any_visible=1-no_detection
                else:
                    # Coarsened observation channel. It does NOT certify empty space.
                    pos=future.positions[:,indices]
                    if pos.shape[2]:
                        d=pos[:,-1]-last[[0,2]];distance=np.linalg.norm(d,axis=-1)
                        forward=d[...,0]*np.cos(last[3])-d[...,1]*np.sin(last[3]);lateral=np.abs(d[...,0]*np.sin(last[3])+d[...,1]*np.cos(last[3]))
                        in_view=(forward>0)&(lateral<forward)&(distance<10)&np.isfinite(distance)
                        pv=np.where(in_view,.75,.03);pv=np.where(np.isfinite(distance),pv,0.)
                        any_visible=1-np.prod(1-pv,axis=1)
                    else:any_visible=np.zeros(len(hidden))
                    # Coarsened event: at least one detection anywhere along the
                    # executed prefix. Use trajectory visibility for tracked humans too.
                    if pos.shape[2]:
                        delta_all=pos-poses[None,:,None,[0,2]]
                        dist_all=np.linalg.norm(delta_all,axis=-1)
                        fw=delta_all[...,0]*np.cos(poses[None,:,None,3])-delta_all[...,1]*np.sin(poses[None,:,None,3])
                        lat=np.abs(delta_all[...,0]*np.sin(poses[None,:,None,3])+delta_all[...,1]*np.cos(poses[None,:,None,3]))
                        seen=np.any((fw[:,1:]>0)&(lat[:,1:]<fw[:,1:])&(dist_all[:,1:]<10),axis=1) if n else np.zeros((len(hidden),pos.shape[2]),bool)
                        pv=np.where(seen,.75,.03)*np.isfinite(pos[:,0,:,0]);any_visible=1-np.prod(1-pv,axis=1) if n else np.zeros(len(hidden))
                    any_visible=1-(1-any_visible)*(1-hidden_appearance)
                channels=[('survived_no_detection',(1-pc)*(1-any_visible),False),
                          ('survived_detection',(1-pc)*any_visible,False),('terminal_contact',pc,True)]
                for name,lik,terminal in channels:
                    ws=world_weights*lik;prob=float(ws.sum())
                    if prob<=0:continue
                    refinement=None
                    if getattr(model,'observation_repair',False) and not terminal and n:
                        if name=='survived_no_detection':
                            positive=np.zeros(len(ws));empty=1/np.maximum(no_last,1e-300)
                        else:
                            positive=1/np.maximum(any_visible,1e-300);empty=(1-no_before)/np.maximum(any_visible,1e-300)
                        refinement=(positive,empty)
                    successor=model.successor_at(future,idx,ws,last,terminal,cursor=endcursor,refinement=refinement)
                    if not terminal and getattr(successor,'carrier',None) is not None:
                        event=(name,float(future.times[idx]),tuple(float(x) for x in last),
                            tuple(float(x) for x in ((1-no_before)*no_last/np.maximum(any_visible,1e-300))) if name=='survived_detection' else ())
                        successor=successor.fork(carrier=replace(successor.carrier,observation_ledger=successor.carrier.observation_ledger+(event,)))
                    if terminal:
                        first_mass=world_weights@first;probabilities=first_mass/first_mass.sum()
                        successor=successor.fork(robot_pose=probabilities@poses[1:],query_cursor=None,terminal_event={
                            'times':(belief.timestamp+times[1:]).tolist(),'poses':poses[1:].tolist(),
                            'probabilities':probabilities.tolist(),
                            'probability_semantics':'conditional on represented component; not the full terminal posterior when residual mass is positive',
                            'represented_posterior_mass':1-successor.unresolved_mass,
                            'unresolved_stopping_time_mass':successor.unresolved_mass,
                            'semantics':'absorbed by predictive clock; actual stopping time and pose remain a distribution'})
                    obs={'kind':name,'timestamp':float(future.times[idx]),'coarsened':True,
                         'probability_interval':[prob,min(1.,prob+u)],
                         'probability_semantics':'represented lower mass; unknown residual allocation is coupled across physical branches',
                         'observation_interval':'entire executed prefix, root excluded',
                         'visibility_model':model.observation_law.version if getattr(model,'observation_repair',False) else 'uncertain_base_heading_sector_v1','certified_empty':False,
                         'terminal_event':successor.terminal_event,'discovery':'unknown_geometry_remains_unknown','detected_position_distribution':'successor mixture, no fabricated exact measurement'}
                    branches.append(ObservationBranch(prob,obs,successor))
                if u:
                    residual=belief.fork(timestamp=float(future.times[idx]),robot_pose=last.copy(),hypotheses=residual_templates,
                        unresolved_mass=1.,query_cursor=copy.deepcopy(endcursor),evidence=copy.deepcopy(belief.evidence))
                    if hasattr(residual,'carrier'):residual=residual.fork(carrier=None)
                    branches.append(ObservationBranch(u,{'kind':'unresolved_physical_outcome',
                        'timestamp':residual.timestamp,'semantics':'epistemic residual; NOT a fabricated survival/contact observation'},residual))
        branch_ms+=(time.perf_counter()-tb)*1000
        conflict={'times_s':times[1:].tolist(),'collision_first_mass':collision_first_mass.tolist(),'psc_first_mass':psc_first_mass.tolist(),'appearance_first_mass':(world_weights@first_appearance).tolist(),'unresolved_first_event_mass':risk_u}
        out.append(QueryPrediction(query['query_id'],collision,psc,(collision_lower,min(1.,collision_lower+risk_u)),(psc_lower,min(1.,psc_lower+risk_u)),
            duration_samples,duration_weights,progress,conflict,0. if supported else 1.,termination,branches,u,
            (belief.timestamp,float(future.times[indices[-1]])),complete,endcursor,
            model_version=future.source.model_version,
            diagnostics={'risk_semantics':'native-label hazard surrogate; reported scalar is conservative upper endpoint',
                         'probability_bound_scope':'omitted-weight accounting conditional on retained surrogate/quadrature; not global projection or native calibration guarantee',
                         'query_geometry_fingerprint':query_fingerprint(query),
                         'duration_last_bin':'unresolved probability; final sample is an interval upper endpoint, not an identified duration law',
                         'continuous_exact_refinement_supported':False,'hidden_appearance_probability':float(world_weights@hidden_appearance),
                         'appearance_semantics':'auxiliary root-untracked native-label proxy; separate from spatial observation law',
                         'observation_process_version':model.observation_law.version if getattr(model,'observation_repair',False) else 'legacy_query_visibility_v1',
                         'static_support_checked':supported,'survival_world_weights':(world_weights*(1-pc)).tolist(),'endpoint_index':int(indices[-1]),'unresolved_survival_weight_interval':[0.,u],'conditional_survival_probability_interval':[max(0.,retained-collision_lower),1-collision_lower],
                         'budget_ms':budget_ms,'budget_exceeded':False,'refined':qi in selected,
                         'population_spatially_unresolved':float(world_weights@hidden)>0}))
        # Paired responses retain common-human-future correlations. There is no
        # invented cross-query coupling of the native event-head randomness.
        out[-1].world_collision_probability=immutable_array(pc)
        out[-1].world_social_probability=immutable_array(pp)
    check_budget('queries:complete')
    elapsed=(time.perf_counter()-started)*1000
    for p in out:p.diagnostics.update(budget_exceeded=False,budget_semantics='cooperative stage checks; no hard real-time guarantee')
    return QueryPredictionBatch(out,{'queries':elapsed,'graph_support':geometry_ms,'branches':branch_ms},future)
