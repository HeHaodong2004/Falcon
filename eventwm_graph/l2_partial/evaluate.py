"""Scene-clustered forecast, conditioning and counterfactual decision evaluation."""
import argparse,collections,json,pathlib,time,hashlib,subprocess
import numpy as np
from scipy.optimize import linear_sum_assignment
from .data import rows,root_data,read
from .model import GraphBeliefWorldModel
from .neural import LearnedDynamics
from .queries import NativeEventModel,event_features

HORIZONS=[.5,1.,2.,3.]

def weighted_energy(samples,truth,weights):
    w=np.asarray(weights);w=w/w.sum()
    first=np.einsum('w,wn->n',w,np.linalg.norm(samples-truth[None],axis=-1))
    pair=np.linalg.norm(samples[:,None]-samples[None,:],axis=-1)
    # Proper Energy Score of the explicit finite quadrature distribution.
    return float(np.mean(first-.5*np.einsum('i,j,ijn->n',w,w,pair)))

def truth_at(data,t):
    times=data['times'];truth=data['truth'];idx=int(np.searchsorted(times,t))
    if idx>=len(times):return None
    if idx==0:return truth[0]
    a=(t-times[idx-1])/(times[idx]-times[idx-1]);return (1-a)*truth[idx-1]+a*truth[idx]

def visible_ids(data):
    visible=[r['position'] for r in data['legal']['history'][-1]['tracks'] if r['visible_now']]
    if not visible or not data['truth'].shape[1]:return []
    p=np.asarray(visible)[:,[0,2]];d=np.linalg.norm(p[:,None]-data['truth'][None,0],axis=-1);ii,jj=linear_sum_assignment(d)
    return [int(j) for i,j in zip(ii,jj) if d[i,j]<1e-5]

def score_future(future,target,ids,initial_truth,absolute_time):
    if not ids or future.weights.sum()<=0:return None
    predictions=[];ww=[]
    ti=int(np.argmin(abs(future.times-absolute_time)))
    if abs(future.times[ti]-absolute_time)>1e-5:raise ValueError('score timestamp mismatch')
    for w,h in enumerate(future.worlds):
        if h.n<len(ids):continue
        distance=np.linalg.norm(future.positions[w,0,:h.n,None]-initial_truth[None,ids],axis=-1)
        # Offline scoring alignment only. Never exposed to the online model.
        ii,jj=linear_sum_assignment(distance);mapping={int(j):int(i) for i,j in zip(ii,jj)}
        if len(mapping)!=len(ids):continue
        predictions.append(future.positions[w,ti,[mapping[j] for j in range(len(ids))]]);ww.append(future.weights[w])
    if not predictions or sum(ww)<=0:return None
    pred=np.asarray(predictions);weights=np.asarray(ww);weights/=weights.sum();truth=target[ids]
    mean=np.einsum('w,wnc->nc',weights,pred);error=np.linalg.norm(mean-truth,axis=-1)
    dist=np.linalg.norm(pred-mean[None],axis=-1);coverage=[];spread=[]
    for j in range(len(ids)):
        order=np.argsort(dist[:,j]);cdf=np.cumsum(weights[order]);radius=dist[order[min(len(order)-1,np.searchsorted(cdf,.9))],j]
        coverage.append(error[j]<=radius);spread.append(radius)
    return {'energy_score':weighted_energy(pred,truth,weights),'mean_error':float(error.mean()),'coverage90':float(np.mean(coverage)),
            'radius90':float(np.mean(spread)),'conditional_scored_mass':float(sum(ww)),'visible_targets':len(ids)}

def scene_ci(records,key,seed=73,replicates=1000):
    grouped=collections.defaultdict(list)
    for r in records:
        if r.get(key) is not None and np.isfinite(r[key]):grouped[r['scene_id']].append(r[key])
    values=np.array([np.mean(v) for v in grouped.values()]);rng=np.random.default_rng(seed)
    if not len(values):return {'mean':None,'lo':None,'hi':None,'roots':0,'scenes':0}
    bootstrap=values[rng.integers(0,len(values),(replicates,len(values)))].mean(1)
    return {'mean':float(values.mean()),'lo':float(np.percentile(bootstrap,2.5)),'hi':float(np.percentile(bootstrap,97.5)),
            'one_sided_upper95':float(np.percentile(bootstrap,95)),'roots':len({r['root_id'] for r in records}),'scenes':len(values)}

def condition_pair(wm,b,data,ids):
    if not ids:return None
    qs=data['queries'];index=max(range(len(qs)),key=lambda i:(len(qs[i]['compiled']['actions']),qs[i]['query_id']))
    q=qs[index];o=data['outcomes'][index];j=10
    if len(o['node_ids'])<=j or data['times'][-1]<b.timestamp+2.-1e-8:return None
    node=data['nodes'][o['node_ids'][j-1]];observation=node['observation'];t=float(observation['timestamp']);T=b.timestamp+2.
    if t>=T:return None
    # Both arms receive the SAME executed-action and survival likelihood.
    actual_times=data['times'][:j+1];f=wm.predict_shared(b,t-b.timestamp,16,seed=91,times=actual_times-b.timestamp)
    hidden=np.array([h.expected_hidden for h in f.worlds]);poses=np.asarray(q['compiled']['planned_poses'][:j+1]);x=event_features(f.positions,f.velocities,poses,hidden,0.,actual_times-b.timestamp)
    hazard=wm.event_model.hazards(x)[:,1:,0];total=hidden+np.array([h.n for h in f.worlds]);hazard[total==0]=0
    survival=np.prod(1-hazard,axis=1)
    prior=wm.propagate(b,t)
    def signature(h):return (tuple(h.slot_ids),tuple(np.round(h.intent.ravel(),5)),tuple(h.mode))
    sums=collections.defaultdict(lambda:[0.,0.])
    for h,w,s in zip(f.worlds,f.weights,survival):
        key=signature(h);sums[key][0]+=w*s;sums[key][1]+=w
    likelihood=np.array([sums[signature(h)][0]/sums[signature(h)][1] if sums[signature(h)][1]>0 else .5 for h in prior.hypotheses])
    # Mixture-weight survival projection keeps analytic within-component
    # covariance; it approximates survival-truncated shapes identically in arms.
    neutral={'timestamp':t,'robot_pose':observation['robot_pose'],'tracks':[],'static_updates':[],'observation_valid':True}
    prior=wm.observe(prior,neutral,{'world_likelihood':likelihood,'unresolved_likelihood':.995**j})
    posterior=wm.observe(prior,observation)
    target=truth_at(data,T);initial=truth_at(data,t)
    a=wm.predict_shared(prior,T-t,16,seed=107,times=[0.,T-t]);c=wm.predict_shared(posterior,T-t,16,seed=107,times=[0.,T-t])
    sa=score_future(a,target,ids,initial,T);sc=score_future(c,target,ids,initial,T)
    if sa is None or sc is None:
        return {'prior':None if sa is None else sa['energy_score'],'posterior':None if sc is None else sc['energy_score'],'delta':None,
                'positive':sum(r['visible_now'] for r in observation['tracks'])>0,'occluded_or_ambiguous':False,'reappearance_proxy':False,'aged_tracks':False,'dense':max(h.n for h in b.hypotheses)>=3,
                'unsupported_pair':True,'prior_unresolved':a.unresolved_mass,'posterior_unresolved':c.unresolved_mass,'target_time':2.}
    positive=sum(r['visible_now'] for r in observation['tracks']);aged=any(np.any(t-h.last_seen>.5) for h in prior.hypotheses if h.n)
    rootvisible={r['track_id'] for r in data['legal']['history'][-1]['tracks'] if r['visible_now']}
    reappearance=any(r['visible_now'] and r['track_id'] not in rootvisible for r in observation['tracks'])
    informative=any(np.min(np.linalg.norm(initial[ids]-np.asarray(r['position'])[[0,2]],axis=-1))<1e-5 for r in observation['tracks'] if r['visible_now'])
    return {'prior':sa['energy_score'],'posterior':sc['energy_score'],'delta':sc['energy_score']-sa['energy_score'],
            'positive':informative,'occluded_or_ambiguous':positive==0,'reappearance_proxy':reappearance,'aged_tracks':aged,
            'dense':max(h.n for h in b.hypotheses)>=3,'observation_time':t-b.timestamp,'target_time':2.,
            'prior_unresolved':a.unresolved_mass,'posterior_unresolved':c.unresolved_mass,
            'survival_projection':'component-weight projection with analytic conditional covariance'}

def evaluate(snapshot,split,checkpoint,population,event_checkpoint,output,limit=0,cv_diffusion=.04):
    learned=LearnedDynamics.load(checkpoint);prior=json.loads(pathlib.Path(population).read_text());event=NativeEventModel(json.loads(pathlib.Path(event_checkpoint).read_text())['coefficients'])
    models={name:GraphBeliefWorldModel(kind=kind,dynamics=learned if name in ('learned_graph','structured_learned') else None,population_prior=prior,event_model=event)
            for name,kind in [('visible_cv','visible_cv'),('memory_cv','memory_cv'),('structured_kinematic','structured'),('learned_graph','structured'),('structured_learned','structured')]}
    rootrows=rows(snapshot,split);rootrows=rootrows[:limit] if limit else rootrows
    forecasts=[];forecast_failures=[];pairs=[];queries=[];consistency=[];started=time.perf_counter()
    for ri,(meta,folder) in enumerate(rootrows):
        data=root_data(folder,True);l=data['legal'];base=models['memory_cv'].initialize(l['history'],l['partial_graph']);ids=visible_ids(data)
        stored={}
        for name,wm in models.items():
            b=base
            if name=='visible_cv':
                hs=[]
                for h in b.hypotheses:
                    mask=np.isclose(h.last_seen,b.timestamp);hh=h.fork(mean=h.mean[mask].copy(),covariance=h.covariance[mask].copy(),last_seen=h.last_seen[mask],slot_ids=h.slot_ids[mask],height=h.height[mask] if h.height is not None else None)
                    for i,p in enumerate(hh.mean[:,:2]):
                        tracks=[r for r in l['history'][-1]['tracks'] if r['visible_now'] and np.linalg.norm(np.asarray(r['position'])[[0,2]]-p)<1e-5]
                        if tracks and tracks[0]['velocity_valid']:hh.mean[i,2:]=np.asarray(tracks[0]['estimated_velocity'])[[0,2]]
                        else:hh.mean[i,2:]=0.
                    hs.append(hh)
                b=b.fork(hypotheses=hs)
            elif name=='learned_graph':
                h=max(b.hypotheses,key=lambda h:h.weight);b=b.fork(hypotheses=[h.fork(weight=1.)],unresolved_mass=0.)
            if name in ('visible_cv','memory_cv'):
                b=wm.prepare(b);b=b.fork(hypotheses=[h.fork(diffusion=np.full(h.n,cv_diffusion)) for h in b.hypotheses])
            times=np.unique(np.r_[np.arange(63)*.048,HORIZONS,3.]);f=wm.predict_shared(b,3.,16,seed=41,times=times)
            for horizon in HORIZONS:
                truth=truth_at(data,b.timestamp+horizon)
                if truth is None:continue
                score=score_future(f,truth,ids,data['truth'][0],b.timestamp+horizon)
                if score:
                    score.update(root_id=meta['root_id'],scene_id=meta['scene_id'],model=name,horizon=horizon,unresolved_mass=f.unresolved_mass)
                    forecasts.append(score)
                elif ids:forecast_failures.append({'root_id':meta['root_id'],'scene_id':meta['scene_id'],'model':name,'horizon':horizon,'unresolved_mass':f.unresolved_mass})
            if name in ('memory_cv','structured_learned'):
                batch=wm.evaluate_queries(b,data['queries'],f);stored[name]=(b,f,batch)
        wm=models['structured_learned'];b,f,pred=stored['structured_learned']
        pair=condition_pair(wm,b,data,ids)
        if pair:pairs.append({**pair,'root_id':meta['root_id'],'scene_id':meta['scene_id']})
        primary=max(b.hypotheses,key=lambda h:h.weight) if b.hypotheses else None;known_ids=set()
        if primary is not None and primary.n and data['truth'].shape[1]:
            distance=np.linalg.norm(primary.mean[:,None,:2]-data['truth'][None,0],axis=-1);ii,jj=linear_sum_assignment(distance)
            known_ids={int(j) for i,j in zip(ii,jj) if distance[i,j]<1.5}
        for qi,(q,o,p) in enumerate(zip(data['queries'],data['outcomes'],pred.predictions)):
            n=min(len(o['node_ids']),62);labels=read(folder/'privileged_step_labels.json.gz') if qi==0 else labels
            labs=[labels[i] for i in o['node_ids'][:n]];elapsed=float(data['times'][n]-b.timestamp) if n else 0.
            # Binary risk is evaluated at the planned horizon, never at an
            # outcome-dependent collision time. Early contact establishes the
            # label; administrative noncontact truncation is right-censored.
            collision=any(v['human_mesh_contact'] for v in labs);psc=any(v['psc_violation'] for v in labs)
            pc=p.collision_probability;pp=p.psc_probability
            planned_n=len(p.time_to_conflict['collision_first_mass'])
            risk_evaluable=bool(collision or n>=planned_n)
            cv=stored['memory_cv'][2].predictions[qi];cc=cv.collision_probability;cp=cv.psc_probability
            nodes=[data['nodes'][i]['observation']['robot_pose'] for i in o['node_ids'][:n]];poses=np.asarray([l['history'][-1]['robot_pose'],*nodes]);progress=float(np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1).sum())
            predicted_duration=float(p.duration_samples@p.duration_weights);predicted_progress=float(p.progress_samples@p.duration_weights)
            appearance=False
            for j,nid in enumerate(o['node_ids'][:n]):
                for track in data['nodes'][nid]['observation']['tracks']:
                    if not track['visible_now'] or not data['truth'].shape[1]:continue
                    distance=np.linalg.norm(data['truth'][j+1]-np.asarray(track['position'])[[0,2]],axis=1);identity=int(distance.argmin())
                    if distance[identity]<1e-5 and identity not in known_ids:appearance=True
            actual_reward=progress-5*collision-psc-.1*elapsed
            queries.append({'root_id':meta['root_id'],'scene_id':meta['scene_id'],'query_id':q['query_id'],'collision':int(collision),'psc':int(psc),'pred_collision':pc,'pred_psc':pp,
                'cv_collision':cc,'cv_psc':cp,'duration':elapsed,'pred_duration':predicted_duration,'progress':progress,'pred_progress':predicted_progress,
                'appearance':int(appearance),'pred_appearance':float(sum(p.time_to_conflict['appearance_first_mass'])),'actual_reward':actual_reward,'pred_reward':predicted_progress-5*p.collision_interval[1]-p.psc_interval[1]-.1*predicted_duration,
                'cv_reward':float(cv.progress_samples@cv.duration_weights)-5*cv.collision_interval[1]-cv.psc_interval[1]-.1*float(cv.duration_samples@cv.duration_weights),
                'depth0_reward':q['compiled'].get('route_distance',q.get('route_distance',0.))-.1*min(3,len(q['compiled']['actions'])*.048),
                'unresolved':p.unresolved_uncertainty,'complete':p.query_complete,'actual_complete':o['summary']['query_complete'],
                'blockage':p.blockage_probability,'actual_blockage':o['summary']['termination']=='route_invalidated','censored_steps':n,
                'planned_steps':planned_n,'risk_evaluable':risk_evaluable,'actual_termination':o['summary']['termination']})
        if ri%10==0 and pred.predictions:
            p=pred.predictions[0];idx=p.diagnostics['endpoint_index'];expected=sum(w*h.expected_hidden for w,h in zip(f.weights,f.worlds))
            marginal=sum(br.probability*sum(h.weight*h.expected_hidden for h in br.successor.hypotheses) for br in p.observation_branches if br.observation['kind']!='unresolved')
            consistency.append({'root_id':meta['root_id'],'scene_id':meta['scene_id'],'branch_probability_error':abs(sum(br.probability for br in p.observation_branches)-1),
                                'count_marginal_error_retained':abs(expected-marginal),'unresolved_mass':f.unresolved_mass})
        if (ri+1)%50==0:print(split,ri+1,'elapsed',round(time.perf_counter()-started),flush=True)
    summary={name:{str(h):scene_ci([r for r in forecasts if r['model']==name and r['horizon']==h],'energy_score') for h in HORIZONS} for name in models}
    obs={}
    for category in ('overall','positive','occluded_or_ambiguous','reappearance_proxy','aged_tracks','dense'):
        subset=pairs if category=='overall' else [r for r in pairs if r[category]];s=scene_ci(subset,'delta')
        s['unsupported_pairs']=sum(r.get('unsupported_pair',False) for r in subset)
        s['decision']='INCONCLUSIVE' if s['unsupported_pairs'] else ('INCONCLUSIVE' if s['scenes']<30 or s['roots']<100 else ('IMPROVES' if s['one_sided_upper95']<0 else ('NON_INFERIOR' if s['one_sided_upper95']<.020 else 'NOT_ESTABLISHED')))
        obs[category]=s
    value={'provenance':{'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in pathlib.Path(__file__).parent.glob('*.py')},'checkpoint_sha256':hashlib.sha256(pathlib.Path(checkpoint).read_bytes()).hexdigest(),'event_checkpoint_sha256':hashlib.sha256(pathlib.Path(event_checkpoint).read_bytes()).hexdigest()},'split':split,'models':summary,'observation_conditioning':obs,'forecasts':forecasts,'forecast_failures':forecast_failures,'conditioning_pairs':pairs,'queries':queries,'consistency':consistency,
           'elapsed_s':time.perf_counter()-started,'roots':len(rootrows),'cv_diffusion':cv_diffusion,'score':'proper finite-quadrature Energy Score; conditional on retained mass',
           'population_prior_source':'train roots only','decision_rule':'path progress - 5*collision upper bound - PSC upper bound - .1*duration; actual reward uses native labels',
           'scope':'designed collection curriculum; no natural-policy or closed-loop RL evaluation'}
    pathlib.Path(output).write_text(json.dumps(value,indent=2));print(json.dumps({'models':summary,'conditioning':obs},indent=2));return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('split');p.add_argument('checkpoint');p.add_argument('population');p.add_argument('event_checkpoint');p.add_argument('output');p.add_argument('--limit',type=int,default=0);p.add_argument('--cv-diffusion',type=float,default=.04);args=p.parse_args();evaluate(**vars(args))
