"""Persistent graph belief, analytic observations, and shared latent futures."""
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math,time,hashlib,json
import numpy as np
from .state import *
from .probability import advance,condition_position,transition_matrices

class GraphBeliefWorldModel:
    def __init__(self,dynamics=None,kind='structured',max_hypotheses=6,population_prior=None,event_model=None):
        self.dynamics=dynamics;self.kind=kind;self.max_hypotheses=max_hypotheses
        self.population_prior=normalized(population_prior if population_prior is not None else [3.**i/math.factorial(i)+1e-6 for i in range(17)])
        self.event_model=event_model;self.graph_cache=OrderedDict();self.timings={}

    def history_filter(self):
        return GraphBeliefWorldModel(kind='memory_cv',max_hypotheses=self.max_hypotheses,population_prior=self.population_prior)

    def initial_hypotheses(self,h):return [h]

    def advance_hypothesis(self,h,dt,start):return advance(h,dt)

    def advance_hypotheses(self,hypotheses,dt,start):
        return [self.advance_hypothesis(h,dt,start) for h in hypotheses]

    def condition_track(self,h,i,z):return condition_position(h.mean[i],h.covariance[i],z)

    def initialize(self,legal_history: Sequence[Dict[str, Any]],observed_graph: Dict[str, Any]) -> BeliefState:
        if not legal_history:raise ValueError('empty legal history')
        graph=ObservedGraph('empty',0,float(observed_graph['resolution']),float(observed_graph['height']),{})
        h=Hypothesis(1.,np.empty((0,4)),np.empty((0,4,4)),np.empty(0),np.empty(0,int),self.population_prior.copy())
        filter_model=self.history_filter()
        b=BeliefState(graph,self.initial_hypotheses(h),float(legal_history[0]['timestamp']),np.asarray(legal_history[0]['robot_pose'],float),
                      model_version=getattr(filter_model,'model_version',MODEL_VERSION))
        for row in legal_history:b=filter_model.observe(b,row)
        b=b.fork(hypotheses=[h.fork(intent=None,diffusion=None,mode=None) for h in b.hypotheses])
        # Snapshot is legal at the final history time; no snapshot geometry was
        # used while assimilating earlier observations.
        g=ObservedGraph.from_snapshot(observed_graph)
        if abs(float(observed_graph['timestamp'])-b.timestamp)>1e-6:raise ValueError('future graph snapshot')
        b=b.fork(graph=g)
        if self.kind=='visible_cv':
            hs=[]
            for h in b.hypotheses:
                mask=np.isclose(h.last_seen,b.timestamp)
                hs.append(h.fork(mean=h.mean[mask],covariance=h.covariance[mask],last_seen=h.last_seen[mask],slot_ids=h.slot_ids[mask],height=h.height[mask] if h.height is not None else None))
            b=b.fork(hypotheses=hs)
        return b.validate()

    def prepare(self,belief: BeliefState) -> BeliefState:
        if belief.terminal:return belief
        hs=[]
        for h in belief.hypotheses:
            if h.intent is not None and not np.any(h.mode<0):hs.append(h);continue
            if self.kind in ('visible_cv','memory_cv'):
                hs.append(h.fork(intent=h.mean[:,2:].copy(),diffusion=np.full(h.n,.16),mode=np.zeros(h.n,int),relaxation=0.));continue
            if self.dynamics is not None and h.n:
                targets,q,probs=self.dynamics.parameters_for(h,belief.graph,belief.timestamp)
            else:
                v=h.mean[:,2:];targets=np.stack([v,.05*v,np.stack([-v[:,1],v[:,0]],axis=-1)],axis=1)
                q=np.full((h.n,3),.25);probs=np.tile([.72,.18,.10],(h.n,1))
            if h.intent is not None:
                existing=h.mode>=0
                targets[existing]=h.intent[existing,None,:];q[existing]=h.diffusion[existing,None]
                probs[existing]=0
                probs[np.flatnonzero(existing),h.mode[existing].astype(int)]=1
            probs=np.asarray(probs,dtype=float)
            if h.n:probs=probs/probs.sum(axis=1,keepdims=True)
            # Store a distribution of persistent intentions. Sampling occurs only
            # when converting this predictive mixture to shared quadrature worlds.
            hs.append(h.fork(intent=targets,diffusion=q,mode=probs,relaxation=.7))
        return belief.fork(hypotheses=hs)

    def propagate(self,belief: BeliefState,timestamp: float) -> BeliefState:
        if timestamp<belief.timestamp-1e-9:raise ValueError('time reversal')
        if belief.terminal:return belief
        b=self.prepare(belief);hs=[];dt=timestamp-b.timestamp
        for h in b.hypotheses:
            if h.intent.ndim==3:
                # Common-U copula: exact categorical marginals, shared intention
                # uncertainty. Expansion is needed only once until real sensing.
                edges=np.unique(np.clip(np.r_[0.,np.cumsum(h.mode,axis=1).ravel(),1.],0.,1.))
                for lo,hi in zip(edges[:-1],edges[1:]):
                    if hi-lo<1e-12:continue
                    modes=(np.cumsum(h.mode,axis=1)<(lo+hi)/2).sum(1).clip(0,2)
                    idx=np.arange(h.n);hh=h.fork(weight=h.weight*(hi-lo),intent=h.intent[idx,modes],diffusion=h.diffusion[idx,modes],mode=modes)
                    hs.append(hh)
            else:hs.append(h)
        # Bound expanded joint intentions; all omitted mass remains unresolved.
        # This is a declared approximation, never normalized into false certainty.
        unresolved=b.unresolved_mass
        if len(hs)>16:
            hs.sort(key=lambda h:-h.weight);unresolved+=sum(h.weight for h in hs[16:]);hs=hs[:16]
        # A transition preserves component weight. Advance only the components
        # retained by the identical weight-only pruning rule; omitted mass is
        # already explicit and no candidate state influences its selection.
        hs=self.advance_hypotheses(hs,dt,b.timestamp)
        return b.fork(hypotheses=hs,timestamp=float(timestamp),unresolved_mass=min(1.,unresolved))

    def observe(self,belief: BeliefState,actual_observation: Dict[str, Any],actual_execution_trace: Optional[Dict[str, Any]]=None) -> BeliefState:
        started=time.perf_counter();row=actual_observation;t=float(row['timestamp'])
        if t<belief.timestamp-1e-8:raise ValueError('old observation')
        if belief.terminal:return belief
        if not row.get('observation_valid',True):return self.propagate(belief,t)
        # Cached native observations are evidence once, not independent redraws.
        # Ignore tracker IDs and arbitrary extra payload fields in this identity.
        empty_boxes=sorted(set(tuple(np.asarray(box,float).ravel()) for box in row.get('certified_visible_empty_boxes',[])))
        payload={'timestamp':t,'robot_pose':row['robot_pose'],
                 'positions':sorted(tuple(r['position']) for r in row.get('tracks',[]) if r.get('visible_now') and r.get('valid',True)),
                 'static_updates':row.get('static_updates',[]),'empty_boxes':empty_boxes,
                 'execution':{k:v for k,v in (actual_execution_trace or {}).items() if k in ('world_likelihood','unresolved_likelihood','episode_terminated')}}
        observation_key=hashlib.sha256(json.dumps(payload,sort_keys=True,default=lambda x:np.asarray(x).tolist()).encode()).hexdigest()
        if belief.evidence.get('last_observation_key')==observation_key:
            self.timings['observe']=(time.perf_counter()-started)*1000
            return belief
        # Actual trace survival uses the identical query-event likelihood when supplied.
        if actual_execution_trace and 'world_likelihood' in actual_execution_trace:
            likelihood=np.asarray(actual_execution_trace['world_likelihood']);hs=belief.hypotheses
            if len(likelihood)!=len(hs):raise ValueError('survival law dimension')
            weights=np.array([h.weight for h in hs])*likelihood
            residual=belief.unresolved_mass*float(actual_execution_trace.get('unresolved_likelihood',.5))
            evidence=float(weights.sum()+residual)
            if evidence>0:belief=belief.fork(hypotheses=[h.fork(weight=float(w/evidence)) for h,w in zip(hs,weights)],unresolved_mass=residual/evidence)
            else:belief=belief.fork(hypotheses=[h.fork(weight=0.) for h in hs],unresolved_mass=1.,evidence={**belief.evidence,'unsupported_survival':True})
        b=self.propagate(belief,t);graph=b.graph.update(row.get('static_updates',[]))
        detections=[r for r in row.get('tracks',[]) if r.get('visible_now') and r.get('valid',True)]
        detections.sort(key=lambda r:tuple(r['position']))
        for r in detections:
            if abs(r['last_seen_time']-t)>1e-6:raise ValueError('visible detection clock')
        # Source tracker identity/estimated velocity is deliberately not evidence.
        pool=[(h,set()) for h in b.hypotheses];u=b.unresolved_mass;pruned=0.;unsupported=False
        next_id=max((int(h.slot_ids.max()) for h,_ in pool if h.n),default=-1)+1
        for j,r in enumerate(detections):
            z=np.asarray(r['position'])[[0,2]];candidates=[]
            for h,used in pool:
                available=h.n-len(used);counts=np.arange(len(h.hidden_pmf));den=available+counts
                inv=np.divide(1.,den,out=np.zeros_like(den,dtype=float),where=den>0)
                track_factor=float(h.hidden_pmf@inv)
                for i in range(h.n):
                    if i in used:continue
                    conditioned,logp=self.condition_track(h,i,z)
                    if conditioned is None:continue
                    mean=h.mean.copy();cov=h.covariance.copy();mean[i],cov[i]=conditioned
                    seen=h.last_seen.copy();seen[i]=t
                    heights=h.height.copy() if h.height is not None else np.full(h.n,graph.height);heights[i]=r['position'][1]
                    pmf=h.hidden_pmf*inv
                    if pmf.sum()<=0:continue
                    hh=h.fork(mean=mean,covariance=cov,last_seen=seen,height=heights,hidden_pmf=pmf/pmf.sum())
                    candidates.append((math.log(max(h.weight,1e-300))+math.log(track_factor)+logp,hh,used|{i}))
                transfer=h.hidden_pmf*counts*inv;birth_factor=float(transfer.sum())
                if birth_factor>0:
                    pmf=np.zeros_like(transfer);pmf[:-1]=transfer[1:]/birth_factor
                    mean=np.concatenate((h.mean,[[*z,0.,0.]]));cov=np.concatenate([h.covariance,np.diag([0.,0.,1.,1.])[None]])
                    hh=h.fork(mean=mean,covariance=cov,last_seen=np.append(h.last_seen,t),slot_ids=np.append(h.slot_ids,next_id+j),height=np.append(h.height if h.height is not None else np.full(h.n,graph.height),r['position'][1]),hidden_pmf=pmf,intent=np.concatenate((h.intent,np.zeros((1,2)))),diffusion=np.append(h.diffusion,.25),mode=np.append(h.mode,-1))
                    # Declared broad sensor-centered birth density, no hidden precise location.
                    distance=np.linalg.norm(z-np.asarray(row['robot_pose'])[[0,2]])
                    log_density=-math.log(2*math.pi*100)-distance**2/200
                    candidates.append((math.log(max(h.weight,1e-300))+math.log(birth_factor)+log_density,hh,used|{h.n}))
            if not candidates:
                unsupported=True;u=1.;pool=[(h.fork(weight=0.),used) for h,used in pool];break
            log_residual=(math.log(u)-math.log(2*math.pi*100)-np.linalg.norm(z-np.asarray(row['robot_pose'])[[0,2]])**2/200) if u>0 else -np.inf
            maximum=max(max(c[0] for c in candidates),log_residual)
            weights=np.exp([c[0]-maximum for c in candidates]);residual=np.exp(log_residual-maximum)
            evidence=float(weights.sum()+residual);weights/=evidence;u=float(residual/evidence)
            # Rao-Blackwell moment compression only within identical associations
            # and exact observed coordinates. It preserves mass and first two
            # marginal moments; cross-human covariance compression is declared.
            groups={}
            for i,(_,hh,used) in enumerate(candidates):
                key=(tuple(hh.slot_ids),tuple(sorted(used)),tuple(np.round(hh.mean[list(sorted(used)),:2].ravel(),9)),tuple(hh.mode.tolist()),tuple(np.round(hh.intent.ravel(),7)) if hh.relaxation else ())
                groups.setdefault(key,[]).append(i)
            merged=[]
            for members in groups.values():
                if len(members)==1:
                    i=members[0];mass=float(weights[i])
                    if mass>1e-300:merged.append((mass,candidates[i][1],candidates[i][2]))
                    continue
                ww=weights[members];mass=float(ww.sum())
                if mass<=1e-300:continue
                ww=ww/mass
                base=candidates[members[0]][1];means=np.stack([candidates[i][1].mean for i in members]);mean=np.einsum('i,ijk->jk',ww,means)
                delta=means-mean;cov=np.einsum('i,ijkl->jkl',ww,np.stack([candidates[i][1].covariance for i in members])+delta[:,:,:,None]*delta[:,:,None,:])
                pmf=np.einsum('i,ij->j',ww,np.stack([candidates[i][1].hidden_pmf for i in members]))
                merged.append((mass,base.fork(mean=mean,covariance=cov,hidden_pmf=pmf),candidates[members[0]][2]))
            merged.sort(key=lambda x:-x[0]);lost=sum(c[0] for c in merged[self.max_hypotheses:])
            pruned+=lost;u=min(1.,max(0.,u+lost))
            keep=merged[:self.max_hypotheses];mass=sum(c[0] for c in keep)
            pool=[(hh.fork(weight=float((1-u)*w/mass)),used) for w,hh,used in keep]
        hs=[h for h,_ in pool]
        # Explicit synthetic certificates only. Actual collector has none.
        empty=[np.asarray(box).reshape(2,2).tolist() for box in empty_boxes]
        empty_diagnostics={}
        if empty:
            from .probability import condition_empty_quadrature
            hs,u,empty_diagnostics=condition_empty_quadrature(hs,empty,u)
        result=b.fork(graph=graph,hypotheses=hs,timestamp=t,robot_pose=np.asarray(row['robot_pose'],float),unresolved_mass=u,
                      evidence={**b.evidence,**empty_diagnostics,'last_detection_count':len(detections),'pruned_mass':b.evidence.get('pruned_mass',0.)+pruned,
                                'unsupported_observation':unsupported,'exact_detections':[r['position'] for r in detections],'last_observation_key':observation_key,
                                'unresolved_observation_law':'Gaussian sensor-centered sigma10m clutter/recovery component','negative_evidence_available':bool(empty)})
        if actual_execution_trace and actual_execution_trace.get('episode_terminated'):result=result.fork(terminal=True)
        result.validate()
        self.timings['observe']=(time.perf_counter()-started)*1000
        return result

    def predict_shared(self,belief: BeliefState,horizon: float,num_worlds: int=16,seed: int=0,times: Optional[Sequence[float]]=None) -> SharedFuture:
        started=time.perf_counter()
        if horizon<0 or num_worlds<1:raise ValueError('invalid prediction budget')
        b=self.propagate(belief,belief.timestamp) if not belief.terminal else belief
        rng=np.random.default_rng(seed)
        rel=np.asarray(times if times is not None else np.r_[0.,np.arange(1,int(math.floor(horizon/.048))+1)*.048,horizon],float)
        rel=np.unique(rel)
        if b.terminal:rel=np.array([0.])
        if rel[0]!=0 or np.any(np.diff(rel)<=0):raise ValueError('invalid future clock')
        count_hypotheses=[]
        for h in b.hypotheses:
            for count,p in enumerate(h.hidden_pmf):
                if p<=0:continue
                pmf=np.zeros_like(h.hidden_pmf);pmf[count]=1.
                count_hypotheses.append(h.fork(weight=h.weight*p,hidden_pmf=pmf))
        if count_hypotheses:b=b.fork(hypotheses=count_hypotheses)
        if not any(h.weight>0 for h in b.hypotheses):
            detections=np.asarray(b.evidence.get('exact_detections',[]),float).reshape(-1,3)
            n=len(detections);mean=np.c_[detections[:,[0,2]],np.zeros((n,2))]
            h=Hypothesis(0.,mean,np.tile(np.diag([0,0,1,1]),(n,1,1)),np.full(n,b.timestamp),np.arange(n),np.array([1.]),
                         intent=np.zeros((n,2)),diffusion=np.ones(n),mode=np.zeros(n),height=detections[:,1])
            b=b.fork(hypotheses=[h],unresolved_mass=1.)
        weights=np.array([h.weight for h in b.hypotheses]);positive=np.flatnonzero(weights>0)
        selected=sorted(positive,key=lambda i:-weights[i])[:num_worlds]
        omitted=float(weights.sum()-weights[selected].sum());unresolved=min(1.,b.unresolved_mass+max(0.,omitted))
        if not selected:selected=[0]
        retained=float(weights[selected].sum());allocation=np.ones(len(selected),int)
        remaining=num_worlds-len(selected)
        if remaining and retained>0:
            ideal=remaining*weights[selected]/retained;base=np.floor(ideal).astype(int);allocation+=base
            for j in np.argsort(-(ideal-base))[:remaining-int(base.sum())]:allocation[j]+=1
        elif remaining:allocation[0]+=remaining
        indices=np.repeat(selected,allocation)
        sample_weights=np.concatenate([np.full(n,weights[i]/n) for i,n in zip(selected,allocation)])
        worlds=[];n=max((h.n for h in b.hypotheses),default=0)
        positions=np.full((num_worlds,len(rel),n,2),np.nan);velocities=positions.copy()
        for w,index in enumerate(indices):
            h=b.hypotheses[index];mean=h.mean.copy()
            if h.intent is None:h=h.fork(intent=h.mean[:,2:].copy(),diffusion=np.zeros(h.n),mode=np.zeros(h.n,int))
            if hasattr(self,'sample_initial'):
                mean=self.sample_initial(h,rng)
            elif h.n:
                eig,U=np.linalg.eigh(h.covariance);mean+=np.einsum('nij,nj->ni',U,rng.normal(size=(h.n,4))*np.sqrt(np.maximum(eig,0)))
            if h.intent.ndim==3:
                modes=(np.cumsum(h.mode,axis=1)<rng.random()).sum(1).clip(0,2);ii=np.arange(h.n)
                intent=h.intent[ii,modes];q=h.diffusion[ii,modes]
            else:modes=h.mode;intent=h.intent;q=h.diffusion
            hidden=int(rng.choice(len(h.hidden_pmf),p=h.hidden_pmf));pmf=np.zeros_like(h.hidden_pmf);pmf[hidden]=1
            hh=h.fork(mean=mean,covariance=np.zeros_like(h.covariance),weight=float(sample_weights[w]),intent=intent.copy(),diffusion=q.copy(),mode=modes.copy(),hidden_pmf=pmf)
            worlds.append(hh);positions[w,0,:h.n]=mean[:,:2];velocities[w,0,:h.n]=mean[:,2:]
        if hasattr(self,'sample_maneuver_paths'):
            positions,velocities,worlds=self.sample_maneuver_paths(worlds,rel,b.timestamp,rng,n)
        else:
            # Vectorized process across worlds/humans. Native time axis remains
            # sequential so one continuous stochastic path is shared across queries.
            means=np.zeros((num_worlds,n,4));targets=np.zeros((num_worlds,n,2));diff=np.zeros((num_worlds,n))
            valid=np.zeros((num_worlds,n),bool);rates=np.array([h.relaxation for h in worlds])
            for w,h in enumerate(worlds):
                means[w,:h.n]=h.mean;targets[w,:h.n]=h.intent;diff[w,:h.n]=h.diffusion;valid[w,:h.n]=True
            factors={}
            for step,dt in enumerate(np.diff(rel),1):
                for rate in np.unique(rates):
                    select=np.flatnonzero(rates==rate);key=(round(float(dt),12),float(rate))
                    if key not in factors:
                        F,Q,a,e=transition_matrices(float(dt),float(rate));factors[key]=(F,np.linalg.cholesky(Q+np.eye(4)*1e-16),a,e)
                    F,L,a,e=factors[key];v=means[select]@F.T
                    v[:,:,:2]+=(dt-a)*targets[select];v[:,:,2:]+=(1-e)*targets[select]
                    v+=(rng.normal(size=v.shape)@L.T)*np.sqrt(diff[select,:,None])
                    if rate>0:v[:,:,:2]+=rng.normal(size=v[:,:,:2].shape)*np.sqrt(.0025*dt)
                    means[select]=v
                positions[:,step]=np.where(valid[...,None],means[:,:,:2],np.nan);velocities[:,step]=np.where(valid[...,None],means[:,:,2:],np.nan)
            worlds=[h.fork(mean=means[w,:h.n].copy()) for w,h in enumerate(worlds)]
        self.timings['forecast']=(time.perf_counter()-started)*1000
        return SharedFuture(b.timestamp+rel,positions,velocities,sample_weights,worlds,b,unresolved,seed,belief.fingerprint())

    def _endpoint_hypothesis(self,future,index,w,h,cov):
        mean=np.empty((h.n,4));mean[:,:2]=future.positions[w,index,:h.n];mean[:,2:]=future.velocities[w,index,:h.n]
        return h.fork(mean=mean,covariance=cov)

    def successor_at(self,future: SharedFuture,index: int,world_weights: np.ndarray,robot_pose: np.ndarray,terminal: bool=False,cursor: Optional[Dict[str, Any]]=None,refinement=None) -> BeliefState:
        weights=np.asarray(world_weights);mass=float(weights.sum());hs=[]
        rp,rv=getattr(self,'successor_regularization',(.005,.02))
        key=(getattr(self,'model_version',future.source.model_version),index,rp,rv)
        if key not in future.endpoint_cache:
            covs={n:np.tile(np.diag([rp**2,rp**2,rv**2,rv**2]),(n,1,1)) for n in {h.n for h in future.worlds}}
            templates=tuple(self._endpoint_hypothesis(future,index,w,h,covs[h.n]) for w,h in enumerate(future.worlds))
            # Forked branches share templates; their arrays must be immutable.
            # Seal inherited latent/count arrays as well as endpoint moments.
            for template in templates:
                for value in vars(template).values():
                    if isinstance(value,np.ndarray):value.setflags(write=False)
            future.endpoint_cache[key]=templates
        for w,h in enumerate(future.endpoint_cache[key]):
            changes={'weight':float(weights[w]/mass) if mass else 0.}
            if refinement is not None:
                changes.update(coarse_observation_time=float(future.times[index]),coarse_positive_refinement=float(refinement[0][w]),coarse_empty_refinement=float(refinement[1][w]))
            hs.append(h.fork(**changes))
        return BeliefState(future.source.graph,hs,float(future.times[index]),np.asarray(robot_pose),0. if mass else 1.,terminal,
                           model_version=future.source.model_version,evidence={'quadrature_worlds':len(hs),'position_regularization_m':rp},query_cursor=cursor)

    def evaluate_queries(self,belief: BeliefState,queries: Sequence[Dict[str, Any]],shared_future: SharedFuture,executor: str=EXECUTOR_VERSION,budget_ms: float=40.,refined_queries: int=4) -> QueryPredictionBatch:
        from .queries import evaluate
        return evaluate(self,belief,queries,shared_future,executor,budget_ms,refined_queries)

    def continue_rollout(self,successor_belief: BeliefState,query: Dict[str, Any],budget_ms: float=40.,horizon: float=3.,seed: int=0) -> QueryPrediction:
        f=self.predict_shared(successor_belief,horizon,seed=seed)
        return self.evaluate_queries(successor_belief,[query],f,budget_ms=budget_ms).predictions[0]

    def online_call(self,belief: BeliefState,actual_observation: Dict[str, Any],queries: Sequence[Dict[str, Any]],horizon: float=3.,num_worlds: int=16,seed: int=0) -> Tuple[BeliefState, QueryPredictionBatch]:
        start=time.perf_counter();b=self.observe(belief,actual_observation)
        f=self.predict_shared(b,horizon,num_worlds,seed);batch=self.evaluate_queries(b,queries,f)
        batch.timings_ms={**self.timings,**batch.timings_ms,'complete':(time.perf_counter()-start)*1000}
        return b,batch
