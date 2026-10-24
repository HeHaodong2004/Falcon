"""Bounded native-endpoint executor SURROGATE and common-world event inference.

Contacts/geodesic PSC are supervised hazards, never mislabeled disk geometry.
"""
import time
import numpy as np
from .state import *

FEATURE_NAMES=('bias','near_04','near_08','near_15','near_30','hidden_count','tracked_count','forward','speed','lateral','age','elapsed')

def event_features(positions,velocities,robot,hidden,age,elapsed):
    # positions W,T,N,2; robot T,4. Empty populations are legitimate.
    W,T,N,_=positions.shape
    if N:
        delta=positions-robot[None,:,None,[0,2]]
        d=np.linalg.norm(delta,axis=-1);d=np.where(np.isfinite(d),d,100.)
        idx=d.argmin(-1);minimum=d.min(-1)
        speed=np.take_along_axis(np.linalg.norm(velocities,axis=-1),idx[...,None],axis=-1)[...,0]
        speed=np.nan_to_num(speed)
        lat=np.abs(delta[...,0]*np.sin(robot[None,:,None,3])+delta[...,1]*np.cos(robot[None,:,None,3]))
        lateral=np.take_along_axis(lat,idx[...,None],axis=-1)[...,0];lateral=np.nan_to_num(lateral,nan=10.)
        count=np.isfinite(positions[...,0]).sum(-1)
    else:minimum=np.full((W,T),100.);speed=np.zeros((W,T));lateral=np.full((W,T),10.);count=np.zeros((W,T))
    forward=np.r_[False,np.linalg.norm(np.diff(robot[:,[0,2]],axis=0),axis=-1)>.05]
    return np.stack([np.ones((W,T)),*[np.exp(-minimum**2/(2*r*r)) for r in (.4,.8,1.5,3.)],
        np.broadcast_to(np.asarray(hidden)[:,None]/4,(W,T)),count/4,np.broadcast_to(forward,(W,T)),speed/3,
        np.exp(-lateral**2/.5),np.broadcast_to(np.minimum(np.asarray(age),5)/5,(W,T)),
        np.broadcast_to(np.asarray(elapsed)/3,(W,T))],axis=-1).astype(np.float32)

def event_features_many(positions,velocities,robots,hidden,age,elapsed):
    """Same features for Q immutable plans; all worlds/endpoints remain present."""
    Q,T,_=robots.shape;W,_,N,_=positions.shape;shape=(Q,W,T)
    if N:
        delta=positions[None]-robots[:,None,:,None,[0,2]]
        distance=np.linalg.norm(delta,axis=-1);distance=np.where(np.isfinite(distance),distance,100.)
        idx=distance.argmin(-1);minimum=distance.min(-1)
        speed=np.take_along_axis(np.linalg.norm(velocities,axis=-1)[None],idx[...,None],axis=-1)[...,0]
        speed=np.nan_to_num(speed)
        lateral=np.abs(delta[...,0]*np.sin(robots[:,None,:,None,3])+delta[...,1]*np.cos(robots[:,None,:,None,3]))
        lateral=np.take_along_axis(lateral,idx[...,None],axis=-1)[...,0];lateral=np.nan_to_num(lateral,nan=10.)
        count=np.isfinite(positions[...,0]).sum(-1)[None]
    else:minimum=np.full(shape,100.);speed=np.zeros(shape);lateral=np.full(shape,10.);count=np.zeros(shape)
    forward=np.concatenate([np.zeros((Q,1),bool),np.linalg.norm(np.diff(robots[:,:,[0,2]],axis=1),axis=-1)>.05],axis=1)
    elapsed=np.asarray(elapsed)
    if elapsed.ndim==2:elapsed=elapsed[:,None,:]
    return np.stack([np.ones(shape),*[np.exp(-minimum**2/(2*r*r)) for r in (.4,.8,1.5,3.)],
        np.broadcast_to(np.asarray(hidden)[None,:,None]/4,shape),np.broadcast_to(count/4,shape),
        np.broadcast_to(forward[:,None],shape),speed/3,np.exp(-lateral**2/.5),
        np.broadcast_to(np.minimum(np.asarray(age),5)/5,shape),np.broadcast_to(elapsed/3,shape)],axis=-1).astype(np.float32)

class NativeEventModel:
    def __init__(self,coefficients=None):
        if coefficients is None:
            coefficients=np.zeros((3,len(FEATURE_NAMES)));coefficients[:,0]=[-7.,-7.,-7.]
            coefficients[0,1:4]=[3.,2.,1.];coefficients[1,1:4]=[2.,3.,2.];coefficients[:,5]=1.
        self.coefficients=np.asarray(coefficients,dtype=float)
    def hazards(self,features):
        logits=features@self.coefficients.T
        return 1/(1+np.exp(-np.clip(logits,-30,30)))

def _support(model,graph,query,start_cursor=0):
    # The swept support footprint depends only on the immutable reference poses,
    # resolution and cursor. Its certificates still depend on the graph version.
    # Hash poses as well as the ID so accidental same-ID mutation cannot reuse a
    # support result. This hashes a bounded action trace, never the full graph.
    poses=np.asarray(query['compiled']['planned_poses'],dtype=float)[start_cursor:]
    r=float(graph.resolution)
    footprint_key=(query['query_id'],int(start_cursor),r,
                   hashlib.blake2b(poses.tobytes(),digest_size=16).digest())
    key=(graph.namespace,footprint_key)
    if key in model.graph_cache:
        model.graph_cache.move_to_end(key);return model.graph_cache[key]
    footprints=getattr(model,'support_footprint_cache',None)
    if footprints is None:
        from collections import OrderedDict
        footprints=model.support_footprint_cache=OrderedDict()
    if footprint_key in footprints:
        footprints.move_to_end(footprint_key)
        unique,ymin,ymax=footprints[footprint_key]
    else:
        # Preserve every endpoint, turn subdivision and forward swept sample.
        # Only aggregation of identical cell witnesses is optimized.
        samples=[poses]
        if len(poses)>1:
            angle=np.arctan2(np.sin(poses[1:,3]-poses[:-1,3]),np.cos(poses[1:,3]-poses[:-1,3]))
            subdivisions=max(8,int(np.ceil(np.max(np.abs(angle))/.005)))
            for a in np.arange(1,subdivisions)/subdivisions:
                p=(1-a)*poses[:-1]+a*poses[1:];p[:,3]=poses[:-1,3]+a*angle;samples.append(p)
        p=np.concatenate(samples);offset=np.linspace(-.646495,.584202,14)
        coords=p[:,None,[0,2]]+offset[None,:,None]*np.stack([np.cos(p[:,3]),-np.sin(p[:,3])],-1)[:,None,:]
        cells=np.floor(coords/r).astype(int).reshape(-1,2)
        packed=(cells[:,0].astype(np.int64)<<32)+(cells[:,1].astype(np.int64)&0xffffffff)
        if np.all(p[:,1]==p[0,1]):
            # Exact constant-height case: no inverse indices or repeated scatter.
            packed=np.unique(packed)
            ymin=np.full(len(packed),p[0,1]);ymax=ymin.copy()
        else:
            heights=np.broadcast_to(p[:,None,1],coords.shape[:2]).reshape(-1)
            order=np.argsort(packed);packed=packed[order];heights=heights[order]
            starts=np.empty(len(packed),bool);starts[0]=True;starts[1:]=packed[1:]!=packed[:-1]
            indices=np.flatnonzero(starts)
            ymin=np.minimum.reduceat(heights,indices);ymax=np.maximum.reduceat(heights,indices)
            packed=packed[indices]
        unique=np.stack([packed>>32,(packed&0xffffffff).astype(np.uint32).view(np.int32)],axis=-1)
        footprints[footprint_key]=(unique,ymin,ymax)
        # Bound both metadata cardinality and retained NumPy buffer storage.
        while len(footprints)>256 or sum(sum(a.nbytes for a in f) for f in footprints.values())>8*1024*1024:
            footprints.popitem(last=False)
    good=True
    for i,k in enumerate(unique):
        state,height=graph.get(k)
        if state!=1 or max(abs(ymin[i]-height),abs(ymax[i]-height))>.22:good=False;break
    model.graph_cache[key]=good
    while len(model.graph_cache)>256:model.graph_cache.popitem(last=False)
    return good

def evaluate(model,belief,queries,future,executor,budget_ms,refined_queries):
    started=time.perf_counter()
    if executor!=EXECUTOR_VERSION:raise ValueError('unknown executor version')
    if abs(future.source.timestamp-belief.timestamp)>1e-6 or future.source.graph.namespace!=belief.graph.namespace:raise ValueError('future/belief mismatch')
    if future.input_fingerprint!=belief.fingerprint():raise ValueError('shared future belongs to a different belief branch')
    if len(queries)>24:raise ValueError('candidate budget 24')
    event=model.event_model or NativeEventModel();out=[];geometry_ms=0.;branch_ms=0.
    hidden=np.array([h.expected_hidden for h in future.worlds]);world_weights=future.weights
    retained=float(world_weights.sum());u=future.unresolved_mass
    prepared=[]
    for qi,query in enumerate(queries):
        if query.get('executor_version')!=executor:raise ValueError('query executor mismatch')
        cursor=belief.query_cursor or {};start_cursor=int(cursor.get('action_index',0)) if cursor.get('query_id')==query['query_id'] else 0
        allposes=np.asarray(query['compiled']['planned_poses'],float)
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
    age=np.mean([max(0.,belief.timestamp-h.last_seen.mean()) if h.n else 0. for h in future.worlds])
    clock_v2=getattr(model,'event_clock_version','')=='native_elapsed_age_v2'
    batched_hazards=None
    if prepared:
        maxn=max(p[4] for p in prepared);fulltimes=np.arange(maxn+1)*.048
        fullindices=np.searchsorted(future.times,belief.timestamp+fulltimes-1e-8).clip(0,len(future.times)-1)
        robots=np.stack([np.concatenate([p[6],np.repeat(p[6][-1:],maxn-p[4],axis=0)]) for p in prepared])
        event_age=age+fulltimes if clock_v2 else age
        event_elapsed=fulltimes[None,:]+np.array([p[1] for p in prepared])[:,None]*.048 if clock_v2 else fulltimes
        feats=event_features_many(future.positions[:,fullindices],future.velocities[:,fullindices],robots,hidden,event_age,event_elapsed)
        batched_hazards=event.hazards(feats)[:,:,1:,:]
    for qi,(query,start_cursor,allposes,remaining,n,supported,poses,times,indices) in enumerate(prepared):
        pos=future.positions[:,indices];vel=future.velocities[:,indices]
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
        # Explicit residual event law, shared by prediction and conditioning.
        # Its broad risk interval remains visible: this is not known geometry.
        residual_first=.995**np.arange(n)*.005
        residual_psc=.995**(2*np.arange(n))*.005
        residual_collision=float(residual_first.sum());residual_psc_probability=float(residual_psc.sum())
        collision=collision_lower+u*residual_collision;psc=psc_lower+u*residual_psc_probability
        collision_first_mass=world_weights@first+u*residual_first
        psc_first_mass=world_weights@first_psc+u*residual_psc
        duration_weights=np.r_[collision_first_mass,1-collision,0.]
        duration_samples=np.r_[times[1:],times[-1],times[-1]]
        # Progress is distance traveled along the immutable plan, not navigation success.
        travel=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1))]
        progress=np.r_[travel[1:],travel[-1],0.]
        complete=n==remaining and supported and not belief.terminal
        endcursor={'query_id':query['query_id'],'action_index':start_cursor+n,'total_actions':len(allposes)-1}
        timeout=start_cursor+n>=120 and n<remaining
        termination={'human_collision':collision,'query_complete':(1-collision) if complete else 0.,
                     'horizon_censored':(1-collision) if not complete and supported and not belief.terminal and not timeout else 0.,
                     'query_timeout':(1-collision) if timeout else 0.,
                     'route_invalidated':1. if not supported else 0.,'unresolved':0.}
        if belief.terminal:termination={k:0. for k in termination};termination['absorbed']=1.
        branches=[];tb=time.perf_counter()
        if qi<refined_queries:
            idx=int(indices[-1]);last=poses[-1]
            if belief.terminal:
                branches=[ObservationBranch(1.,{'kind':'terminal_absorbed'},belief)]
            else:
                if getattr(model,'observation_repair',False):
                    no_detection,no_before,no_last=model.observation_law.no_detection(future,indices,poses,return_parts=True)
                    any_visible=1-no_detection
                else:
                    # Coarsened observation channel. It does NOT certify empty space.
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
                    if prob<1e-15:continue
                    refinement=None
                    if getattr(model,'observation_repair',False) and not terminal and n:
                        if name=='survived_no_detection':
                            positive=np.zeros(len(ws));empty=1/np.maximum(no_last,1e-300)
                        else:
                            positive=1/np.maximum(any_visible,1e-300);empty=(1-no_before)/np.maximum(any_visible,1e-300)
                        refinement=(positive,empty)
                    successor=model.successor_at(future,idx,ws,last,terminal,cursor=endcursor,refinement=refinement)
                    if terminal:
                        first_mass=world_weights@first;probabilities=first_mass/first_mass.sum()
                        successor=successor.fork(robot_pose=probabilities@poses[1:],query_cursor=None,terminal_event={
                            'times':(belief.timestamp+times[1:]).tolist(),'poses':poses[1:].tolist(),
                            'probabilities':probabilities.tolist(),'semantics':'absorbed by predictive clock; actual stopping time and pose remain a distribution'})
                    obs={'kind':name,'timestamp':float(future.times[idx]),'coarsened':True,'observation_interval':'entire executed prefix, root excluded',
                         'visibility_model':model.observation_law.version if getattr(model,'observation_repair',False) else 'uncertain_base_heading_sector_v1','certified_empty':False,
                         'terminal_event':successor.terminal_event,'discovery':'unknown_geometry_remains_unknown','detected_position_distribution':'successor mixture, no fabricated exact measurement'}
                    branches.append(ObservationBranch(prob,obs,successor))
                if u:
                    residual=belief.fork(timestamp=float(future.times[idx]),robot_pose=last,hypotheses=[h.fork(weight=0.) for h in belief.hypotheses],unresolved_mass=1.,query_cursor=endcursor)
                    if residual_collision<1:
                        branches.append(ObservationBranch(u*(1-residual_collision),{'kind':'unresolved_survival','timestamp':residual.timestamp},residual))
                    if residual_collision>0:
                        rw=residual_first/residual_collision
                        terminal=residual.fork(terminal=True,robot_pose=rw@poses[1:],query_cursor=None,terminal_event={'times':(belief.timestamp+times[1:]).tolist(),'poses':poses[1:].tolist(),'probabilities':rw.tolist(),'semantics':'unresolved contact law, absorbing by predictive clock'})
                        branches.append(ObservationBranch(u*residual_collision,{'kind':'unresolved_contact','terminal_event':terminal.terminal_event},terminal))
        branch_ms+=(time.perf_counter()-tb)*1000
        conflict={'times_s':times[1:].tolist(),'collision_first_mass':collision_first_mass.tolist(),'psc_first_mass':psc_first_mass.tolist(),'appearance_first_mass':(world_weights@first_appearance+u*residual_psc).tolist()}
        out.append(QueryPrediction(query['query_id'],collision,psc,(collision_lower,min(1.,collision_lower+u)),(psc_lower,min(1.,psc_lower+u)),
            duration_samples,duration_weights,progress,conflict,0. if supported else 1.,termination,branches,u,
            (belief.timestamp,float(future.times[indices[-1]])),complete,endcursor,
            model_version=future.source.model_version,
            diagnostics={'risk_semantics':'native-label hazard surrogate','hidden_appearance_probability':float(world_weights@hidden_appearance),
                         'appearance_semantics':'auxiliary root-untracked native-label proxy; separate from spatial observation law',
                         'observation_process_version':model.observation_law.version if getattr(model,'observation_repair',False) else 'legacy_query_visibility_v1',
                         'static_support_checked':supported,'survival_world_weights':(world_weights*(1-pc)).tolist(),'endpoint_index':int(indices[-1]),'unresolved_survival_weight':u*(1-residual_collision),'conditional_survival_probability':1-collision,
                         'budget_ms':budget_ms,'budget_exceeded':False,'refined':qi<refined_queries,
                         'population_spatially_unresolved':float(world_weights@hidden)>0}))
    elapsed=(time.perf_counter()-started)*1000
    for p in out:p.diagnostics['budget_exceeded']=elapsed>budget_ms
    return QueryPredictionBatch(out,{'queries':elapsed,'graph_support':geometry_ms,'branches':branch_ms},future)

def integration_features(batch):
    """Minimal L2.5 adapter; no planning policy or graph compiler."""
    return np.asarray([[p.collision_probability,p.psc_probability,float(p.duration_weights@p.duration_samples),
                        float(p.duration_weights@p.progress_samples),p.blockage_probability,p.unresolved_uncertainty,
                        len(p.observation_branches),float(p.query_complete)] for p in batch.predictions],np.float32)
