"""Bounded multi-Bernoulli/count set update with explicit lost assignment mass.

The small subset dynamic program computes the complete set likelihood; a beam
only selects which association posteriors to materialize. Thus pruning cannot
silently renormalize alternative people away. Gaussian miss conditioning is a
declared positive-weight quadrature and moment approximation.
"""
import hashlib,json,math,time
from functools import lru_cache
import numpy as np
from .repair_model import REPAIR_VERSION

@lru_cache(maxsize=12)
def _subsets(D):
    masks=np.arange(1<<D);bits=(masks[:,None]&(1<<np.arange(D)))!=0
    targets=tuple(np.flatnonzero(bits[:,j]) for j in range(D))
    sources=tuple(targets[j]^(1<<j) for j in range(D))
    return bits,targets,sources,D-bits.sum(1)

def set_evidence(assign,miss,hidden_pmf,p_hidden,birth):
    """Sum all injective assignments and hidden-person permutations.

    assign[N,D] is p_detect(z)*position_density(z); miss[N] is integrated
    non-detection. Hidden people are exchangeable, first seen at birth density.
    Reference measure is the standard unordered finite-set measure (1/D!).
    """
    N,D=assign.shape;size=1<<D;dp=np.zeros(size);dp[0]=1.
    bits,targets,sources,remaining_count=_subsets(D)
    for i in range(N):
        new=dp*miss[i]
        for j in range(D):
            new[targets[j]]+=dp[sources[j]]*assign[i,j]
        dp=new
    count=np.arange(len(hidden_pmf));fall=np.ones_like(count,dtype=float);factors=[]
    for m in range(D+1):
        if m:fall*=np.maximum(count-m+1,0)
        factors.append(float(hidden_pmf@(fall*p_hidden**m*(1-p_hidden)**np.maximum(count-m,0))))
    births=np.prod(np.where(bits,1.,np.asarray(birth)[None,:]),axis=1)
    return float(dp@(np.asarray(factors)[remaining_count]*births))

def _apply_survival(b,trace):
    if not trace or 'world_likelihood' not in trace:return b
    lik=np.asarray(trace['world_likelihood'],float)
    if lik.shape!=(len(b.hypotheses),) or np.any((lik<0)|(lik>1)) or not np.isfinite(lik).all():raise ValueError('survival law dimension/probability')
    ws=np.array([h.weight for h in b.hypotheses])*lik
    u=b.unresolved_mass*float(trace.get('unresolved_likelihood',.5));den=ws.sum()+u
    if den<=0:return b.fork(hypotheses=[h.fork(weight=0.) for h in b.hypotheses],unresolved_mass=1.,evidence={**b.evidence,'unsupported_survival':True})
    return b.fork(hypotheses=[h.fork(weight=float(w/den)) for h,w in zip(b.hypotheses,ws)],unresolved_mass=float(u/den))

def condition_all_empty(hypotheses,boxes,unresolved_mass):
    """Apply a legal certificate to tracked AND never-localized people.

    Expand count alternatives only for this uncommon explicit certificate API.
    Quadrature points in an unknown region are latent support, not certificates.
    All failed evidence and quadrature mass remain accounted for.
    """
    from .probability import condition_empty_quadrature
    candidates=[];empirical=analytic=0.;points=0
    for h in hypotheses:
        counts=enumerate(h.hidden_pmf) if h.hidden_exchangeable else [(int(round(h.expected_hidden)),1.)]
        for k,p in counts:
            if p<=0 or h.weight<=0:continue
            hm=np.repeat(h.hidden_mean,k,axis=0) if h.hidden_exchangeable else h.hidden_mean
            hc=np.repeat(h.hidden_covariance,k,axis=0) if h.hidden_exchangeable else h.hidden_covariance
            augmented=h.fork(weight=h.weight*p,mean=np.concatenate([h.mean,hm]),covariance=np.concatenate([h.covariance,hc]),
                last_seen=np.r_[h.last_seen,np.zeros(k)],slot_ids=np.arange(h.n+k),hidden_pmf=np.array([1.]))
            result,_,diag=condition_empty_quadrature([augmented],boxes,0.)
            mass=float(diag.get('empty_evidence_quadrature',0.));empirical+=mass
            analytic+=float(diag.get('empty_evidence_cdf',mass));points+=int(diag.get('empty_quadrature_points',0))
            for a in result:
                if a.weight<=0 or mass<=0:continue
                pmf=np.zeros_like(h.hidden_pmf);pmf[k]=1.
                candidates.append(h.fork(weight=a.weight*mass,mean=a.mean[:h.n],covariance=a.covariance[:h.n],
                    hidden_mean=a.mean[h.n:],hidden_covariance=a.covariance[h.n:],hidden_pmf=pmf,hidden_exchangeable=False,
                    maneuver_mean=None,maneuver_covariance=None,maneuver_probability=None,
                    hidden_maneuver_mean=None,hidden_maneuver_covariance=None,hidden_maneuver_probability=None))
    residual=unresolved_mass*.5**len(boxes);den=empirical+residual
    if den<=0:return [h.fork(weight=0.) for h in hypotheses],1.,{'unsupported_empty':True}
    return [h.fork(weight=h.weight/den) for h in candidates],residual/den,{
        'empty_evidence_quadrature':empirical,'empty_evidence_cdf':analytic,
        'empty_evidence_absolute_error':abs(empirical-analytic),'empty_quadrature_points':points,
        'empty_includes_hidden_population':True}

def observe(wm,belief,row,trace=None):
    started=time.perf_counter();t=float(row['timestamp'])
    if t<belief.timestamp-1e-8:raise ValueError('old observation')
    if belief.terminal:return belief
    if not row.get('observation_valid',True):return wm.propagate(belief,t)
    detections=sorted([r for r in row.get('tracks',[]) if r.get('visible_now') and r.get('valid',True)],key=lambda r:tuple(r['position']))
    if any(abs(float(r['last_seen_time'])-t)>1e-6 for r in detections):raise ValueError('visible detection clock')
    boxes=sorted(set(tuple(np.asarray(b,float).ravel()) for b in row.get('certified_visible_empty_boxes',[])))
    payload={'timestamp':t,'robot_pose':row['robot_pose'],'positions':[r['position'] for r in detections],
             'updates':row.get('static_updates',[]),'empty':boxes,'trace':trace or {},'sensor':row.get('human_sensor_available',True)}
    key=hashlib.sha256(json.dumps(payload,sort_keys=True,default=lambda x:np.asarray(x).tolist()).encode()).hexdigest()
    if belief.evidence.get('last_observation_key')==key:
        wm.timings['observe']=(time.perf_counter()-started)*1000;return belief
    b=wm.propagate(_apply_survival(belief,trace),t);graph=b.graph.update(row.get('static_updates',[]))
    if not row.get('human_sensor_available',True):
        return b.fork(graph=graph,robot_pose=np.asarray(row['robot_pose'],float),evidence={**b.evidence,'last_observation_key':key})
    D=len(detections);law=wm.observation_law;z=np.asarray([r['position'] for r in detections],float).reshape(-1,3)
    # A real exact endpoint observation can refine a previously conditioned
    # coarse prefix event. Divide out precisely its already-counted emission;
    # for an empty endpoint retain the probability of an earlier detection.
    b=b.fork(hypotheses=[h.fork(weight=h.weight*(h.coarse_positive_refinement if D else h.coarse_empty_refinement))
        if abs(h.coarse_observation_time-t)<1e-8 else h for h in b.hypotheses])
    xy=z[:,[0,2]];clutter=law.birth_density(xy,row['robot_pose'])
    from .hidden import hypothesis_emissions
    from .maneuver import condition_components_many,support_ranks
    exact_dp=D<=10;infos=[];scales=np.log(np.maximum(clutter,1e-300));hidden_cache={}
    for h in b.hypotheses:
        miss,missmean,misscov=law.missed_moments(h,row['robot_pose'])
        loga=np.full((h.n,D),-np.inf);rank=np.full((h.n,D),3,np.int8);conditioned={};innov=[]
        if h.n and D:
            if h.maneuver_mean is not None:
                gm,gc,gp=h.maneuver_mean,h.maneuver_covariance,h.maneuver_probability
            else:gm,gc,gp=h.mean[:,None,:],h.covariance[:,None,:,:],np.ones((h.n,1))
            (cm,cv),lp=condition_components_many(gm,gc,gp,xy)
            rank=support_ranks(gm,gc,gp,xy)
            pd=law.detection(xy,row['robot_pose'],h.visibility_mode)
            loga=lp+np.log(np.maximum(pd,1e-300))[None,:]
            for i,j in zip(*np.nonzero(np.isfinite(lp))):conditioned[i,j]=(cm[i,j],cv[i,j])
            delta=xy[None]-h.mean[:,None,:2]
            inv=np.linalg.pinv(h.covariance[:,:2,:2]+np.eye(2)*1e-12)
            chi=np.einsum('ndi,nij,ndj->nd',delta,inv,delta)
            innov=[float(chi[i,j]) for i,j in zip(*np.nonzero(np.isfinite(lp)))]
        cachekey=(h.visibility_mode,)+tuple(None if v is None else v.tobytes() for v in
            (h.hidden_mean,h.hidden_covariance,h.hidden_maneuver_mean,h.hidden_maneuver_covariance,h.hidden_maneuver_probability))
        if cachekey not in hidden_cache:
            hidden_cache[cachekey]=hypothesis_emissions(h,xy,row['robot_pose'],law)
        hm,hmm,hcc,hc,hloga=hidden_cache[cachekey]
        if h.hidden_maneuver_mean is None:
            hrank=support_ranks(h.hidden_mean[:,None],h.hidden_covariance[:,None],np.ones((len(h.hidden_mean),1)),xy)
        else:hrank=support_ranks(h.hidden_maneuver_mean,h.hidden_maneuver_covariance,h.hidden_maneuver_probability,xy)
        if not h.hidden_exchangeable:
            for k,j in zip(*np.nonzero(np.isfinite(hloga))):conditioned[h.n+k,j]=(hc[0][k,j],hc[1][k,j])
            loga=np.concatenate([loga,hloga]);miss=np.r_[miss,hm]
            rank=np.concatenate([rank,hrank])
        infos.append({'h':h,'miss':miss,'mm':missmean,'cc':misscov,'loga':loga,'conditioned':conditioned,
            'rank':rank,'hrank':hrank,'innov':innov,'hm':hm,'hmm':hmm,'hcc':hcc,'hc':hc,'hloga':hloga.copy()})
    # One common dominating measure across assignments/hypotheses. A continuous
    # density contributes zero at a supported atom (or on a supported line).
    # Combining those numbers directly creates false people even at dt=0.
    measure=np.full(D,2 if b.unresolved_mass>0 else 3,np.int8)
    for info in infos:
        h=info['h']
        if h.weight<=0:continue
        if len(info['rank']):measure=np.minimum(measure,info['rank'].min(0))
        if h.hidden_exchangeable and h.expected_hidden>0:measure=np.minimum(measure,info['hrank'][0])
    logclutter=np.log(np.maximum(clutter,1e-300));logclutter=np.where(measure>=2,logclutter,-np.inf)
    scales=np.where(np.isfinite(logclutter),logclutter,-700.)
    for info in infos:
        info['loga']=np.where(info['rank']==measure[None],info['loga'],-np.inf)
        info['hloga']=np.where(info['hrank']==measure[None],info['hloga'],-np.inf)
        if info['h'].weight<=0:continue
        if len(info['loga']):scales=np.maximum(scales,info['loga'].max(0))
        if info['h'].hidden_exchangeable and info['h'].expected_hidden>0:scales=np.maximum(scales,info['hloga'][0])
    evidence=0.;candidates=[];diagnostics=[]
    next_id=max((int(h.slot_ids.max()) for h in b.hypotheses if h.n),default=-1)+1
    for hi,info in enumerate(infos):
        h=info['h'];miss=info['miss'];a=np.exp(info['loga']-scales);conditioned=info['conditioned'];C=len(a)
        if h.hidden_exchangeable:
            ph=float(1-info['hm'][0]);ph=float(np.clip(ph,1e-12,1-1e-12))
            birth=np.exp(info['hloga'][0]-scales)/ph;count_pmf=h.hidden_pmf
        else:ph=0.;birth=np.zeros(D);count_pmf=np.array([1.])
        if exact_dp:likelihood=set_evidence(a,miss,count_pmf,ph,birth)
        else:
            # Fixed-count overcount first, then count mixing; replacing a random
            # K with E[K] would NOT bound positively dependent detections.
            counts=np.arange(len(count_pmf))
            likelihood=float(count_pmf@np.prod(a.sum(0)[None,:]+counts[:,None]*ph*birth[None,:],axis=1))
        evidence+=h.weight*likelihood
        pmf=count_pmf*(1-ph)**np.arange(len(count_pmf));base=float(pmf.sum()*np.prod(miss))
        if base<=0 or h.weight<=0:continue
        beam=[(h.weight*base,(),0,pmf/pmf.sum())]
        for j in range(D):
            choices=[]
            for w,assignment,used,pmf in beam:
                for i in range(C):
                    if used&(1<<i) or (i,j) not in conditioned:continue
                    factor=a[i,j]/max(miss[i],1e-300)
                    if factor>0:choices.append((w*factor,assignment+(i,),used|(1<<i),pmf))
                if h.hidden_exchangeable:
                    k=np.arange(len(pmf));transfer=pmf*k*ph/(1-ph);mass=float(transfer.sum())
                    if mass>0:
                        posterior=np.zeros_like(pmf);posterior[:-1]=transfer[1:]/mass
                        choices.append((w*mass*birth[j],assignment+(-1,),used,posterior))
            choices.sort(key=lambda x:-x[0]);beam=choices[:wm.max_hypotheses]
            if not beam:break
        for w,assignment,used,pmf in beam:candidates.append((w,hi,assignment,used,pmf))
        diagnostics.append({'prior_weight':float(h.weight),'visibility_mode':h.visibility_mode,'slots':h.n,
            'set_evidence_scaled':likelihood,'minimum_innovation_chi2':float(min(info['innov'],default=0.))})
    scaled_clutter=np.exp(logclutter-scales)
    residual=b.unresolved_mass*math.exp(-1.)*float(np.prod(scaled_clutter))
    den=evidence+residual;candidates.sort(key=lambda x:-x[0]);keep=candidates[:wm.max_hypotheses]
    retained=sum(c[0] for c in keep);omitted=max(0.,evidence-retained);hs=[];ancestry=[]
    if den>0 and np.isfinite(den):
        u=float(min(1.,max(0.,(residual+omitted)/den)))
        for w,hi,assignment,used,pmf in keep:
            if w<=0:continue
            info=infos[hi];h=info['h'];conditioned=info['conditioned']
            mean=info['mm'].copy();cov=info['cc'].copy();seen=h.last_seen.copy()
            height=h.height.copy() if h.height is not None else np.full(h.n,graph.height)
            ids=h.slot_ids.copy();intent=h.intent.copy();diff=h.diffusion.copy();mode=h.mode.copy()
            bt=h.birth_time.copy() if h.birth_time is not None else h.last_seen.copy()
            bp=h.birth_position.copy() if h.birth_position is not None else h.mean[:,:2].copy()
            born=[];matched=[];transferred=[]
            for j,i in enumerate(assignment):
                if 0<=i<h.n:
                    mean[i],cov[i]=conditioned[i,j];seen[i]=t;height[i]=z[j,1];matched.append((int(ids[i]),j))
                else:
                    if i==-1:bm,bc=info['hc'][0][0,j],info['hc'][1][0,j]
                    else:bm,bc=conditioned[i,j];transferred.append(i-h.n)
                    mean=np.concatenate([mean,bm[None]]);cov=np.concatenate([cov,bc[None]])
                    seen=np.r_[seen,t];height=np.r_[height,z[j,1]];ids=np.r_[ids,next_id+j]
                    # The first-localization component retains the declared
                    # hidden target/diffusion; sensing does not invent a target.
                    intent=np.concatenate([intent,np.zeros((1,2))]);diff=np.r_[diff,.25];mode=np.r_[mode,0]
                    bt=np.r_[bt,t];bp=np.concatenate([bp,xy[j:j+1]]);born.append(int(next_id+j))
            if h.hidden_exchangeable:hm,hcov=info['hmm'],info['hcc']
            else:
                remaining=np.ones(len(h.hidden_mean),bool);remaining[transferred]=False
                hm,hcov=info['hmm'][remaining],info['hcc'][remaining]
                pmf=np.zeros_like(h.hidden_pmf);pmf[len(hm)]=1.
            hh=h.fork(weight=float(w/den),mean=mean,covariance=cov,last_seen=seen,height=height,slot_ids=ids,hidden_pmf=pmf,
                intent=intent,diffusion=diff,mode=mode,birth_time=bt,birth_position=bp,hidden_mean=hm,hidden_covariance=hcov,
                maneuver_mean=None,maneuver_covariance=None,maneuver_probability=None,
                hidden_maneuver_mean=None,hidden_maneuver_covariance=None,hidden_maneuver_probability=None,
                coarse_observation_time=-1.,coarse_positive_refinement=1.,coarse_empty_refinement=1.)
            hs.append(hh);ancestry.append({'posterior_weight':hh.weight,'matched_slots':matched,'new_slots':born})
    else:hs=[h.fork(weight=0.) for h in b.hypotheses];u=1.
    empty_diagnostics={}
    if boxes:
        hs,u,empty_diagnostics=condition_all_empty(hs,[np.asarray(x).reshape(2,2).tolist() for x in boxes],u)
    result=b.fork(graph=graph,hypotheses=hs,timestamp=t,robot_pose=np.asarray(row['robot_pose'],float),unresolved_mass=u,model_version=wm.model_version,
        evidence={**b.evidence,**empty_diagnostics,'last_observation_key':key,'last_detection_count':D,'exact_detections':z.tolist(),
            'unsupported_observation':den<=0,'pruned_mass':float(omitted/den) if den>0 else 1.,
            'observation_law':law.version,'assignment_evidence_exact':exact_dp,'association_alternatives':ancestry,
            'association_diagnostics':diagnostics,'birth_probability':float(sum(a['posterior_weight'] for a in ancestry if a['new_slots'])),
            'negative_evidence_available':bool(boxes),'soft_missed_detection_evidence':True,
            'miss_integration':'position Gauss-Hermite9x9/15x15; analytic conditional velocity; moment matched; not certified empty'})
    if trace and trace.get('episode_terminated'):result=result.fork(terminal=True)
    result.validate();wm.timings['observe']=(time.perf_counter()-started)*1000
    return result
