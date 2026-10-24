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

def observe_sensor(wm,b,row,elapsed=0.):
    """One endpoint sensor event; projection/quadrature errors remain explicit."""
    started=time.perf_counter();t=float(row['timestamp']);graph=b.graph
    detections=sorted([r for r in row.get('tracks',[]) if r.get('visible_now') and r.get('valid',True)],key=lambda r:tuple(r['position']))
    boxes=sorted(set(tuple(np.asarray(box,float).ravel()) for box in row.get('certified_visible_empty_boxes',[])))
    D=len(detections);law=wm.observation_law;z=np.asarray([r['position'] for r in detections],float).reshape(-1,3)
    from .structural_budget import check_budget
    check_budget('observe:emission_workspace')
    xy=z[:,[0,2]];clutter=law.birth_density(xy,row['robot_pose'])
    from .hidden import hypothesis_emissions
    from .maneuver import condition_components_many,support_ranks
    exact_dp=D<=10;infos=[];scales=np.log(np.maximum(clutter,1e-300));hidden_cache={}
    from .structural_kernels import ObservationWorkspace
    workspace=ObservationWorkspace(b.hypotheses,xy,row['robot_pose'],law) if wm.optimized else None
    for hi,h in enumerate(b.hypotheses):
        check_budget('observe:component_emissions')
        miss,missmean,misscov=workspace.tracked_misses[hi] if workspace is not None else law.missed_moments(h,row['robot_pose'])
        loga=np.full((h.n,D),-np.inf);rank=np.full((h.n,D),3,np.int8);conditioned={};innov=[]
        if h.n and D:
            if h.maneuver_mean is not None:
                gm,gc,gp=h.maneuver_mean,h.maneuver_covariance,h.maneuver_probability
            else:gm,gc,gp=h.mean[:,None,:],h.covariance[:,None,:,:],np.ones((h.n,1))
            (cm,cv),lp=condition_components_many(gm,gc,gp,xy)
            rank=support_ranks(gm,gc,gp,xy)
            pd=law.detection(xy,row['robot_pose'],h.visibility_mode)
            with np.errstate(divide='ignore'):
                loga=lp+np.log(pd)[None,:]
            for i,j in zip(*np.nonzero(np.isfinite(lp))):conditioned[i,j]=(cm[i,j],cv[i,j])
            delta=xy[None]-h.mean[:,None,:2]
            inv=np.linalg.pinv(h.covariance[:,:2,:2]+np.eye(2)*1e-12)
            chi=np.einsum('ndi,nij,ndj->nd',delta,inv,delta)
            innov=[float(chi[i,j]) for i,j in zip(*np.nonzero(np.isfinite(lp)))]
        cachekey=(h.visibility_mode,)+tuple(None if v is None else v.tobytes() for v in
            (h.hidden_mean,h.hidden_covariance,h.hidden_maneuver_mean,h.hidden_maneuver_covariance,h.hidden_maneuver_probability))
        if cachekey not in hidden_cache:
            hidden_cache[cachekey]=workspace.hidden_emissions[hi] if workspace is not None else hypothesis_emissions(h,xy,row['robot_pose'],law)
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
    evidence=0.;candidates=[];diagnostics=[];association_pruned=False
    next_id=max((int(h.slot_ids.max()) for h in b.hypotheses if h.n),default=-1)+1
    for hi,info in enumerate(infos):
        check_budget('observe:association_component')
        h=info['h'];miss=info['miss'];a=np.exp(info['loga']-scales);conditioned=info['conditioned'];C=len(a)
        if h.hidden_exchangeable:
            ph=float(1-info['hm'][0]);ph=float(np.clip(ph,0.,1.))
            birth=np.exp(info['hloga'][0]-scales)/ph if ph>0 else np.zeros(D);count_pmf=h.hidden_pmf
        else:ph=0.;birth=np.zeros(D);count_pmf=np.array([1.])
        if exact_dp:likelihood=set_evidence(a,miss,count_pmf,ph,birth)
        else:
            # Fixed-count overcount first, then count mixing; replacing a random
            # K with E[K] would NOT bound positively dependent detections.
            counts=np.arange(len(count_pmf))
            likelihood=float(count_pmf@np.prod(a.sum(0)[None,:]+counts[:,None]*ph*birth[None,:],axis=1))
        evidence+=h.weight*likelihood
        if h.weight<=0:continue
        # Ratio form is fast when every denominator is strictly positive.
        # Degenerate pD=1/zero-miss cases use the division-free path below.
        if np.all(miss>0) and ph<1:
            pmf=count_pmf*(1-ph)**np.arange(len(count_pmf));base=float(pmf.sum()*np.prod(miss))
            beam=[(h.weight*base,(),0,pmf/pmf.sum())] if base>0 else []
            for j in range(D):
                choices=[]
                for w,assignment,used,pmf in beam:
                    for i in range(C):
                        if used&(1<<i) or (i,j) not in conditioned:continue
                        factor=a[i,j]/miss[i]
                        if factor>0:choices.append((w*factor,assignment+(i,),used|(1<<i),pmf))
                    if h.hidden_exchangeable:
                        transfer=pmf*np.arange(len(pmf))*ph/(1-ph);mass=float(transfer.sum())
                        if mass>0:
                            posterior=np.zeros_like(pmf);posterior[:-1]=transfer[1:]/mass
                            choices.append((w*mass*birth[j],assignment+(-1,),used,posterior))
                choices.sort(key=lambda x:-x[0]);association_pruned|=len(choices)>wm.max_hypotheses;beam=choices[:wm.max_hypotheses]
                if not beam:break
            for w,assignment,used,pmf in beam:candidates.append((w,hi,assignment,used,pmf))
        else:
            # Assign first, miss later: pD=1 positives must not disappear through
            # an initial prod(miss)=0 or a detection/miss 0/0 ratio.
            beam=[(h.weight,(),0,count_pmf.copy())]
            for j in range(D):
                choices=[]
                for w,assignment,used,pmf in beam:
                    for i in range(C):
                        if used&(1<<i) or (i,j) not in conditioned:continue
                        if a[i,j]>0:choices.append((w*a[i,j],assignment+(i,),used|(1<<i),pmf))
                    if h.hidden_exchangeable:
                        transfer=pmf*np.arange(len(pmf));mass=float(transfer.sum());emission=ph*birth[j]
                        if mass>0 and emission>0:
                            posterior=np.zeros_like(pmf);posterior[:-1]=transfer[1:]/mass
                            choices.append((w*mass*emission,assignment+(-1,),used,posterior))
                def beam_score(item):
                    w,_,used,pmf=item
                    return w*float(pmf@((1-ph)**np.arange(len(pmf))))*float(np.prod([miss[i] for i in range(C) if not used&(1<<i)]))
                choices.sort(key=lambda item:-beam_score(item));association_pruned|=len(choices)>wm.max_hypotheses;beam=choices[:wm.max_hypotheses]
                if not beam:break
            for w,assignment,used,pmf in beam:
                pmf=pmf*(1-ph)**np.arange(len(pmf));mass=float(pmf.sum())
                missed=float(np.prod([miss[i] for i in range(C) if not used&(1<<i)]))
                if mass>0 and missed>0:candidates.append((w*mass*missed,hi,assignment,used,pmf/mass))
        diagnostics.append({'prior_weight':float(h.weight),'visibility_mode':h.visibility_mode,'slots':h.n,
            'set_evidence_scaled':likelihood,'minimum_innovation_chi2':float(min(info['innov'],default=0.))})
    from .structural_probability import (CountSupport,ObservationMeasure,
        finite_set_evidence_bound,gaussian_noise_log_supremum)
    reference=ObservationMeasure('set',tuple(int(x) for x in measure),'endpoint-common-reference:'+hashlib.sha256(xy.tobytes()).hexdigest())
    retained_support={h.n+int(k) for h in b.hypotheses if h.weight>0 for k in np.flatnonzero(h.hidden_pmf>0)}
    previous_support=b.evidence.get('residual_count_support')
    unknown_upper=b.unresolved_mass>0 and previous_support is None
    incoming_support=set(int(k) for k in previous_support) if b.unresolved_mass>0 and previous_support is not None else set()
    incoming_support|=retained_support
    floor=max(D,int(b.evidence.get('maximum_simultaneous_detections',0)))
    surviving_support=None if unknown_upper else tuple(sorted(k for k in incoming_support if k>=floor))
    maximum=max(incoming_support,default=-1)
    support=CountSupport(None if unknown_upper else tuple(sorted(incoming_support)),floor)
    if D and np.all(measure==2):
        contract=b.evidence.get('transition_noise_contract',{})
        certified=contract.get('kind')=='independent_position_diffusion' and contract.get('model_version')==wm.model_version
        variance=float(contract.get('variance_rate',0.))*elapsed if certified else 0.
        supremum=gaussian_noise_log_supremum(np.eye(2)*max(0.,variance),elapsed_time=max(0.,elapsed))
        bounds=np.full(D,supremum)
    elif D and np.all(measure==0):bounds=np.zeros(D)
    else:bounds=np.full(D,np.inf)
    bound=finite_set_evidence_bound(support,bounds,reference,joint_domination_certified=True,
        provenance='independent additive kernel noise and declared finite count support; LOCAL, projection error excluded')
    if wm.residual_policy=='legacy_parity_only':
        scaled_clutter=np.exp(logclutter-scales)
        residual=b.unresolved_mass*math.exp(-1.)*float(np.prod(scaled_clutter))
    elif b.unresolved_mass==0 or bound.log_upper==-math.inf:residual=0.
    else:
        log_residual=math.log(b.unresolved_mass)+bound.log_upper-float(scales.sum())
        residual=math.exp(log_residual) if log_residual<700 else math.inf
    candidates.sort(key=lambda x:-x[0]);keep=candidates[:wm.max_hypotheses]
    retained=sum(c[0] for c in keep);numerical_evidence_difference=0.
    if exact_dp and not association_pruned and len(candidates)<=wm.max_hypotheses:
        delta=evidence-retained
        tolerance=np.finfo(float).eps*128*(1+D+sum(h.n for h in b.hypotheses))*max(evidence,retained)
        if abs(delta)<=tolerance:
            # Complete materialization is the same finite sum as the DP. Use
            # one shared normalizer instead of interpreting reduction roundoff
            # as an omitted physical hypothesis with an arbitrary posterior.
            numerical_evidence_difference=float(delta);evidence=retained
    den=evidence+residual;omitted=max(0.,evidence-retained);hs=[];ancestry=[]
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
        hs,u,empty_diagnostics=condition_empty_bound(wm,hs,[np.asarray(x).reshape(2,2).tolist() for x in boxes],u)
    result=b.fork(graph=graph,hypotheses=hs,timestamp=t,robot_pose=np.asarray(row['robot_pose'],float),unresolved_mass=u,model_version=wm.model_version,
        evidence={**b.evidence,**empty_diagnostics,'last_sensor_timestamp':t,'last_detection_count':D,'exact_detections':z.tolist(),
            'unsupported_observation':den<=0 or not np.isfinite(den),'pruned_mass':float(omitted/den) if den>0 and np.isfinite(den) else 1.,
            'residual_bound_status':bound.status,'residual_point_law':wm.residual_policy=='legacy_parity_only',
            'probability_certificate_scope':'LOCAL only; Gaussian projections and quadrature are NOT globally bounded',
            'residual_count_support':None if surviving_support is None or not surviving_support else list(surviving_support),'maximum_simultaneous_detections':floor,
            'posterior_status':'LOCAL_APPROXIMATE' if den>0 and np.isfinite(den) else 'VACUOUS_BOUND_OR_UNSUPPORTED_PRIOR_RECOVERY',
            'observation_law':law.version,'assignment_evidence_exact':exact_dp,'numerical_set_evidence_difference':numerical_evidence_difference,'association_alternatives':ancestry,
            'mass_accounting':{'incoming_unresolved_mass':float(b.unresolved_mass),
                'represented_evidence_scaled':float(evidence),
                'retained_assignment_evidence_scaled':float(retained),
                'omitted_assignment_evidence_scaled':float(omitted),
                'residual_evidence_scaled':float(residual) if np.isfinite(residual) else None,
                'log_observation_density_scale':float(scales.sum()),
                'residual_log_likelihood_upper':float(bound.log_upper) if np.isfinite(bound.log_upper) else None,
                'residual_bound_status':bound.status,
                'post_sensor_unresolved_mass':float(u),
                'association_beam_pruned':bool(association_pruned),
                'scope':'local retained surrogate; projection/quadrature error unbounded'},
            'association_diagnostics':diagnostics,'birth_probability':float(sum(a['posterior_weight'] for a in ancestry if a['new_slots'])),
            'negative_evidence_available':bool(boxes),'soft_missed_detection_evidence':True,
            'miss_integration':'position Gauss-Hermite9x9/15x15; analytic conditional velocity; moment matched; not certified empty'})
    # Public observe() validates the completed state once, after terminal and
    # receipt updates. Repeating the same dense eigensystem checks here added
    # no probability information and dominated certificate latency.
    wm.timings['observe']=(time.perf_counter()-started)*1000
    return result


def condition_empty_bound(wm,hypotheses,boxes,unresolved_mass):
    """Condition the union once; residual event likelihood is bounded by one."""
    from .structural_kernels import condition_all_empty_fast
    from .repair_filter import condition_all_empty
    kernel=condition_all_empty_fast if wm.optimized else condition_all_empty
    hs,_,diag=kernel(hypotheses,boxes,0.)
    empirical=float(diag.get('empty_evidence_quadrature',0.))
    residual=unresolved_mass*.5**len(boxes) if wm.residual_policy=='legacy_parity_only' else unresolved_mass
    den=empirical+residual
    if den<=0:return [h.fork(weight=0.) for h in hypotheses],1.,{**diag,'unsupported_empty':True}
    return [h.fork(weight=h.weight*empirical/den) for h in hs],residual/den,{
        **diag,'empty_residual_semantics':'upper event likelihood 1; local approximation scope',
        'empty_union_conditioned_once':True}
