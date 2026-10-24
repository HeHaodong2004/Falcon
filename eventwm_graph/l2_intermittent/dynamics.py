"""Continuous-time Markov Stop/Move law; Gaussian conditional on event history.

No independent position diffusion. Event clocks persist across API partitions.
"""
from dataclasses import dataclass, replace
import hashlib
import math
import numpy as np
from eventwm_graph.l2_partial.probability import transition_matrices as _legacy_transition
from .random_kernel import KeyedWorkspace

LAW_VERSION = 'ctmc-stop-move-integrated-ou-1'
DEFAULT = dict(stop_rate=.8, restart_rate=1.3, move_relaxation=.7,
               stop_relaxation=8., stop_diffusion=.0036, stop_reset_std=.015,
               restart_variance_factor=.1)


def transition_matrices(dt,k,q=1.):
    F,Q,a,e=_legacy_transition(dt,k,q)
    if k>0 and k*dt<1e-3:
        z=k*dt
        Q[:2,:2]=np.eye(2)*(q*dt**3*(1/3-z/4+7*z*z/60-z**3/24))
        Q[:2,2:]=Q[2:,:2]=np.eye(2)*(q*(-math.expm1(-z)/k)**2/2)
    return F,Q,a,e


def keyed_rng(seed, slot, counter):
    raw=hashlib.blake2b(('%s:%s:%s' % (seed,slot,counter)).encode(),digest_size=16).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(raw,'little')))


def categorical(rng,probabilities):
    """Scalar Generator.choice(p=...) with its float64 normalized CDF."""
    original=np.asarray(probabilities);p=np.asarray(original,dtype=float)
    tolerance=max(math.sqrt(np.finfo(float).eps),math.sqrt(np.finfo(original.dtype).eps)) if original.dtype.kind=='f' else math.sqrt(np.finfo(float).eps)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or np.any(p<0) or abs(math.fsum(p)-1)>tolerance:
        raise ValueError('invalid categorical probabilities')
    cdf=p.cumsum();cdf/=cdf[-1]
    return int(cdf.searchsorted(rng.random(),side='right'))


def root_psd(p):
    e,u=np.linalg.eigh(p)
    if e.size and e.min() < -1e-7:
        raise ValueError('indefinite covariance')
    return u*np.sqrt(np.maximum(e,0))[...,None,:]


@dataclass(frozen=True)
class Component:
    weight: float
    mean: np.ndarray
    covariance: np.ndarray
    slots: np.ndarray
    last_seen: np.ndarray
    birth: np.ndarray
    last_position: np.ndarray
    stopped: np.ndarray
    intent: np.ndarray
    diffusion: np.ndarray
    next_switch: np.ndarray
    counters: np.ndarray
    seed: int
    hidden_pmf: np.ndarray
    visibility_mode: int=1
    origin: int=0
    hidden_mean: object=None
    hidden_covariance: object=None
    last_position_valid: object=None

    def fork(self, **kw):
        return replace(self, **kw)

    @property
    def n(self):
        return len(self.slots)


def schedule(c, now, law,workspace=None):
    workspace=KeyedWorkspace() if workspace is None else workspace
    nxt=np.array([now+workspace.generator(c.seed,int(s),int(k)).exponential(1/(law['restart_rate'] if st else law['stop_rate']))
                  for s,k,st in zip(c.slots,c.counters,c.stopped)])
    return c.fork(next_switch=nxt)


def evolve(c, start, end, law, rng=None, record=None):
    """Exact Gaussian evolution conditional on retained CTMC event schedules.

    rng=None evolves Rao-Blackwell components. A supplied RNG samples that SAME
    conditional Gaussian law; callers first sample the initial Gaussian state.
    """
    if end < start-1e-9 or not math.isfinite(end):
        raise ValueError('invalid physical time')
    # Float32 cache storage must not round the continuous state at each API
    # boundary: that alone creates a small partition-dependent drift.
    m=c.mean.astype(float);P=c.covariance.astype(float);st=c.stopped.copy()
    nxt=c.next_switch.copy();counter=c.counters.copy()
    for i in range(c.n):
        t=float(start)
        while t < end-1e-12:
            u=min(float(end),float(nxt[i]));dt=max(0.,u-t)
            k=law['stop_relaxation'] if st[i] else law['move_relaxation']
            q=law['stop_diffusion'] if st[i] else c.diffusion[i]
            target=np.zeros(2) if st[i] else np.asarray(c.intent[i],dtype=float)
            F,Q,a,e=transition_matrices(dt,k,q)
            m[i]=F@m[i]+np.r_[(dt-a)*target,(1-e)*target]
            if rng is None:
                P[i]=F@P[i]@F.T+Q
            else:
                m[i]+=root_psd(Q)@rng.normal(size=4)
            t=u
            if nxt[i]<=end+1e-12:
                st[i]=not st[i]
                var=law['stop_reset_std']**2 if st[i] else law['restart_variance_factor']*c.diffusion[i]
                m[i,2:]=0. if st[i] else c.intent[i]
                P[i,2:,:]=0.;P[i,:,2:]=0.;P[i,2:,2:]=np.eye(2)*var
                if rng is not None:
                    m[i,2:]+=rng.normal(size=2)*math.sqrt(var);P[i,2:,2:]=0.
                counter[i]+=1
                nxt[i]=t+keyed_rng(c.seed,int(c.slots[i]),int(counter[i])).exponential(1/(law['restart_rate'] if st[i] else law['stop_rate']))
                if record is not None:
                    record.append((i,t,bool(st[i])))
    hm,hc=c.hidden_mean,c.hidden_covariance
    if hm is not None:
        F,_,_,_ = transition_matrices(end-start,0.)
        hm=F@hm;hc=F@hc@F.T
    return c.fork(mean=m,covariance=P,stopped=st,next_switch=nxt,counters=counter,hidden_mean=hm,hidden_covariance=hc)


def evolve_many(components,start,end,law):
    """Batch exact conditional Gaussian algebra; retain every component/clock.

    Scalar transition coefficients use the reference math functions. Batched
    four-dimensional matmul follows the same arithmetic as each reference
    matmul. Nonstandard event-storage dtypes retain the scalar implementation.
    """
    if end<start-1e-9 or not math.isfinite(end):raise ValueError('invalid physical time')
    if any(c.next_switch.dtype!=np.dtype(float) or c.counters.dtype!=np.dtype(np.int64) for c in components):
        return tuple(evolve(c,start,end,law) for c in components)
    sizes=[c.n for c in components];offsets=np.r_[0,np.cumsum(sizes)]
    total=int(offsets[-1]);m=np.concatenate([c.mean for c in components]).astype(float)
    P=np.concatenate([c.covariance for c in components]).astype(float)
    st=np.concatenate([c.stopped for c in components]);nxt=np.concatenate([c.next_switch for c in components])
    counter=np.concatenate([c.counters for c in components]);intent=np.concatenate([c.intent for c in components]).astype(float)
    diffusion=np.concatenate([c.diffusion for c in components]).astype(float)
    seeds=[c.seed for c in components for _ in range(c.n)];slots=np.concatenate([c.slots for c in components])
    clock=np.full(total,float(start));matrix_cache={};workspace=KeyedWorkspace()
    while True:
        indices=np.flatnonzero(clock<end-1e-12)
        if not len(indices):break
        u=np.minimum(float(end),nxt[indices]);dt=np.maximum(0.,u-clock[indices])
        k=np.where(st[indices],law['stop_relaxation'],law['move_relaxation'])
        q=np.where(st[indices],law['stop_diffusion'],diffusion[indices])
        F=np.empty((len(indices),4,4));Q=np.empty_like(F);a=np.empty(len(indices));e=np.empty(len(indices))
        groups={}
        for j,key in enumerate(zip(dt,k)):groups.setdefault(key,[]).append(j)
        for key,js in groups.items():
            if key not in matrix_cache:matrix_cache[key]=transition_matrices(float(key[0]),float(key[1]),1.)
            f,base_q,aa,ee=matrix_cache[key];F[js]=f;Q[js]=base_q*q[js,None,None];a[js]=aa;e[js]=ee
            if key[1]>0 and key[1]*key[0]<1e-3:
                # Match the reference's order of q multiplication in its small
                # physical interval correction, rather than scaling a moment.
                d,kk=map(float,key);z=kk*d
                pp=q[js]*d**3*(1/3-z/4+7*z*z/60-z**3/24)
                pv=q[js]*(-math.expm1(-z)/kk)**2/2
                Q[js,0,0]=pp;Q[js,1,1]=pp
                Q[js,0,2]=pv;Q[js,1,3]=pv;Q[js,2,0]=pv;Q[js,3,1]=pv
        target=np.where(st[indices,None],0.,intent[indices])
        shift=np.concatenate([(dt-a)[:,None]*target,(1-e)[:,None]*target],1)
        m[indices]=(F@m[indices,:,None])[...,0]+shift
        P[indices]=F@P[indices]@F.swapaxes(-1,-2)+Q
        clock[indices]=u
        changed=indices[nxt[indices]<=end+1e-12]
        for i in changed:
            st[i]=not st[i];var=law['stop_reset_std']**2 if st[i] else law['restart_variance_factor']*diffusion[i]
            m[i,2:]=0. if st[i] else intent[i];P[i,2:,:]=0.;P[i,:,2:]=0.;P[i,2:,2:]=np.eye(2)*var
            counter[i]+=1
            nxt[i]=clock[i]+workspace.generator(seeds[i],int(slots[i]),int(counter[i])).exponential(1/(law['restart_rate'] if st[i] else law['stop_rate']))
    result=[]
    hidden_indices=[j for j,c in enumerate(components) if c.hidden_mean is not None]
    hidden={}
    if hidden_indices:
        f,_,_,_=transition_matrices(end-start,0.)
        hm=np.stack([components[j].hidden_mean for j in hidden_indices]);hc=np.stack([components[j].hidden_covariance for j in hidden_indices])
        hm=(f@hm[...,None])[...,0];hc=f@hc@f.T
        hidden={j:(hm[i],hc[i]) for i,j in enumerate(hidden_indices)}
    for j,c in enumerate(components):
        sl=slice(int(offsets[j]),int(offsets[j+1]));hm,hc=hidden.get(j,(c.hidden_mean,c.hidden_covariance))
        result.append(c.fork(mean=m[sl],covariance=P[sl],stopped=st[sl],next_switch=nxt[sl],counters=counter[sl],hidden_mean=hm,hidden_covariance=hc))
    return tuple(result)


def _sample_joint_paths_reference(worlds,rel,start,law,rng,cv_diffusion=None):
    """Vectorized SAME conditional law; no support/world/time reduction."""
    W=len(worlds);N=max((c.n for c in worlds),default=0);T=len(rel)
    m=np.zeros((W,N,4));intent=np.zeros((W,N,2));q=np.zeros((W,N));st=np.zeros((W,N),bool)
    nxt=np.full((W,N),np.inf);counter=np.zeros((W,N),int);valid=np.zeros((W,N),bool)
    for w,c in enumerate(worlds):
        n=c.n;m[w,:n]=c.mean;intent[w,:n]=c.intent;q[w,:n]=c.diffusion;st[w,:n]=c.stopped;nxt[w,:n]=c.next_switch;counter[w,:n]=c.counters;valid[w,:n]=True
    pos=np.full((W,T,N,2),np.nan);vel=pos.copy();stops=np.zeros((W,T,N),bool)
    next_series=np.zeros((W,T,N));counter_series=np.zeros((W,T,N),int)
    for ti,t in enumerate(start+rel):
        clock=np.full((W,N),start+rel[ti-1] if ti else start)
        while ti and np.any(valid&(clock<t-1e-12)):
            active=valid&(clock<t-1e-12)
            u=np.minimum(nxt,t) if cv_diffusion is None else np.full((W,N),t)
            dt=np.where(active,np.maximum(0,u-clock),0.)
            if cv_diffusion is None:
                k=np.where(st,law['stop_relaxation'],law['move_relaxation']);e=np.exp(-k*dt);a=-np.expm1(-k*dt)/k
                vv=-np.expm1(-2*k*dt)/(2*k);pv=(-np.expm1(-k*dt)/k)**2/2
                pp=(dt-2*a+vv)/(k*k);z=k*dt;pp=np.where(z<1e-3,dt**3*(1/3-z/4+7*z*z/60-z**3/24),pp)
                scale=np.sqrt(np.where(st,law['stop_diffusion'],q));target=intent*(~st)[...,None]
            else:
                e=np.ones_like(dt);a=dt;vv=dt;pv=dt**2/2;pp=dt**3/3;scale=np.full_like(dt,math.sqrt(cv_diffusion));target=np.zeros_like(intent)
            lv=np.sqrt(np.maximum(vv,0));lp=np.divide(pv,lv,out=np.zeros_like(pv),where=lv>0);lpp=np.sqrt(np.maximum(pp-lp*lp,0))
            z=rng.normal(size=(W,N,2,2));np_=lp[...,None]*z[...,0,:]+lpp[...,None]*z[...,1,:];nv=lv[...,None]*z[...,0,:]
            m[...,:2]+=a[...,None]*m[...,2:]+(dt-a)[...,None]*target+scale[...,None]*np_
            m[...,2:]=e[...,None]*m[...,2:]+(1-e)[...,None]*target+scale[...,None]*nv
            clock=np.where(active,u,clock)
            change=active&(nxt<=t+1e-12) if cv_diffusion is None else np.zeros_like(valid)
            st^=change
            for w,i in zip(*np.where(change)):
                variance=law['stop_reset_std']**2 if st[w,i] else law['restart_variance_factor']*q[w,i]
                m[w,i,2:]=(0. if st[w,i] else intent[w,i])+rng.normal(size=2)*math.sqrt(variance)
                counter[w,i]+=1
                nxt[w,i]=clock[w,i]+keyed_rng(worlds[w].seed,int(worlds[w].slots[i]),int(counter[w,i])).exponential(1/(law['restart_rate'] if st[w,i] else law['stop_rate']))
        pos[:,ti]=np.where(valid[...,None],m[...,:2],np.nan);vel[:,ti]=np.where(valid[...,None],m[...,2:],np.nan);stops[:,ti]=st&valid
        if cv_diffusion is not None:nxt[:]=t+1e10
        next_series[:,ti]=nxt;counter_series[:,ti]=counter
    return pos,vel,stops,next_series,counter_series


def sample_joint_paths(worlds,rel,start,law,rng,cv_diffusion=None):
    """Same conditional paths/draw order, with within-call clock sharing."""
    from .sampling_kernel import fused_paths
    W=len(worlds);N=max((c.n for c in worlds),default=0)
    m=np.zeros((W,N,4));intent=np.zeros((W,N,2));q=np.zeros((W,N));st=np.zeros((W,N),bool)
    nxt=np.full((W,N),np.inf);counter=np.zeros((W,N),np.int64);valid=np.zeros((W,N),bool)
    schedules={};references={};absolute_times=start+rel;workspace=KeyedWorkspace()
    for w,c in enumerate(worlds):
        n=c.n;m[w,:n]=c.mean;intent[w,:n]=c.intent;q[w,:n]=c.diffusion;st[w,:n]=c.stopped
        nxt[w,:n]=c.next_switch;counter[w,:n]=c.counters;valid[w,:n]=True
        if cv_diffusion is not None:continue
        for i in range(n):
            key=(c.seed,int(c.slots[i]),int(counter[w,i]),bool(st[w,i]),float(nxt[w,i]))
            if key not in schedules:
                tau=key[-1];state=key[-2];count=key[-3];times=[]
                for ti in range(1,len(absolute_times)):
                    t=float(absolute_times[ti-1]);end=float(absolute_times[ti])
                    while t<end-1e-12:
                        t=min(end,tau)
                        if tau<=end+1e-12:
                            state=not state;count+=1
                            tau=t+workspace.generator(c.seed,int(c.slots[i]),count).exponential(1/(law['restart_rate'] if state else law['stop_rate']))
                            times.append(tau)
                schedules[key]=times
            references[(w,i)]=schedules[key]
    depth=max((len(v) for v in schedules.values()),default=0)
    event_next=np.empty((W,N,max(1,depth)))
    for (w,i),times in references.items():event_next[w,i,:len(times)]=times
    configuration=np.array([law[k] for k in ('stop_rate','restart_rate','move_relaxation','stop_relaxation',
        'stop_diffusion','stop_reset_std','restart_variance_factor')])
    return fused_paths(m,intent,q,st,nxt,counter,valid,np.asarray(rel,float),float(start),configuration,
        event_next,rng,-1. if cv_diffusion is None else float(cv_diffusion))
