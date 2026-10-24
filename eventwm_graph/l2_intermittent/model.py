"""Persistent observed-human World Model with bounded joint Gaussian carriers.

Online inputs are anonymous centers, arrived pose/static graph, and timestamps.
Physical identities and offline truth readers never enter this module.
"""
import copy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import weakref
from pathlib import Path
from types import MappingProxyType
import numpy as np
from scipy.special import logsumexp
from eventwm_graph.l2_partial.state import ObservedGraph, Hypothesis
from eventwm_graph.l2_partial.observation import EndpointObservationLaw
from eventwm_graph.l2_partial.probability import consume_hidden
from .dynamics import Component, DEFAULT, LAW_VERSION, schedule, evolve, evolve_many, root_psd, keyed_rng, categorical, sample_joint_paths
from .neural import LearnedIntermittent, IntermittentDynamics
from .association import prepare as assignment_prepare, sample_assignment, association_marginals
from .random_kernel import KeyedWorkspace


def freeze(a):
    a=np.asarray(a)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def sealed(c):
    return c.fork(**{k:freeze(v) for k,v in vars(c).items() if isinstance(v,np.ndarray) and v.flags.writeable})


def freeze_graph(g):
    if isinstance(g.cells,MappingProxyType):return g
    return replace(g,cells=MappingProxyType(dict(g.cells)),parent=freeze_graph(g.parent) if g.parent is not None else None)


def detections(ob):
    if not math.isfinite(float(ob['timestamp'])):raise ValueError('invalid observation timestamp')
    if not ob.get('human_sensor_available',True) or not ob.get('observation_valid',True):
        return np.empty((0,2))
    for track in ob.get('tracks',[]):
        if track.get('visible_now',False) and track.get('valid',True):
            position=np.asarray(track['position'],float)
            seen=float(track.get('last_seen_time',ob['timestamp']))
            if position.shape!=(3,) or not np.isfinite(position).all() or not math.isfinite(seen) or abs(seen-float(ob['timestamp']))>=1e-7:
                raise ValueError('inconsistent current-positive position/timestamp')
    return np.asarray([np.asarray(t['position'],float)[[0,2]] for t in ob.get('tracks',[])
        if t.get('visible_now',False) and t.get('valid',True)
        and abs(float(t.get('last_seen_time',ob['timestamp']))-float(ob['timestamp']))<1e-7]).reshape(-1,2)


def position_condition(m,P,z):
    """Exact Gaussian disintegration with explicit singular support dimension."""
    e,U=np.linalg.eigh(P[:2,:2]);tol=max(1e-20,float(e.max())*1e-10)
    if e.min() < -tol:raise ValueError('invalid predictive covariance')
    active=e>tol;rank=int(active.sum());d=np.asarray(z)-m[:2];coords=U.T@d
    if np.linalg.norm(coords[~active])>1e-7:return None,-math.inf,rank
    inv=(U[:,active]/e[active])@U[:,active].T
    gain=P[:,:2]@inv;mean=m+gain@d;cov=P-gain@P[:2,:]
    mean[:2]=z;cov[:2,:]=0;cov[:,:2]=0;cov=(cov+cov.T)/2
    return (mean,cov),float(-rank*.5*math.log(2*math.pi)-.5*np.log(e[active]).sum()-.5*d@inv@d),rank


def position_conditions(means,covariances,centers):
    """Batch the same singular Gaussian disintegration over tracks/centers."""
    e,U=np.linalg.eigh(covariances[:,:2,:2])
    tol=np.maximum(1e-20,e.max(-1).astype(float)*1e-10)
    if np.any(e.min(-1)<-tol):raise ValueError('invalid predictive covariance')
    active=e>tol[:,None];rank=active.sum(-1)
    inverse=(np.divide(U,e[:,None,:],out=np.zeros_like(U),where=active[:,None,:])@
             U.swapaxes(-1,-2))
    d=centers[None]-means[:,None,:2]
    coords=(U.swapaxes(-1,-2)[:,None]@d[...,None])[...,0]
    supported=np.linalg.norm(np.where(active[:,None,:],0.,coords),axis=-1)<=1e-7
    gain=covariances[:,:,:2]@inverse
    posterior_mean=means[:,None,:]+(gain[:,None]@d[...,None])[...,0]
    posterior_mean[:,:,:2]=centers
    posterior_covariance=covariances-gain@covariances[:,:2,:]
    posterior_covariance[:,:2,:]=0.;posterior_covariance[:,:,:2]=0.
    posterior_covariance=(posterior_covariance+posterior_covariance.swapaxes(-1,-2))/2
    logdet=np.log(np.where(active,e,1.)).sum(-1).astype(float)
    quadratic=((d[...,None,:]@inverse[:,None])@d[...,None])[...,0,0]
    logdensity=(-rank[:,None]*.5*math.log(2*math.pi)-.5*logdet[:,None]-.5*quadratic)
    logdensity=np.where(supported,logdensity,-np.inf)
    return posterior_mean,posterior_covariance,logdensity,rank,supported


@dataclass(frozen=True)
class Belief:
    graph: ObservedGraph
    components: tuple
    timestamp: float
    robot_pose: np.ndarray
    model_version: str
    diagnostics: dict
    role: str='actual'
    terminal: bool=False
    receipt: str=''
    carrier: object=None
    stochastic_source: str=''
    next_slot: int=-1
    lineage_namespace: str=''
    _cached_fingerprint: object=field(default=None,init=False,repr=False,compare=False)

    def __post_init__(self):
        # Compatibility for callers constructing a new belief from components.
        # Existing forks retain their high-water mark and cannot recycle slots.
        if self.next_slot==-1:
            object.__setattr__(self,'next_slot',max((int(s)+1 for c in self.components for s in c.slots),default=0))

    def fork(self,**kw):return replace(self,**kw)

    def fingerprint(self):
        cached=self._cached_fingerprint
        if cached is not None and cached[0]() is self:return cached[1]
        h=hashlib.blake2b(digest_size=20)
        h.update(repr((self.graph.namespace,self.timestamp,self.model_version,self.role,self.terminal,self.receipt,self.stochastic_source,self.next_slot,self.lineage_namespace)).encode())
        h.update(self.robot_pose.tobytes())
        for c in self.components:
            for k,v in vars(c).items():h.update(v.tobytes() if isinstance(v,np.ndarray) else repr(v).encode())
        return h.hexdigest()


@dataclass(frozen=True)
class EndpointSequence:
    worlds: tuple
    positions: np.ndarray
    velocities: np.ndarray
    stopped: np.ndarray
    switches: np.ndarray
    counters: np.ndarray

    def __len__(self):return self.positions.shape[1]

    def __getitem__(self,index):
        if isinstance(index,slice):return tuple(self[j] for j in range(len(self))[index])
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return tuple(sealed(c.fork(mean=np.concatenate([self.positions[w,index,:c.n],self.velocities[w,index,:c.n]],-1),
            covariance=np.zeros_like(c.covariance),stopped=self.stopped[w,index,:c.n],next_switch=self.switches[w,index,:c.n],
            counters=self.counters[w,index,:c.n])) for w,c in enumerate(self.worlds))

    def subset(self,indices):
        return EndpointSequence(self.worlds,freeze(self.positions[:,indices]),freeze(self.velocities[:,indices]),
            freeze(self.stopped[:,indices]),freeze(self.switches[:,indices]),freeze(self.counters[:,indices]))


@dataclass(frozen=True)
class JointFuture:
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    stopped: np.ndarray
    weights: np.ndarray
    slots: tuple
    hidden_positions: np.ndarray
    source: Belief
    endpoints: tuple
    seed: int
    rng_state: dict
    source_fingerprint: str
    model_version: str
    diagnostics: dict
    integrity: str=''
    parent_tape: object=None
    hidden_velocities: object=None

    @property
    def track_keys(self):
        worlds=self.endpoints.worlds if isinstance(self.endpoints,EndpointSequence) else self.endpoints[0]
        return tuple(tuple((self.source.lineage_namespace,int(slot),float(birth)) for slot,birth in zip(c.slots,c.birth)) for c in worlds)

    def identity(self):
        return self.integrity or self._digest()

    def _digest(self):
        h=hashlib.blake2b(digest_size=20)
        for x in (self.times,self.positions,self.velocities,self.stopped,self.weights,self.hidden_positions,*self.slots):h.update(x.tobytes())
        if self.hidden_velocities is not None:h.update(self.hidden_velocities.tobytes())
        if isinstance(self.endpoints,EndpointSequence):
            h.update(self.endpoints.switches.tobytes());h.update(self.endpoints.counters.tobytes());endpoint_list=(self.endpoints.worlds,)
        else:endpoint_list=self.endpoints
        for endpoint in endpoint_list:
            for c in endpoint:
                for v in vars(c).values():h.update(v.tobytes() if isinstance(v,np.ndarray) else repr(v).encode())
        h.update((self.source_fingerprint+self.model_version+str(self.rng_state)).encode())
        if self.parent_tape is not None:h.update(self.parent_tape.identity().encode())
        return h.hexdigest()


class IntermittentWorldModel:
    def __init__(self,learned=None,law=None,max_components=32,observation_law=None,population=None,seed=221):
        if not isinstance(max_components,int) or not 4<=max_components<=256:raise ValueError('invalid carrier budget')
        if learned is None:
            import torch
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(seed))
                learned=LearnedIntermittent(IntermittentDynamics(64))
        self.learned=learned
        self.law={**DEFAULT,**(law or {})}
        if any(not math.isfinite(v) or v<=0 for v in self.law.values()):raise ValueError('invalid motion law')
        self.max_components=max_components;self.seed=int(seed)
        baseline=Path(__file__).resolve().parents[2]/'docs/l2/graph_wm_120k_20261021/artifacts'
        self.observation_law=observation_law or EndpointObservationLaw(json.loads((baseline/'checkpoints/frozen_observation_fit.json').read_text())['parameters'])
        self.population=np.asarray(population if population is not None else json.loads((baseline/'population_prior.json').read_text()),float)
        if not np.isfinite(self.population).all() or np.any(self.population<0) or self.population.sum()<=0:raise ValueError('invalid population')
        self.population=freeze(self.population/self.population.sum())
        digest=hashlib.sha256(json.dumps(self.law,sort_keys=True).encode());stochastic=digest.copy()
        for name in ('model.py','dynamics.py','neural.py','fast_kernels.py','sampling_kernel.py','random_kernel.py','association.py','observations.py'):digest.update(Path(__file__).with_name(name).read_bytes())
        reference=Path(__file__).resolve().parents[2]/'docs/l2/intermittent_wm_20261022/artifacts/speed_reference_34425709'
        identities=dict(model='34425709c014231148a6028e55ddf68ae0d4c48582774c894f1ccd97daee9473',
            dynamics='a527b0950a824955da33a3310183a664d71fd7043c0619c09cd8f5e33d60f74b',
            neural='d8acc1826f3f25498c21b94a88ab7803beedce592b69d20b3d0c2a3f986dd6b3')
        for name,expected in identities.items():
            payload=(reference/(name+'.py')).read_bytes()
            if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('changed stochastic identity reference')
            stochastic.update(payload)
        for k,v in self.learned.network.state_dict().items():
            payload=k.encode()+v.numpy().tobytes();digest.update(payload);stochastic.update(payload)
        configuration=repr((LAW_VERSION,max_components,seed,tuple(self.population),vars(self.observation_law))).encode()
        digest.update(configuration);stochastic.update(configuration)
        self.model_version='l2-intermittent-1-'+digest.hexdigest()[:20]
        self.stochastic_version='l2-intermittent-1-'+stochastic.hexdigest()[:20]
        self.law=MappingProxyType(self.law)
        self._parameter_versions=tuple(p._version for p in self.learned.network.state_dict().values())
        self._optical_signature=repr((vars(self.observation_law),self.observation_law.hidden_detection,self.observation_law.birth_sigma))
        self._futures=weakref.WeakValueDictionary()
        self._beliefs=weakref.WeakValueDictionary()

    @classmethod
    def load(cls,path,law=None,max_components=32):
        ck=__import__('torch').load(path,map_location='cpu')
        return cls(LearnedIntermittent.load(path),law or ck['configuration']['law'],max_components)

    def _check(self,b,actual=False):
        if tuple(p._version for p in self.learned.network.state_dict().values())!=self._parameter_versions:raise ValueError('model parameters/buffers changed; reload as a new model version')
        if repr((vars(self.observation_law),self.observation_law.hidden_detection,self.observation_law.birth_sigma))!=self._optical_signature:raise ValueError('optical law changed; reload as a new model version')
        if not isinstance(b,Belief) or b.model_version!=self.model_version:raise ValueError('foreign belief')
        if actual and b.role!='actual':raise ValueError('actual evidence requires separately retained continuous actual belief')
        if self._beliefs.get(id(b)) is b:return
        w=np.asarray([c.weight for c in b.components])
        if not len(w) or not np.isfinite(w).all() or np.any(w<0) or abs(w.sum()-1)>1e-9:raise ValueError('invalid normalized carrier')
        if not math.isfinite(b.timestamp) or b.robot_pose.shape!=(4,) or not np.isfinite(b.robot_pose).all():raise ValueError('invalid pose/time')
        if not isinstance(b.next_slot,(int,np.integer)) or b.next_slot<max((int(s)+1 for c in b.components for s in c.slots),default=0):raise ValueError('invalid persistent slot allocator')
        for c in b.components:
            if c.mean.shape!=(c.n,4) or c.covariance.shape!=(c.n,4,4) or len(set(c.slots))!=c.n:raise ValueError('invalid track state')
            if not np.isfinite(c.mean).all() or not np.isfinite(c.covariance).all() or not np.isfinite(c.last_seen).all() or np.any(c.last_seen>b.timestamp+1e-8):raise ValueError('invalid continuous state')
            if c.n and np.linalg.eigvalsh(c.covariance).min()<-1e-8:raise ValueError('indefinite continuous covariance')
            if c.n and np.max(abs(c.covariance-c.covariance.swapaxes(-1,-2)))>1e-9:raise ValueError('nonsymmetric covariance')
            for a in (c.slots,c.last_seen,c.birth,c.stopped,c.diffusion,c.next_switch,c.counters):
                if a.shape!=(c.n,):raise ValueError('invalid latent shape')
            if (c.slots.dtype.kind not in 'iu' or c.counters.dtype.kind not in 'iu'
                    or np.any(c.slots<0) or np.any(c.counters<0) or c.stopped.dtype.kind!='b'):
                raise ValueError('track slots/counters must be nonnegative integers; modes must be Boolean')
            if c.last_position.shape!=(c.n,2) or not np.isfinite(c.last_position).all():raise ValueError('invalid last measured position')
            if c.last_position_valid is not None and (c.last_position_valid.shape!=(c.n,) or c.last_position_valid.dtype.kind!='b'):raise ValueError('invalid observation provenance')
            if c.intent.shape!=(c.n,2) or not np.isfinite(c.intent).all() or not np.isfinite(c.next_switch).all() or (not b.terminal and np.any(c.next_switch<b.timestamp-1e-8)):raise ValueError('invalid latent clock/intent')
            if c.hidden_pmf.ndim!=1 or not np.isfinite(c.hidden_pmf).all() or np.any(c.hidden_pmf<0) or abs(c.hidden_pmf.sum()-1)>1e-9 or not np.isfinite(c.diffusion).all() or np.any(c.diffusion<=0):raise ValueError('invalid hidden mass/diffusion')
            if not np.isfinite(c.birth).all() or np.any(c.birth>c.last_seen+1e-8):raise ValueError('invalid track birth')
            if c.hidden_mean is not None and (c.hidden_mean.shape!=(4,) or c.hidden_covariance.shape!=(4,4) or not np.isfinite(c.hidden_mean).all() or not np.isfinite(c.hidden_covariance).all()):raise ValueError('invalid unlocalized spatial prior')
            if c.hidden_mean is not None and (np.max(abs(c.hidden_covariance-c.hidden_covariance.T))>1e-9 or np.linalg.eigvalsh(c.hidden_covariance).min()<-1e-8):raise ValueError('invalid unlocalized covariance')
        # Cache validation only for internally owned, byte-backed immutable data.
        arrays=[b.robot_pose]+[a for c in b.components for a in vars(c).values() if isinstance(a,np.ndarray)]
        def owned(a):
            if a.flags.writeable:return False
            while isinstance(a.base,np.ndarray):a=a.base
            return isinstance(a.base,bytes)
        if isinstance(b.graph.cells,MappingProxyType) and all(owned(a) for a in arrays):
            # Only fully checked immutable state receives this cache. dataclass
            # replacement clears it, including every legal graph/pose fork.
            object.__setattr__(b,'_cached_fingerprint',(weakref.ref(b),b.fingerprint()));self._beliefs[id(b)]=b

    def from_legacy(self,b,seed=None):
        """Explicit legal-history bootstrap; never accepts unresolved bound mass."""
        if b.unresolved_mass!=0 or abs(sum(h.weight for h in b.hypotheses)-1)>1e-9:raise ValueError('incomplete bootstrap measure')
        seed=self.seed if seed is None else int(seed);rng=np.random.default_rng(seed);cs=[]
        weights=np.array([h.weight for h in b.hypotheses]);weights/=weights.sum()
        # Stratified finite prior-carrier approximation; no posterior mass bound.
        choices=np.searchsorted(np.cumsum(weights),(np.arange(self.max_components)+rng.random())/self.max_components)
        for j,hi in enumerate(choices):
            h=b.hypotheses[hi];m=h.mean.copy();P=h.covariance.copy()
            if h.maneuver_mean is not None:
                pick=(rng.random((h.n,1))>np.cumsum(h.maneuver_probability,axis=-1)).sum(-1)
                m=h.maneuver_mean[np.arange(h.n),pick].copy();P=h.maneuver_covariance[np.arange(h.n),pick].copy()
            target,q,pr,ps=self.learned.parameters(h.mean,b.timestamp-h.last_seen,b.graph)
            mode=(rng.random((h.n,1))>np.cumsum(pr,axis=-1)).sum(-1).clip(0,2)
            st=rng.random(h.n)<ps
            m[st,2:]=0.;P[st,2:,:]=0.;P[st,:,2:]=0.;P[st,2:,2:]=np.eye(2)*self.law['stop_reset_std']**2
            cs.append(sealed(schedule(Component(1/self.max_components,m,P,h.slot_ids.copy(),h.last_seen.copy(),
                h.birth_time.copy() if h.birth_time is not None else h.last_seen.copy(),h.mean[:,:2].copy(),st,
                target[np.arange(h.n),mode],q[np.arange(h.n),mode],np.zeros(h.n),np.zeros(h.n,int),
                int(rng.integers(0,2**63)),h.hidden_pmf.copy(),h.visibility_mode,j,
                h.hidden_mean[0].copy() if h.hidden_mean is not None else np.r_[b.robot_pose[[0,2]],[0.,0.]],
                h.hidden_covariance[0].copy() if h.hidden_covariance is not None else np.diag([100.,100.,.36,.36]),
                np.isclose(h.last_seen,b.timestamp,atol=1e-8,rtol=0)&(np.max(abs(h.covariance[:,:2,:2]),axis=(1,2))<1e-12)),b.timestamp,self.law)))
        result=Belief(freeze_graph(b.graph),tuple(cs),float(b.timestamp),freeze(b.robot_pose),self.model_version,
                      dict(bootstrap='frozen normalized legal-history estimator',carrier_components=len(cs),probability_status='normalized finite CTMC mixture approximation',
                           bootstrap_ledger=b.evidence.get('normalized_approximation'),ancestry='anonymous generation-qualified slots',
                           reliability='UNKNOWN: finite support and uncalibrated hidden geometry; absence is not safety'),terminal=b.terminal,
                      next_slot=max((int(s)+1 for c in cs for s in c.slots),default=0),
                      lineage_namespace=hashlib.blake2b((b.graph.namespace+str(seed)).encode(),digest_size=12).hexdigest())
        self._check(result);return result

    def initialize(self,legal_history,observed_graph,episode_namespace=None):
        if not legal_history:raise ValueError('empty legal history')
        history=[];receipts={};last_time=-math.inf;duplicate_prefix=False
        for row in legal_history:
            t=float(row['timestamp']);z=detections(row)
            if len(z)>1 and np.min(np.linalg.norm(z[:,None]-z[None,:]+np.eye(len(z))[:,:,None]*1e8,axis=-1))<1e-9:
                duplicate_prefix=True
            if t<last_time:raise ValueError('noncausal legal history')
            last_time=t
            available=row.get('human_sensor_available',True) and row.get('observation_valid',True)
            if available and t in receipts:
                if receipts[t]!=self._receipt(row):raise ValueError('conflicting same-time history evidence')
                # An exact retransmission must not change the bootstrap seed,
                # estimator, or amount of evidence. Preserve distinct legal
                # graph/pose arrivals at that time without repeating humans.
                if any(repr(previous)==repr(row) for previous in history if float(previous['timestamp'])==t):continue
                # Keep legal graph/pose arrivals while suppressing repeat humans.
                row={**row,'human_sensor_available':False,'tracks':[]}
            elif available:receipts[t]=self._receipt(row)
            history.append(row)
        # Keep the validated network-independent legal-prefix estimator. A
        # repaired-CTMC replay from an empty prior was evaluated and rejected:
        # it inflated track support in dense native histories. Subsequent
        # persistent updates use the exact finite association sum-product law.
        return self._initialize_legal_prefix(history,observed_graph,receipts,episode_namespace,duplicate_prefix)

    def _initialize_legal_prefix(self,history,observed_graph,receipts,episode_namespace,duplicate_prefix=False):
        from eventwm_graph.l2_120k.model import load_model
        baseline=Path(__file__).resolve().parents[2]/'docs/l2/graph_wm_120k_20261021/artifacts'
        legacy=load_model(population=self.population,motion=json.loads((baseline/'motion.json').read_text()),
                          observation=str(baseline/'checkpoints/frozen_observation_fit.json'))
        b=legacy._initialize_history(history,observed_graph)
        seed=int.from_bytes(hashlib.blake2b((str(self.seed)+b.fingerprint()).encode(),digest_size=8).digest(),'little')
        result=self.from_legacy(b,seed=seed);components=[]
        for c in result.components:
            loc=c.last_position.copy();valid=np.zeros(c.n,bool)
            for i,seen in enumerate(c.last_seen):
                frames=[detections(row) for row in history if abs(float(row['timestamp'])-seen)<1e-8]
                zs=np.concatenate(frames,axis=0) if frames else np.empty((0,2))
                if len(zs)==1:loc[i]=zs[0];valid[i]=True
                elif abs(seen-result.timestamp)<1e-8:
                    match=np.flatnonzero(np.linalg.norm(zs-c.mean[i,:2],axis=-1)<1e-7)
                    if len(match)==1:loc[i]=zs[match[0]];valid[i]=True
            components.append(sealed(c.fork(last_position=loc,last_position_valid=valid)))
        lineage=hashlib.blake2b((result.graph.namespace+str(seed)+str(episode_namespace)).encode(),digest_size=12).hexdigest()
        return result.fork(components=tuple(components),receipt=receipts.get(result.timestamp,''),lineage_namespace=lineage,
             diagnostics={**result.diagnostics,'bootstrap':'validated network-independent legal-prefix estimator',
                          'bootstrap_support_status':('UNKNOWN_UNSUPPORTED_DUPLICATE_PREFIX_LEGACY_APPROXIMATION' if duplicate_prefix else 'UNKNOWN_LEGACY_PREFIX_APPROXIMATION'),
                          'bootstrap_duplicate_fallback':duplicate_prefix,'association_history_search_complete':False,
                          'reliability':'UNKNOWN: approximate legacy prefix support; finite posterior carrier; no identity/count certification'})

    def propagate(self,b,timestamp):
        self._check(b);t=float(timestamp)
        if not math.isfinite(t) or t<b.timestamp-1e-9 or t-b.timestamp>30:raise ValueError('invalid propagation time')
        if t==b.timestamp:return b
        if b.terminal:return b.fork(timestamp=t)
        if b.carrier is not None:
            f,index=b.carrier
            f=f.parent_tape if f.parent_tape is not None else f
            match=np.flatnonzero(np.isclose(f.times,t,rtol=0,atol=1e-9))
            if not len(match):
                if t<=f.times[-1]+1e-8:raise ValueError('finite successor supports held interior endpoints only')
                continued=self.predict_shared(b,t-b.timestamp,len(b.components),f.seed,times=[0.,t-b.timestamp])
                return self.successor_at(continued,1,np.array([c.weight for c in b.components]),b.robot_pose)
            return self.successor_at(f,int(match[0]),b.components_weights if hasattr(b,'components_weights') else np.array([c.weight for c in b.components]),b.robot_pose)
        propagated=(evolve_many(b.components,b.timestamp,t,self.law) if type(self).evolve is IntermittentWorldModel.evolve
                    else tuple(self.evolve(c,b.timestamp,t) for c in b.components))
        return b.fork(timestamp=t,receipt='',components=tuple(sealed(c) for c in propagated),
                      diagnostics={**b.diagnostics,'propagation':'retained conditional Gaussian CTMC histories; no moment collapse'})

    @staticmethod
    def _receipt(ob):
        z=detections(ob)
        if len(z):z=z[np.lexsort((z[:,1],z[:,0]))]
        return hashlib.blake2b(repr((float(ob['timestamp']),bool(ob.get('human_sensor_available',True)),bool(ob.get('observation_valid',True)))).encode()+z.tobytes(),digest_size=16).hexdigest()

    def observe(self,b,ob):
        self._check(b,actual=True)
        t=float(ob['timestamp']);pose=np.asarray(ob['robot_pose'],float)
        if pose.shape!=(4,) or not np.isfinite(pose).all():raise ValueError('invalid arrived pose')
        z=detections(ob)
        if not np.isfinite(z).all() or len(z)>16:raise ValueError('invalid detection domain')
        if ob.get('certified_empty_boxes'):raise ValueError('native empty frame is not certified spatial emptiness')
        for u in ob.get('static_updates',[]):
            if u['state'] not in (0,1,2) or len(u['cell'])!=2 or not np.isfinite(u['height']) or float(u.get('timestamp',t))>t+1e-8:raise ValueError('invalid/future static arrival')
        updated=self.propagate(b,t).fork(graph=freeze_graph(b.graph.update(ob.get('static_updates',[]))),robot_pose=freeze(pose))
        receipt=self._receipt(ob)
        if receipt==b.receipt or b.terminal:return updated
        if not ob.get('human_sensor_available',True) or not ob.get('observation_valid',True):
            return updated.fork(diagnostics={**updated.diagnostics,'human_evidence':'unavailable; no negative evidence'})
        if abs(t-b.timestamp)<1e-10 and b.receipt:raise ValueError('conflicting second human emission at same timestamp')
        if len(z)>1 and np.min(np.linalg.norm(z[:,None]-z[None,:]+np.eye(len(z))[:,:,None]*1e8,axis=-1))<1e-9:raise ValueError('duplicate exact centers unsupported')
        # Canonical ordering makes seeded posterior samples equivariant to input
        # serialization. Exact sum-product below, rather than sorting, removes
        # the former probabilistic order bias.
        if len(z):z=z[np.lexsort((z[:,1],z[:,0]))]
        nextslot=max(updated.next_slot,max((int(s)+1 for c in updated.components for s in c.slots),default=0))
        pd={reg:np.asarray(self.observation_law.detection(z,pose,reg),float).reshape(-1) for reg in (0,1)}
        positives={};corrections={};tables={};hidden_posts={};log_evidence=[];dimensions=[]
        birth_cache={};hidden_cache={}
        # Choose quadrature per track, independently of which OTHER tracks emit.
        # This gives a well-defined factored approximate miss likelihood for DP.
        groups={};aliases={};unique={}
        for ci,c in enumerate(updated.components):
            if c.weight<=0:continue
            for i in range(c.n):
                m=c.mean[i].astype(float);P=c.covariance[i].astype(float)
                order=9 if np.linalg.eigvalsh(P[:2,:2]).max()<=.25 else 15
                key=(c.visibility_mode,order,m.tobytes(),P.tobytes())
                if type(self.observation_law) is not EndpointObservationLaw:key=(ci,i)
                if key not in unique:
                    unique[key]=(m,P);groups.setdefault((c.visibility_mode,order),[]).append(key)
                aliases[(ci,i)]=key
        for (reg,order),keys in groups.items():
            ms=np.stack([unique[k][0] for k in keys]);Ps=np.stack([unique[k][1] for k in keys])
            h=Hypothesis(1.,ms,Ps,np.zeros(len(ms)),np.arange(len(ms)),np.array([1.]),visibility_mode=reg)
            mass,mm,cc=self.observation_law.missed_moments(h,pose)
            if np.any(mass<=0) or np.any(mass>1+1e-12) or not np.isfinite(mass).all():raise ValueError('invalid miss likelihood')
            for j,key in enumerate(keys):corrections[key]=(float(mass[j]),mm[j],cc[j])
        # Batch positive disintegrations; the DP only reads numerical tables.
        eligible=[(ci,c) for ci,c in enumerate(updated.components) if c.weight>0 and c.n]
        if eligible and len(z):
            ms=np.concatenate([c.mean for _,c in eligible]).astype(float)
            Ps=np.concatenate([c.covariance for _,c in eligible]).astype(float)
            mm,cc,ll,rr,ok=position_conditions(ms,Ps,z);offset=0
            for ci,c in eligible:
                positives[ci]=(mm[offset:offset+c.n],cc[offset:offset+c.n],ll[offset:offset+c.n],rr[offset:offset+c.n])
                offset+=c.n
        for ci,c in enumerate(updated.components):
            if c.weight<=0:
                dimensions.append(100000);log_evidence.append(-math.inf);continue
            hidden_key=(c.visibility_mode,c.hidden_pmf.dtype.str,c.hidden_pmf.tobytes())
            if hidden_key not in hidden_cache:
                hp=[c.hidden_pmf];den=[]
                for _ in z:
                    post,value=consume_hidden(hp[-1])
                    if post is None:break
                    hp.append(post);den.append(value)
                factor=(1-self.observation_law.hidden_detection[c.visibility_mode])**np.arange(len(c.hidden_pmf))
                hlogs=[];hposts=[]
                for prior in hp:
                    value=float(prior@factor)
                    hlogs.append(math.log(value) if value>0 else -math.inf)
                    hposts.append(prior*factor/value if value>0 else prior)
                hidden_cache[hidden_key]=(den,hlogs,hposts)
            den,hlogs,hposts=hidden_cache[hidden_key]
            hidden_posts[ci]=hposts
            birth_rank=2
            if len(z) and len(den):
                if c.hidden_mean is not None:
                    birth_key=(c.hidden_mean.dtype.str,c.hidden_covariance.dtype.str,c.hidden_mean.tobytes(),c.hidden_covariance.tobytes())
                    if birth_key not in birth_cache:birth_cache[birth_key]=position_conditions(c.hidden_mean[None],c.hidden_covariance[None],z)
                    birth_conditions=birth_cache[birth_key]
                    bd=birth_conditions[2][0];birth_rank=int(birth_conditions[3][0])
                else:bd=np.log(self.observation_law.birth_density(z,pose))
                bd=bd+math.log(self.observation_law.hidden_detection[c.visibility_mode])
            else:bd=np.full(len(z),-np.inf)
            if ci in positives:_,_,ll,rr=positives[ci]
            else:ll=np.empty((c.n,len(z)));rr=np.zeros(c.n,np.int64)
            miss=np.array([corrections[aliases[(ci,i)]][0] for i in range(c.n)])
            with np.errstate(divide='ignore'):
                positive,lmiss,table=assignment_prepare(ll,rr,np.log(pd[c.visibility_mode]),miss,bd,den,hlogs,birth_rank)
            tables[ci]=(positive,lmiss,rr,table)
            dimensions.append(int(table[0][0,0]));log_evidence.append(math.log(c.weight)+float(table[1][0,0]))
        best=min((r for r,l in zip(dimensions,log_evidence) if math.isfinite(l)),default=100000)
        logs=np.array([l if r==best else -math.inf for l,r in zip(log_evidence,dimensions)])
        total=logsumexp(logs)
        if not math.isfinite(total):raise ValueError('zero represented exact-observation support')
        weights=np.exp(logs-total)
        stochastic_fingerprint=updated.fork(model_version=self.stochastic_version).fingerprint()
        rng=np.random.default_rng(int(hashlib.blake2b((stochastic_fingerprint+receipt).encode(),digest_size=8).hexdigest(),16))
        chosen=np.searchsorted(np.cumsum(weights),(np.arange(self.max_components)+rng.random())/self.max_components)
        sampled=[];materialized={};selected_mass={};assignments={}
        for ci in chosen:
            ci=int(ci);c=updated.components[ci];positive,lmiss,ranks,table=tables[ci]
            indices,logp=sample_assignment(positive,ranks,lmiss,table[0],table[1],rng.random(c.n))
            assign=tuple(int(c.slots[i]) if i>=0 else nextslot+zi for zi,i in enumerate(indices))
            key=(ci,assign);sampled.append(key)
            selected_mass[key]=float(weights[ci]*math.exp(logp))
            assignments[assign]=assignments.get(assign,0.)+1/self.max_components
            if key in materialized:continue
            m=c.mean.copy();P=c.covariance.copy();last=c.last_seen.copy();loc=c.last_position.copy()
            valid=c.last_position_valid.copy() if c.last_position_valid is not None else np.zeros(c.n,bool)
            used=set(int(i) for i in indices if i>=0);nb=int(np.sum(indices<0))
            for zi,i in enumerate(indices):
                if i<0:continue
                m[i]=positives[ci][0][i,zi];P[i]=positives[ci][1][i];last[i]=t;loc[i]=z[zi];valid[i]=True
            for i in range(c.n):
                if i not in used:_,m[i],P[i]=corrections[aliases[(ci,i)]]
            h=c.fork(mean=m,covariance=P,last_seen=last,last_position=loc,last_position_valid=valid,hidden_pmf=hidden_posts[ci][nb])
            for zi,i in enumerate(indices):
                if i<0:
                    piece=self._birth_piece(c,z[zi],t,nextslot+zi,updated.graph,zi)
                    piece['last_position_valid']=np.ones(1,bool)
                    h=h.fork(**{k:np.concatenate([getattr(h,k),v],axis=0) for k,v in piece.items()})
            materialized[key]=h
        selected=[materialized[k] for k in sampled]
        params=self.learned.parameters_many(selected,t,updated.graph)
        result=[];workspace=KeyedWorkspace()
        for outi,h in enumerate(selected):
            ix=np.flatnonzero(np.isclose(h.last_seen,t,atol=1e-8,rtol=0))
            if len(ix):
                target,q,pr,ps=params[outi];intent=h.intent.copy();diff=h.diffusion.copy()
                probabilities=pr[ix]/pr[ix].sum(-1,keepdims=True)
                if not np.isfinite(probabilities).all() or np.any(probabilities<0):raise ValueError('invalid categorical probabilities')
                p=probabilities.astype(float);tolerance=max(math.sqrt(np.finfo(float).eps),math.sqrt(np.finfo(probabilities.dtype).eps))
                if np.any(abs(p.sum(-1)-1)>tolerance):raise ValueError('invalid categorical probabilities')
                cdf=p.cumsum(-1);cdf/=cdf[:,-1:]
                uniforms=np.empty(len(ix))
                for j,i in enumerate(ix):
                    mode_rng=workspace.generator(h.seed,int(h.slots[i]),int(round(t*1e6))+100000+outi*1000000007)
                    uniforms[j]=mode_rng.random()
                modes=(uniforms[:,None]>=cdf).sum(-1)
                intent[ix]=target[ix,modes];diff[ix]=q[ix,modes]
                h=h.fork(intent=intent,diffusion=diff)
            h=h.fork(weight=1/self.max_components,seed=int(rng.integers(0,2**63)),counters=np.zeros(h.n,int))
            result.append(sealed(schedule(h,t,self.law,workspace)))
        entropy=float(-sum(w*math.log(w) for w in weights if w))
        conditional_entropy=sum(weights[ci]*tables[ci][3][2] for ci in tables if weights[ci]>0)
        diag={**updated.diagnostics,'human_evidence':'positive' if len(z) else 'available_empty_not_spatial_certificate',
              'association_assignments':[(list(k),v) for k,v in assignments.items()],
              'association_assignment_status':'empirical final carrier; exact pre-carrier evidence and conditional entropy reported separately',
              'association_entropy':float(-sum(p*math.log(p) for p in assignments.values() if p)),
              'association_entropy_semantics':'joint assignment entropy of sampled final carrier; finite support only',
              'association_entropy_conditional_on_prior_component':float(conditional_entropy),
              'component_assignment_entropy':float(entropy+conditional_entropy),
              'association_search_complete':True,'association_evidence_log_coefficient':float(total),
              'association_small_ball_dimension':int(best),'beam_retained_fraction_of_enumerated':1.,
              'enumerated_candidates':float(sum(tables[ci][3][3] for ci in tables if weights[ci]>0)),
              'component_entropy':entropy,'resampled_ancestor_count':len(materialized),'future_clock_count':len(result),
              'pre_resampling_effective_components':float(1/(weights@weights)),
              'resampling_omitted_probability':float(max(0.,1-sum(selected_mass.values()))),
              'approximation':'exact finite-prior assignment sum; isotropic product small-ball conditioning; finite CTMC carrier sampling; per-track GH miss moments; no posterior support guarantee'}
        out=updated.fork(components=tuple(result),receipt=receipt,next_slot=nextslot+len(z),diagnostics=diag)
        self._check(out);return out

    def _birth_piece(self,c,z,t,slot,graph,offset):
        if type(self)._birth is not IntermittentWorldModel._birth:
            template=self._birth(c,z,t,slot,c.hidden_pmf,graph,offset)
            return {k:v[-1:] for k,v in vars(template).items() if isinstance(v,np.ndarray)
                and k in ('mean','covariance','slots','last_seen','birth','last_position','stopped','intent','diffusion','next_switch','counters')}
        rng=keyed_rng(c.seed,slot,0)
        st=bool(rng.random()<self.law['stop_rate']/(self.law['stop_rate']+self.law['restart_rate']))
        covariance=np.diag([0.,0.,.64,.64])[None]
        if st:covariance[0,2:,2:]=np.eye(2)*self.law['stop_reset_std']**2
        rate=self.law['restart_rate'] if st else self.law['stop_rate']
        return dict(mean=np.r_[z,[0.,0.]][None],covariance=covariance,
            slots=np.array([slot],dtype=np.result_type(c.slots,slot)),
            last_seen=np.array([t],dtype=np.result_type(c.last_seen,t)),
            birth=np.array([t],dtype=np.result_type(c.birth,t)),last_position=z[None],
            stopped=np.array([st]),intent=np.zeros((1,2)),diffusion=np.array([.05],dtype=np.result_type(c.diffusion,.05)),
            next_switch=np.array([t+rng.exponential(1/rate)]),counters=np.array([0],dtype=np.result_type(c.counters,0)))

    def _birth(self,c,z,t,slot,hp,graph,offset):
        m=np.r_[c.mean,np.r_[z,[0.,0.]][None]];P=np.concatenate([c.covariance,np.diag([0.,0.,.64,.64])[None]])
        # Encoder parameters are installed for ALL selected positive tracks in
        # one batch after association. No transition occurs inside this arrival.
        rng=keyed_rng(c.seed,slot,0)
        st=bool(rng.random()<self.law['stop_rate']/(self.law['stop_rate']+self.law['restart_rate']))
        if st:P[-1,2:,2:]=np.eye(2)*self.law['stop_reset_std']**2
        rate=self.law['restart_rate'] if st else self.law['stop_rate']
        return c.fork(mean=m,covariance=P,slots=np.r_[c.slots,slot],last_seen=np.r_[c.last_seen,t],birth=np.r_[c.birth,t],last_position=np.r_[c.last_position,z[None]],
            stopped=np.r_[c.stopped,st],intent=np.r_[c.intent,np.zeros((1,2))],diffusion=np.r_[c.diffusion,.05],
            next_switch=np.r_[c.next_switch,t+rng.exponential(1/rate)],counters=np.r_[c.counters,0],hidden_pmf=hp)

    def predict_shared(self,b,horizon=3.,num_worlds=16,seed=0,times=None):
        self._check(b)
        if b.role=='imagined' and b.stochastic_source and b.carrier is None:raise ValueError('detached future-source metadata is not an operable successor; use successor_at')
        if not math.isfinite(horizon) or not 0<=horizon<=30 or isinstance(num_worlds,bool) or not isinstance(num_worlds,(int,np.integer)) or not 1<=num_worlds<=256:raise ValueError('invalid forecast budget')
        rel=np.asarray(times if times is not None else np.unique(np.r_[0.,np.arange(1,int(horizon/.048)+1)*.048,horizon]),float)
        if rel.ndim!=1 or not len(rel) or rel[0]!=0 or not np.isfinite(rel).all() or np.any(np.diff(rel)<=0) or abs(rel[-1]-horizon)>1e-8:raise ValueError('invalid forecast clock')
        if b.terminal:rel=np.array([0.])
        if b.carrier is not None:
            return self._reuse_carrier(b,rel,num_worlds,seed)
        rng=np.random.default_rng(seed);ix=rng.choice(len(b.components),size=num_worlds,p=[c.weight for c in b.components]);worlds=[]
        for i in ix:
            c=b.components[i];m=c.mean+np.einsum('nij,nj->ni',root_psd(c.covariance),rng.normal(size=c.mean.shape))
            # Retain the component's immutable CTMC history. Redrawing clocks here
            # would change the finite represented law after API segmentation.
            c=c.fork(weight=1/num_worlds,mean=m,covariance=np.zeros_like(c.covariance))
            worlds.append(c)
        N=max((c.n for c in worlds),default=0);T=len(rel)
        pos=np.full((num_worlds,T,N,2),np.nan);vel=pos.copy();st=np.zeros((num_worlds,T,N),bool);ends=[]
        hidden=[];hidden_velocity=[]
        for w,c in enumerate(worlds):
            count=int(rng.choice(len(c.hidden_pmf),p=c.hidden_pmf))
            pmf=np.zeros_like(c.hidden_pmf);pmf[count]=1.;worlds[w]=c.fork(hidden_pmf=pmf)
            hm=c.hidden_mean if c.hidden_mean is not None else np.r_[b.robot_pose[[0,2]],[0.,0.]]
            hc=c.hidden_covariance if c.hidden_covariance is not None else np.diag([100.,100.,.36,.36])
            state=hm+rng.normal(size=(count,4))@root_psd(hc).T
            hidden.append(state[None,:,:2]+rel[:,None,None]*state[None,:,2:])
            hidden_velocity.append(state[:,2:])
        nh=max((h.shape[1] for h in hidden),default=0);hp=np.full((num_worlds,T,nh,2),np.nan)
        hv=np.full((num_worlds,nh,2),np.nan)
        for w,h in enumerate(hidden):hp[w,:,:h.shape[1]]=h
        for w,h in enumerate(hidden_velocity):hv[w,:len(h)]=h
        pos,vel,st,next_series,counter_series=sample_joint_paths(worlds,rel,b.timestamp,self.law,rng,getattr(self,'cv_diffusion',None))
        pos,vel,st=freeze(pos),freeze(vel),freeze(st)
        ends=EndpointSequence(tuple(sealed(c) for c in worlds),pos,vel,st,freeze(next_series),freeze(counter_series))
        return self._seal_future(JointFuture(freeze(b.timestamp+rel),pos,vel,st,freeze(np.full(num_worlds,1/num_worlds)),
            tuple(freeze(c.slots) for c in worlds),freeze(hp),b,ends,int(seed),json.dumps(rng.bit_generator.state,sort_keys=True),b.fingerprint(),self.model_version,
            dict(law=LAW_VERSION,shared_joint_worlds=True,hidden='inherited broad unlocalized pool; not calibrated spatial absence',
                 stochastic_model_version=self.stochastic_version,implementation_version=self.model_version,backend='single-thread compiled loops; unchanged support/draw order',
                 temporal='conditional Gaussian CTMC paths; latent velocity intention persistent without evidence',world_source_indices=ix.tolist()),hidden_velocities=freeze(hv)))

    def get_previously_observed_tracks(self,b):
        self._check(b);result=[]
        for slot,birth in sorted({(int(s),float(bt)) for c in b.components for s,bt in zip(c.slots,c.birth)}):
            pairs=[(c,i) for c in b.components if c.weight>0 for i in np.flatnonzero((c.slots==slot)&(c.birth==birth))];mass=sum(c.weight for c,i in pairs)
            if mass<=0:continue
            m=sum(c.weight*c.mean[i] for c,i in pairs)/mass
            P=sum(c.weight*(c.covariance[i]+np.outer(c.mean[i]-m,c.mean[i]-m)) for c,i in pairs)/mass
            measurements=[dict(timestamp=float(c.last_seen[i]),position=c.last_position[i].tolist(),conditional_weight=float(c.weight/mass))
                          for c,i in pairs if c.last_position_valid is not None and c.last_position_valid[i]]
            result.append(dict(slot_id=slot,track_key=(b.lineage_namespace,slot,birth),generation=(slot,birth),birth_timestamp=birth,
                ancestry_status='anonymous lifetime; competing associations remain probabilistic',
                state_provenance=('legal_observation' if len(measurements)==len(pairs) else 'UNSUPPORTED_OBSERVATION_PROVENANCE') if all(abs(c.last_seen[i]-b.timestamp)<1e-8 for c,i in pairs) else 'propagated_prediction',
                last_observed_position_status='SUPPORTED' if len(measurements)==len(pairs) else 'UNKNOWN_OR_PARTIALLY_SUPPORTED',
                last_observed_measurements=measurements,last_seen_time=max(float(c.last_seen[i]) for c,i in pairs),
                bootstrap_support_status=b.diagnostics.get('bootstrap_support_status','FINITE_APPROXIMATION'),
                reliability_status='FINITE_APPROXIMATION_UNKNOWN_SAFETY',presence_probability=mass,mean=m,covariance=P,
                last_observed_timestamps=[float(c.last_seen[i]) for c,i in pairs],
                age=sum(c.weight*(b.timestamp-c.last_seen[i]) for c,i in pairs)/mass,
                track_age=sum(c.weight*(b.timestamp-c.birth[i]) for c,i in pairs)/mass,
                stop_probability=sum(c.weight*bool(c.stopped[i]) for c,i in pairs)/mass,
                association_uncertainty=b.diagnostics.get('association_entropy',0.),
                association_uncertainty_status=b.diagnostics.get('association_entropy_semantics','bootstrap approximation; joint assignment entropy unavailable'),
                association_entropy_upper_bound=b.diagnostics.get('component_assignment_entropy')))
        return result

    def project_future_distribution(self,future,spatial_regions,times):
        self._check_future(future);regions=np.asarray(spatial_regions,float);ts=np.asarray(times,float)
        if regions.ndim!=2 or regions.shape[1]!=4 or not np.isfinite(regions).all() or np.any(regions[:,:2]>regions[:,2:]):raise ValueError('invalid supplied regions')
        if ts.ndim!=1 or not np.isfinite(ts).all():raise ValueError('invalid projection times')
        indices=[]
        for t in ts:
            j=int(np.argmin(abs(future.times-t)))
            if abs(future.times[j]-t)>1e-8:raise ValueError('projection requires stored future timestamps')
            indices.append(j)
        x=future.positions[:,indices];h=future.hidden_positions[:,indices];allx=np.concatenate([x,h],2)
        inside=np.all((allx[:,:,:,None,:]>=regions[None,None,None,:,:2])&(allx[:,:,:,None,:]<=regions[None,None,None,:,2:]),axis=-1)
        return dict(times=ts,at_least_one_probability=np.einsum('w,wtr->tr',future.weights,inside.any(2)),
                    expected_count=np.einsum('w,wtr->tr',future.weights,inside.sum(2)),
                    observed_track_expected_count=np.einsum('w,wtr->tr',future.weights,inside[:,:,:x.shape[2]].sum(2)),
                    semantics='weighted joint-world center membership; overlapping regions need not sum to1; no static-free/visibility assumption',
                    hidden_population_status='broad retained prior included; zero sampled membership is not certified zero unknown-space risk',
                    reliability_status='FINITE_APPROXIMATION_UNKNOWN_SAFETY',worlds=len(future.weights),
                    source_fingerprint=future.source_fingerprint,future_identity=future.identity())

    def _check_future(self,f):
        if not isinstance(f,JointFuture) or f.model_version!=self.model_version or f.source_fingerprint!=f.source.fingerprint():raise ValueError('foreign or stale joint future')
        if not np.isfinite(f.weights).all() or np.any(f.weights<0) or abs(f.weights.sum()-1)>1e-9:raise ValueError('invalid future mass')
        if self._futures.get(id(f)) is not f:
            if f.parent_tape is not None:self._check_future(f.parent_tape)
            if not f.integrity or f.integrity!=f._digest():raise ValueError('altered joint stochastic carrier')
            self._futures[id(f)]=f

    def _seal_future(self,f):
        f=replace(f,integrity=f._digest());self._futures[id(f)]=f;return f

    def successor_at(self,f,index,masses=None,robot_pose=None):
        self._check_future(f)
        if not isinstance(index,(int,np.integer)) or not 0<=index<len(f.times):raise ValueError('invalid endpoint')
        weights=np.asarray(f.weights if masses is None else masses,float)
        if weights.shape!=f.weights.shape or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:raise ValueError('zero/invalid event support')
        weights=weights/weights.sum();cs=tuple(sealed(c.fork(weight=float(w))) for c,w in zip(f.endpoints[index],weights))
        pose=np.asarray(f.source.robot_pose if robot_pose is None else robot_pose,float)
        if pose.shape!=(4,) or not np.isfinite(pose).all():raise ValueError('invalid successor pose')
        return Belief(f.source.graph,cs,float(f.times[index]),freeze(pose),self.model_version,
                      dict(probability_status='finite conditional shared-world measure',exact_imagined_refinement='unsupported',
                           bootstrap_support_status=f.source.diagnostics.get('bootstrap_support_status','FINITE_APPROXIMATION'),
                           bootstrap_duplicate_fallback=f.source.diagnostics.get('bootstrap_duplicate_fallback',False)),role='imagined',carrier=(f,index),terminal=f.source.terminal,
                      stochastic_source=f.identity()+':'+str(index),next_slot=f.source.next_slot,lineage_namespace=f.source.lineage_namespace)

    def _reuse_carrier(self,b,rel,num_worlds,seed):
        clipped,start=b.carrier
        f=clipped.parent_tape if clipped.parent_tape is not None else clipped
        indices=[];extra=[]
        for t in b.timestamp+rel:
            if t>f.times[-1]+1e-8:
                extra.append(float(t));continue
            j=int(np.argmin(abs(f.times-t)))
            if abs(f.times[j]-t)>1e-8:raise ValueError('finite imagined continuation requires stored endpoint support')
            indices.append(j)
        if num_worlds!=len(b.components):raise ValueError('finite carrier world count is fixed; no silent resampling')
        if extra:
            # Continue the held stochastic tape using its own retained RNG state.
            # No Gaussian re-projection, replacement particles or new hidden draw.
            rng=np.random.default_rng();rng.bit_generator.state=json.loads(f.rng_state)
            tail_rel=np.r_[0.,np.asarray(extra)-f.times[-1]]
            worlds=f.endpoints[-1]
            xp,vp,sp,ns,cs=sample_joint_paths(worlds,tail_rel,float(f.times[-1]),self.law,rng,getattr(self,'cv_diffusion',None))
            retained=np.flatnonzero(f.times>=b.timestamp-1e-8)
            x=freeze(np.concatenate([f.positions[:,retained],xp[:,1:]],1));v=freeze(np.concatenate([f.velocities[:,retained],vp[:,1:]],1));st=freeze(np.concatenate([f.stopped[:,retained],sp[:,1:]],1))
            original=f.endpoints
            if not isinstance(original,EndpointSequence):raise ValueError('continuation needs compact stochastic carrier')
            ends=EndpointSequence(original.worlds,x,v,st,freeze(np.concatenate([original.switches[:,retained],ns[:,1:]],1)),freeze(np.concatenate([original.counters[:,retained],cs[:,1:]],1)))
            if f.hidden_velocities is None:
                if f.hidden_positions.shape[2]:raise ValueError('hidden continuation requires retained sampled velocities')
                hv=np.empty((len(f.weights),0,2))
            else:hv=f.hidden_velocities
            hidden=f.hidden_positions[:,-1,None]+(np.asarray(extra)-f.times[-1])[None,:,None,None]*hv[:,None]
            hidden=freeze(np.concatenate([f.hidden_positions[:,retained],hidden],1))
            canonical=self._seal_future(JointFuture(freeze(np.r_[f.times[retained],extra]),x,v,st,freeze([c.weight for c in b.components]),f.slots,hidden,b.fork(carrier=None),ends,seed,
                json.dumps(rng.bit_generator.state,sort_keys=True),b.fingerprint(),self.model_version,
                dict(f.diagnostics,held_tape_reuse=True,continuation='parent RNG state; seed argument does not reseed an existing physical tape'),hidden_velocities=freeze(hv)))
            wanted=np.searchsorted(canonical.times,b.timestamp+rel-1e-9)
            return self._seal_future(JointFuture(freeze(b.timestamp+rel),freeze(canonical.positions[:,wanted]),freeze(canonical.velocities[:,wanted]),freeze(canonical.stopped[:,wanted]),
                canonical.weights,canonical.slots,freeze(canonical.hidden_positions[:,wanted]),b.fork(carrier=None),canonical.endpoints.subset(wanted),seed,canonical.rng_state,
                b.fingerprint(),self.model_version,dict(canonical.diagnostics),parent_tape=canonical,hidden_velocities=canonical.hidden_velocities))
        return self._seal_future(JointFuture(freeze(b.timestamp+rel),freeze(f.positions[:,indices]),freeze(f.velocities[:,indices]),freeze(f.stopped[:,indices]),
              freeze([c.weight for c in b.components]),f.slots,freeze(f.hidden_positions[:,indices]),b.fork(carrier=None),f.endpoints.subset(indices) if isinstance(f.endpoints,EndpointSequence) else tuple(f.endpoints[j] for j in indices),seed,
              copy.deepcopy(f.rng_state),b.fingerprint(),self.model_version,dict(f.diagnostics,held_tape_reuse=True),parent_tape=f,hidden_velocities=f.hidden_velocities))

    def evolve(self,c,start,end,rng=None):
        return evolve(c,start,end,self.law,rng)

    def serialize_actual_belief(self,b):
        self._check(b,actual=True)
        return dict(schema='l2_actual_belief_2',next_slot=b.next_slot,lineage_namespace=b.lineage_namespace,version=self.model_version,timestamp=b.timestamp,robot_pose=b.robot_pose.tolist(),terminal=b.terminal,receipt=b.receipt,
            graph=dict(namespace=b.graph.namespace,version=b.graph.version,resolution=b.graph.resolution,height=b.graph.height,
                       cells=[[int(k[0]),int(k[1]),int(v[0]),float(v[1])] for k,v in sorted(b.graph.materialize().items())]),
            components=[{k:dict(data=v.tolist(),dtype=str(v.dtype),shape=list(v.shape)) if isinstance(v,np.ndarray) else v for k,v in vars(c).items()} for c in b.components],diagnostics=b.diagnostics)

    def deserialize_actual_belief(self,data):
        if data['version']!=self.model_version:raise ValueError('foreign serialized model version')
        g=dict(data['graph']);g['cells']={(x,z):(s,h) for x,z,s,h in g['cells']};cs=[]
        for raw in data['components']:
            raw=dict(raw)
            for k in ('mean','covariance','slots','last_seen','birth','last_position','stopped','intent','diffusion','next_switch','counters','hidden_pmf','hidden_mean','hidden_covariance','last_position_valid'):
                v=raw.get(k)
                if isinstance(v,dict):raw[k]=np.asarray(v['data'],dtype=v['dtype']).reshape(v['shape'])
                elif v is not None:raw[k]=np.asarray(v,dtype=bool if k in ('stopped','last_position_valid') else int if k in ('slots','counters') else float)
            raw['mean']=raw['mean'].reshape(-1,4);raw['covariance']=raw['covariance'].reshape(-1,4,4);raw['intent']=raw['intent'].reshape(-1,2);raw['last_position']=raw['last_position'].reshape(-1,2)
            for k in ('hidden_mean','hidden_covariance'):
                if raw.get(k) is not None:raw[k]=np.asarray(raw[k],float)
            cs.append(sealed(Component(**raw)))
        b=Belief(freeze_graph(ObservedGraph(**g)),tuple(cs),float(data['timestamp']),freeze(data['robot_pose']),self.model_version,dict(data['diagnostics']),
                 terminal=bool(data['terminal']),receipt=data['receipt'],next_slot=int(data.get('next_slot',max((int(s)+1 for c in cs for s in c.slots),default=0))),lineage_namespace=data.get('lineage_namespace',''))
        self._check(b,actual=True);return b
