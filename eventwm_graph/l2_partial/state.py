"""Serializable explicit probability state; all arrays use world X/Z meters."""
from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional, Any
import hashlib, json
import numpy as np

MODEL_VERSION='partial-graph-belief-0.1'
EXECUTOR_VERSION='known_space_discrete_route_follower_1'

def normalized(x):
    x=np.asarray(x,dtype=float)
    if np.any(x<0) or not np.isfinite(x).all() or x.sum()<=0:raise ValueError('invalid probability')
    return x/x.sum()

@dataclass(frozen=True)
class ObservedGraph:
    """Copy-on-write local overlays. Unknown defaults to 0, never free."""
    namespace: str
    version: int
    resolution: float
    height: float
    cells: Dict
    parent: Optional[Any]=None
    depth: int=0

    @classmethod
    def from_snapshot(cls,snapshot):
        raw=json.dumps(snapshot,sort_keys=True,separators=(',',':')).encode()
        cells={tuple(c['cell']):(int(c['state']),float(c['height'])) for c in snapshot['cells']}
        return cls(hashlib.sha256(raw).hexdigest(),int(snapshot['revision']),float(snapshot['resolution']),float(snapshot['height']),cells)

    def get(self,key):
        v=self.cells.get(tuple(key))
        return v if v is not None else (self.parent.get(key) if self.parent else (0,self.height))

    def materialize(self):
        d=self.parent.materialize() if self.parent else {};d.update(self.cells);return d

    def update(self,rows):
        delta={}
        occupied={tuple(u['cell']) for u in rows if u['state']==2}
        for u in rows:
            if u['source']=='occupied_ego_base_support':continue
            k=tuple(u['cell']);v=(int(u['state']),float(u['height']))
            if v[0]==1 and k in occupied:continue
            if v!=self.get(k):delta[k]=v
        if not delta:return self
        # Hash identifies alternate legal branches with the same integer revision.
        key=hashlib.sha256((self.namespace+repr(sorted(delta.items()))).encode()).hexdigest()
        if self.depth>=8:
            cells=self.materialize();cells.update(delta);return ObservedGraph(key,self.version+1,self.resolution,self.height,cells)
        return ObservedGraph(key,self.version+1,self.resolution,self.height,delta,self,self.depth+1)

    def local_features(self,positions):
        p=np.asarray(positions);shape=p.shape[:-1];flat=p.reshape(-1,2); out=np.zeros((len(flat),18),np.float32)
        # Two rings are a bounded local observed spatial graph, never full geometry.
        offsets=((0,0),(1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2))
        keys=np.floor(flat/self.resolution).astype(int)
        for i,k in enumerate(keys):
            for j,d in enumerate(offsets):
                state,_=self.get((k[0]+d[0],k[1]+d[1]));out[i,2*j]=state==1;out[i,2*j+1]=state==2
        return out.reshape(*shape,18)

@dataclass
class Hypothesis:
    weight: float
    mean: np.ndarray                 # N x 4: x,z,vx,vz
    covariance: np.ndarray           # N x 4 x 4
    last_seen: np.ndarray
    slot_ids: np.ndarray             # internal only, NOT collector identity
    hidden_pmf: np.ndarray           # remaining never-localized population
    intent: Optional[np.ndarray]=None
    diffusion: Optional[np.ndarray]=None
    mode: Optional[np.ndarray]=None
    relaxation: float=0.
    height: Optional[np.ndarray]=None
    visibility_mode: int=1          # persistent coarse optical regime, not a map
    birth_time: Optional[np.ndarray]=None
    birth_position: Optional[np.ndarray]=None
    maneuver_mean: Optional[np.ndarray]=None
    maneuver_covariance: Optional[np.ndarray]=None
    maneuver_probability: Optional[np.ndarray]=None
    maneuver_until: float=-1.
    hidden_mean: Optional[np.ndarray]=None
    hidden_covariance: Optional[np.ndarray]=None
    hidden_exchangeable: bool=True
    hidden_maneuver_mean: Optional[np.ndarray]=None
    hidden_maneuver_covariance: Optional[np.ndarray]=None
    hidden_maneuver_probability: Optional[np.ndarray]=None
    coarse_observation_time: float=-1.
    coarse_positive_refinement: float=1.
    coarse_empty_refinement: float=1.

    def fork(self,**changes):return replace(self,**changes)
    @property
    def n(self):return len(self.mean)
    @property
    def positions_xyz(self):
        y=self.height if self.height is not None else np.zeros(self.n)
        return np.stack([self.mean[:,0],y,self.mean[:,1]],-1)
    @property
    def expected_hidden(self):return float(self.hidden_pmf @ np.arange(len(self.hidden_pmf)))

@dataclass
class BeliefState:
    graph: ObservedGraph
    hypotheses: List[Hypothesis]
    timestamp: float
    robot_pose: np.ndarray
    unresolved_mass: float=0.
    terminal: bool=False
    model_version: str='unbound'
    evidence: Dict=field(default_factory=dict)
    query_cursor: Optional[Dict]=None
    terminal_event: Optional[Dict]=None

    def fork(self,**changes):return replace(self,**changes)
    def fingerprint(self):
        """Small belief-only digest; never hashes/materializes the full graph."""
        digest=hashlib.blake2b(digest_size=16)
        digest.update(repr((self.graph.namespace,self.timestamp,self.unresolved_mass,self.terminal,self.model_version,self.query_cursor)).encode())
        digest.update(np.asarray(self.robot_pose).tobytes())
        for h in self.hypotheses:
            digest.update(repr((h.weight,h.relaxation,h.visibility_mode,h.maneuver_until,h.hidden_exchangeable,h.coarse_observation_time,h.coarse_positive_refinement,h.coarse_empty_refinement)).encode())
            for k in ('mean','covariance','last_seen','slot_ids','hidden_pmf','intent','diffusion','mode','height','birth_time','birth_position','maneuver_mean','maneuver_covariance','maneuver_probability','hidden_mean','hidden_covariance','hidden_maneuver_mean','hidden_maneuver_covariance','hidden_maneuver_probability'):
                v=getattr(h,k);digest.update(b'None' if v is None else np.asarray(v).tobytes())
        digest.update(repr(self.evidence.get('exact_detections',[])).encode())
        return digest.hexdigest()
    def validate(self):
        weights=np.array([h.weight for h in self.hypotheses])
        assert np.isfinite(weights).all() and np.all(weights>=0)
        assert 0<=self.unresolved_mass<=1 and abs(weights.sum()+self.unresolved_mass-1)<1e-7
        for h in self.hypotheses:
            assert h.mean.shape==(h.n,4) and h.covariance.shape==(h.n,4,4)
            assert len(set(h.slot_ids.tolist()))==h.n
            assert np.isfinite(h.mean).all() and np.isfinite(h.covariance).all()
            assert np.all(h.hidden_pmf>=0) and abs(h.hidden_pmf.sum()-1)<1e-8
            assert np.all(h.last_seen<=self.timestamp+1e-7)
            if h.n:assert np.linalg.eigvalsh(h.covariance).min()>-1e-7
            if h.hidden_mean is not None:
                assert h.hidden_mean.ndim==2 and h.hidden_mean.shape[1]==4
                assert h.hidden_covariance.shape==(len(h.hidden_mean),4,4)
                assert np.isfinite(h.hidden_mean).all() and np.isfinite(h.hidden_covariance).all()
                if h.hidden_exchangeable:assert len(h.hidden_mean)==1
                else:
                    assert np.count_nonzero(h.hidden_pmf>1e-10)==1
                    assert len(h.hidden_mean)==int(np.argmax(h.hidden_pmf))
        return self
    def to_dict(self):
        def cv(v):return v.tolist() if isinstance(v,np.ndarray) else v
        g=self.graph
        return {'graph':{'namespace':g.namespace,'version':g.version,'resolution':g.resolution,'height':g.height,
                         'cells':[[*k,*v] for k,v in sorted(g.materialize().items())]},
                'hypotheses':[{k:cv(v) for k,v in h.__dict__.items()} for h in self.hypotheses],
                **{k:cv(v) for k,v in self.__dict__.items() if k not in ('graph','hypotheses')}}
    @classmethod
    def from_dict(cls,v):
        v=dict(v);g=dict(v.pop('graph'));g['cells']={(x,z):(s,y) for x,z,s,y in g['cells']};g=ObservedGraph(**g)
        hs=[]
        for raw in v.pop('hypotheses'):
            d=dict(raw)
            for k in ('mean','covariance','last_seen','slot_ids','hidden_pmf','intent','diffusion','mode','height','birth_time','birth_position','maneuver_mean','maneuver_covariance','maneuver_probability','hidden_mean','hidden_covariance','hidden_maneuver_mean','hidden_maneuver_covariance','hidden_maneuver_probability'):
                if d.get(k) is not None:d[k]=np.asarray(d[k],dtype=int if k=='slot_ids' else float)
            d['mean']=d['mean'].reshape(-1,4);d['covariance']=d['covariance'].reshape(-1,4,4)
            hs.append(Hypothesis(**d))
        v['robot_pose']=np.asarray(v['robot_pose']);return cls(g,hs,**v).validate()

@dataclass
class SharedFuture:
    times: np.ndarray
    positions: np.ndarray            # W,T,N,2; tracked then private hidden, NaN padding
    velocities: np.ndarray
    weights: np.ndarray              # absolute retained weights
    worlds: List[Hypothesis]
    source: BeliefState
    unresolved_mass: float
    seed: int
    input_fingerprint: Optional[str]=None
    endpoint_cache: Dict=field(default_factory=dict,repr=False)

    @property
    def localized_mask(self):
        """Distinguish sampled hidden states from formerly observed slots."""
        return np.arange(self.positions.shape[2])[None,:]<np.array([h.n for h in self.worlds])[:,None]

@dataclass
class ObservationBranch:
    probability: float
    observation: Dict
    successor: BeliefState

@dataclass
class QueryPrediction:
    query_id: str
    collision_probability: float
    psc_probability: float
    collision_interval: tuple
    psc_interval: tuple
    duration_samples: np.ndarray
    duration_weights: np.ndarray
    progress_samples: np.ndarray
    time_to_conflict: Dict
    blockage_probability: float
    termination_probabilities: Dict
    observation_branches: List[ObservationBranch]
    unresolved_uncertainty: float
    evaluated_interval: tuple
    query_complete: bool
    remaining_query_cursor: Dict
    executor_version: str=EXECUTOR_VERSION
    model_version: str=MODEL_VERSION
    diagnostics: Dict=field(default_factory=dict)

@dataclass
class QueryPredictionBatch:
    predictions: List[QueryPrediction]
    timings_ms: Dict
    shared_future: SharedFuture
