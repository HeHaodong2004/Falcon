"""Explicit finite-world carriers, without adding endpoint smoothing noise.

The carrier is a declared finite quadrature measure. It preserves its own
weights/paths under coarse conditioning; it is NOT a continuous posterior
bridge. Exact off-grid detections require a different operation and are refused.
"""
from dataclasses import dataclass, field, fields
import hashlib
import json
import copy
from types import MappingProxyType
import numpy as np
from .state import BeliefState, SharedFuture, Hypothesis, ObservedGraph


def immutable_array(value):
    value = np.array(value, copy=True)
    value.setflags(write=False)
    return value


class ImmutableHypothesis(Hypothesis):
    """Private immutable carrier template; fork creates an ordinary hypothesis."""
    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):
            raise AttributeError('immutable carrier hypothesis; use fork')
        object.__setattr__(self,name,value)

    def fork(self,**changes):
        return Hypothesis(**{**{f.name:getattr(self,f.name) for f in fields(Hypothesis)},**changes})


def freeze_hypothesis(h):
    if isinstance(h,ImmutableHypothesis):return h
    d={f.name:getattr(h,f.name) for f in fields(Hypothesis)}
    d={k:immutable_array(v) if isinstance(v,np.ndarray) else v for k,v in d.items()}
    result=ImmutableHypothesis(**d)
    object.__setattr__(result,'_sealed',True)
    return result


def freeze_hypotheses(hypotheses, *, weight=None):
    """Snapshot shared buffers once, preserving all zero-weight support templates.

    Count/certificate expansions share many input arrays. Copying them once per
    component was both unnecessary and expensive. The memo is call-local: live
    caller-owned arrays never acquire an unsafe persistent validity cache.
    """
    arrays = {}
    result = []
    names = tuple(f.name for f in fields(Hypothesis))
    for h in hypotheses:
        d = {name: getattr(h, name) for name in names}
        for name, value in d.items():
            if isinstance(value, np.ndarray):
                if not isinstance(h, ImmutableHypothesis):
                    key = id(value)
                    if key not in arrays:
                        arrays[key] = immutable_array(value)
                    d[name] = arrays[key]
        if weight is not None:
            d['weight'] = float(weight)
        frozen = ImmutableHypothesis(**d)
        object.__setattr__(frozen, '_sealed', True)
        result.append(frozen)
    return tuple(result)


def _restore_empty_hidden_shapes(value):
    for h in value.get('hypotheses',[]):
        for key,shape in (('hidden_mean',(0,4)),('hidden_covariance',(0,4,4))):
            if h.get(key) is not None and np.asarray(h[key]).size==0:
                h[key]=np.empty(shape)
    return value


@dataclass(frozen=True)
class _TapeValidation:
    """Internal certificate for already checked immutable numerical storage."""
    identities: tuple
    digest: str


@dataclass(frozen=True)
class StructuralObservedGraph(ObservedGraph):
    """Versioned graph overlays with privately owned, read-only cell storage."""
    def __post_init__(self):
        object.__setattr__(self,'cells',MappingProxyType(dict(self.cells)))

    @classmethod
    def freeze(cls,g):
        if isinstance(g,cls):return g
        return cls(g.namespace,g.version,g.resolution,g.height,g.cells,
                   None if g.parent is None else cls.freeze(g.parent),g.depth)

    def update(self,rows):
        return self.freeze(super().update(rows))


@dataclass(frozen=True)
class WorldCarrier:
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    worlds: tuple
    weights: np.ndarray
    unresolved_mass: float
    seed: int
    origin_fingerprint: str
    index: int
    observation_ledger: tuple = ()
    tape_digest: str = None
    rng_state: object = None
    _tape_validation: object = field(default=None,repr=False,compare=False)

    def __post_init__(self):
        if self.rng_state is not None and not isinstance(self.rng_state,str):
            object.__setattr__(self,'rng_state',json.dumps(self.rng_state,sort_keys=True,separators=(',',':')))
        for key in ('times', 'positions', 'velocities', 'weights'):
            value = getattr(self, key)
            if value.flags.writeable:
                object.__setattr__(self, key, immutable_array(value))
        if not isinstance(self.worlds,tuple) or any(not isinstance(h,ImmutableHypothesis) for h in self.worlds):
            object.__setattr__(self,'worlds',tuple(freeze_hypothesis(h) for h in self.worlds))
        identities=tuple(id(getattr(self,k)) for k in ('times','positions','velocities'))+tuple(id(h) for h in self.worlds)
        certificate=self._tape_validation
        trusted=isinstance(certificate,_TapeValidation) and certificate.identities==identities
        if self.weights.ndim!=1 or not np.isfinite(self.weights).all() or np.any(self.weights<0) or not 0<=self.unresolved_mass<=1 or abs(self.weights.sum()+self.unresolved_mass-1)>1e-9:
            raise ValueError('invalid carrier probability mass')
        if isinstance(self.index,bool) or int(self.index)!=self.index or not 0<=self.index<len(self.times):
            raise ValueError('invalid carrier endpoint')
        if trusted:
            object.__setattr__(self,'tape_digest',certificate.digest)
            return
        if self.positions.shape!=self.velocities.shape or self.positions.ndim!=4 or self.positions.shape[:2]!=(len(self.weights),len(self.times)) or self.positions.shape[-1]!=2:
            raise ValueError('invalid finite path tensor shapes')
        if self.times.ndim!=1 or not len(self.times) or not np.isfinite(self.times).all() or np.any(np.diff(self.times)<=0) or isinstance(self.index,bool) or int(self.index)!=self.index or not 0<=self.index<len(self.times):
            raise ValueError('invalid carrier clock')
        if self.weights.ndim!=1 or not np.isfinite(self.weights).all() or np.any(self.weights<0) or not 0<=self.unresolved_mass<=1 or abs(self.weights.sum()+self.unresolved_mass-1)>1e-9:
            raise ValueError('invalid carrier probability mass')
        if len(self.worlds)!=len(self.weights):raise ValueError('carrier world count mismatch')
        for w,h in enumerate(self.worlds):
            if np.count_nonzero(h.hidden_pmf>0)!=1 or abs(h.hidden_pmf.sum()-1)>1e-10:
                raise ValueError('finite world requires one fixed population count')
            n=h.n+int(np.argmax(h.hidden_pmf))
            if n>self.positions.shape[2] or not np.isfinite(self.positions[w,:,:n]).all() or not np.isfinite(self.velocities[w,:,:n]).all():
                raise ValueError('nonfinite person state in finite carrier')
            if not np.isnan(self.positions[w,:,n:]).all() or not np.isnan(self.velocities[w,:,n:]).all():
                raise ValueError('invalid population padding in finite carrier')
        if not trusted:
            h=hashlib.blake2b(digest_size=16)
            for value in (self.times,self.positions,self.velocities):h.update(value.tobytes())
            for world in self.worlds:
                h.update(repr((world.relaxation,world.visibility_mode)).encode())
                for f in fields(Hypothesis):
                    if f.name=='weight':continue
                    value=getattr(world,f.name)
                    h.update(value.tobytes() if isinstance(value,np.ndarray) else repr(value).encode())
            object.__setattr__(self,'tape_digest',h.hexdigest())
            object.__setattr__(self,'_tape_validation',_TapeValidation(identities,h.hexdigest()))

    def digest(self):
        h = hashlib.blake2b(digest_size=16)
        h.update(repr((self.origin_fingerprint, self.index, self.seed,
                       self.unresolved_mass, self.observation_ledger,self.tape_digest,self.rng_state)).encode())
        h.update(self.weights.tobytes())
        # Origin includes the generating belief; the tape hash distinguishes
        # externally restored or edited carriers, without hashing it per query.
        h.update(self.times.tobytes())
        return h.hexdigest()

    def to_dict(self):
        return {k: np.where(np.isfinite(getattr(self,k)),getattr(self,k),None).tolist() for k in ('times','positions','velocities','weights')} | {
            'worlds': [_hypothesis_dict(h) for h in self.worlds],
            'world_array_metadata':[_array_metadata(h) for h in self.worlds],
            'unresolved_mass': self.unresolved_mass, 'seed': self.seed,
            'origin_fingerprint': self.origin_fingerprint, 'index': self.index,
            'observation_ledger': list(self.observation_ledger),'tape_digest':self.tape_digest,
            'rng_state':self.rng_state}

    @classmethod
    def from_dict(cls, d, graph):
        raw = copy.deepcopy(d)
        expected_digest=raw.pop('tape_digest',None)
        world_rows=raw.pop('worlds')
        metadata=raw.pop('world_array_metadata',None)
        if metadata is not None and len(metadata)!=len(world_rows):raise ValueError('world dtype metadata count mismatch')
        hs=[_restore_hypothesis(row,None if metadata is None else metadata[i]) for i,row in enumerate(world_rows)]
        # Hypotheses are templates; their stored original weights need not sum
        # to one after branch weighting. Parse via an individually normalized
        # temporary belief, then restore their weights in the carrier itself.
        BeliefState(graph,[h.fork(weight=0.) for h in hs],float(raw['times'][-1]),np.zeros(4),unresolved_mass=1.).validate()
        for key in ('times','positions','velocities','weights'):
            raw[key] = immutable_array(np.asarray(raw[key],float))
        raw['worlds'] = tuple(hs)
        def tuples(value):return tuple(tuples(v) for v in value) if isinstance(value,list) else value
        raw['observation_ledger'] = tuples(raw.get('observation_ledger', ()))
        result=cls(**raw)
        if expected_digest is not None and expected_digest!=result.tape_digest:
            raise ValueError('serialized carrier tape checksum mismatch')
        return result


def _hypothesis_dict(h):
    return {f.name:getattr(h,f.name).tolist() if isinstance(getattr(h,f.name),np.ndarray) else getattr(h,f.name) for f in fields(Hypothesis)}


_ARRAY_FIELDS=('mean','covariance','last_seen','slot_ids','hidden_pmf','intent','diffusion','mode','height',
    'birth_time','birth_position','maneuver_mean','maneuver_covariance','maneuver_probability',
    'hidden_mean','hidden_covariance','hidden_maneuver_mean','hidden_maneuver_covariance','hidden_maneuver_probability')


def _array_metadata(h):
    return {key:{'dtype':getattr(h,key).dtype.str,'shape':list(getattr(h,key).shape)}
            for key in _ARRAY_FIELDS if isinstance(getattr(h,key),np.ndarray)}


def _restore_hypothesis(raw,metadata=None):
    d=copy.deepcopy(raw);metadata=metadata or {}
    for key in _ARRAY_FIELDS:
        if d.get(key) is None:continue
        entry=metadata.get(key)
        dtype=np.dtype(entry['dtype']) if entry is not None else np.dtype(int if key=='slot_ids' else float)
        if dtype.kind not in 'fiu b'.replace(' ',''):raise ValueError('unsupported numeric state dtype')
        value=np.asarray(d[key],dtype=dtype)
        if entry is not None:value=value.reshape(tuple(entry['shape']))
        elif key in ('mean','hidden_mean'):value=value.reshape(-1,4)
        elif key in ('covariance','hidden_covariance'):value=value.reshape(-1,4,4)
        d[key]=value
    return Hypothesis(**d)


@dataclass
class StructuralBeliefState(BeliefState):
    carrier: object = None

    def __post_init__(self):
        # Base initialize() replaces its graph after filtering history; every
        # construction/fork must enforce storage ownership, not only wrap().
        self.graph=StructuralObservedGraph.freeze(self.graph)

    @classmethod
    def wrap(cls, b, **changes):
        if isinstance(b, cls):
            return b.fork(**changes)
        d={**vars(b),**changes}
        d['graph']=StructuralObservedGraph.freeze(d['graph'])
        return cls(**d)

    def fingerprint(self):
        physical = super().fingerprint()
        if self.carrier is None:
            return physical
        return hashlib.blake2b((physical + self.carrier.digest()).encode(), digest_size=16).hexdigest()

    def to_dict(self):
        d = super().to_dict()
        d.pop('carrier',None)
        d['hypotheses']=[_hypothesis_dict(h) for h in self.hypotheses]
        d=copy.deepcopy(d)
        d['hypothesis_array_metadata']=[_array_metadata(h) for h in self.hypotheses]
        d['serialization_schema']='partial_graph_structural_state_2'
        d['carrier'] = self.carrier.to_dict() if self.carrier is not None else None
        return d

    @classmethod
    def from_dict(cls, value):
        value = copy.deepcopy(value)
        carrier = value.pop('carrier', None)
        schema=value.pop('serialization_schema','partial_graph_structural_state_2')
        if schema!='partial_graph_structural_state_2':raise ValueError('unsupported structural state schema')
        metadata=value.pop('hypothesis_array_metadata',None)
        g=value.pop('graph');g['cells']={(x,z):(s,y) for x,z,s,y in g['cells']}
        graph=StructuralObservedGraph(**g)
        raw=value.pop('hypotheses')
        if metadata is not None and len(metadata)!=len(raw):raise ValueError('state dtype metadata count mismatch')
        hs=[_restore_hypothesis(row,None if metadata is None else metadata[i]) for i,row in enumerate(raw)]
        value['robot_pose']=np.asarray(value['robot_pose'],float)
        b=cls(graph,hs,**value)
        return cls.wrap(b, carrier=None if carrier is None else WorldCarrier.from_dict(carrier, b.graph)).validate()

    def validate(self):
        from .structural_validation import validate_batched
        return validate_batched(self)


def seal_future(future):
    """Seal newly generated numerical buffers, not a caller's belief arrays."""
    for key in ('times','positions','velocities','weights'):
        getattr(future, key).setflags(write=False)
    return future
