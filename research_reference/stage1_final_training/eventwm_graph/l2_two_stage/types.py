"""Strict legal boundary and typed, query-independent human-world output."""
from dataclasses import dataclass
from typing import Tuple
from collections.abc import Mapping
from types import MappingProxyType
import math
import numpy as np

FRAME_KEYS = frozenset(('timestamp','robot_pose','camera_origin','camera_rotation',
    'tracks','static_updates','sensor_protocol','observation_valid','human_sensor_available'))
TRACK_KEYS = frozenset(('track_id','position','estimated_velocity','first_seen_time',
    'last_seen_time','num_observations','observation_age','valid','velocity_valid','visible_now'))
STATIC_KEYS = frozenset(('cell','height','source','state','timestamp'))
STATIC_SOURCES = frozenset(('visible_native_cspace_square_and_clearance_halo',
    'observed_static_wall_inflated_025m','occupied_ego_base_support'))
_CERTIFICATION=object()
_BOOLEAN_TYPES=(bool,np.bool_)


def finite(v):
    a = np.asarray(v, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite legal input')
    return a


def validate_frame(ob):
    if isinstance(ob,SealedObservation):
        if getattr(ob,'_certificate',None) is not _CERTIFICATION:raise ValueError('uncertified observation object')
        return ob['timestamp']
    if set(ob) - FRAME_KEYS:
        raise ValueError('unknown observation fields: '+repr(set(ob)-FRAME_KEYS))
    for flag in ('observation_valid','human_sensor_available'):
        if flag in ob and not isinstance(ob[flag],_BOOLEAN_TYPES):raise ValueError('nonboolean observation validity flag')
    t = float(ob['timestamp'])
    if not math.isfinite(t): raise ValueError('nonfinite clock')
    if finite(ob['robot_pose']).shape != (4,): raise ValueError('pose shape')
    if finite(ob['camera_origin']).shape != (3,): raise ValueError('camera shape')
    if finite(ob['camera_rotation']).shape != (3,3): raise ValueError('rotation shape')
    rotation=finite(ob['camera_rotation'])
    if not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-5) or abs(np.linalg.det(rotation)-1.)>1e-5:
        raise ValueError('camera rotation is not a proper orthonormal transform')
    if ob.get('sensor_protocol')!='ideal_frustum_observed_cspace_dynamic_occlusion_1':
        raise ValueError('unsupported installed sensor protocol')
    if not ob.get('observation_valid',True) and len(ob['static_updates']):
        raise ValueError('invalid whole frame cannot publish static certificates')
    for tr in ob['tracks']:
        if set(tr)-TRACK_KEYS: raise ValueError('unknown track fields')
        for flag in ('valid','visible_now','velocity_valid'):
            if flag in tr and not isinstance(tr[flag],_BOOLEAN_TYPES):raise ValueError('nonboolean track validity flag')
        if finite(tr['position']).shape != (3,): raise ValueError('position shape')
        velocity=tr.get('estimated_velocity')
        if velocity is None:
            if tr.get('velocity_valid',False):raise ValueError('valid velocity requires supplied evidence')
        elif finite(velocity).shape!=(3,):raise ValueError('velocity shape')
        last = float(tr['last_seen_time']);first=float(tr['first_seen_time']);age=float(tr['observation_age'])
        if not (math.isfinite(first) and math.isfinite(last) and math.isfinite(age)) or last > t+1e-7 or first > last:
            raise ValueError('noncausal track clock')
        if abs(age-(t-last)) > 1e-6:
            raise ValueError('incorrect observation age')
        if ob.get('observation_valid',True) and ob.get('human_sensor_available',True) and tr.get('valid',True) and tr.get('visible_now') and abs(last-t)>1e-7:
            raise ValueError('stale positive')
    for cell in ob['static_updates']:
        if cell.keys() != STATIC_KEYS:
            raise ValueError('unknown static update fields')
        timestamp=float(cell['timestamp'])
        if not math.isfinite(timestamp) or abs(timestamp-t)>1e-8 or cell['state'] not in (1,2):
            raise ValueError('noncausal/unknown static evidence')
        if cell['source'] not in STATIC_SOURCES:raise ValueError('unknown static sensor certificate')
        if len(cell['cell'])!=2 or not all(isinstance(v,int) and not isinstance(v,bool) for v in cell['cell']) or not math.isfinite(cell['height']):
            raise ValueError('invalid static coordinate/height')
    return t


def freeze(value):
    if isinstance(value,Mapping):return MappingProxyType({k:freeze(v) for k,v in value.items()})
    if isinstance(value,(list,tuple)):return tuple(freeze(v) for v in value)
    if isinstance(value,np.ndarray):return tuple(freeze(v) for v in value.tolist())
    if isinstance(value,np.generic):return freeze(value.item())
    if value is None or type(value) in (str,int,float,bool,bytes):return value
    raise ValueError('unsupported mutable legal input type: '+type(value).__name__)


_STATIC_NAMES=('visible_native_cspace_square_and_clearance_halo','observed_static_wall_inflated_025m','occupied_ego_base_support')


def packed_cell_keys(cells):
    cells=np.asarray(cells,np.int64).reshape(-1,2)
    if np.any(cells<-(1<<31)) or np.any(cells>=(1<<31)):return None
    bits=cells.view(np.uint64)
    return (bits[:,0]<<np.uint64(32)) | (bits[:,1]&np.uint64(0xffffffff))


@dataclass(frozen=True,init=False,eq=False)
class StaticCertificateTable:
    """Immutable numeric columns, preserving every arrived static certificate.

    Rows are materialized only while consumed. Historical frames no longer keep
    one GC-tracked dictionary/proxy per repeated cell observation.
    """
    cells: np.ndarray
    heights: np.ndarray
    states: np.ndarray
    sources: np.ndarray
    timestamp: float

    @classmethod
    def from_validated(cls,rows,timestamp):
        n=len(rows)
        try:cells=np.asarray([r['cell'] for r in rows],dtype=np.int64).reshape(n,2)
        except (OverflowError,ValueError) as e:raise ValueError('static cell index outside supported integer range') from e
        values=dict(cells=immutable_array(cells),heights=immutable_array(np.asarray([float(r['height']) for r in rows],np.float64)),
            states=immutable_array(np.asarray([int(r['state']) for r in rows],np.uint8)),
            sources=immutable_array(np.asarray([_STATIC_NAMES.index(r['source']) for r in rows],np.uint8)),timestamp=float(timestamp))
        result=object.__new__(cls)
        for key,value in values.items():object.__setattr__(result,key,value)
        return result

    def __len__(self):return len(self.states)

    def __eq__(self,other):
        return isinstance(other,StaticCertificateTable) and self.timestamp==other.timestamp and all(
            np.array_equal(getattr(self,key),getattr(other,key)) for key in ('cells','heights','states','sources'))

    def __getitem__(self,index):
        if isinstance(index,slice):return tuple(self[i] for i in range(*index.indices(len(self))))
        cell=self.cells[index]
        return MappingProxyType(dict(cell=(int(cell[0]),int(cell[1])),height=float(self.heights[index]),
            state=int(self.states[index]),source=_STATIC_NAMES[int(self.sources[index])],timestamp=self.timestamp))

    def __iter__(self):
        for i in range(len(self)):yield self[i]


@dataclass(frozen=True,init=False,eq=False)
class SealedObservation(Mapping):
    """Immutable arrived sensor facts, validated once at arrival.

    Free/occupied cell membership is part of the stored sensor record, not a
    prediction cache. Cold model caches cannot invent or modify these facts.
    """
    record: Mapping
    free_support: frozenset
    occupied_support: frozenset
    free_codes: np.ndarray

    def __getitem__(self,k):return self.record[k]
    def __iter__(self):return iter(self.record)
    def __len__(self):return len(self.record)
    def __eq__(self,other):return isinstance(other,SealedObservation) and self.record==other.record

    @classmethod
    def from_raw(cls,ob):
        if isinstance(ob,cls):validate_frame(ob);return ob
        validate_frame(ob)
        free=frozenset(tuple(u['cell']) for u in ob['static_updates'] if u['state']==1 and u['source']!='occupied_ego_base_support')
        occupied=frozenset(tuple(u['cell']) for u in ob['static_updates'] if u['state']==2)
        # The fixed validated static schema contains only immutable scalars and
        # one2D cell coordinate. Avoid recursively rediscovering those types.
        record={k:freeze(v) for k,v in ob.items() if k!='static_updates'}
        record['static_updates']=StaticCertificateTable.from_validated(ob['static_updates'],ob['timestamp'])
        table=record['static_updates'];codes=packed_cell_keys(table.cells)
        free_codes=None if codes is None else immutable_array(np.setdiff1d(codes[(table.states==1)&(table.sources!=2)],codes[table.states==2]))
        result=object.__new__(cls)
        for k,v in dict(record=MappingProxyType(record),free_support=free,occupied_support=occupied,free_codes=free_codes,_certificate=_CERTIFICATION).items():object.__setattr__(result,k,v)
        return result


def update_cells(cells,ob,freeze_new=False):
    occupied=ob.occupied_support
    timestamp=ob['timestamp']
    changed=False
    table=ob['static_updates']
    if isinstance(table,StaticCertificateTable):
        facts=(((x,z),state,height,_STATIC_NAMES[source]) for x,z,state,height,source in zip(
            table.cells[:,0].tolist(),table.cells[:,1].tolist(),table.states.tolist(),table.heights.tolist(),table.sources.tolist()) if source!=2)
    else:
        facts=((tuple(u['cell']),u['state'],u['height'],u['source']) for u in table if u['source']!='occupied_ego_base_support')
    for k,state,height,source in facts:
        if state==1 and k in occupied:continue
        old=cells.get(k,{})
        if not changed:changed=old.get('state')!=state or abs(old.get('height',height)-height)>1e-8
        arrived=dict(cell=k,state=state,height=height,source=source,
            first_seen=old.get('first_seen',timestamp),last_seen=timestamp)
        cells[k]=MappingProxyType(arrived) if freeze_new else arrived
    return int(changed)


def freeze_cells(cells):
    return MappingProxyType({k:c if isinstance(c,MappingProxyType) else MappingProxyType(c) for k,c in cells.items()})


@dataclass(frozen=True,init=False)
class LegalHistory:
    frames: tuple
    cells: Mapping
    revision: int
    version: str = 'immutable_arrived_history_two_stage_1'

    def __iter__(self):return iter(self.frames)
    def __len__(self):return len(self.frames)

    @classmethod
    def from_frames(cls,frames):
        sealed=tuple(SealedObservation.from_raw(ob) for ob in frames);cells={};revision=0
        for i,ob in enumerate(sealed):
            if i and ob['timestamp']<=sealed[i-1]['timestamp']:raise ValueError('noncausal history')
            revision+=update_cells(cells,ob)
        return cls._from_verified(sealed,freeze_cells(cells),revision)

    @classmethod
    def _from_verified(cls,frames,cells,revision):
        result=object.__new__(cls)
        for k,v in dict(frames=frames,cells=cells,revision=revision,version='immutable_arrived_history_two_stage_1',_certificate=_CERTIFICATION).items():object.__setattr__(result,k,v)
        return result

    def append(self,ob):
        if getattr(self,'_certificate',None) is not _CERTIFICATION:raise ValueError('uncertified history object')
        ob=SealedObservation.from_raw(ob)
        if self.frames and ob['timestamp']<=self.frames[-1]['timestamp']:raise ValueError('noncausal append')
        # Existing values are certified immutable proxies. Freeze only updated
        # values in this newly owned dictionary instead of scanning/copying all
        # accumulated cells a second time.
        cells=dict(self.cells);increment=update_cells(cells,ob,freeze_new=True)
        return LegalHistory._from_verified(self.frames+(ob,),MappingProxyType(cells),self.revision+increment)


@dataclass(frozen=True)
class LegalContext:
    """Arrived history. Final observation occurs exactly once.

    Mapping inputs are validated at every public call; callers must not mutate
    an in-use context. Feature caches are keyed by content, not integer revision.
    """
    history: tuple
    actions: tuple
    partial_graph: dict
    version: str = 'legal_context_two_stage_1'

    @classmethod
    def from_inputs(cls, history, obs, partial_graph, actions=()):
        certified=history if isinstance(history,LegalHistory) else LegalHistory.from_frames(history)
        if getattr(certified,'_certificate',None) is not _CERTIFICATION:raise ValueError('uncertified history object')
        rows=certified.frames
        if not rows or float(rows[-1]['timestamp']) < float(obs['timestamp'])-1e-8:
            certified=certified.append(obs);rows=certified.frames
        else:
            arrived=SealedObservation.from_raw(obs)
            if rows[-1].record != arrived.record:raise ValueError('conflicting current observation')
        times = [validate_frame(row) for row in rows]
        if any(b <= a for a,b in zip(times,times[1:])):
            raise ValueError('history must be strictly chronological')
        if abs(times[-1]-float(partial_graph['timestamp']))>1e-7:
            raise ValueError('graph clock differs from arrived observation')
        allowed = {'schema','resolution','height','clearance','timestamp','revision',
                   'cells','visited','ego_support','global_nodes','global_edges'}
        if set(partial_graph)-allowed: raise ValueError('unknown/privileged graph field')
        if partial_graph['schema'] != 'causal_partial_cspace_graph_1':
            raise ValueError('unknown graph contract')
        graph_keys={tuple(c['cell']) for c in partial_graph['cells']}
        if len(graph_keys)!=len(partial_graph['cells']) or graph_keys!=set(certified.cells):
            raise ValueError('duplicated, omitted, or unobserved graph cell')
        for c in partial_graph['cells']:
            if set(c)-{'cell','height','state','first_seen','last_seen','source'}:
                raise ValueError('unknown/privileged cell field')
            if not (math.isfinite(c['last_seen']) and c['first_seen'] <= c['last_seen'] <= times[-1]+1e-7):
                raise ValueError('future graph evidence')
            if c['state'] not in (1,2): raise ValueError('explicit cells must be observed')
            source=certified.cells.get(tuple(c['cell']))
            if source is None or (c['state'],c['height'],c['source'],c['first_seen'],c['last_seen']) != (source['state'],source['height'],source['source'],source['first_seen'],source['last_seen']):
                raise ValueError('graph cell has no arrived observation provenance')
        if len(partial_graph['cells'])!=len(certified.cells) or partial_graph['revision']!=certified.revision:
            raise ValueError('graph contents/revision differ from arrived history')
        for node in partial_graph['global_nodes'].values():
            if node['created_at'] > times[-1]+1e-7: raise ValueError('future node')
            c=certified.cells.get(tuple(node['cell']))
            if c is None or c['state']!=1:raise ValueError('topology node outside observed free support')
            center=[(node['cell'][0]+.5)*partial_graph['resolution'],c['height'],(node['cell'][1]+.5)*partial_graph['resolution']]
            if np.linalg.norm(np.asarray(node['position'])-center)>1e-8:raise ValueError('uncertified topology node location')
        for edge in partial_graph['global_edges']:
            if edge['revision'] > partial_graph['revision']: raise ValueError('future edge revision')
        if actions and len(actions) != len(rows)-1: raise ValueError('action alignment')
        if any(a not in ('forward','left','right','wait','stop') for a in actions):
            raise ValueError('unknown executed action')
        return cls(rows,tuple(actions),partial_graph)


@dataclass(frozen=True)
class JointHumanLaw:
    """Complete continuous joint law underlying the Monte Carlo world bank.

    One shared categorical mode, coherent Gaussian/t velocity coefficients,
    source-episode Bernoulli existence, spatial/motion Poisson marks, and the
    declared historical count conditioning. No finite-bank zero certifies safety.
    """
    parameters: Mapping
    origin: np.ndarray
    velocity: np.ndarray
    age: np.ndarray
    robotxy: np.ndarray
    minimum_count: int
    tail: str
    independent_initial: bool
    persistent_offset_velocity: bool = False
    compulsory_unique_episode: bool = False
    max_speed: float = None
    compulsory_observed_clique: bool = False
    knots: tuple = (0.,.5,1.,2.,3.)
    common_future_coefficient: float = .2
    native_endpoint_extension: float = 3.028
    version: str = 'continuous_joint_human_law_1'

    def count_log_probability(self,count):
        from .count import count_log_probability,certain_sources
        return count_log_probability(self.parameters['presence_logits'][:len(self.age)],self.age,
            float(self.parameters['rate']),count,self.minimum_count,
            certain=certain_sources(self.age,self.minimum_count,self.compulsory_unique_episode,self.compulsory_observed_clique))


def immutable_array(value):
    a=np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class JointHumanWorlds:
    """A Monte Carlo approximation to one joint random finite path set.

    present is constant through a sampled future, including out-of-view people.
    local episode keys are not oracle identities. Hidden keys are sample-local.
    The residual Poisson law has unbounded count support; finite draws do not.
    All paths in one row share a persistent mode and common innovation.
    """
    timestamp: float
    times: np.ndarray
    weights: np.ndarray
    positions: np.ndarray  # [world,time,person,xy]
    present: np.ndarray  # [world,person]
    track_keys: Tuple[str,...]
    unseen: np.ndarray
    poisson_rate: float
    support: Tuple[str,...]
    minimum_count: int = 0
    law: JointHumanLaw = None
    version: str = 'joint_human_worlds_two_stage_1'

    def validate(self):
        k,t,n,d = self.positions.shape
        assert d == 2 and self.present.shape == (k,n)
        assert len(self.times)==t and len(self.track_keys)==n
        assert np.all(np.diff(self.times)>0) and abs(self.times[0])<1e-8
        assert np.isfinite(self.positions).all()
        assert np.isfinite(self.weights).all() and (self.weights>=0).all()
        assert abs(self.weights.sum()-1)<1e-8
        assert self.poisson_rate>0 and self.unseen.shape==(n,)
        assert (self.present.sum(-1)>=self.minimum_count).all()
        return self


def require_stage1_pass(gate):
    """Stage 2 construction must call this guard, never merely test file existence."""
    if gate.get('decision') != 'STAGE1_PASS' or not all(gate.get('gates',{}).get(k,{}).get('status')=='PASS' for k in 'ABCD'):
        raise RuntimeError('Stage 2 blocked: mandatory Stage 1 gate has not passed')
