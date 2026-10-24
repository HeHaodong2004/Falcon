"""Canonical immutable eventwm.l25l4.contract.v1 reference compiler.

World XYZ plus yaw; yaw zero faces +X, positive yaw turns toward -Z.
No simulator, human identity or human future is accepted here.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from eventwm_graph.config import digest
from eventwm_graph.partial.query import FORWARD_STEP, TURN_STEP, wrap, yaw_to

CONTRACT = 'eventwm.l25l4.contract.v1'
COMPILER_VERSION = 'l4.reference.1'


@dataclass(frozen=True)
class ActionSpec:
    forward: float = FORWARD_STEP
    turn: float = TURN_STEP
    nominal_dt: float = .048
    min_dt: float = .048
    max_dt: float = .056
    actions: tuple = ('forward', 'left', 'right')
    stop_terminates: bool = True

    def __post_init__(self):
        if self.actions != ('forward', 'left', 'right') or not self.stop_terminates:
            raise ValueError('Unverified native action protocol')
        if not 0 < self.min_dt <= self.nominal_dt <= self.max_dt:
            raise ValueError('Invalid action timing')

    def step(self, pose, action):
        if action not in self.actions: raise ValueError('Illegal execution action')
        p = np.array(pose, dtype=float, copy=True)
        if action == 'forward':
            p[0] += self.forward * math.cos(p[3]); p[2] -= self.forward * math.sin(p[3])
        else: p[3] = wrap(p[3] + (self.turn if action == 'left' else -self.turn))
        return p


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    reference_route: tuple
    start_pose: tuple
    terminal_orientation: float
    timestamp: float = 0.
    position_tolerance: float = .155
    orientation_tolerance: float = TURN_STEP / 2 + 1e-5
    execution_limit: int = 400
    duration_limit: float = 22.4
    completion_rule: str = 'ordered_route_endpoint_and_orientation'

    def __post_init__(self):
        object.__setattr__(self, 'reference_route', tuple(tuple(map(float, p)) for p in self.reference_route))
        object.__setattr__(self, 'start_pose', tuple(map(float, self.start_pose)))
        r=np.asarray(self.reference_route)
        if (not self.query_id or r.ndim!=2 or r.shape[1]!=3 or len(r)==0 or
                len(self.start_pose)!=4 or not np.isfinite(r).all() or
                not np.isfinite(self.start_pose).all() or not math.isfinite(self.terminal_orientation)):
            raise ValueError('Invalid reference')
        if np.linalg.norm(r[0]-self.start_pose[:3]) > 1e-5: raise ValueError('Route must start at supplied pose')
        if not 0<self.position_tolerance<.21 or not 0<self.orientation_tolerance<.1:
            raise ValueError('Unsupported completion tolerance')
        if self.execution_limit<=0 or self.duration_limit<=0: raise ValueError('Invalid execution budget')
        if not all(math.isfinite(x) for x in (self.timestamp,self.duration_limit,self.position_tolerance,self.orientation_tolerance)):
            raise ValueError('Nonfinite query timing or tolerance')
        if self.completion_rule!='ordered_route_endpoint_and_orientation': raise ValueError('Unknown completion rule')

    @property
    def target(self): return self.reference_route[-1]

    @property
    def reference_hash(self): return digest(asdict(self))


@dataclass(frozen=True)
class CompiledReference:
    query: QuerySpec
    nominal_waypoints: tuple
    native_action_schedule: tuple
    timestamped_nominal_poses: tuple
    reference_hash: str
    graph_version: str
    nominal_static_supported: bool = True
    compiler_version: str = COMPILER_VERSION
    contract: str = CONTRACT

    def l2_segment(self, first=0, count=40):
        """Deterministic fixed-query adapter; never an imagined actual packet."""
        if first<0 or count<1 or first>len(self.native_action_schedule): raise ValueError('Invalid segment')
        end=min(first+count,len(self.native_action_schedule))
        rows=self.timestamped_nominal_poses[first:end+1]
        from eventwm_graph.l2_partial.state import EXECUTOR_VERSION
        return dict(query_id=f'{self.query.query_id}:{first}:{end}', executor_version=EXECUTOR_VERSION,
                    waypoints=[list(rows[-1][1:4])], final_yaw=rows[-1][4],
                    parent_reference_hash=self.reference_hash,
                    compiled=dict(planned_poses=[list(p[1:]) for p in rows],
                                  actions=list(self.native_action_schedule[first:end]), graph_revision=self.graph_version),
                    nominal_absolute_times=[p[0] for p in rows])


def compile_reference(query, legal_graph, action_spec=ActionSpec()):
    """Compile exactly the supplied route; reject unsupported native steps.

    legal_graph supplies segment_free(a,b), motion_free(a,b) and version.
    Stage 1 callers can supply its compiled route through from_stage1 below.
    This function never changes connectivity or substitutes another objective.
    """
    if not isinstance(query,QuerySpec): raise TypeError('QuerySpec required')
    route=np.asarray(query.reference_route); pose=np.array(query.start_pose)
    if any(not legal_graph.segment_free(a,b) for a,b in zip(route[:-1],route[1:])):
        raise ValueError('Reference segment not certified')
    actions=[]; poses=[(query.timestamp,*pose)]; visited={}; nominal_supported=True
    target=1 if len(route)>1 else 0
    while True:
        while target<len(route)-1 and np.linalg.norm(pose[:3]-route[target])<=query.position_tolerance:
            target+=1
        arrived=target==len(route)-1 and np.linalg.norm(pose[:3]-route[-1])<=query.position_tolerance
        desired=query.terminal_orientation if arrived else yaw_to(pose[:3],route[target])
        angle=wrap(desired-pose[3])
        if arrived and abs(angle)<=query.orientation_tolerance: break
        if len(actions)>=query.execution_limit: raise ValueError('Nominal compilation exceeds query budget')
        action=('left' if angle>0 else 'right') if abs(angle)>query.orientation_tolerance else 'forward'
        nxt=legal_graph.predict(pose,action,action_spec) if hasattr(legal_graph,'predict') else action_spec.step(pose,action)
        # Nominal reference is an immutable target, not an authorization to
        # execute an unsafe step. L4 must screen every actual candidate.
        nominal_supported=nominal_supported and legal_graph.motion_free(pose,nxt)
        pose=nxt; actions.append(action); poses.append((query.timestamp+len(actions)*action_spec.nominal_dt,*pose))
        key=tuple(np.round(pose,4))+(target,); visited[key]=visited.get(key,0)+1
        if visited[key]>2: raise ValueError('Nominal discrete route cycle')
    graph_version=str(legal_graph.version)
    identity=digest(dict(contract=CONTRACT,compiler=COMPILER_VERSION,query=asdict(query),
                         action_spec=asdict(action_spec),graph_version=graph_version,
                         actions=actions,poses=poses,nominal_static_supported=nominal_supported))
    return CompiledReference(query,query.reference_route,tuple(actions),tuple(poses),identity,graph_version,nominal_supported)


def from_stage1(query, observation, execution_limit=400):
    """Preserve the established Stage 1 route and terminal-yaw semantics."""
    c=query['compiled']
    if not c.get('ok'): raise ValueError('Stage 1 query lacks successful compilation')
    route=[list(p) for p in c['route']]
    if np.linalg.norm(np.asarray(route[0])-observation['robot_pose'][:3])>1e-5:
        raise ValueError('Stale Stage 1 reference start')
    return QuerySpec(query['query_id'],route,observation['robot_pose'],query['final_yaw'],
                     observation['timestamp'],execution_limit=execution_limit)
