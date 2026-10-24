"""Actual-transition execution API. This module cannot access a simulator."""
from dataclasses import dataclass, field, replace
from enum import Enum
import math
import time
import numpy as np
from .reference import ActionSpec, QuerySpec, CompiledReference
from .progress import RouteProgress
from .reactive import nominal_action, near_term_risk, choose_maneuver, static_escape, clearance
from .geometry import swept_free, body_free


class Mode(str,Enum):
    FOLLOW='FOLLOW';YIELD='YIELD';AVOID='AVOID';REJOIN='REJOIN'


class Status(str,Enum):
    RUNNING='RUNNING';QUERY_COMPLETED='QUERY_COMPLETED';BLOCKED='BLOCKED'
    EXECUTOR_LIMIT='EXECUTOR_LIMIT';INTERRUPTED='INTERRUPTED'


@dataclass(frozen=True)
class RuntimeContext:
    geometry: object
    action_spec: ActionSpec = ActionSpec()
    # Deprecated constructor fields retained for old callers; no search or
    # local lifetime watchdog uses these values in progress-first execution.
    horizon: int = 32
    beam_width: int = 16
    no_progress_seconds: float = 5.
    profile_id: str = 'l4.progress_first.obs.v2'
    reactive: bool = True

    def __post_init__(self):
        if not 1<=self.horizon<=40 or not 3<=self.beam_width<=32: raise ValueError('Search budget out of range')


@dataclass(frozen=True)
class ActionDecision:
    action: object
    planned_ego_trajectory: tuple
    mode: str
    collision_check: dict
    forecast_source: str
    forecast_validity: str
    route_target: tuple
    reason: str
    control_ms: float


@dataclass
class ExecutionHandle:
    reference: CompiledReference
    context: RuntimeContext
    tracker: RouteProgress
    status: Status = Status.RUNNING
    mode: Mode = Mode.FOLLOW
    reason: str = ''
    actions: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    poses: list = field(default_factory=list)
    observation_refs: list = field(default_factory=list)
    pending: object = None
    preferred: int = 0
    episode_termination: bool = False
    episode_truncation: bool = False
    mode_history: list = field(default_factory=list)
    maneuver: list = field(default_factory=list)
    cooldown_distance: float = 0.
    avoidance_attempts: int = 0
    avoidance_failures: int = 0
    transition_events: list = field(default_factory=list)


class Executor:
    def update_geometry(self,handle,legal_geometry):
        """Install arrived static evidence without changing the reference.

        The unified observation owner supplies this certified map. Changes are
        prohibited while a native action is awaiting acknowledgment.
        """
        if handle.pending is not None:raise ValueError('Resolve actual transition before static update')
        handle.context=replace(handle.context,geometry=legal_geometry)

    def start(self,query,compiled_reference,runtime_context):
        if query!=compiled_reference.query: raise ValueError('Query/reference identity mismatch')
        if str(runtime_context.geometry.version)!=compiled_reference.graph_version:
            raise ValueError('Compiled static graph differs from runtime graph')
        handle=ExecutionHandle(compiled_reference,runtime_context,RouteProgress(compiled_reference))
        handle.timestamps=[query.timestamp];handle.poses=[query.start_pose]
        if handle.tracker.update(query.start_pose,query.timestamp): handle.status=Status.QUERY_COMPLETED
        return handle

    @staticmethod
    def _people(observation,now):
        rows=[]
        for t in observation.get('tracks',[]):
            if not t.get('valid',True): continue
            seen=float(t['last_seen_time']);age=now-seen
            if age< -1e-7: raise ValueError('Future human observation')
            if t.get('visible_now',False) and abs(age)>1e-7: raise ValueError('Stale track mislabeled current')
            p=np.asarray(t['position'],float)
            valid=bool(t.get('velocity_valid',False))
            v=np.asarray(t.get('estimated_velocity') if valid else [0,0,0],float)
            if p.shape!=(3,) or v.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(v).all():
                raise ValueError('Invalid measured track')
            # Reject unreliable associations without modifying exact measured position.
            if np.linalg.norm(v[[0,2]])>3.:v=np.zeros(3);valid=False
            rows.append([p[0],p[2],v[0],v[2],max(0.,age),float(valid)])
        return np.asarray(rows,dtype=float).reshape(-1,6)

    def act(self,handle,live_state,actual_observations,forecast_view=None):
        h=handle;began=time.perf_counter();now=float(live_state['timestamp'])
        pose=np.asarray(live_state['robot_pose'],float);ctx=h.context;q=h.reference.query
        if pose.shape!=(4,) or not np.isfinite(pose).all() or not math.isfinite(now): raise ValueError('Invalid actual state')
        if abs(now-float(actual_observations['timestamp']))>1e-7: raise ValueError('Observation/state time mismatch')
        if abs(now-h.timestamps[-1])>1e-7 or np.linalg.norm(pose-np.asarray(h.poses[-1]))>1e-5:
            raise ValueError('Actual state must arrive through on_transition')
        if h.pending is not None: return h.pending
        # Query scheduling belongs to L3. Legacy budgets remain metadata; only
        # explicit interrupt, native episode end, or actual completion ends work.
        target=h.tracker.route[h.tracker.next_waypoint].copy()
        if h.status!=Status.RUNNING:
            return ActionDecision(None,(),h.mode.value,{},'none','not_requested',tuple(target),h.reason or h.status.value,0.)
        nominal=nominal_action(pose,target,q,h.tracker.next_waypoint==len(h.tracker.route)-1)
        tracking_ms=(time.perf_counter()-began)*1000
        people=self._people(actual_observations,now)
        source='observed_motion';validity='not_requested'
        forecast=np.empty((0,0,2));times=np.empty(0)
        if forecast_view is not None:
            validity,forecast,times,source=forecast_view.for_control(now)
        action=nominal;reason='nominal_tracking';expanded=1;reactive_ms=0.
        risk=False
        before=time.perf_counter()
        if ctx.reactive:
            risk=near_term_risk(pose,nominal,people,ctx.action_spec)
            if h.maneuver:
                # Execute the short committed side consistently. Recheck only
                # the next motion; a failed check resumes nominal immediately.
                candidate=h.maneuver[0]
                candidate_risk=near_term_risk(pose,candidate,people,ctx.action_spec,horizon=1)
                candidate_clearance=clearance(ctx.action_spec.step(pose,candidate),people,ctx.action_spec.max_dt)
                nominal_clearance=clearance(ctx.action_spec.step(pose,nominal),people,ctx.action_spec.max_dt)
                if candidate_risk and candidate_clearance<nominal_clearance-.05:
                    h.maneuver.clear();h.cooldown_distance=h.tracker.path_length+.8
                    reason='avoidance_no_longer_useful_nominal'
                else:
                    action=h.maneuver.pop(0);reason='committed_avoidance'
                    if not h.maneuver:h.cooldown_distance=h.tracker.path_length+.8
            elif risk and h.tracker.path_length>=h.cooldown_distance:
                h.avoidance_attempts+=1
                sequence,checks=choose_maneuver(pose,target,people,ctx.geometry,ctx.action_spec,h.preferred,
                                               forecast,times,now)
                expanded+=checks
                if sequence:
                    h.preferred=1 if sequence[0]=='left' else 2
                    action=sequence[0];h.maneuver=list(sequence[1:]);reason='reactive_avoidance'
                else:
                    h.avoidance_failures+=1;h.cooldown_distance=h.tracker.path_length+.8
                    reason='avoidance_unavailable_nominal'
        reactive_ms=(time.perf_counter()-before)*1000
        before=time.perf_counter()
        # The grid result is best-effort support evidence, not a permission gate.
        # No online mesh/FK fallback is called. Native sliding/contact is real.
        predicted=ctx.geometry.predict(pose,action,ctx.action_spec)
        static_supported=bool(swept_free(pose,predicted,*ctx.geometry.kernel_args()))
        if (ctx.reactive and action==nominal=='forward' and not static_supported
                and h.tracker.path_length>=h.cooldown_distance and not h.maneuver
                and body_free(pose,*ctx.geometry.kernel_args())):
            sequence,checks=static_escape(pose,target,people,ctx.geometry,ctx.action_spec)
            expanded+=checks
            if sequence:
                action=sequence[0];h.maneuver=list(sequence[1:]);reason='static_nudge'
                predicted=ctx.geometry.predict(pose,action,ctx.action_spec)
                static_supported=bool(swept_free(pose,predicted,*ctx.geometry.kernel_args()))
        static_ms=(time.perf_counter()-before)*1000
        path=np.array([pose,predicted])
        mode=Mode.AVOID if reason in ('reactive_avoidance','committed_avoidance','static_nudge') else (
            Mode.REJOIN if h.tracker.deviation>.2 else Mode.FOLLOW)
        if mode!=h.mode:h.mode_history.append(dict(timestamp=now,mode=mode.value));h.mode=mode
        elapsed=(time.perf_counter()-began)*1000
        decision=ActionDecision(action,tuple(tuple(map(float,p)) for p in path),h.mode.value,
            dict(expanded=int(expanded),nominal_action=nominal,predicted_human_risk=bool(risk),
                 static_supported=static_supported,precise_static_fallback=False,
                 tracking_ms=tracking_ms,reactive_ms=reactive_ms,static_ms=static_ms,
                 actual_graph_version=str(ctx.geometry.version),
                 static_scope='best_effort_certified_grid_no_full_body_guarantee',
                 human_scope='short_observed_motion_risk_not_actual_collision'),
            source,validity,tuple(target),reason,elapsed)
        if action is not None:h.pending=decision
        return decision

    def on_transition(self,handle,actual_transition):
        h=handle;t=actual_transition
        if h.pending is None or t['action']!=h.pending.action: raise ValueError('Unsolicited or mismatched actual transition')
        stamp=float(t['timestamp']);pose=tuple(map(float,t['robot_pose']))
        if not math.isfinite(stamp) or stamp<=h.timestamps[-1] or len(pose)!=4 or not np.isfinite(pose).all():
            raise ValueError('Actual transition must advance finite native state/time')
        dt=stamp-h.timestamps[-1];predicted=np.asarray(h.pending.planned_ego_trajectory[1])
        h.actions.append(t['action']);h.timestamps.append(stamp);h.poses.append(pose)
        h.observation_refs.append(str(t['observation_ref']));h.pending=None
        completed=h.tracker.update(pose,stamp)
        h.episode_termination=bool(t.get('episode_termination',False))
        h.episode_truncation=bool(t.get('episode_truncation',False))
        # Contact and sliding are environment outcomes, not executor handoffs.
        # Keep every event even if the native episode continues after contact.
        event=dict(timestamp=stamp,human_collision=bool(t.get('human_collision')),
                   static_collision=bool(t.get('static_collision')),
                   motion_error_m=float(np.linalg.norm((np.asarray(pose)-predicted)[[0,2]])),
                   clock_outside_verified_bounds=not h.context.action_spec.min_dt-1e-6<=dt<=h.context.action_spec.max_dt+1e-6)
        h.transition_events.append(event)
        if h.episode_termination or h.episode_truncation:
            h.status=Status.INTERRUPTED
            h.reason='native_collision' if event['human_collision'] else 'native_episode_ended'
        elif completed and not event['human_collision']:
            h.status=Status.QUERY_COMPLETED
        return self.feedback(h)

    def interrupt(self,handle,reason='L3_handoff'):
        if handle.pending is not None: raise ValueError('Resolve in-flight native action before interrupting')
        handle.status=Status.INTERRUPTED;handle.reason=reason
        return self.feedback(handle)

    @staticmethod
    def feedback(h):
        return dict(contract='eventwm.l25l4.contract.v1',query_id=h.reference.query.query_id,
            reference_hash=h.reference.reference_hash,executor_profile_id=h.context.profile_id,
            actual_actions=list(h.actions),actual_timestamps=list(h.timestamps),actual_poses=list(h.poses),
            actual_duration=h.timestamps[-1]-h.timestamps[0],route_progress=h.tracker.progress,
            route_deviation=h.tracker.deviation,query_status=h.status.value,blocked_reason=h.reason,
            actual_observation_refs=list(h.observation_refs),episode_termination=h.episode_termination,
            episode_truncation=h.episode_truncation,detours=h.tracker.detours,rejoins=h.tracker.rejoins,
            path_length=h.tracker.path_length,mode_history=list(h.mode_history),
            transition_events=list(h.transition_events),avoidance_attempts=h.avoidance_attempts,
            avoidance_failures=h.avoidance_failures)
