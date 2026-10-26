"""Episode runtime: branch states for the Oracle and authoritative replay.

The *same* agent step (legal query execution -> command -> environment step
-> legal observation) runs either against the privileged ``BranchSimulator``
(Oracle search) or the unchanged ``habitat.Env.step`` (replay).  Branch
states own their legal history, sensor state, planner state and query
progress; nothing observed in one branch can enter another.
"""
from dataclasses import dataclass, field
import copy
import math
import time
import numpy as np

from .native import (NativeTask, HumanTape, BranchSimulator, classify_contacts, MAX_EPISODE_STEPS,
                     SUCCESS_DISTANCE, BODY_PROXY_M)
from .legal import LegalState, LegalSensor
from .executor import QueryExecution, Query, command_for, next_dyn_state
from .planners import make_planner
from .dynamics import STEP_SECONDS


@dataclass
class BranchState:
    robot: object              # native RobotState (None in real replay)
    index: int                 # native steps taken (= tape index)
    sensor: object             # LegalSensor state
    legal: LegalState
    dyn_state: tuple
    execution: object          # QueryExecution or None
    decisions: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    outcome: str = None        # None | success | human_collision | static_collision | stop_fail | timeout
    static_contacts: int = 0
    body_overlaps: int = 0                # steps violating the declared body-proxy criterion
    min_rect_dist: float = math.inf       # min robot-rectangle to human-centre distance
    min_center_dist: float = math.inf     # privileged diagnostic (robot-human centre)
    psc_steps: int = 0                    # steps with any human centre < 1 m (Euclidean, privileged)
    path_length: float = 0.
    geodesic: float = math.nan            # privileged distance-to-goal at this state

    def copy(self):
        return BranchState(self.robot, self.index, self.sensor, self.legal.copy(), self.dyn_state,
                           self.execution.copy() if self.execution is not None else None,
                           list(self.decisions), list(self.actions), self.outcome, self.static_contacts,
                           self.body_overlaps, self.min_rect_dist,
                           self.min_center_dist, self.psc_steps, self.path_length, self.geodesic)

    @property
    def terminal(self):
        return self.outcome is not None


class EpisodeRuntime:
    """One benchmark episode with a fixed planner and dynamics model."""

    def __init__(self, spec, planner_name, dyn, planner_kw=None, tape_steps=MAX_EPISODE_STEPS,
                 strict=True, trace=False):
        t0 = time.perf_counter()
        self.spec = spec
        self.dyn = dyn
        self.planner_name = planner_name
        self.planner_kw = planner_kw or {}
        self.strict = strict
        self.trace = [] if trace else None
        self.task = NativeTask(spec, velocity_action=not dyn.native_actions)
        self.private_cache = {}
        self.sensor = LegalSensor(self.task, private_cache=self.private_cache)
        self.sim = BranchSimulator(self.task, None)
        root_robot = self.sim.root()
        row = self.sensor.observe(self.task.world_time())
        legal = LegalState(self.task.goal, row['robot_pose'], row['timestamp'])
        legal.map.apply(row)
        legal.tracks = row['tracks']
        legal.track_log.append((legal.time, row['tracks']))
        self.root = BranchState(root_robot, 0, self.sensor.state(), legal, (0., 0.), None)
        self.root.geodesic = self.task.geodesic_to_goal()
        self.start_geodesic = self.root.geodesic
        self.sim.tape = HumanTape.record(self.task, tape_steps)
        self.tape = self.sim.tape
        self._loaded = None
        self.setup_s = time.perf_counter()-t0
        self.perf = dict(steps=0, sim_s=0., sensor_s=0., planner_s=0., legal_s=0., copy_s=0.)

    def new_execution(self, query):
        return QueryExecution(query, make_planner(self.planner_name, **self.planner_kw), self.dyn)

    def _load(self, state):
        if self._loaded is not state:
            self.sim.load(state.robot, state.index)
            self.sensor.restore(state.sensor)
            self._loaded = state

    def run(self, parent, query=None, horizon=20, cont=False, geodesic=True):
        """Execute ``query`` (or continue the parent's execution) from ``parent``.

        Returns a new terminal-or-decision BranchState; ``parent`` is untouched.
        """
        if parent.terminal:
            raise ValueError('Terminal branch')
        if query is not None and query.kind == 'continue':
            query, cont = None, True
        t = time.perf_counter()
        st = parent.copy()
        self.perf['copy_s'] += time.perf_counter()-t
        self._load(parent)
        if cont:
            if st.execution is None or st.execution.status != 'running':
                raise ValueError('Nothing to continue')
            ex = st.execution
            ex.decision_boundary()
        else:
            ex = self.new_execution(query)
        st.execution = ex
        record = dict(query=ex.query, start=st.index, cont=bool(cont), horizon=int(horizon))
        sim, sensor, dyn = self.sim, self.sensor, self.dyn
        for _ in range(horizon):
            t = time.perf_counter()
            prim = ex.step(st.legal, st.dyn_state)
            self.perf['planner_s'] += time.perf_counter()-t
            cmd = command_for(dyn, prim)
            t = time.perf_counter()
            res = sim.step(cmd, compute_geodesic=False)
            self.perf['sim_s'] += time.perf_counter()-t
            self.perf['steps'] += 1
            st.index = res.index
            st.robot = sim.state
            st.actions.append(cmd if isinstance(cmd, str) else tuple(map(float, cmd)))
            prev = st.legal.pose
            st.path_length += float(math.hypot(res.pose[0]-prev[0], res.pose[2]-prev[2]))
            hp = self.tape.positions[res.index]
            if len(hp):
                d = np.hypot(hp[:, 0]-res.pose[0], hp[:, 2]-res.pose[2])
                st.min_center_dist = min(st.min_center_dist, float(d.min()))
                st.psc_steps += int(bool((d < 1.0).any()))
            if res.static_contact:
                st.static_contacts += 1
            st.body_overlaps += int(res.body_overlap)
            st.min_rect_dist = min(st.min_rect_dist, res.min_rect_distance)
            if res.human_collision:
                st.outcome = 'human_collision'
            elif res.body_overlap and self.strict:
                st.outcome = 'body_overlap'
            elif res.static_contact and self.strict:
                st.outcome = 'static_collision'
            elif res.stop:
                st.geodesic = res.geodesic
                st.outcome = 'success' if res.success else 'stop_fail'
            elif res.index >= MAX_EPISODE_STEPS:
                st.outcome = 'timeout'
            if st.outcome is None:
                t = time.perf_counter()
                row = sensor.observe(res.time)
                self.perf['sensor_s'] += time.perf_counter()-t
                t = time.perf_counter()
                st.legal.observe(row, prim)
                self.perf['legal_s'] += time.perf_counter()-t
                st.dyn_state = next_dyn_state(dyn, prim, st.dyn_state)
            if self.trace is not None:
                self.trace.append(dict(index=res.index, pose=res.pose.tolist(), prim=prim if isinstance(prim, str) else list(prim),
                                       static=bool(res.static_contact), overlap=res.body_overlap,
                                       rect=float(res.min_rect_distance), human_collision=res.human_collision,
                                       route_len=ex.route_len, label=ex.query.label, kind=ex.query.kind,
                                       mode=getattr(ex.planner, 'mode', None),
                                       target=list(ex.target) if ex.target is not None else None))
            if st.outcome is not None or ex.status != 'running':
                break
        st.sensor = sensor.state()
        ex.decision_boundary()
        self._loaded = st
        if geodesic and st.outcome is None:
            st.geodesic = self.task.geodesic_to_goal()
        record.update(end=st.index, status=ex.status, reason=ex.reason, outcome=st.outcome)
        st.decisions.append(record)
        return st

    def close(self):
        self.task.close()


def replay(spec, planner_name, dyn, decisions, planner_kw=None, verbose=False):
    """Authoritative continuous replay with the unchanged ``habitat.Env.step``.

    Re-executes the recorded *query sequence* through the live executor from
    the episode reset (no tape, no restoration) and returns official metrics,
    static contacts and the action sequence for comparison with the Oracle.
    """
    task = NativeTask(spec, velocity_action=not dyn.native_actions)
    try:
        sensor = LegalSensor(task)
        row = sensor.observe(task.world_time())
        legal = LegalState(task.goal, row['robot_pose'], row['timestamp'])
        legal.map.apply(row)
        legal.tracks = row['tracks']
        dyn_state = (0., 0.)
        actions, statics, per_decision = [], 0, []
        overlaps, min_rect = 0, math.inf
        ex = None
        for rec in decisions:
            if rec['cont']:
                if ex is None or ex.status != 'running':
                    raise RuntimeError('Replay cannot continue a finished execution')
                ex.decision_boundary()
            else:
                ex = QueryExecution(rec['query'], make_planner(planner_name, **(planner_kw or {})), dyn)
            n = 0
            for _ in range(rec['horizon']):
                if task.episode_over:
                    break
                prim = ex.step(legal, dyn_state)
                cmd = command_for(dyn, prim)
                task.step(cmd)
                n += 1
                actions.append(cmd if isinstance(cmd, str) else tuple(map(float, cmd)))
                c = classify_contacts(task, gate_humans=False, need_static=True)
                statics += int(bool(c['static']))
                overlaps += int(c['min_rect_distance'] < BODY_PROXY_M)
                min_rect = min(min_rect, c['min_rect_distance'])
                if task.episode_over:
                    break
                row = sensor.observe(task.world_time())
                legal.observe(row, prim)
                dyn_state = next_dyn_state(dyn, prim, dyn_state)
                if ex.status != 'running':
                    break
            per_decision.append(n)
            ex.decision_boundary()
        metrics = task.metrics()
        return dict(metrics=metrics, actions=actions, static_contact_steps=statics,
                    body_overlap_steps=overlaps, min_rect_distance=min_rect,
                    per_decision_steps=per_decision, episode_over=task.episode_over, steps=task.steps,
                    final_geodesic=task.geodesic_to_goal())
    finally:
        task.close()
