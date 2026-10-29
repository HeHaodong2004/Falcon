"""Real-episode driver for the World Model Sandbox (tests, benchmarks, the RL integration reference).

``RealEpisode`` runs one benchmark episode in the UNCHANGED ``habitat.Env`` (querygen ``NativeTask``) and keeps the
two legal agent-side views the sandbox consumes, updated after every environment step from the same observation:

* environment side (a simulator, privileged by construction -- like any environment): the env, the legal geometry
  sensor (querygen ``LegalSensor``: legal observation rows) and the range sensor (``ranges.py``: one range frame per
  row at the same camera pose);
* agent side (legal only): ``legal`` (querygen ``LegalState``: the executor's view, with bump memory), ``stream``
  (``context.LegalStream``: frames, causal partial graph and range packet for Layer 2), the running
  ``QueryExecution`` and the robot's speed level.

The sandbox never sees the environment side.  ``generator(config)`` returns the frozen querygen candidate generator
of a named configuration (``CONFIGS``: the v14 K=10 / K=6 sets).
"""
import json
import math
import time

import numpy as np

from eventwm_graph.querygen.dynamics import FAMILIES
from eventwm_graph.querygen.executor import QueryExecution, Query, command_for
from eventwm_graph.querygen.planners import make_planner
from .context import LegalStream, RESOLUTION

# v14 frozen generator configurations (scripts/querygen_dev_v14.sh): the v12 legal default-policy foundation +
# goal_unknown_cost 1.25 + '_th' copies in the K=10 sets
FOUNDATION = dict(goal_unstick=6, bump_memory_m=0.25, evade=True, unstick_keep=True, bump_front=True, static_humans=True,
                  stop_strict=True, unstick_progress=True, goal_relax_m=0.5, stop_closest_m=1.2, goal_unknown_cost=1.25)
CONFIGS = {
    'k10': dict(tc_variants=['goal', 'st_goal'], th_variants=['goal', 'st_goal'], st_goal=True,
                maneuver_spawn_only=False, **FOUNDATION),
    'k6': dict(select='compact', budget=6, tc_variants=['goal'], st_goal=True, maneuver_spawn_only=False,
               compact_long_turns=True, compact_human_first=True, **FOUNDATION),
}


def generator(config, dyn):
    """The querygen QueryGenerator of a named (or explicit dict) configuration.  Construction sets the process-wide
    legal options the executor shares with the generator (bump memory radius, blocked-target relaxation)."""
    from eventwm_graph.querygen.generator import QueryGenerator, GeneratorConfig
    kw = dict(CONFIGS[config]) if isinstance(config, str) else dict(config)
    for k in ('tc_variants', 'th_variants', 'vc_variants', 'xc_variants', 'kturns'):
        if k in kw:
            kw[k] = tuple(kw[k])
    return QueryGenerator(FAMILIES[dyn] if isinstance(dyn, str) else dyn, GeneratorConfig(**kw))


def freeze_gc():
    """Long-running workers: move every live object (simulator, models, modules) to the permanent GC generation, so
    cyclic collections scan only objects created afterwards (full collections otherwise pause 150-215 ms with
    ~300k-750k live objects, measured).  Pair with ``thaw_gc`` when the episode's objects are released."""
    import gc
    gc.collect()
    gc.freeze()


def thaw_gc():
    import gc
    gc.unfreeze()
    gc.collect()


def make_stream(stream, spec, global_graph=False):
    """The episode's legal stream: a name (a local ``LegalStream``) or a stream object, e.g. a
    ``service.RemoteStream`` whose Layer 2 view a sandbox server keeps."""
    if hasattr(stream, 'observe'):
        return stream
    return LegalStream(stream or spec.key, RESOLUTION, global_graph=global_graph)


def close_stream(stream):
    """Close a stream or sensor that holds resources (a remote stream's server state, a range sensor's GPU buffers)."""
    close = getattr(stream, 'close', None)
    if close is not None:
        close()


def default_query(gen_kw=None):
    """The legal default-policy goal query of the querygen executor-only floor (same meta)."""
    g = dict(FOUNDATION if gen_kw is None else gen_kw)
    n = int(g.get('goal_unstick', 0))
    meta = dict(unstick=n) if n else {}
    if g.get('evade'):
        meta['evade'] = 1
    if n and g.get('unstick_keep'):
        meta['ukeep'] = 1
    if n and g.get('static_humans'):
        meta['shum'] = 1
    if n and g.get('unstick_progress'):
        meta['uprog'] = 1
    if g.get('bump_front') and meta:
        meta['bumpf'] = 1
    if g.get('stop_strict'):
        meta['stop19'] = 1
    if g.get('goal_unknown_cost'):
        meta['unknown_cost'] = float(g['goal_unknown_cost'])
    return Query('goal', label='goal', meta=meta)


class RealEpisode:
    def __init__(self, spec, planner='orca', dynamics='native', range_sensor='native', stream=None,
                 global_graph=False, gc_freeze=False):
        from eventwm_graph.querygen.native import NativeTask
        from eventwm_graph.querygen.legal import LegalSensor, LegalState
        from .ranges import make_range_sensor
        t0 = time.perf_counter()
        self.spec = spec
        self.dyn = FAMILIES[dynamics] if isinstance(dynamics, str) else dynamics
        self.planner = planner
        self.task = NativeTask(spec, velocity_action=not self.dyn.native_actions)
        self.sensor = LegalSensor(self.task, private_cache={}, pure_caches=True, fast=True)
        self.ranges = make_range_sensor(range_sensor, self.task, self.sensor.config)
        self.stream = make_stream(stream, spec, global_graph)
        self.perf = dict(steps=0, env_s=0., sensor_s=0., range_s=0., legal_s=0., stream_s=0., planner_s=0.)
        row, rf = self._sense()
        self.legal = LegalState(self.task.goal, row['robot_pose'], row['timestamp'])
        self.legal.map.apply(row)
        self.legal.tracks = row['tracks']
        self.legal.track_log.append((self.legal.time, row['tracks']))
        t = time.perf_counter()
        self.stream.observe(row, rf)
        self.perf['stream_s'] += time.perf_counter()-t
        self.execution = None
        self.level = 0
        self.actions = []
        self.decisions = []
        self.gc_freeze = bool(gc_freeze)
        if self.gc_freeze:
            freeze_gc()
        self.setup_s = time.perf_counter()-t0

    # ------------------------------------------------------------------ sensing
    def _sense(self):
        t = time.perf_counter()
        row = self.sensor.observe(self.task.world_time())
        t1 = time.perf_counter()
        rf = self.ranges.measure()
        t2 = time.perf_counter()
        self.perf['sensor_s'] += t1-t; self.perf['range_s'] += t2-t1
        return row, rf

    @property
    def over(self):
        return self.task.episode_over

    @property
    def steps(self):
        return int(self.task.steps)

    def decision_state(self, **metadata):
        return self.stream.decision_state(dict(metadata, key=self.spec.key, steps=self.task.steps))

    # ------------------------------------------------------------------ acting
    def run(self, query=None, horizon=16, on_step=None):
        """Execute ``query`` (None = continue the running execution) for at most ``horizon`` real steps."""
        if self.over:
            raise RuntimeError('episode is over')
        if query is None:
            if self.execution is None or self.execution.status != 'running':
                raise RuntimeError('nothing to continue')
            self.execution.decision_boundary()
        else:
            prev = self.execution
            self.execution = QueryExecution(query, make_planner(self.planner), self.dyn)
            if QueryExecution.INHERIT and prev is not None:
                self.execution.inherit(prev)
        ex = self.execution
        rec = dict(query=ex.query, cont=query is None, horizon=int(horizon), start=self.task.steps)
        n = 0
        for _ in range(int(horizon)):
            if self.over:
                break
            t = time.perf_counter()
            prim, level = ex.step(self.legal, self.level)
            self.perf['planner_s'] += time.perf_counter()-t
            cmd = command_for(self.dyn, prim)
            t = time.perf_counter()
            self.task.step(cmd)
            self.perf['env_s'] += time.perf_counter()-t
            self.perf['steps'] += 1
            n += 1
            self.actions.append(cmd if isinstance(cmd, str) else tuple(map(float, cmd)))
            if self.over:
                break
            row, rf = self._sense()
            t = time.perf_counter()
            self.legal.observe(row, prim)
            t1 = time.perf_counter()
            self.stream.observe(row, rf)
            self.perf['legal_s'] += t1-t; self.perf['stream_s'] += time.perf_counter()-t1
            self.level = level
            if on_step is not None:
                on_step(self, row, rf)
            if ex.status != 'running':
                break
        rec.update(steps=n, end=self.task.steps, status=ex.status, reason=ex.reason)
        self.decisions.append(rec)
        ex.decision_boundary()
        return rec

    def metrics(self):
        return self.task.metrics()

    def close(self):
        self.task.close()
        close_stream(self.stream)
        close_stream(self.ranges)
        if self.gc_freeze:
            thaw_gc()


class BranchEpisode:
    """``RealEpisode``'s interface on querygen's exact branch simulator (``EpisodeRuntime``: the ``Env.step`` mirror
    with the humans replayed from the recorded tape; bit-identical poses and actions to the unchanged habitat.Env,
    verified by every Oracle replay).  This is the environment a Layer 3 learner steps (querygen ``QueryEnv``): no live
    human controllers or physics sub-steps, so per-step cost is a fraction of ``habitat.Env``'s.  The tape cache is
    ``QG_TAPE_CACHE`` (default ``output/sandbox_cache/tapes``; a miss records the episode's tape once).
    """

    def __init__(self, spec, planner='orca', dynamics='native', range_sensor='fast', stream=None, global_graph=False,
                 gc_freeze=False):
        import os
        from pathlib import Path
        from eventwm_graph.querygen.episode import EpisodeRuntime
        from .ranges import make_range_sensor
        t0 = time.perf_counter()
        os.environ.setdefault('QG_TAPE_CACHE', str(Path(__file__).resolve().parents[2]/'output/sandbox_cache/tapes'))
        os.environ.setdefault('QG_FAST_SENSE', '1')      # the exact fast legal sensor (identical rows; RealEpisode: fast=True)
        Path(os.environ['QG_TAPE_CACHE']).mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.dyn = FAMILIES[dynamics] if isinstance(dynamics, str) else dynamics
        self.planner = planner
        self.rt = EpisodeRuntime(spec, planner, self.dyn, strict=False, velocity_action=not self.dyn.native_actions)
        self.task = self.rt.task
        self.sensor = self.rt.sensor
        self.ranges = make_range_sensor(range_sensor, self.task, self.sensor.config)
        self.stream = make_stream(stream, spec, global_graph)
        self.perf = dict(steps=0, env_s=0., sensor_s=0., range_s=0., legal_s=0., stream_s=0., planner_s=0.)
        self._observe = self.rt.sensor.observe
        self.rt._load(self.rt.root)                 # robot at the reset, humans at tape index 0
        root_row = self.rt.root.sensor[3]
        t = time.perf_counter()
        rf = self.ranges.measure(timestamp=root_row['timestamp'])
        self.perf['range_s'] += time.perf_counter()-t
        self.stream.observe(root_row, rf)
        self.rt.sensor.observe = self._sense
        self.state = self.rt.root
        self.decisions = []
        self.gc_freeze = bool(gc_freeze)
        if self.gc_freeze:
            freeze_gc()
        self.setup_s = time.perf_counter()-t0

    def _sense(self, ts):
        row = self._observe(ts)
        t = time.perf_counter()
        rf = self.ranges.measure(timestamp=row['timestamp'])
        t1 = time.perf_counter()
        self.stream.observe(row, rf)
        self.perf['range_s'] += t1-t; self.perf['stream_s'] += time.perf_counter()-t1
        return row

    @property
    def legal(self):
        return self.state.legal

    @property
    def execution(self):
        return self.state.execution

    @property
    def level(self):
        return self.state.dyn_state

    @property
    def over(self):
        return self.state.terminal

    @property
    def steps(self):
        return int(self.state.index)

    def decision_state(self, **metadata):
        return self.stream.decision_state(dict(metadata, key=self.spec.key, steps=self.state.index))

    def run(self, query=None, horizon=16, on_step=None):
        if self.over:
            raise RuntimeError('episode is over')
        p = dict(self.rt.perf)
        r0, s0 = self.perf['range_s'], self.perf['stream_s']
        cont = query is None
        if cont and (self.state.execution is None or self.state.execution.status != 'running'):
            raise RuntimeError('nothing to continue')
        start = self.state.index
        self.state = self.rt.run(self.state, query, horizon=int(horizon), cont=cont, geodesic=False)
        q = self.rt.perf
        self.perf['steps'] += q['steps']-p['steps']
        self.perf['env_s'] += (q['sim_s']-p['sim_s'])+(q['copy_s']-p['copy_s'])
        # the runtime times our wrapped observe: legal sensor + range frame + stream update
        self.perf['sensor_s'] += (q['sensor_s']-p['sensor_s'])-(self.perf['range_s']-r0)-(self.perf['stream_s']-s0)
        self.perf['legal_s'] += q['legal_s']-p['legal_s']
        self.perf['planner_s'] += q['planner_s']-p['planner_s']
        rec = dict(query=self.state.decisions[-1]['query'], cont=cont, horizon=int(horizon), start=start,
                   steps=self.state.index-start, end=self.state.index, status=self.state.decisions[-1]['status'],
                   reason=self.state.decisions[-1]['reason'], outcome=self.state.outcome)
        self.decisions.append(rec)
        return rec

    def metrics(self):
        return dict(steps=float(self.state.index), success=float(self.state.outcome == 'success'),
                    human_collision=float(self.state.outcome == 'human_collision'))

    def close(self):
        self.rt.close()
        close_stream(self.stream)
        close_stream(self.ranges)
        if self.gc_freeze:
            thaw_gc()
