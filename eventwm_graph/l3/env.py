"""Layer 3 decision environment: one Falcon episode = a sequence of Query selections.

At every decision state the frozen querygen generator proposes K candidate Queries for the robot's own dynamics,
the World Model Sandbox imagines each of them with the robot's own planner and dynamics, the Graph Compiler turns the
imagination into the consequence graph, a selector chooses one candidate, and the fixed low-level stack executes it
for one decision interval in querygen's exact branch simulator (the ``habitat.Env.step`` mirror; humans replayed from
the recorded tape).

Configurations (``ConfigSpec``): planner (orca | dwa | sfm), dynamics family (querygen ``FAMILIES``) and generator
set (k10 | k6).  ``official`` iff dynamics == native (the benchmark's discrete action set); other families use the
continuous base-velocity action (modified robot dynamics, outside the official benchmark).

Reward components per decision (training signal only; they may use privileged simulator quantities):
  steps, path (m), social steps (any human centre < 1 m: the official PSC criterion), geodesic progress (m),
  terminal outcome (success | human_collision | stop_fail | timeout).
Episode metrics follow the official definitions: success, SPL = success * L*/max(L, L*), PSC = 1 - social/steps,
human collision.
"""
from dataclasses import dataclass, asdict
import math
import time

import numpy as np

HORIZON = 16


@dataclass(frozen=True)
class ConfigSpec:
    planner: str = 'orca'
    dynamics: str = 'native'
    gen: str = 'k10'

    @property
    def name(self):
        return '%s-%s-%s' % (self.planner, self.dynamics, self.gen)

    @property
    def official(self):
        return self.dynamics == 'native'

    @classmethod
    def parse(cls, text):
        p, d, g = text.split('-')
        return cls(p, d, g)


class Decision:
    """One decision state: candidates, the compiled graph and timings."""
    __slots__ = ('queries', 'candidates', 'labels', 'graph', 'diag', 'result', 'timing', 'index', 'steps', 'default')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def default_index(labels):
    """The no-selection floor: keep the running query, else the goal query."""
    if 'continue' in labels:
        return labels.index('continue')
    return labels.index('goal') if 'goal' in labels else 0


class L3Episode:
    def __init__(self, spec, cfg, sandbox, compiler=None, range_sensor='fast-driver', stream=None, horizon=HORIZON,
                 gc_freeze=True, imagination=True):
        from eventwm_graph.sandbox.episode import BranchEpisode, generator
        from .compiler import GraphCompiler
        t0 = time.perf_counter()
        self.spec, self.cfg = spec, cfg
        self.sb = sandbox
        self.compiler = compiler or GraphCompiler()
        self.horizon = int(horizon)
        self.imagination = bool(imagination)
        self.gen = generator(cfg.gen, cfg.dynamics)
        if stream is None and sandbox is not None and hasattr(sandbox, 'open_stream'):
            stream = sandbox.open_stream(spec.key)
        self.ep = BranchEpisode(spec, cfg.planner, cfg.dynamics, range_sensor=range_sensor, gc_freeze=gc_freeze,
                                stream=stream)
        self.rt = self.ep.rt
        self.start_geodesic = float(self.rt.start_geodesic)
        self.state = self.ep.state
        self.state.geodesic = self.start_geodesic
        self.decisions = 0
        self.log = []
        self.setup_s = time.perf_counter()-t0
        self.current = None

    # ------------------------------------------------------------------ observe
    @property
    def over(self):
        return self.ep.over

    def observe(self):
        from eventwm_graph.sandbox.core import Candidate
        ep, cfg = self.ep, self.cfg
        T = time.perf_counter
        tm = {}
        t = T()
        queries, diag = self.gen.propose(ep.legal, ep.execution)
        tm['generator_ms'] = 1000*(T()-t)
        labels = [q.label for q in queries]
        cands = [Candidate(None, cfg.planner, cfg.dynamics, execution=ep.execution, level=ep.level, query_id='continue')
                 if q.kind == 'continue' else Candidate(q, cfg.planner, cfg.dynamics, level=ep.level, query_id=q.label)
                 for q in queries]
        result, graph = None, None
        if self.imagination and self.sb is not None:
            t = T()
            state = ep.decision_state()
            tm['state_ms'] = 1000*(T()-t)
            t = T()
            result = self.sb.evaluate(state, ep.legal, cands)
            tm['sandbox_ms'] = 1000*(T()-t)
            t = T()
            graph = self.compiler.compile(result, ep.legal, queries, diag, ep.steps)
            tm['compile_ms'] = 1000*(T()-t)
        else:
            t = T()
            graph = self.compiler.compile_blind(ep.legal, queries, diag, ep.steps)
            tm['compile_ms'] = 1000*(T()-t)
        d = Decision(queries=queries, candidates=cands, labels=labels, graph=graph, diag=diag, result=result, timing=tm,
                     index=self.decisions, steps=ep.steps, default=default_index(labels))
        self.current = d
        return d

    # ------------------------------------------------------------------ act
    def act(self, k):
        from eventwm_graph.querygen.oracle import horizon_for
        from eventwm_graph.querygen.native import MAX_EPISODE_STEPS
        d = self.current
        q = d.queries[int(k)]
        parent = self.ep.state
        t = time.perf_counter()
        h = min(horizon_for(q, _H(self.horizon)), MAX_EPISODE_STEPS-parent.index)
        cont = q.kind == 'continue'
        self.ep.run(None if cont else q, horizon=h)
        child = self.ep.state
        if child.outcome is None:
            child.geodesic = self.rt.task.geodesic_to_goal()
        env_ms = 1000*(time.perf_counter()-t)
        g0 = float(parent.geodesic) if parent.geodesic == parent.geodesic else self.start_geodesic
        g1 = float(child.geodesic) if child.geodesic == child.geodesic else (0. if child.outcome == 'success' else g0)
        r = dict(steps=int(child.index-parent.index), path=float(child.path_length-parent.path_length),
                 social=int(child.psc_steps-parent.psc_steps), progress=g0-g1, outcome=child.outcome,
                 label=q.label, index=int(k), env_ms=env_ms)
        self.log.append(dict(decision=self.decisions, step=int(parent.index), label=q.label, index=int(k),
                             default=int(d.default), n=len(d.queries), **{x: r[x] for x in ('steps', 'path', 'social', 'progress', 'outcome')}))
        self.decisions += 1
        self.current = None
        return r

    # ------------------------------------------------------------------ results
    def metrics(self):
        st = self.ep.state
        steps = max(int(st.index), 1)
        success = float(st.outcome == 'success')
        L, Ls = float(st.path_length), self.start_geodesic
        spl = success*Ls/max(L, Ls) if L > 0 else success
        return dict(key=self.spec.key, config=self.cfg.name, official=self.cfg.official, outcome=st.outcome or 'running',
                    success=success, spl=spl, psc=1.-st.psc_steps/steps, human_collision=float(st.outcome == 'human_collision'),
                    steps=int(st.index), path=L, start_geodesic=Ls, decisions=self.decisions,
                    min_center_dist=float(st.min_center_dist) if math.isfinite(st.min_center_dist) else None)

    def decision_list(self):
        """The executed decision sequence (querygen replay format) for official replay verification."""
        return self.ep.state.decisions

    def close(self):
        self.ep.close()


class _H:
    def __init__(self, horizon):
        self.horizon = horizon
