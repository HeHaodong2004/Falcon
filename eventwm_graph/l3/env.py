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
ORACLE_BANK = __import__('os').environ.get('L3_ORACLE_BANK')    # analysis only: 'tracked' | 'all' (oraclebank.py)
ORACLE_BANK_STEPS = int(__import__('os').environ.get('L3_ORACLE_BANK_STEPS', '0'))     # analysis only: true horizon
ORACLE_NOISE = float(__import__('os').environ.get('L3_ORACLE_NOISE', '0'))      # analysis only: velocity error (m/s)
ORACLE_NOISE_WORLDS = int(__import__('os').environ.get('L3_ORACLE_NOISE_WORLDS', '6'))
LEGAL_BANK = __import__('os').environ.get('L3_MW_BANK')      # Layer 3 option: 'cv' = constant-velocity tracks, 'mean' = Stage 1 mean of worlds, 'fc1' / 'fck' = map-aware forecaster (peoplemap.py; legal)
SCORE_TRUTH = __import__('os').environ.get('L3_MW_SCORE_TRUTH') == '1'   # analysis only (privileged scoring)
# speed option (default off): with a people bank that does not come from the released Sandbox pass (L3_MW_BANK=cv |
# mean, L3_ORACLE_BANK) the decision skips that pass (imagination + Stage 2 of every candidate) and the compiled graph
# (the blind layout is recorded instead); for selectors that read only ``result.humans`` (mwroll)
LIGHT = __import__('os').environ.get('L3_MW_LIGHT') == '1'


@dataclass(frozen=True)
class BankOnlyResult:
    """The part of a ``SandboxResult`` a bank-only selector reads (``L3_MW_LIGHT``)."""
    humans: object


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
                 gc_freeze=True, imagination=True, near_interval=0, near_m=3.0, scan=False):
        from eventwm_graph.sandbox.episode import BranchEpisode, generator
        from .compiler import GraphCompiler
        t0 = time.perf_counter()
        self.spec, self.cfg = spec, cfg
        self.sb = sandbox
        self.compiler = compiler or GraphCompiler()
        self.horizon = int(horizon)
        self.base_horizon = int(horizon)
        # adaptive decision frequency: a shorter interval while a tracked person is within near_m (legal tracks only)
        self.near_interval, self.near_m = int(near_interval), float(near_m)
        self.scan = bool(scan)
        self.imagination = bool(imagination)
        self.gen = generator(cfg.gen, cfg.dynamics)
        if not self.imagination:
            # no Layer 2 input is needed: skip the range frames and the legal Layer 2 stream (cheaper environment)
            range_sensor, stream = _NullRange(), _NullStream()
        elif stream is None and sandbox is not None and hasattr(sandbox, 'open_stream'):
            stream = sandbox.open_stream(spec.key)
        self.ep = BranchEpisode(spec, cfg.planner, cfg.dynamics, range_sensor=range_sensor, gc_freeze=gc_freeze,
                                stream=stream)
        self.rt = self.ep.rt
        from . import sensorcache
        sensorcache.attach(self.rt)          # exact cross-episode support-certificate cache (sensorcache.py)
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

    def _interval_now(self):
        if not self.near_interval:
            return self.base_horizon
        from eventwm_graph.querygen.control import human_estimates
        legal = self.ep.legal
        p = np.asarray(legal.pose, np.float64)
        for h in human_estimates(legal.tracks, legal.time):
            # planar estimate (x, -z) of control.human_estimates
            if math.hypot(float(h.position[0])-p[0], float(-h.position[1])-p[2]) <= self.near_m:
                return self.near_interval
        return self.base_horizon

    def observe(self):
        from eventwm_graph.sandbox.core import Candidate
        ep, cfg = self.ep, self.cfg
        T = time.perf_counter
        tm = {}
        self.horizon = self._interval_now()
        if self.sb is not None and hasattr(self.sb, 'config'):
            self.sb.config.interval = self.horizon        # imagine the interval that will be executed
        t = T()
        queries, diag = self.gen.propose(ep.legal, ep.execution)
        if self.scan:
            queries = list(queries)+scan_queries(ep.legal, self.gen.dyn, [q.label for q in queries])
        tm['generator_ms'] = 1000*(T()-t)
        labels = [q.label for q in queries]
        running = ep.execution            # any status: a resumed goal query inherits what the last one suspended
        cands = [Candidate(None, cfg.planner, cfg.dynamics, execution=ep.execution, level=ep.level, query_id='continue')
                 if q.kind == 'continue' else Candidate(q, cfg.planner, cfg.dynamics, level=ep.level, query_id=q.label,
                                                        execution=running)
                 for q in queries]
        result, graph = None, None
        running = ep.execution.query if ep.execution is not None else None
        if self.imagination and self.sb is not None:
            t = T()
            state = ep.decision_state()
            self.last_state = state          # (multi-world selector's Stage 2 discovery calls; same decision only)
            tm['state_ms'] = 1000*(T()-t)
            t = T()
            light = LIGHT and bool(ORACLE_BANK or LEGAL_BANK in ('cv', 'mean', 'fc1', 'fck'))
            if light:
                # (L3_MW_LIGHT) no released pass; Stage 1 only where the bank needs it (MeanBank)
                self.sb.check_context(state, ep.legal)
                result = BankOnlyResult(self.sb.l2.predict_humans(state, worlds=self.sb.config.worlds,
                                                                  seed=self.sb.config.seed)
                                        if LEGAL_BANK == 'mean' and not ORACLE_BANK else None)
            else:
                result = self.sb.evaluate(state, ep.legal, cands)
            tm['sandbox_ms'] = 1000*(T()-t)
            if ORACLE_BANK:
                # ANALYSIS ONLY (privileged): true human futures in place of Stage 1's bank (oraclebank.py)
                import dataclasses
                from .oraclebank import OracleBank
                ob = OracleBank(self.rt.tape.positions, ep.state.index, ep.legal, ORACLE_BANK, long_steps=ORACLE_BANK_STEPS)
                if ORACLE_NOISE > 0:
                    # prediction-error study: noisy copies of the truth (oraclebank.OracleBank.add_noise)
                    ob.add_noise(ORACLE_NOISE, ORACLE_NOISE_WORLDS, seed=ep.state.index+1000*self.decisions)
                result = dataclasses.replace(result, humans=ob)
            elif LEGAL_BANK == 'mean':
                # LEGAL: Stage 1's tracked people at the weighted mean of its worlds (oraclebank.MeanBank)
                import dataclasses
                from .oraclebank import MeanBank
                result = dataclasses.replace(result, humans=MeanBank(result.humans, ep.legal))
            elif LEGAL_BANK in ('fc1', 'fck'):
                # LEGAL: the map-aware multimodal forecaster of the tracked people (peoplemap.FCBank; model
                # L3_FC_MODEL): 'fc1' one world (weighted mean or top mode), 'fck' the L3_MW_WORLDS most probable
                # joint mode combinations, weighted; constant velocity for tracks without a recent detection
                import dataclasses
                from .peoplemap import FCBank
                result = dataclasses.replace(result, humans=FCBank(
                    ep.legal, mode=LEGAL_BANK, worlds=int(__import__('os').environ.get('L3_MW_WORLDS', '6'))))
            elif LEGAL_BANK == 'cv':
                # LEGAL baseline predictor: the tracked people at the tracker's constant velocity (oraclebank.CVBank)
                import dataclasses
                from .oraclebank import CVBank
                cvb = CVBank(ep.legal)
                if SCORE_TRUTH:
                    # ANALYSIS ONLY (privileged, L3_MW_SCORE_TRUTH=1): the TRUE futures of the people NOT tracked at t,
                    # for the selector's contact scoring only (the imagined planner never perceives them) -- the value
                    # of a perfect hidden-people model on top of the legal loop
                    from .oraclebank import OracleBank
                    cvb.score_extra = OracleBank(self.rt.tape.positions, ep.state.index, ep.legal, 'all')
                result = dataclasses.replace(result, humans=cvb)
            t = T()
            graph = (self.compiler.compile_blind(ep.legal, queries, diag, ep.steps, running=running) if light else
                     self.compiler.compile(result, ep.legal, queries, diag, ep.steps, running=running))
            tm['compile_ms'] = 1000*(T()-t)
        else:
            t = T()
            graph = self.compiler.compile_blind(ep.legal, queries, diag, ep.steps, running=running)
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
        # official definitions (falcon benchmark config): SPL over the 3D path of the robot base, PSC with the navmesh
        # geodesic robot-human distance (querygen BranchState.psc_official_steps); the planar / Euclidean variants are
        # kept as psc_euclid / spl_planar
        L, Ls = float(getattr(st, 'path_length_3d', 0.) or st.path_length), self.start_geodesic
        if not math.isfinite(Ls):
            spl = 0.                         # no navmesh path from the start: SPL undefined, counted as 0
        else:
            spl = success*Ls/max(L, Ls) if L > 0 else success
        Lp = float(st.path_length)
        spl_planar = (success*Ls/max(Lp, Ls) if Lp > 0 else success) if math.isfinite(Ls) else 0.
        return dict(key=self.spec.key, config=self.cfg.name, official=self.cfg.official, outcome=st.outcome or 'running',
                    success=success, spl=spl, psc=1.-getattr(st, 'psc_official_steps', st.psc_steps)/steps,
                    psc_euclid=1.-st.psc_steps/steps, spl_planar=spl_planar,
                    human_collision=float(st.outcome == 'human_collision'),
                    steps=int(st.index), path=L, start_geodesic=Ls, decisions=self.decisions,
                    min_center_dist=float(st.min_center_dist) if math.isfinite(st.min_center_dist) else None)

    def decision_list(self):
        """The executed decision sequence (querygen replay format) for official replay verification."""
        return self.ep.state.decisions

    def close(self):
        from . import sensorcache
        try:
            sensorcache.save(self.rt)
        except Exception:
            pass
        self.ep.close()


SCAN_DEG = 80.


def scan_queries(legal, dyn, labels):
    """Layer 3 information-gathering queries (``scan``): in-place rotations by SCAN_DEG to either side (one decision
    interval reaches ~76 deg for the native turn rate; slower robots continue them), and a rotation toward the most
    recently lost tracked person when the generator did not offer one.  The Oracle never needs them (it knows every
    person); under the 90 deg camera they are how the robot looks around."""
    from eventwm_graph.querygen.executor import Query
    from eventwm_graph.querygen.control import wrap
    pose = legal.pose
    yaw = float(pose[3])
    out = []
    for side, sgn in (('left', 1.), ('right', -1.)):
        n = int(math.ceil(math.radians(SCAN_DEG)/dyn.turn_rad))+1
        out.append(Query('rotate', heading=wrap(yaw+sgn*math.radians(SCAN_DEG)), duration=n, label=f'scan_{side}',
                         meta=dict(direction=side)))
    if 'look_lost_human' not in labels:
        lost = [t for t in legal.tracks if not t.get('visible_now') and legal.time-float(t['last_seen_time']) < 5.0]
        if lost:
            t = min(lost, key=lambda t: math.hypot(t['position'][0]-pose[0], t['position'][2]-pose[2]))
            ldir = math.atan2(-(t['position'][2]-pose[2]), t['position'][0]-pose[0])
            if abs(wrap(ldir-yaw)) > math.radians(35):
                out.append(Query('rotate', heading=ldir, duration=40, label='look_lost_human'))
    return out


class _H:
    def __init__(self, horizon):
        self.horizon = horizon


class _NullRange:
    """No range frames (selectors that do not use Layer 2)."""
    def measure(self, timestamp=None):
        return None

    def close(self):
        pass


class _NullStream:
    def observe(self, row, range_frame):
        pass

    def close(self):
        pass
