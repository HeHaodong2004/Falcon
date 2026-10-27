"""Privileged offline Oracle over generator-provided queries.

Query *generation* is legal (``QueryGenerator`` sees only the branch's own
legal state); *selection* is privileged: the Oracle inspects true simulator
outcomes and the true geodesic distance; *execution* is real (each candidate
runs through the fixed planner, dynamics and the native-equivalent branch
simulator).  The search is a weighted best-first search over decision nodes
for a complete successful query sequence: STOP within the success radius
with no human collision, body-proxy overlap or static contact, inside the
episode limit.  Locally worse decisions (waiting, retreating, detours) are
kept in the frontier, so the first success need not be greedy.  Every success
must then pass ``episode.replay`` (unchanged Env.step from the reset).
"""
from dataclasses import dataclass, asdict, field
import heapq
import math
import time
import numpy as np

from .native import MAX_EPISODE_STEPS


@dataclass
class OracleConfig:
    horizon: int = 16                 # max control steps per movement decision
    max_expansions: int = 500
    max_seconds: float = 1800.
    weight: float = 2.0               # weighted A*: f = g + w * h
    speed_est: float = 0.17           # m per step for the remaining-time estimate
    clearance_soft: float = 0.6       # m rectangle-to-human clearance below which f is penalised
    clearance_cost: float = 30.       # steps per metre below clearance_soft
    max_open: int = 3000
    dedup_cell: float = 0.3
    dedup_heading_bins: int = 8
    dedup_time: int = 10
    human_near_m: float = 4.0         # time matters for de-duplication only near humans
    danger_horizon: int = 60          # steps of true human future checked against a held pose
    danger_m: float = 0.8             # rectangle clearance below which a held pose is 'doomed'
    danger_cost: float = 60.          # f penalty per metre below danger_m
    lazy: bool = True                 # deferred edge evaluation (simulate an edge when popped)
    max_edges: int = 4000             # simulated edges budget in lazy mode


def horizon_for(query, cfg):
    if query.kind == 'stop':
        return 1
    if query.kind == 'wait':
        return max(1, int(query.duration))
    return cfg.horizon          # movement and rotation: one decision interval, continuable


class Oracle:
    def __init__(self, runtime, generator, config=None):
        self.rt = runtime
        self.gen = generator
        self.cfg = config or OracleConfig()
        self.stats = dict(expansions=0, children=0, outcomes={}, by_label={}, gen_s=0., run_s=0.,
                          raw_candidates=0, kept_candidates=0, gen_ms=[], edge_ms=[], edge_steps=[])

    def _danger(self, st):
        """Privileged: closest approach of any human's true future to the held pose."""
        from .native import same_floor_rect_distance
        tape = self.rt.tape
        hp = tape.positions[st.index:min(len(tape), st.index+self.cfg.danger_horizon+1)]
        if hp.size == 0 or hp.shape[1] == 0:
            return math.inf
        p = st.legal.pose
        return float(same_floor_rect_distance(p, hp.reshape(-1, 3)).min())

    def _f(self, st):
        cfg = self.cfg
        h = st.geodesic/cfg.speed_est if st.geodesic == st.geodesic else 1e3
        pen = max(0., cfg.clearance_soft-st.min_rect_dist)*cfg.clearance_cost
        pen += max(0., cfg.danger_m-self._danger(st))*cfg.danger_cost
        return st.index+cfg.weight*h+pen

    def _sig(self, st):
        cfg = self.cfg
        p = st.legal.pose
        hb = int(((p[3] % (2*math.pi))/(2*math.pi))*cfg.dedup_heading_bins)
        hp = self.rt.tape.positions[st.index]
        near = len(hp) and float(np.min(np.hypot(hp[:, 0]-p[0], hp[:, 2]-p[2]))) < cfg.human_near_m
        tb = st.index//cfg.dedup_time if near else -1
        return (int(math.floor(p[0]/cfg.dedup_cell)), int(math.floor(p[2]/cfg.dedup_cell)), hb, tb)

    # Generic (episode-independent) edge priors in steps for lazy ordering.
    PRIOR = dict(continue_=0., goal_slow=1., goal=0., stop=0., approach=1., pass_=2., sidestep=3., yield_=3., wait_short=2.,
                 wait_long=4., retreat=5., turn=2., advance=2., alt_route=3., frontier=5., look=5., rot_=3.)

    def _prior(self, q):
        lab = q.label
        for k, v in self.PRIOR.items():
            if lab.startswith(k.rstrip('_')):
                return v
        return 4.

    def search(self):
        if self.cfg.lazy:
            return self._search_lazy()
        return self._search_eager()

    def _search_lazy(self):
        """Best-first over *edges*: an edge (node, query) is simulated when popped."""
        cfg = self.cfg
        t0 = time.perf_counter()
        root = self.rt.root
        counter = 0
        heap = []
        best = root
        seen = set()

        def push_children(node):
            nonlocal counter
            t = time.perf_counter()
            queries, diag = self.gen.propose(node.legal, node.execution)
            dt = time.perf_counter()-t
            self.stats['gen_s'] += dt
            self.stats['gen_ms'].append(round(1000*dt, 2))
            self.stats['raw_candidates'] += sum(diag.get('raw', {}).values())+diag.get('fixed', 0)
            self.stats['kept_candidates'] += len(queries)
            self.stats['expansions'] += 1
            fn = self._f(node)
            for q in queries:
                counter += 1
                heapq.heappush(heap, (fn+self._prior(q), counter, node, q))
        push_children(root)
        while heap and self.stats['children'] < cfg.max_edges and time.perf_counter()-t0 < cfg.max_seconds:
            _, _, node, q = heapq.heappop(heap)
            if MAX_EPISODE_STEPS-node.index <= 0:
                continue
            t = time.perf_counter()
            child = self.rt.run(node, q, horizon=min(horizon_for(q, cfg), MAX_EPISODE_STEPS-node.index))
            dt = time.perf_counter()-t
            self.stats['run_s'] += dt
            self.stats['edge_ms'].append(round(1000*dt, 1))
            self.stats['edge_steps'].append(child.index-node.index)
            self.stats['children'] += 1
            lab = self.stats['by_label'].setdefault(q.label.rstrip('0123456789'), dict(n=0, fail=0))
            lab['n'] += 1
            if child.outcome is not None:
                self.stats['outcomes'][child.outcome] = self.stats['outcomes'].get(child.outcome, 0)+1
                if child.outcome == 'success':
                    return self._result(child, t0, True)
                lab['fail'] += 1
                continue
            if child.geodesic < best.geodesic:
                best = child
            sig = self._sig(child)
            if sig in seen:
                continue
            seen.add(sig)
            push_children(child)
            if len(heap) > cfg.max_open*4:
                heap = heapq.nsmallest(cfg.max_open*2, heap)
                heapq.heapify(heap)
        return self._result(best, t0, False)

    def _search_eager(self):
        cfg = self.cfg
        t0 = time.perf_counter()
        root = self.rt.root
        counter = 0
        heap = [(self._f(root), counter, root)]
        seen = set()
        best = root
        while heap and self.stats['expansions'] < cfg.max_expansions and time.perf_counter()-t0 < cfg.max_seconds:
            _, _, node = heapq.heappop(heap)
            sig = self._sig(node)
            if sig in seen:
                continue
            seen.add(sig)
            self.stats['expansions'] += 1
            t = time.perf_counter()
            queries, diag = self.gen.propose(node.legal, node.execution)
            self.stats['gen_s'] += time.perf_counter()-t
            self.stats['raw_candidates'] += sum(diag.get('raw', {}).values())+diag.get('fixed', 0)
            self.stats['kept_candidates'] += len(queries)
            for q in queries:
                if MAX_EPISODE_STEPS-node.index <= 0:
                    break
                t = time.perf_counter()
                child = self.rt.run(node, q, horizon=min(horizon_for(q, cfg), MAX_EPISODE_STEPS-node.index))
                self.stats['run_s'] += time.perf_counter()-t
                self.stats['children'] += 1
                lab = self.stats['by_label'].setdefault(q.label.rstrip('0123456789'), dict(n=0, fail=0))
                lab['n'] += 1
                if child.outcome is not None:
                    self.stats['outcomes'][child.outcome] = self.stats['outcomes'].get(child.outcome, 0)+1
                    if child.outcome == 'success':
                        return self._result(child, t0, True)
                    lab['fail'] += 1
                    continue
                if child.geodesic < best.geodesic:
                    best = child
                counter += 1
                heapq.heappush(heap, (self._f(child), counter, child))
            if len(heap) > cfg.max_open:
                heap = heapq.nsmallest(cfg.max_open//2, heap)
                heapq.heapify(heap)
        return self._result(best, t0, False)

    def _result(self, st, t0, success):
        return dict(success=success, state=st, steps=st.index, outcome=st.outcome,
                    final_geodesic=st.geodesic, decisions=st.decisions,
                    seconds=time.perf_counter()-t0, stats=self.stats)
