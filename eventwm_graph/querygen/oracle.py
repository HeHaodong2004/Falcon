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
    fine_cell: float = 0.05           # de-duplication at nodes where the generator's legal stuck test fires
    fine_heading_bins: int = 75       # (multi-point turns need cm / single-turn-step resolution)
    predict_dups: bool = False        # sandbox-predict an edge's end signature (true human future) and
    dup_penalty: float = 25.          # defer predicted duplicates by this f penalty instead of simulating
    maneuver_prior: float = 10.       # extra f for exact manoeuvre edges (used when ordinary queries do not help)
    fine_by_edge: bool = True         # fine de-duplication for states reached by manoeuvre edges (else: stuck nodes)
    tie_bucket: float = 0.            # >0: f is bucketed and ties prefer the later node (cross waiting plateaus forward in time)
    block_weight: float = 0.          # privileged time-to-clear term (steps) added to f (0 = off)
    block_dist: float = 0.5           # centre distance to the geodesic path that blocks it (> half-width + body proxy)
    block_horizon_m: float = 8.0      # path length checked ahead
    block_step: int = 4               # departure-delay resolution (steps)
    block_max: int = 240              # cap (steps) when no clear departure is found


def horizon_for(query, cfg):
    if query.kind == 'stop':
        return 1
    if query.kind in ('wait', 'wait_clear', 'creep'):
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

    def _block_time(self, st):
        """Privileged: smallest departure delay (steps) after which the robot could
        follow the true geodesic path at ``speed_est`` for ``block_horizon_m``
        without passing within ``block_dist`` of any same-floor human (true tape)."""
        import habitat_sim
        cfg = self.cfg
        task = self.rt.task
        p = np.asarray(st.legal.pose[:3], dtype=np.float32)
        sp = habitat_sim.ShortestPath()
        sp.requested_start = p
        sp.requested_end = np.asarray(task.goal, dtype=np.float32)
        if not task.sim.pathfinder.find_path(sp) or len(sp.points) < 2:
            return 0.
        pts = np.asarray(sp.points, dtype=float)
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 2]))
        cum = np.concatenate([[0.], np.cumsum(seg)])
        L = min(float(cum[-1]), cfg.block_horizon_m)
        ds = np.arange(0., L+1e-9, 0.2)
        P = np.stack([np.interp(ds, cum, pts[:, 0]), np.interp(ds, cum, pts[:, 1]), np.interp(ds, cum, pts[:, 2])], 1)
        tape = self.rt.tape.positions
        if tape.shape[1] == 0:
            return 0.
        arrive = np.round(ds/cfg.speed_est).astype(int)
        for tau in range(0, cfg.block_max+1, cfg.block_step):
            idx = np.minimum(st.index+tau+arrive, len(tape)-1)
            H = tape[idx]                                   # [K, N, 3]
            d = np.hypot(H[..., 0]-P[:, None, 0], H[..., 2]-P[:, None, 2])
            same = np.abs(H[..., 1]-P[:, None, 1]) < 0.5
            if not (same & (d < cfg.block_dist)).any():
                return float(tau)
        return float(cfg.block_max)

    def _f(self, st):
        cfg = self.cfg
        h = st.geodesic/cfg.speed_est if st.geodesic == st.geodesic else 1e3
        pen = max(0., cfg.clearance_soft-st.min_rect_dist)*cfg.clearance_cost
        pen += max(0., cfg.danger_m-self._danger(st))*cfg.danger_cost
        if cfg.block_weight:
            pen += cfg.block_weight*self._block_time(st)
        return st.index+cfg.weight*h+pen

    def _sig(self, st, fine=False):
        return self._sig_pose(st.legal.pose, st.index, fine)

    def _predict_sig(self, node, q, fine):
        """Privileged, approximate: end signature of edge (node, q) from one kinematic
        sandbox rollout of the same executor on the node's legal map against the
        true human future (used only to order/defer edges, never to accept one)."""
        from .sandbox import rollout
        steps = min(horizon_for(q, self.cfg), MAX_EPISODE_STEPS-node.index)
        tape = self.rt.tape
        F = tape.positions[node.index:min(len(tape), node.index+steps+1)]
        ex = None
        if q.kind == 'continue':
            ex = node.execution.copy()
            ex.decision_boundary()
        elif q.kind == 'stop':
            return None
        traj = rollout(node.legal, None if ex is not None else q, self.rt.planner_name, self.rt.dyn, F,
                       len(F)-1, execution=ex, level=int(node.dyn_state) if not isinstance(node.dyn_state, tuple) else 0)
        return self._sig_pose(traj[-1], node.index+len(traj)-1, fine)

    def _sig_pose(self, p, index, fine=False):
        cfg = self.cfg
        cell, bins = (cfg.fine_cell, cfg.fine_heading_bins) if fine else (cfg.dedup_cell, cfg.dedup_heading_bins)
        hb = int(round(((p[3] % (2*math.pi))/(2*math.pi))*bins)) % bins if fine else \
            int(((p[3] % (2*math.pi))/(2*math.pi))*bins)
        hp = self.rt.tape.positions[min(index, len(self.rt.tape)-1)]
        near = len(hp) and float(np.min(np.hypot(hp[:, 0]-p[0], hp[:, 2]-p[2]))) < cfg.human_near_m
        tb = index//cfg.dedup_time if near else -1
        return (bool(fine), int(math.floor(p[0]/cell)), int(math.floor(p[2]/cell)), hb, tb)

    # Generic (episode-independent) edge priors in steps for lazy ordering.
    PRIOR = dict(continue_=0., goal_slow=1., goal=0., stop=0., approach=1., pass_=2., sidestep=3., yield_=3., wait_short=2.,
                 wait_long=4., wait_clear=3., retreat=5., turn=2., advance=2., alt_route=3., frontier=5., look=5., rot_=3.,
                 creep=2.)

    def _key(self, f, node):
        """Heap key: f, or (bucketed f, -index) so equal-cost nodes prefer later times."""
        b = self.cfg.tie_bucket
        return (round(f/b), -node.index) if b > 0 else (f, 0)

    def _defer(self, key, penalty):
        b = self.cfg.tie_bucket
        return (key[0]+round(penalty/b), key[1]) if b > 0 else (key[0]+penalty, key[1])

    @staticmethod
    def is_maneuver(q):
        lab = q.label
        return q.kind == 'creep' or (q.kind == 'rotate' and lab.rstrip('0123456789') in ('rot_left', 'rot_right')
                                     and lab[-1].isdigit())

    def _prior(self, q):
        lab = q.label
        if self.is_maneuver(q):
            return self.cfg.maneuver_prior
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

        def propose(node):
            t = time.perf_counter()
            queries, diag = self.gen.propose(node.legal, node.execution)
            dt = time.perf_counter()-t
            self.stats['gen_s'] += dt
            self.stats['gen_ms'].append(round(1000*dt, 2))
            return queries, diag

        def push_children(node, queries, diag):
            nonlocal counter
            self.stats['raw_candidates'] += sum(diag.get('raw', {}).values())+diag.get('fixed', 0)
            self.stats['kept_candidates'] += len(queries)
            self.stats['expansions'] += 1
            fn = self._f(node)
            self.stats.setdefault('exp_log', []).append(
                (int(node.index), round(float(node.geodesic), 3), round(float(fn), 1), bool(diag.get('stuck')),
                 round(time.perf_counter()-t0, 1)))
            fine = bool(diag.get('stuck'))
            for q in queries:
                counter += 1
                heapq.heappush(heap, (self._key(fn+self._prior(q), node), counter, node, q, False, fine))
        push_children(root, *propose(root))
        while heap and self.stats['children'] < cfg.max_edges and time.perf_counter()-t0 < cfg.max_seconds:
            prio, _, node, q, deferred, node_fine = heapq.heappop(heap)
            if MAX_EPISODE_STEPS-node.index <= 0:
                continue
            if cfg.predict_dups and not deferred:
                t = time.perf_counter()
                psig = self._predict_sig(node, q, self.is_maneuver(q) if cfg.fine_by_edge else node_fine)
                self.stats['predict_s'] = self.stats.get('predict_s', 0.)+time.perf_counter()-t
                if psig is not None and psig in seen:
                    counter += 1
                    heapq.heappush(heap, (self._defer(prio, cfg.dup_penalty), counter, node, q, True, node_fine))
                    self.stats['deferred'] = self.stats.get('deferred', 0)+1
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
            queries, diag = propose(child)
            # cm/turn-step resolution only for states reached by an exact manoeuvre
            # (multi-point turns); ordinary exploration keeps the coarse key
            sig = self._sig(child, fine=self.is_maneuver(q) if cfg.fine_by_edge else bool(diag.get('stuck')))
            if sig in seen:
                self.stats['duplicates'] = self.stats.get('duplicates', 0)+1
                continue
            seen.add(sig)
            push_children(child, queries, diag)
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
