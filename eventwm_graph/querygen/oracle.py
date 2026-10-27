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

from .native import MAX_EPISODE_STEPS, SUCCESS_DISTANCE


@dataclass
class OracleConfig:
    horizon: int = 16                 # max control steps per movement decision
    max_expansions: int = 500
    max_seconds: float = 9000.      # safety cap only; the binding budget is max_edges (deterministic)
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
    max_edges: int = 6000             # simulated edges budget in lazy mode (deterministic, load-independent)
    fine_cell: float = 0.05           # de-duplication at nodes where the generator's legal stuck test fires
    fine_heading_bins: int = 75       # (multi-point turns need cm / single-turn-step resolution)
    predict_dups: bool = False        # sandbox-predict an edge's end signature (true human future) and
    dup_penalty: float = 25.          # defer predicted duplicates by this f penalty instead of simulating
    maneuver_prior: float = 10.       # extra f for exact manoeuvre edges (used when ordinary queries do not help)
    maneuver_pad: bool = True         # also charge short manoeuvres up to a full decision interval
    maneuver_gate: float = 120.       # >0: entering manoeuvre mode costs this much extra unless every ordinary
                                      # movement query tried from the parent failed natively (privileged gate)
    fine_by_edge: bool = True         # fine de-duplication for states reached by manoeuvre edges (else: stuck nodes)
    tie_bucket: float = 0.            # >0: f is bucketed and ties prefer the later node (cross waiting plateaus forward in time)
    witness_dir: str = ''             # privileged witness guidance: directory with witness.py results (by episode)
    witness_late_cost: float = 0.5    # steps of penalty per step of arriving later than the witness at a join point
    witness_heading: bool = True      # near a witness pose also charge the heading mismatch (turn steps)
    goal_test: bool = True            # v7b: try the generator's STOP at once on any new state inside the success radius
    goal_fine_m: float = 1.0          # v7b: fine de-duplication for states this close to the goal (geodesic)
    escape_penalty: float = 0.        # v9: once any branch has left the spawn region (escape_m from the start), manoeuvre
                                      # edges from nodes still inside it cost this much extra (the spawn manoeuvre space
                                      # otherwise absorbed 72-79% of the edge budget in YRUkbU5xsYj / 3t8DB4Uzvkt)
    escape_m: float = 1.5
    escape_h: bool = False            # v9: privileged spawn-escape heuristic (exact primitive BFS around the spawn;
                                      # steps-to-escape per 2 cm cell x turn index); only for tight spawns
    escape_h_sims: int = 20000        # simulated primitive steps for the escape A* (one-off, ~6 ms each)
    prefix_cache: int = 0             # v9: exact prefix sharing among sibling edges (runtime keeps caches of this many
                                      # most recent parents; 0 = off).  Search results are identical with or without.
    heading_h: bool = False           # v9: add the in-place turn steps needed to face the true geodesic path
                                      # (0.6 m look-ahead) to h -- a turn-in-place robot must turn before it moves
    block_weight: float = 0.          # privileged time-to-clear term (steps) added to f (0 = off)
    st_heuristic: bool = True         # privileged space-time cost-to-go (SpaceTimeHeuristic) replaces h + block term
    st_fallback: float = 300.         # extra steps when the space-time model finds no arrival
    block_dist: float = 0.5           # centre distance to the geodesic path that blocks it (> half-width + body proxy)
    block_horizon_m: float = 8.0      # path length checked ahead
    block_step: int = 4               # departure-delay resolution (steps)
    block_max: int = 240              # cap (steps) when no clear departure is found


class SpaceTimeHeuristic:
    """Privileged time-dependent cost-to-go for the Oracle (never used online).

    Grid over the navmesh top-down view (``cell`` m); a cell is unsafe at step t
    if a same-floor human centre (true tape) is within ``radius``.  Backward
    dynamic programming over (cell, step) with unit-step moves to the 8
    neighbours or waiting gives T[t, cell] = earliest arrival (steps from t) at
    a goal cell.  The model ignores turning and Spot's body (optimistic, like
    the geodesic), but it knows *when* passages are blocked, so waiting on the
    optimal schedule lowers T one-for-one instead of looking like lost time.
    """

    def __init__(self, rt, cell=0.2, radius=0.6, horizon=None, goal_m=0.3):
        from .native import MAX_EPISODE_STEPS
        task = rt.task
        pf = task.sim.pathfinder
        y = float(task.robot.base_pos[1])
        nav = np.asarray(pf.get_topdown_view(cell, y), dtype=bool)          # [z, x]
        lb = pf.get_bounds()[0]
        self.x0, self.z0, self.cell = float(lb[0]), float(lb[2]), cell
        self.nav = nav
        T_steps = min(len(rt.tape), (horizon or MAX_EPISODE_STEPS)+1)
        nz, nx = nav.shape
        zc = self.z0+(np.arange(nz)+.5)*cell
        xc = self.x0+(np.arange(nx)+.5)*cell
        g = task.goal
        goal = nav & (np.hypot(xc[None, :]-g[0], zc[:, None]-g[2]) <= goal_m)
        if not goal.any():
            iz, ix = self._nearest_nav(int((g[2]-self.z0)/cell), int((g[0]-self.x0)/cell))
            goal = np.zeros_like(nav); goal[iz, ix] = True
        INF = np.float32(1e6)
        H = rt.tape.positions[:T_steps]                                       # [T, N, 3]
        T = np.full(nav.shape, INF, dtype=np.float32)
        T[goal] = 0.
        self.table = np.empty((T_steps,)+nav.shape, dtype=np.float32)
        for t in range(T_steps-1, -1, -1):
            if t < T_steps-1:
                nxt = self.table[t+1]
                best = nxt+1.                                                 # wait
                for dz in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dz == 0 and dx == 0:
                            continue
                        sh = np.full(nav.shape, INF, dtype=np.float32)
                        zs = slice(max(0, -dz), nz-max(0, dz)); zd = slice(max(0, dz), nz-max(0, -dz))
                        xs = slice(max(0, -dx), nx-max(0, dx)); xd = slice(max(0, dx), nx-max(0, -dx))
                        sh[zs, xs] = nxt[zd, xd]+1.
                        best = np.minimum(best, sh)
                T = best
            else:
                T = np.full(nav.shape, INF, dtype=np.float32)
            T = np.where(nav, T, INF)
            unsafe = np.zeros(nav.shape, dtype=bool)
            for h in H[t]:
                if abs(float(h[1])-y) > 0.5:
                    continue
                unsafe |= (np.hypot(xc[None, :]-h[0], zc[:, None]-h[2]) < radius)
            T = np.where(unsafe, INF, T)
            T = np.where(goal & ~unsafe, 0., T)
            self.table[t] = T
        self.INF = float(INF)

    def _nearest_nav(self, iz, ix, r=3):
        nz, nx = self.nav.shape
        best = None
        for a in range(iz-r, iz+r+1):
            for b in range(ix-r, ix+r+1):
                if 0 <= a < nz and 0 <= b < nx and self.nav[a, b]:
                    d = (a-iz)**2+(b-ix)**2
                    if best is None or d < best[0]:
                        best = (d, a, b)
        return (iz, ix) if best is None else (best[1], best[2])

    def value(self, x, z, t):
        """Steps to the goal from world (x, z) at step t (None if unreachable in the model)."""
        iz, ix = int((z-self.z0)/self.cell), int((x-self.x0)/self.cell)
        nz, nx = self.nav.shape
        if not (0 <= iz < nz and 0 <= ix < nx) or not self.nav[iz, ix]:
            iz, ix = self._nearest_nav(min(max(iz, 0), nz-1), min(max(ix, 0), nx-1))
        v = float(self.table[min(t, len(self.table)-1), iz, ix])
        return None if v >= self.INF/2 else v


def horizon_for(query, cfg):
    if query.kind == 'stop':
        return 1
    if query.kind in ('wait', 'wait_clear', 'creep', 'kturn'):
        return max(1, int(query.duration))
    return cfg.horizon          # movement and rotation: one decision interval, continuable


class EscapeMap:
    """Privileged (Oracle-only) exact escape path out of a confined spawn.

    1. Confinement test: breadth-first over the native primitives to depth 3 (true static contacts,
       declared spawn-escape rule); the spawn counts as confined if any primitive within that depth
       touches static geometry.
    2. Weighted A* over the primitives (2 cm cells x exact turn index x escape-window flag; humans
       ignored -- a static ranking aid) toward the first state ``escape_m`` from the start.
    3. Every state on the found path gets its remaining primitive steps to the escape; explored spawn
       states off the path get ``OFF_PATH`` extra steps (they need a detour back to the path).
    """

    OFF_PATH = 25

    def __init__(self, rt, escape_m=1.0, max_sims=20000, weight=2.0):
        import heapq as _hq
        from .native import escape_exempt
        t0 = time.perf_counter()
        task, sim = rt.task, rt.sim
        dyn = rt.dyn
        self.n_turn = int(round(2*math.pi/dyn.turn_rad))
        self.turn = dyn.turn_rad
        root = rt.root
        p0 = root.legal.pose
        self.p0 = (float(p0[0]), float(p0[2]), float(p0[3]))
        C0 = rt.spawn_links
        acts = ('forward', 'left', 'right')
        k0 = self._key(p0[0], p0[2], 0, bool(C0))
        sims = 0
        blocked_early = 0
        layer = [(root.robot, 0, (float(p0[0]), float(p0[2])), bool(C0), 0.)]
        for d in range(3):                         # 1. confinement test
            nxt = []
            for st, ti, xz, esc, trav in layer:
                for a in acts:
                    sim.load(st, d)
                    r = sim.step(a, compute_geodesic=False)
                    sims += 1
                    trav2 = trav+math.hypot(r.pose[0]-xz[0], r.pose[2]-xz[1])
                    if r.static_contact and not escape_exempt(r.static_links, C0, d+1, trav2):
                        blocked_early += 1
                        continue
                    ti2 = ti+(1 if a == 'left' else -1 if a == 'right' else 0)
                    nxt.append((sim.state, ti2, (float(r.pose[0]), float(r.pose[2])), esc and bool(r.static_contact), trav2))
            layer = nxt[:60]
        self.blocked_early = blocked_early
        self.path = {}
        self.known = set()
        self.first = None
        if blocked_early:                          # 2. weighted A* toward displacement
            step = dyn.step_m
            heap = [(weight*escape_m/step, 0, 0, root.robot, 0, (float(p0[0]), float(p0[2])), bool(C0), 0., k0)]
            parent = {k0: None}
            g_of = {k0: 0}
            counter = 0
            goal = None
            while heap and sims < max_sims:
                _, _, g, st, ti, xz, esc, trav, k = _hq.heappop(heap)
                if g_of.get(k, 1e9) < g:
                    continue
                for a in acts:
                    sim.load(st, g)
                    r = sim.step(a, compute_geodesic=False)
                    sims += 1
                    trav2 = trav+math.hypot(r.pose[0]-xz[0], r.pose[2]-xz[1])
                    if r.static_contact and not escape_exempt(r.static_links, C0, g+1, trav2):
                        continue
                    esc2 = esc and bool(r.static_contact)
                    ti2 = ti+(1 if a == 'left' else -1 if a == 'right' else 0)
                    k2 = self._key(r.pose[0], r.pose[2], ti2, esc2)
                    if g_of.get(k2, 1e9) <= g+1:
                        continue
                    g_of[k2] = g+1
                    parent[k2] = k
                    disp = math.hypot(r.pose[0]-p0[0], r.pose[2]-p0[2])
                    if not esc2 and disp >= escape_m:
                        goal = k2
                        break
                    counter += 1
                    _hq.heappush(heap, ((g+1)+weight*max(0., escape_m-disp)/step, counter, g+1, sim.state, ti2,
                                        (float(r.pose[0]), float(r.pose[2])), esc2, trav2, k2))
                if goal is not None:
                    break
            self.known = set(g_of)
            if goal is not None:                   # 3. label the path
                chain = []
                k = goal
                while k is not None:
                    chain.append(k)
                    k = parent[k]
                chain.reverse()
                self.first = len(chain)-1
                for i, k in enumerate(chain):
                    self.path[k] = len(chain)-1-i
        rt._loaded = None
        self.sims = sims
        self.states = len(self.known)
        self.active = bool(self.path)
        self.seconds = time.perf_counter()-t0

    def _key(self, x, z, ti, esc):
        return (round(float(x)/0.02), round(float(z)/0.02), int(ti) % self.n_turn, bool(esc))

    def lookup(self, pose, escape_open):
        """Remaining steps along the escape path; OFF_PATH extra for other explored spawn states."""
        dyaw = math.atan2(math.sin(float(pose[3])-self.p0[2]), math.cos(float(pose[3])-self.p0[2]))
        ti = int(round(dyaw/self.turn))
        known = False
        for esc in (bool(escape_open), not escape_open):
            k = self._key(pose[0], pose[2], ti, esc)
            if k in self.path:
                return self.path[k]
            known = known or k in self.known
        return (self.first or 0)+self.OFF_PATH if known else None


class Oracle:
    def __init__(self, runtime, generator, config=None):
        self.rt = runtime
        self.gen = generator
        self.cfg = config or OracleConfig()
        self._st = None
        self._witness = None
        self._escaped = False
        self._escmap = None
        runtime.prefix_cache = int(self.cfg.prefix_cache)
        p0 = runtime.root.legal.pose
        self._start_xz = (float(p0[0]), float(p0[2]))
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
        hw = self._witness_h(st)
        if hw is not None:
            h = max(h, hw)
        if cfg.escape_h:
            if self._escmap is None:
                self._escmap = EscapeMap(self.rt, max_sims=cfg.escape_h_sims)
                em = self._escmap
                self.stats['escape_map'] = dict(active=em.active, first_escape=em.first, states=em.states,
                                                blocked_early=em.blocked_early,
                                                sims=em.sims, seconds=round(em.seconds, 1))
            if self._escmap.active and self._in_spawn(st, 1.2):
                k = self._escmap.lookup(st.legal.pose, st.escape_open)
                if k is not None:
                    # remaining exact primitive steps to leave the spawn, then the usual estimate
                    h = h+k
        hs = 0.
        if cfg.heading_h and st.geodesic == st.geodesic and st.geodesic > 0.3:
            hs = self._heading_steps(st.legal.pose)
            h += hs
        if cfg.st_heuristic:
            if self._st is None:
                t = time.perf_counter()
                self._st = SpaceTimeHeuristic(self.rt)
                self.stats['st_build_s'] = time.perf_counter()-t
            p = st.legal.pose
            v = self._st.value(float(p[0]), float(p[2]), st.index)
            # time-to-go in steps: the space-time arrival (it already includes the
            # forced waits); never below the geodesic estimate
            h = max(h, v+hs) if v is not None else h+cfg.st_fallback
        elif cfg.block_weight:
            pen += cfg.block_weight*self._block_time(st)
        return st.index+cfg.weight*h+pen

    def _in_spawn(self, node, radius=1.0):
        p = node.legal.pose
        return math.hypot(float(p[0])-self._start_xz[0], float(p[2])-self._start_xz[1]) < radius

    def _heading_steps(self, pose):
        """Privileged: in-place turn steps to face the true geodesic path (0.6 m look-ahead)."""
        import habitat_sim
        task = self.rt.task
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(pose[:3], dtype=np.float32)
        sp.requested_end = np.asarray(task.goal, dtype=np.float32)
        if not task.sim.pathfinder.find_path(sp) or len(sp.points) < 2:
            return 0.
        P = np.asarray(sp.points, dtype=float)
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 2]))
        cum = np.concatenate([[0.], np.cumsum(seg)])
        L = min(0.6, float(cum[-1]))
        x = np.interp(L, cum, P[:, 0]); z = np.interp(L, cum, P[:, 2])
        if math.hypot(x-pose[0], z-pose[2]) < 1e-6:
            return 0.
        want = math.atan2(-(z-pose[2]), x-pose[0])
        return abs(math.atan2(math.sin(want-pose[3]), math.cos(want-pose[3])))/self.rt.dyn.turn_rad

    def _witness_h(self, st):
        """Privileged: remaining steps if the node joins a known feasible witness trajectory.

        Joining at index k needs d/v steps (v = native step 0.208 m); early arrivals wait, late
        arrivals pay ``witness_late_cost`` per step (the witness's human timing no longer holds).
        """
        cfg = self.cfg
        if not cfg.witness_dir:
            return None
        if self._witness is None:
            self._witness = False
            import json as _json
            from pathlib import Path as _Path
            f = _Path(cfg.witness_dir)/f"{self.rt.spec.key.replace('/', '_')}.json"
            if f.exists():
                w = _json.loads(f.read_text())
                if w.get('found') and w.get('verified', {}).get('ok') and \
                        w.get('dynamics', 'native') == self.rt.dyn.name:
                    from .witness import load_witness_poses
                    P = load_witness_poses(self.rt.spec, w['actions'], w.get('dynamics', 'native'))
                    self._witness = P[:, [0, 2, 3]]
                    self.stats['witness_steps'] = len(P)-1
        if self._witness is False:
            return None
        W = self._witness
        T = len(W)-1
        p = st.legal.pose
        t = st.index
        dist = np.hypot(W[:, 0]-p[0], W[:, 1]-p[2])
        d = dist/0.208
        if cfg.witness_heading:
            # near a witness pose the heading must match too (turn steps; fades out by 0.6 m)
            dyaw = np.abs(np.arctan2(np.sin(W[:, 2]-p[3]), np.cos(W[:, 2]-p[3])))
            d = d+dyaw/self.rt.dyn.turn_rad*np.clip(1.-dist/0.6, 0., 1.)
        k = np.arange(T+1)
        total = np.maximum(T-t, d+(T-k))+cfg.witness_late_cost*np.maximum(0., t+d-k)
        return float(total.min())

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
    PRIOR = dict(continue_=0., goal_slow=1., goal_tc=1., goal=0., st_goal_tc=1., st_goal=0., stop=0., approach=1., pass_=2., sidestep=3., yield_=3., wait_short=2.,
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
        from .generator import is_maneuver
        return is_maneuver(q)

    def _prior(self, q):
        lab = q.label
        if self.is_maneuver(q):
            # a manoeuvre is charged at least one full decision interval (short
            # edges must not look cheaper than ordinary decisions) plus the prior
            pad = max(0, self.cfg.horizon-int(q.duration)) if self.cfg.maneuver_pad else 0
            return self.cfg.maneuver_prior+pad
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

        # Bookkeeping lives on the branch objects (no id() reuse) and never keeps a
        # parent alive: parent._qg_ordinary = [tried, failed] (a small shared list),
        # child._qg_origin = (that list, reached-by-manoeuvre flag).

        def gate_open(node):
            """Manoeuvres are justified at ``node`` if it was reached by a manoeuvre, or
            every ordinary movement query tried from its parent failed natively."""
            par = getattr(node, '_qg_origin', None)
            if par is None:
                return False
            rec, by_maneuver = par
            if by_maneuver:
                return True
            return rec[0] > 0 and rec[1] == rec[0]

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
            gate = cfg.maneuver_gate if (cfg.maneuver_gate and not gate_open(node)) else 0.
            # once some branch has left the spawn region, manoeuvre edges from spawn nodes wait behind the
            # escaped frontier (finite).  Ordinary edges are not penalised: waiting near the start can be the
            # right strategy (3t8DB4Uzvkt's solution holds there for the corridor to clear; penalising every
            # spawn edge lost it, and did not help YRUkbU5xsYj).
            if cfg.escape_penalty and self._escaped and self._in_spawn(node):
                gate += cfg.escape_penalty
            for q in queries:
                counter += 1
                extra = gate if self.is_maneuver(q) else 0.
                heapq.heappush(heap, (self._key(fn+self._prior(q)+extra, node), counter, node, q, False, fine))
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
            if not hasattr(node, '_qg_ordinary'):
                node._qg_ordinary = [0, 0]
            child._qg_origin = (node._qg_ordinary, self.is_maneuver(q))
            if q.kind in ('goal', 'goto', 'rotate') and not self.is_maneuver(q):
                node._qg_ordinary[0] += 1
                node._qg_ordinary[1] += int(child.outcome is not None and child.outcome != 'success')
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
            if cfg.escape_penalty and not self._escaped and not self._in_spawn(child, cfg.escape_m):
                self._escaped = True
                self.stats['escaped_at_edge'] = self.stats['children']
            # Exact shortcut: with edge-type fine de-duplication the signature does not depend on the
            # generator's output, so a duplicate child (55-75% of edges) is dropped before the generator
            # runs (~100 ms) -- unless it is inside the success radius (goal test first, as before).
            early_sig = None
            if cfg.fine_by_edge and not (cfg.goal_test and child.geodesic < SUCCESS_DISTANCE):
                early_fine = self.is_maneuver(q) or (cfg.goal_fine_m > 0 and child.geodesic < cfg.goal_fine_m)
                early_sig = self._sig(child, fine=early_fine)
                if early_sig in seen:
                    self.stats['duplicates'] = self.stats.get('duplicates', 0)+1
                    dl = self.stats.setdefault('dup_by_label', {})
                    dl[q.label.rstrip('0123456789')] = dl.get(q.label.rstrip('0123456789'), 0)+1
                    continue
            queries, diag = propose(child)
            if cfg.goal_test and child.geodesic < SUCCESS_DISTANCE:
                # privileged goal test on generation: a state inside the success
                # radius tries the generator's STOP immediately (before any
                # de-duplication could merge it with a farther state)
                stop_q = next((x for x in queries if x.kind == 'stop'), None)
                if stop_q is not None:
                    t = time.perf_counter()
                    done = self.rt.run(child, stop_q, horizon=1)
                    self.stats['run_s'] += time.perf_counter()-t
                    self.stats['children'] += 1
                    self.stats['goal_tests'] = self.stats.get('goal_tests', 0)+1
                    if done.outcome == 'success':
                        self.stats['outcomes']['success'] = self.stats['outcomes'].get('success', 0)+1
                        return self._result(done, t0, True)
            # cm/turn-step resolution only for states reached by an exact manoeuvre
            # (multi-point turns) or close to the goal; ordinary exploration keeps the coarse key
            fine = (self.is_maneuver(q) if cfg.fine_by_edge else bool(diag.get('stuck'))) or \
                (cfg.goal_fine_m > 0 and child.geodesic < cfg.goal_fine_m)
            sig = self._sig(child, fine=fine)
            if sig in seen:
                self.stats['duplicates'] = self.stats.get('duplicates', 0)+1
                dl = self.stats.setdefault('dup_by_label', {})
                dl[q.label.rstrip('0123456789')] = dl.get(q.label.rstrip('0123456789'), 0)+1
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
