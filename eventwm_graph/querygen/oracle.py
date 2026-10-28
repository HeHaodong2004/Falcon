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
    escape_h_min_blocked: int = 4     # confinement = at least this many static contacts within 3 primitive layers
                                      # (3t8DB4Uzvkt: 1 contact; the escape path then misdirected 58% of edges)
    prefix_cache: int = 0             # v9: exact prefix sharing among sibling edges (runtime keeps caches of this many
                                      # most recent parents; 0 = off).  Search results are identical with or without.
    heading_h: bool = False           # v9: add the in-place turn steps needed to face the true geodesic path
                                      # (0.6 m look-ahead) to h -- a turn-in-place robot must turn before it moves
    body_h: str = ''                  # v10 privileged body-aware static cost-to-go (bodyh.py, navmesh, exact native
                                      # primitives over cell x heading): 'heading' = its heading cost replaces the
                                      # geodesic-facing turn term; 'full' = also h >= body_scale * cost-to-go
    body_band_cost: float = 1.0       # price per body-axis sample off the navmesh (dilated-wall band)
    time_key_slack: float = 0.        # v10: also key states on time when the privileged space-time table shows at
                                      # least this much forced waiting ahead (0 = only when a human is within
                                      # human_near_m; v9).  Without it, waiting where no human is near returns to
                                      # an already-seen state and is pruned, so time can only be spent by
                                      # pose-changing manoeuvres (TEEsavR23oF: 1,763 spawn expansions)
    body_scale: float = 1.0
    block_weight: float = 0.          # privileged time-to-clear term (steps) added to f (0 = off)
    st_heuristic: bool = True         # privileged space-time cost-to-go (SpaceTimeHeuristic) replaces h + block term
    st_fallback: float = 300.         # extra steps when the space-time model finds no arrival
    st_radius: float = 0.5            # human radius (m) of the space-time model (v9: 0.6).  0.5 = the strict rule's lateral
                                      # bound (0.25 m rectangle clearance + 0.243 m half width); 0.6 declared corridors
                                      # the witnesses pass through unreachable, and st_fallback then pushed every later
                                      # state behind all earlier ones (TEEsavR23oF: unreachable from t=20)
    st_radius_opt: float = -1.        # v10: where the st_radius table finds no arrival, use a second table with the
                                      # cell-sound radius (strict bound 0.4882 m minus the 0.2 m cell's half
                                      # diagonal; -1 = that value, 0 = off).  A 0.5 m cell-centre test still
                                      # declared W16Bm4ysK8v unreachable from t=50, yet a witness arrives at t=108
    block_dist: float = 0.5           # centre distance to the geodesic path that blocks it (> half-width + body proxy)
    block_horizon_m: float = 8.0      # path length checked ahead
    block_step: int = 4               # departure-delay resolution (steps)
    block_max: int = 240              # cap (steps) when no clear departure is found
    portfolio: list = field(default_factory=list)
                                      # v10: [] = one search.  Otherwise one OracleConfig override dict per member
                                      # search (e.g. [{}, {"weight": 3.5}]): the members are independent best-first
                                      # searches over the same generator queries from the same reset, run
                                      # interleaved (the member with the fewest simulated edges per share goes
                                      # next) under the shared max_edges / max_seconds budget; the first success
                                      # wins.  Search cost is heavy-tailed (small, unrelated changes moved single
                                      # episodes between 30 and 12,000 edges), which differently ordered
                                      # searches cut
    portfolio_shares: list = field(default_factory=list)   # relative edge shares of the members (default equal)
    improve_below_spl: float = 0.     # v14 candidate: refine only first solutions whose path efficiency (start geodesic /
                                      # path length, the official SPL of a success) is below this (0 = always)
    improve_lambda: float = 0.        # v14 candidate: the refinement's objective is steps + improve_lambda x forward
                                      # steps (path length / native step): among equally fast branches the shorter
                                      # route wins (SPL).  The time-only A* fills unavoidable waits with detours
                                      # (cV4RVeZvu5T: drives 1.4 m away and back, SPL 0.47).  0 = steps only
    improve_edges: int = 0            # anytime refinement: after the first success, a second bounded search (weight
    improve_weight: float = 1.5       # improve_weight) spends up to this many extra edges looking for a branch with
                                      # fewer steps (nodes whose step lower bound -- index + geodesic / step + 1 --
                                      # cannot beat the incumbent are pruned).  SR and edges-to-first-success unchanged
    intervention_cost: float = 0.     # learnability diagnostic: f penalty (steps) per decision that deviates from the
                                      # default policy (Oracle.is_default: continue, 'goal' or the final STOP); large
                                      # values give the solution with the fewest non-default ("critical") decisions
    noise: float = 0.                 # v10: deterministic pseudo-random edge priority offset in [0, noise) steps (a
    noise_seed: int = 0               # randomised tie-break / restart member of a portfolio; 0 = off)
    prim_search: bool = False         # v13: the ordinary search may expand the generator's exact primitive queries
                                      # (kind 'prim'); False = they are never expanded (search identical to v12)
    prim_fallback_sims: int = 0       # v13 privileged fallback phase: when the ordinary search fails, a native witness
                                      # search (witness.py: weighted A* over the native primitives with the true human
                                      # tape) from the reset for up to this many simulated primitive steps (the binding,
                                      # deterministic budget); a witness is executed through the runtime as the
                                      # generator's primitive queries (needs GeneratorConfig.primitives) and verified
                                      # like any branch.  0 = off
    prim_fallback_s: float = 14400.   # wall-clock safety cap of the fallback search only
    prim_fallback_weight: float = 3.0
    prim_takeover: bool = False       # v13 option: keep only the witness prefix up to the earliest step (every
    prim_takeover_every: int = 8      # prim_takeover_every steps) from which the no-selection default policy succeeds
    prim_takeover_spl: float = 0.9    # with path efficiency >= this x the witness's (fewest primitive decisions).  Off:
                                      # on VqCaAuuoeWk the default policy can take over only after witness step 64 (it
                                      # meets a human from every earlier step), saving 30 of 94 primitive decisions for
                                      # SPL 0.989 -> 0.909 and +50 s


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
        # human-free cost-to-go (same moves): slack = table - free > 0 where the human schedule forces waits
        from collections import deque
        free = np.full(nav.shape, INF, dtype=np.float32)
        dq = deque()
        for iz, ix in zip(*np.nonzero(goal)):
            free[iz, ix] = 0.
            dq.append((iz, ix))
        while dq:
            iz, ix = dq.popleft()
            d = free[iz, ix]+1.
            for dz in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    a, b = iz+dz, ix+dx
                    if (dz or dx) and 0 <= a < nz and 0 <= b < nx and nav[a, b] and free[a, b] > d:
                        free[a, b] = d
                        dq.append((a, b))
        self.free = free

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

    def slack(self, x, z, t):
        """Forced waiting (steps) on the best schedule from (x, z) at step t: time-indexed minus human-free
        cost-to-go; None if unreachable in the model."""
        iz, ix = int((z-self.z0)/self.cell), int((x-self.x0)/self.cell)
        nz, nx = self.nav.shape
        if not (0 <= iz < nz and 0 <= ix < nx) or not self.nav[iz, ix]:
            iz, ix = self._nearest_nav(min(max(iz, 0), nz-1), min(max(ix, 0), nx-1))
        v = float(self.table[min(t, len(self.table)-1), iz, ix])
        f = float(self.free[iz, ix])
        return None if v >= self.INF/2 or f >= self.INF/2 else v-f


def horizon_for(query, cfg):
    if query.kind == 'stop':
        return 8      # 1 for instant-speed robots; robots with momentum brake to rest first (executor)
    if query.kind in ('wait', 'wait_clear', 'creep', 'kturn'):
        return max(1, int(query.duration))
    return cfg.horizon          # movement and rotation: one decision interval, continuable


class EscapeMap:
    """Privileged (Oracle-only) steps-to-escape around a confined spawn (a static ranking aid).

    1. Breadth-first search over the native primitives from the reset (branch simulator, true static
       contacts, declared spawn-escape rule; humans ignored), keyed on 2 cm cells x exact turn index x
       escape-window flag; skipped for open spawns (no static contact within 3 primitive layers).  If it
       reaches states ``escape_m`` from the start within ``bfs_sims``, a reverse BFS gives *every*
       recorded state its exact primitive steps to the nearest escape (dead ends: ``DEAD``).
    2. Otherwise, for clearly confined spawns (>= ``min_blocked`` contacts), a weighted A* toward
       ``escape_m`` finds one escape path (deep escapes such as YRUkbU5xsYj's 49 primitives); path states
       get their remaining steps, other explored spawn states ``first + OFF_PATH``.
    """

    DEAD = 120
    OFF_PATH = 25

    def __init__(self, rt, escape_m=1.0, extra_depth=8, open_depth=4, bfs_sims=12000, astar_sims=20000,
                 weight=2.0, min_blocked=4):
        from collections import deque
        import heapq as _hq
        from .native import escape_exempt
        t0 = time.perf_counter()
        sim = rt.sim
        dyn = rt.dyn
        self.n_turn = int(round(2*math.pi/dyn.turn_rad))
        self.turn = dyn.turn_rad
        root = rt.root
        p0 = root.legal.pose
        self.p0 = (float(p0[0]), float(p0[2]), float(p0[3]))
        C0 = rt.spawn_links
        acts = ('forward', 'left', 'right')
        k0 = self._key(p0[0], p0[2], 0, bool(C0))
        # ---- 1. breadth-first map
        depth = {k0: 0}
        preds = {k0: set()}
        q = deque([(root.robot, 0, 0, (float(p0[0]), float(p0[2])), bool(C0), 0., k0)])
        escapes = set()
        first = None
        sims = 0
        blocked_early = 0
        while q and sims < bfs_sims:
            st, d, ti, xz, esc, trav, kp = q.popleft()
            if first is not None and d >= first+extra_depth:
                break
            if d >= 3 and blocked_early == 0:
                break
            for a in acts:
                sim.load(st, d)
                r = sim.step(a, compute_geodesic=False)
                sims += 1
                trav2 = trav+math.hypot(r.pose[0]-xz[0], r.pose[2]-xz[1])
                if r.static_contact and not escape_exempt(r.static_links, C0, d+1, trav2):
                    if d <= 2:
                        blocked_early += 1
                    continue
                esc2 = esc and bool(r.static_contact)
                ti2 = ti+(1 if a == 'left' else -1 if a == 'right' else 0)
                k = self._key(r.pose[0], r.pose[2], ti2, esc2)
                preds.setdefault(k, set()).add(kp)
                if k in depth:
                    continue
                depth[k] = d+1
                if not esc2 and math.hypot(r.pose[0]-p0[0], r.pose[2]-p0[2]) >= escape_m:
                    escapes.add(k)
                    if first is None:
                        first = d+1
                    continue
                q.append((sim.state, d+1, ti2, (float(r.pose[0]), float(r.pose[2])), esc2, trav2, k))
            if first is not None and first <= open_depth:
                break
        self.blocked_early = blocked_early
        self.first = first
        self.mode = None
        self.steps = {}
        self.known = set()
        if first is not None and first > open_depth and blocked_early > 0:
            self.mode = 'bfs'
            self.known = set(depth)
            dq = deque()
            for k in escapes:
                self.steps[k] = 0
                dq.append(k)
            while dq:
                k = dq.popleft()
                for kp in preds.get(k, ()):
                    if kp not in self.steps:
                        self.steps[kp] = self.steps[k]+1
                        dq.append(kp)
        elif first is None and blocked_early >= min_blocked:
            # ---- 2. weighted A* toward displacement (deep escapes)
            step = dyn.step_m
            heap = [(weight*escape_m/step, 0, 0, root.robot, 0, (float(p0[0]), float(p0[2])), bool(C0), 0., k0)]
            parent = {k0: None}
            g_of = {k0: 0}
            counter = 0
            goal = None
            budget = sims+astar_sims
            while heap and sims < budget:
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
            if goal is not None:
                self.mode = 'astar'
                self.known = set(g_of)
                chain = []
                k = goal
                while k is not None:
                    chain.append(k)
                    k = parent[k]
                chain.reverse()
                self.first = len(chain)-1
                for i, k in enumerate(chain):
                    self.steps[k] = len(chain)-1-i
        rt._loaded = None
        self.sims = sims
        self.states = len(self.known)
        self.active = self.mode is not None
        self.seconds = time.perf_counter()-t0

    def _key(self, x, z, ti, esc):
        return (round(float(x)/0.02), round(float(z)/0.02), int(ti) % self.n_turn, bool(esc))

    def lookup(self, pose, escape_open):
        """Steps to escape for a node pose; None when the pose is not a recorded spawn state."""
        dyaw = math.atan2(math.sin(float(pose[3])-self.p0[2]), math.cos(float(pose[3])-self.p0[2]))
        ti = int(round(dyaw/self.turn))
        known = False
        for esc in (bool(escape_open), not escape_open):
            k = self._key(pose[0], pose[2], ti, esc)
            if k in self.steps:
                return self.steps[k]
            known = known or k in self.known
        if not known:
            return None
        return self.DEAD if self.mode == 'bfs' else (self.first or 0)+self.OFF_PATH


class Oracle:
    def __init__(self, runtime, generator, config=None, shared=None, member=0):
        self.rt = runtime
        self.gen = generator
        self.cfg = config or OracleConfig()
        # heavy privileged tables (space-time, escape map, body cost-to-go) shared by portfolio members
        self._shared = shared if shared is not None else {}
        self.member = int(member)
        self._ord_attr = '_qg_ordinary' if not self.member else f'_qg_ordinary{self.member}'
        self._org_attr = '_qg_origin' if not self.member else f'_qg_origin{self.member}'
        self._best = None
        self._bound = None          # anytime refinement: incumbent step count (prune nodes that cannot beat it)
        self._incumbent = None
        self._improvements = 0
        self._st = None
        self._st_opt = None
        self._witness = None
        self._escaped = False
        self._escmap = None
        self._body = None
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

    def _efficiency(self, st):
        g0 = float(self.rt.start_geodesic)
        return g0/max(float(st.path_length), g0) if g0 > 0 else 1.

    def _obj(self, st):
        """Refinement objective of a branch: steps (+ improve_lambda x path length in native forward steps)."""
        lam = self.cfg.improve_lambda if getattr(self, '_improving', False) else 0.
        return st.index+lam*st.path_length/float(self.rt.dyn.step_m)

    def _obj_lb(self, st):
        """Lower bound of the objective of any completion of ``st`` (remaining forward steps >= geodesic / step)."""
        lam = self.cfg.improve_lambda if getattr(self, '_improving', False) else 0.
        step_m = float(self.rt.dyn.step_m)
        return self._obj(st)+(1.+lam)*st.geodesic/step_m+1

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
                self._escmap = self._cached(('escmap', cfg.escape_h_sims, cfg.escape_h_min_blocked), lambda: EscapeMap(
                    self.rt, astar_sims=cfg.escape_h_sims, min_blocked=cfg.escape_h_min_blocked))
                em = self._escmap
                self.stats['escape_map'] = dict(active=em.active, mode=em.mode, first_escape=em.first, states=em.states,
                                                blocked_early=em.blocked_early,
                                                sims=em.sims, seconds=round(em.seconds, 1))
            if self._escmap.active and self._in_spawn(st, 1.2):
                k = self._escmap.lookup(st.legal.pose, st.escape_open)
                if k is not None:
                    # remaining exact primitive steps to leave the spawn, then the usual estimate
                    h = h+k
        hs = 0.
        if cfg.body_h and self._body is None:
            from .bodyh import BodyCostToGo
            self._body = self._cached(('body', cfg.body_band_cost), lambda: BodyCostToGo(self.rt, band_cost=cfg.body_band_cost))
            self.stats['body_h'] = dict(shape=list(self._body.shape), seconds=round(self._body.seconds, 2))
        if cfg.heading_h and st.geodesic == st.geodesic and st.geodesic > 0.3:
            hb = self._body.heading_cost(st.legal.pose) if cfg.body_h else None
            hs = hb if hb is not None else self._heading_steps(st.legal.pose)
            h += hs
        if cfg.body_h == 'full':
            v = self._body.value(st.legal.pose)
            if v is not None:
                h = max(h, cfg.body_scale*v)
        if cfg.st_heuristic:
            if self._st is None:
                t = time.perf_counter()
                self._st = self._cached(('st', cfg.st_radius), lambda: SpaceTimeHeuristic(self.rt, radius=cfg.st_radius))
                if cfg.st_radius_opt:
                    from .native import BODY_PROXY_M, ROBOT_BOUNDS
                    r = cfg.st_radius_opt if cfg.st_radius_opt > 0 else \
                        BODY_PROXY_M+min(abs(b) for b in ROBOT_BOUNDS)-self._st.cell*math.sqrt(2)/2
                    self._st_opt = self._cached(('st', r), lambda: SpaceTimeHeuristic(self.rt, radius=r))
                self.stats['st_build_s'] = time.perf_counter()-t
            p = st.legal.pose
            v = self._st.value(float(p[0]), float(p[2]), st.index)
            if v is None and self._st_opt is not None:
                v = self._st_opt.value(float(p[0]), float(p[2]), st.index)
            # time-to-go in steps: the space-time arrival (it already includes the
            # forced waits); never below the geodesic estimate
            h = max(h, v+hs) if v is not None else h+cfg.st_fallback
        elif cfg.block_weight:
            pen += cfg.block_weight*self._block_time(st)
        if cfg.improve_lambda and getattr(self, '_improving', False):
            lam = cfg.improve_lambda/float(self.rt.dyn.step_m)
            return st.index+lam*st.path_length+cfg.weight*(h+lam*st.geodesic)+pen
        return st.index+cfg.weight*h+pen

    def _noise(self, node, q):
        """Deterministic uniform [0, 1) from (seed, node pose/time, query) -- process-independent (crc32)."""
        import zlib
        p = node.legal.pose
        k = f"{self.cfg.noise_seed}|{node.index}|{p[0]:.3f}|{p[2]:.3f}|{p[3]:.3f}|{q.label}|{q.kind}"
        return zlib.crc32(k.encode())/4294967296.

    def _cached(self, key, make):
        if key not in self._shared:
            self._shared[key] = make()
        return self._shared[key]

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
        if not near and cfg.time_key_slack > 0 and self._st is not None:
            sl = self._st.slack(float(p[0]), float(p[2]), index)
            near = sl is not None and sl >= cfg.time_key_slack
        tb = index//cfg.dedup_time if near else -1
        return (bool(fine), int(math.floor(p[0]/cell)), int(math.floor(p[2]/cell)), hb, tb)

    # Generic (episode-independent) edge priors in steps for lazy ordering.
    PRIOR = dict(continue_=0., goal_slow=1., goal_tc=1., goal_vc=1.5, goal_xc=1.5, goal=0., st_goal_tc=1., st_goal_vc=1.5,
                 st_goal_xc=1.5, st_goal=0.,
                 stop=0., approach=1., pass_=2., sidestep=3., yield_=3., wait_short=2.,
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
    def is_default(q):
        """The no-selection default policy: keep the running query, (re)start the goal query, STOP at the goal."""
        return q.kind in ('continue', 'stop') or q.label == 'goal'

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
        if self.cfg.portfolio:
            res = self._search_portfolio()
        elif self.cfg.lazy:
            res = self._search_lazy()
        else:
            return self._search_eager()
        if res['success'] and self.cfg.improve_edges > 0 and (
                not self.cfg.improve_below_spl or self._efficiency(res['state']) < self.cfg.improve_below_spl):
            res = self._improve(res)
        if not res['success'] and self.cfg.prim_fallback_sims > 0:
            res = self._prim_fallback(res)
        return res

    def _prim_fallback(self, res):
        """v13 privileged fallback: native witness search from the reset (true human tape; selection only), executed
        through the runtime as exact primitive queries -- every step simulated and legally observed again, so the
        branch is an ordinary branch (authoritative replay as usual)."""
        from .witness import find_witness
        from .generator import primitive_queries
        cfg = self.cfg
        rt = self.rt
        t0 = time.perf_counter()
        info = dict(found=False)
        stats = res['stats']
        stats['prim_fallback'] = info
        if not rt.dyn.native_actions:
            info['skipped'] = 'not a native-action robot'
            return res
        w = find_witness(rt.spec, budget_s=cfg.prim_fallback_s, max_sims=cfg.prim_fallback_sims,
                         weight=cfg.prim_fallback_weight, dynamics='native', rules='native', runtime=rt, verify=False)
        rt._loaded = None                   # the witness search moved the shared simulator
        info.update(found=bool(w['found']), sims=int(w['sims']), states=int(w['states']),
                    witness_steps=int(w['steps']), best_geodesic=round(float(w['best_geodesic']), 3),
                    search_s=round(time.perf_counter()-t0, 1))
        if not w['found']:
            res['seconds'] += time.perf_counter()-t0
            return res
        prims = {q.meta['action'][0]: q for q in primitive_queries()}
        node = rt.root
        chain = [node]
        for ch in w['actions']:
            node = rt.run(node, prims[ch], horizon=1)
            chain.append(node)
            if node.terminal:
                break
        info['outcome'] = node.outcome
        stats['children'] += len(node.decisions)
        stats['outcomes'][f'prim_{node.outcome}'] = stats['outcomes'].get(f'prim_{node.outcome}', 0)+1
        ok = node.outcome == 'success'
        if ok and cfg.prim_takeover:
            node = self._prim_takeover(chain, node, info)
        info['primitive_decisions'] = sum(1 for d in node.decisions if d['query'].kind == 'prim')
        info['seconds'] = round(time.perf_counter()-t0, 1)
        return dict(success=ok, state=node if ok else res['state'], steps=node.index if ok else res['steps'],
                    outcome=node.outcome if ok else res['outcome'],
                    final_geodesic=node.geodesic if ok else res['final_geodesic'],
                    decisions=node.decisions if ok else res['decisions'],
                    seconds=res['seconds']+time.perf_counter()-t0, stats=stats)

    def _default_rollout(self, node):
        """The no-selection default policy from ``node`` (continue / the generator's 'goal' query / its STOP where
        the privileged goal test says success), through the runtime.  Returns the terminal (or last) branch."""
        rt, cfg = self.rt, self.cfg
        while not node.terminal and node.index < MAX_EPISODE_STEPS:
            queries, _ = self.gen.propose(node.legal, node.execution)
            by = {(q.kind if q.kind in ('continue', 'stop') else q.label): q for q in queries}
            if 'stop' in by and node.geodesic < SUCCESS_DISTANCE:
                q = by['stop']
            elif 'continue' in by:
                q = by['continue']
            elif 'goal' in by:
                q = by['goal']
            else:
                break
            node = rt.run(node, q, horizon=min(horizon_for(q, cfg), MAX_EPISODE_STEPS-node.index))
        return node

    def _prim_takeover(self, chain, witness_end, info):
        """Fewest primitive decisions: the earliest witness step (grid ``prim_takeover_every``) from which the
        default policy succeeds with SPL >= ``prim_takeover_spl`` x the witness's own path efficiency; else the
        whole witness."""
        cfg = self.cfg
        geo0 = float(self.rt.start_geodesic)
        def eff(st):
            return geo0/max(st.path_length, geo0) if st.outcome == 'success' else 0.
        target = cfg.prim_takeover_spl*eff(witness_end)
        tried = []
        for k in range(0, len(chain)-1, max(1, int(cfg.prim_takeover_every))):
            st = self._default_rollout(chain[k])
            tried.append((k, st.outcome, round(eff(st), 3)))
            if st.outcome == 'success' and eff(st) >= target:
                info['takeover'] = dict(step=k, tried=tried, witness_efficiency=round(eff(witness_end), 3))
                return st
        info['takeover'] = dict(step=None, tried=tried, witness_efficiency=round(eff(witness_end), 3))
        return witness_end

    def _improve(self, res):
        """Anytime refinement: bounded search for a branch with fewer steps than the first solution."""
        from dataclasses import replace
        base = self.cfg
        t0 = time.perf_counter()
        first = res['state']
        cfg = replace(base, portfolio=[], portfolio_shares=[], lazy=True, weight=base.improve_weight,
                      max_edges=base.improve_edges, improve_edges=0)
        m = Oracle(self.rt, self.gen, cfg, shared=self._shared, member=99)
        m._member_over = dict(improve=True)
        m._improving = True
        m._bound = m._obj(first)
        it = m._lazy_iter()
        while True:
            try:
                next(it)
            except StopIteration:
                break
        stats = res['stats']
        stats['edges_first_success'] = stats['children']
        stats['improve'] = dict(edges=m.stats['children'], steps_first=int(first.index),
                                steps_best=int(m._incumbent.index) if m._incumbent is not None else int(first.index),
                                path_first=round(float(first.path_length), 3),
                                path_best=round(float((m._incumbent or first).path_length), 3),
                                improvements=m._improvements, seconds=round(time.perf_counter()-t0, 1))
        stats['children'] += m.stats['children']
        if m._incumbent is not None and m._obj(m._incumbent) < m._obj(first):
            st = m._incumbent
            return dict(success=True, state=st, steps=st.index, outcome=st.outcome, final_geodesic=st.geodesic,
                        decisions=st.decisions, seconds=res['seconds']+time.perf_counter()-t0, stats=stats)
        res['seconds'] += time.perf_counter()-t0
        return res

    def _search_lazy(self):
        it = self._lazy_iter()
        while True:
            try:
                next(it)
            except StopIteration as stop:
                return stop.value

    def _search_portfolio(self):
        """Interleaved independent searches (``OracleConfig.portfolio``) under one edge budget.

        Resumable: the member searches persist on the Oracle, so after a budget stop the caller may raise
        ``self.cfg.max_edges`` / ``max_seconds`` and call ``search()`` again to continue the same searches (expert
        generation: a capped first pass, then an extension for the few hard episodes -- no restart).  The members'
        own caps are unbounded; the portfolio loop enforces the shared budget (a member's count never exceeds the
        total, so single-call results are unchanged)."""
        from dataclasses import replace
        base = self.cfg
        pf = getattr(self, '_pf', None)
        if pf is None:
            n = len(base.portfolio)
            shares = [float(x) for x in base.portfolio_shares] if base.portfolio_shares else [1.]*n
            assert len(shares) == n and min(shares) > 0
            members = []
            for i, over in enumerate(base.portfolio):
                cfg = replace(base, portfolio=[], portfolio_shares=[], lazy=True, max_edges=10**12, max_seconds=1e12,
                              **over)
                members.append(Oracle(self.rt, self.gen, cfg, shared=self._shared, member=i))
                members[-1]._member_over = dict(over)
            pf = self._pf = dict(members=members, its=[m._lazy_iter() for m in members], status=['running']*n,
                                 shares=shares, t0=time.perf_counter(), win=None)
        members, its, status, shares, t0 = pf['members'], pf['its'], pf['status'], pf['shares'], pf['t0']
        n = len(members)
        win = pf['win']
        while win is None and time.perf_counter()-t0 < base.max_seconds:
            used = sum(m.stats['children'] for m in members)
            if used >= base.max_edges:
                break
            live = [i for i in range(n) if status[i] == 'running']
            if not live:
                break
            i = min(live, key=lambda j: (members[j].stats['children']/shares[j], j))
            try:
                next(its[i])
            except StopIteration as stop:
                res = stop.value
                if res['success']:
                    status[i] = 'success'
                    win = pf['win'] = (i, res)
                    break
                status[i] = 'exhausted'
        stats = self._merge_stats(members, ['budget' if x == 'running' else x for x in status], shares)
        if win is not None:
            stats['portfolio_winner'] = win[0]
            st, ok = win[1]['state'], True
        else:
            bests = [m._best for m in members if m._best is not None]
            st, ok = min(bests, key=lambda b: b.geodesic) if bests else self.rt.root, False
        self.stats = stats
        return self._result(st, t0, ok)

    @staticmethod
    def _merge_stats(members, status, shares):
        S = dict(expansions=0, children=0, outcomes={}, by_label={}, gen_s=0., run_s=0., raw_candidates=0,
                 kept_candidates=0, gen_ms=[], edge_ms=[], edge_steps=[], exp_log=[])
        for i, m in enumerate(members):
            s = m.stats
            for k in ('expansions', 'children', 'gen_s', 'run_s', 'raw_candidates', 'kept_candidates', 'duplicates',
                      'goal_tests', 'deferred', 'predict_s', 'trims'):
                if k in s:
                    S[k] = S.get(k, 0)+s[k]
            for k in ('gen_ms', 'edge_ms', 'edge_steps'):
                S[k].extend(s[k])
            S['exp_log'].extend(tuple(e)+(i,) for e in s.get('exp_log', []))
            for k in ('outcomes', 'dup_by_label'):
                for a, b in s.get(k, {}).items():
                    S.setdefault(k, {})[a] = S.setdefault(k, {}).get(a, 0)+b
            for a, b in s['by_label'].items():
                d = S['by_label'].setdefault(a, dict(n=0, fail=0))
                d['n'] += b['n']; d['fail'] += b['fail']
            for k in ('escape_map', 'body_h', 'witness_steps'):
                if k in s and k not in S:
                    S[k] = s[k]
            S['st_build_s'] = S.get('st_build_s', 0.)+s.get('st_build_s', 0.)
        S['portfolio'] = [dict(member=i, overrides=m._member_over, share=shares[i], status=status[i],
                               children=m.stats['children'], expansions=m.stats['expansions'],
                               best_geodesic=None if m._best is None else round(float(m._best.geodesic), 3),
                               escaped_at_edge=m.stats.get('escaped_at_edge'))
                          for i, m in enumerate(members)]
        return S

    def _lazy_iter(self):
        """Best-first over *edges*: an edge (node, query) is simulated when popped.  A generator that
        yields before every pop (portfolio interleaving) and returns the search result."""
        cfg = self.cfg
        t0 = time.perf_counter()
        root = self.rt.root
        counter = 0
        heap = []
        best = self._best = root
        seen = set()

        def propose(node):
            t = time.perf_counter()
            queries, diag = self.gen.propose(node.legal, node.execution)
            if not cfg.prim_search:
                queries = [q for q in queries if q.kind != 'prim']
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
            par = getattr(node, self._org_attr, None)
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
            n_int = getattr(node, '_qg_interv', 0)
            for q in queries:
                counter += 1
                extra = gate if self.is_maneuver(q) else 0.
                if cfg.intervention_cost:
                    extra += cfg.intervention_cost*(n_int+(0 if self.is_default(q) else 1))
                if cfg.noise > 0:
                    extra += cfg.noise*self._noise(node, q)
                heapq.heappush(heap, (self._key(fn+self._prior(q)+extra, node), counter, node, q, False, fine))
        push_children(root, *propose(root))
        step_m = float(self.rt.dyn.step_m)
        while heap and self.stats['children'] < cfg.max_edges and time.perf_counter()-t0 < cfg.max_seconds:
            yield
            prio, _, node, q, deferred, node_fine = heapq.heappop(heap)
            if MAX_EPISODE_STEPS-node.index <= 0:
                continue
            if self._bound is not None and self._obj_lb(node) >= self._bound:
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
            child._qg_interv = getattr(node, '_qg_interv', 0)+int(not self.is_default(q))
            self.stats['edge_ms'].append(round(1000*dt, 1))
            self.stats['edge_steps'].append(child.index-node.index)
            self.stats['children'] += 1
            rec = getattr(node, self._ord_attr, None)
            if rec is None:
                rec = [0, 0]
                setattr(node, self._ord_attr, rec)
            setattr(child, self._org_attr, (rec, self.is_maneuver(q)))
            if q.kind in ('goal', 'goto', 'rotate') and not self.is_maneuver(q):
                rec[0] += 1
                rec[1] += int(child.outcome is not None and child.outcome != 'success')
            lab = self.stats['by_label'].setdefault(q.label.rstrip('0123456789'), dict(n=0, fail=0))
            lab['n'] += 1
            if child.outcome is not None:
                self.stats['outcomes'][child.outcome] = self.stats['outcomes'].get(child.outcome, 0)+1
                if child.outcome == 'success':
                    if self._bound is None:
                        return self._result(child, t0, True)
                    if self._obj(child) < self._bound:
                        self._incumbent, self._bound = child, self._obj(child)
                        self._improvements += 1
                    continue
                lab['fail'] += 1
                continue
            if child.geodesic < best.geodesic:
                best = self._best = child
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
                    # a robot with momentum brakes to rest before STOP (executor): allow those steps
                    dyn = self.rt.dyn
                    lvl = int(child.dyn_state) if not isinstance(child.dyn_state, tuple) else 0
                    done = self.rt.run(child, stop_q, horizon=1+(dyn.stop_steps(lvl) if dyn.stateful else 0))
                    self.stats['run_s'] += time.perf_counter()-t
                    self.stats['children'] += 1
                    self.stats['goal_tests'] = self.stats.get('goal_tests', 0)+1
                    if done.outcome == 'success':
                        self.stats['outcomes']['success'] = self.stats['outcomes'].get('success', 0)+1
                        if self._bound is None:
                            return self._result(done, t0, True)
                        if self._obj(done) < self._bound:
                            self._incumbent, self._bound = done, self._obj(done)
                            self._improvements += 1
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
                # diagnostics only: how often the open list is cut and the priority at the cut
                self.stats['trims'] = self.stats.get('trims', 0)+1
                self.stats['trim_key'] = round(float(heap[-1][0][0]), 1) if heap else None
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
            if not cfg.prim_search:
                queries = [q for q in queries if q.kind != 'prim']
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
