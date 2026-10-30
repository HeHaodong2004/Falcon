"""Closed-loop imagination for the World Model Sandbox (option; the released core ``core.py`` is unchanged).

The released Sandbox imagines one trajectory per candidate against the Stage 1 MEAN people future, hands every
tracked person to the planner as freshly observed at every step, keeps the legal map frozen and calls Stage 2 only
after the trajectory (``docs/l3/SANDBOX_AUDIT.md``).  ``imagine_closed_loop`` closes the loop that a short-horizon
query choice depends on:

* per world: every candidate is re-imagined in each of the W heaviest Stage 1 worlds -- the real executor and planner
  react to THAT world's people, step by step;
* perception: the imagined planner perceives people only through an imagined 90 deg camera from the imagined pose
  (line of sight on the imagined map, the tracker's 2 s memory; the world's hidden people included) --
  ``l3.perceive.VisibleWorld``;
* map loop (``map_loop``): the imagined map is updated as the imagined camera certifies cells -- 'stage2_union': one
  batched Stage 2 call on every candidate's first-pass trajectory gives a predicted local map (legal, hypothetical);
  every candidate's closed-loop pass discovers it through the imagined frustum (``l3.perceive.Discovery``), so the
  imagined planner re-plans on predicted geometry as the real one re-plans on observed geometry; 'off': frozen map;
* continuation fidelity: after the candidate's interval the default policy continues with a decision boundary every
  interval and goal-query inheritance, as the real runtime does.

People do not react to the robot in this benchmark (exogenous, replayed), so a sampled world's people futures need no
re-prediction inside the loop; what closes is robot <-> imagined observation <-> imagined map <-> dynamics.  Physics
stays nominal (no navmesh contact).  Everything returned is hypothetical and legal (inputs: the legal state, the
Stage 1 bank, Stage 2 predictions).

    out = imagine_closed_loop(sb, state, legal, candidates, bank, ClosedLoopConfig(window=32))
    out.worlds[j].rollouts[k]   # ImaginedRollout of candidate k in world j (with .leaf)
    out.contact(k, steps=32, body=0.2)   # world-weighted contact probability
"""
from dataclasses import dataclass
import math
import time

import numpy as np

from .imagine import imagine, frozen_map, failed_rollout


@dataclass
class ClosedLoopConfig:
    window: int = 32                 # imagined steps from the decision (interval + default continuation)
    worlds: int = 6                  # heaviest Stage 1 worlds imagined (each candidate re-imagined per world)
    perception: str = 'camera'       # 'camera' (imagined 90 deg camera + tracker memory) | 'all' (released behaviour)
    map_loop: str = 'stage2_union'   # 'stage2_union' | 'off'
    fidelity: bool = True            # continuation decision boundaries + goal inheritance
    boundary: int = 16
    keep_leaf: bool = True
    reweight: object = None          # optional world weighter (l3.worldweights.WorldWeighter): learned weights of the
                                     # Stage 1 worlds from the tracked people's predicted paths (legal), else uniform
    selection: str = 'heaviest'      # 'heaviest' (the W highest-weight worlds) | 'cluster': k-means of all worlds by the
                                     # TRACKED people's futures (what the robot can react to); one closed-loop
                                     # representative per cluster; contacts scored against EVERY world's people


@dataclass
class WorldRollouts:
    world: int
    weight: float
    rollouts: list                   # ImaginedRollout per candidate
    people: np.ndarray               # [T, N, 2] the world's people (world x, z), +inf absent
    members: tuple = ()              # 'cluster' selection: (world weight, people [T, N, 2]) of every member world


@dataclass
class ClosedLoopResult:
    worlds: list
    profile: dict
    hypothetical: bool = True

    def _seq(self, ro):
        poses = np.asarray(ro.poses, np.float64)
        c = getattr(ro, 'cont', None)
        seq = np.concatenate([poses[1:], c]) if c is not None and len(c) else poses[1:]
        return seq if len(seq) else poses[:1]

    @staticmethod
    def _min_clr(seq, people, steps):
        from eventwm_graph.l3.heuristics import _rect_clearance
        n = min(len(seq), len(people), int(steps))
        if n <= 0 or not people.shape[1]:
            return math.inf
        d = _rect_clearance(seq[:n], people[:n])
        d = np.where(np.isfinite(d), d, np.inf)
        return float(d.min()) if d.size else math.inf

    def clearance(self, k, steps):
        """Per imagined (representative) world: min body-rectangle clearance of candidate k to that world's people
        within ``steps`` ('cluster' selection: the cluster's weighted mean over its member worlds is in
        ``member_clearance``)."""
        return np.asarray([self._min_clr(self._seq(w.rollouts[k]), w.people, steps) for w in self.worlds])

    def member_clearance(self, k, steps):
        """[(weight, clearance)] over EVERY world: each member world's people against its representative's imagined
        trajectory ('heaviest' selection: the imagined worlds themselves)."""
        out = []
        for w in self.worlds:
            seq = self._seq(w.rollouts[k])
            if w.members:
                out += [(mw, self._min_clr(seq, mp, steps)) for mw, mp in w.members]
            else:
                out.append((w.weight, self._min_clr(seq, w.people, steps)))
        return out

    def contact(self, k, steps, body=0.2):
        """Probability that candidate k comes within ``body`` m (body rectangle) of a person, over every scored
        world."""
        mc = self.member_clearance(k, steps)
        tot = sum(w for w, _ in mc)
        return float(sum(w for w, c in mc if c < body)/max(tot, 1e-9))


def _world_people(bank, world, T, step_s=0.048):
    rel = np.arange(1, T+1)*step_s
    inside = rel[rel <= float(bank.times[-1])+1e-9]
    HP = np.asarray(bank.sandbox_positions(inside), np.float64)[int(world)]
    if len(inside) < T and HP.shape[0] >= 2:
        vel = (HP[-1]-HP[-2])/step_s
        extra = np.arange(1, T-len(inside)+1)*step_s
        HP = np.concatenate([HP, HP[-1][None]+extra[:, None, None]*vel[None]], 0)
    return HP


def _cluster_worlds(bank, W, C, T, times=(12, 24, 36, 48, 62), iters=20):
    """k-means (farthest-point init, deterministic) of all worlds by the TRACKED people's positions at a few horizons;
    returns (representative world per cluster = the member nearest the centroid, cluster mass, member lists)."""
    keys = [str(k) for k in getattr(bank, 'track_keys', ())]
    tracked = np.asarray([k.startswith('track:') for k in keys], bool)
    nW = len(W)
    if not tracked.any():
        reps = np.arange(min(C, nW))
        mem = [[int(r)] for r in reps]
        rest = [i for i in range(nW) if i not in set(reps.tolist())]
        for i, r in enumerate(rest):
            mem[i % len(mem)].append(int(r))
        return reps, np.asarray([W[m].sum() for m in mem]), mem
    rel = np.asarray([min(t, T) for t in times])*0.048
    rel = rel[rel <= float(bank.times[-1])+1e-9]
    P = np.asarray(bank.sandbox_positions(rel), np.float64)[:, :, tracked]          # [W, t, Nt, 2]
    X = np.where(np.isfinite(P), P, 0.).reshape(nW, -1)
    X = np.concatenate([X, (~np.isfinite(P)).reshape(nW, -1)*5.], 1)
    cent = [X[0]]
    for _ in range(1, min(C, nW)):
        d = np.min([np.square(X-c).sum(1) for c in cent], axis=0)
        cent.append(X[int(np.argmax(d))])
    cent = np.asarray(cent)
    for _ in range(iters):
        lab = np.argmin(np.square(X[:, None]-cent[None]).sum(2), 1)
        new = np.asarray([X[lab == c].mean(0) if (lab == c).any() else cent[c] for c in range(len(cent))])
        if np.allclose(new, cent):
            break
        cent = new
    reps, mass, mem = [], [], []
    for c in range(len(cent)):
        idx = np.flatnonzero(lab == c)
        if not len(idx):
            continue
        r = idx[int(np.argmin(np.square(X[idx]-cent[c]).sum(1)))]
        reps.append(int(r)); mass.append(float(W[idx].sum())); mem.append([int(i) for i in idx])
    return np.asarray(reps), np.asarray(mass)/max(sum(mass), 1e-9), mem


def imagine_closed_loop(sb, state, legal, candidates, bank, cfg=None):
    """Closed-loop imagination of ``candidates`` (sandbox ``Candidate``s) at one legal decision state."""
    from eventwm_graph.sandbox.episode import default_query
    from eventwm_graph.sandbox import accel
    from eventwm_graph.l3.perceive import VisibleWorld, Discovery, predicted_truth_union
    from eventwm_graph.l3.infofeat import _blocked_lookup
    from eventwm_graph.layer2.state import trajectory_from_poses
    from .imagine import NominalHumans
    cfg = cfg or ClosedLoopConfig()
    t0 = time.perf_counter()
    W = np.asarray(bank.weights, np.float64)
    W = W/W.sum() if W.sum() > 0 else np.full(len(W), 1./max(len(W), 1))
    if cfg.reweight is not None:
        W = np.asarray(cfg.reweight.weights(legal, bank), np.float64)
    members = None
    if cfg.selection == 'cluster' and len(W) > cfg.worlds:
        worlds, ww, members = _cluster_worlds(bank, W, cfg.worlds, T=int(cfg.window))
    else:
        order = np.argsort(-W) if len(np.unique(W)) > 1 else np.arange(len(W))
        worlds = order[:cfg.worlds]
        ww = W[worlds]/W[worlds].sum()
    cq = default_query()
    gmap = frozen_map(legal.map)
    bnd = int(cfg.boundary) if cfg.fidelity else 0
    T = int(cfg.window)

    def run(c, humans, vmap):
        ex, dyn = sb._execution(c)
        if isinstance(ex, Exception):
            return failed_rollout(legal, '%s: %s' % (type(ex).__name__, ex))
        h = sb.config.steps_for(c)
        return imagine(legal, ex, dyn, humans, min(h, T), level=int(c.level), gmap=vmap,
                       continuation=max(0, T-min(h, T)), cont_query=cq, planner=c.planner,
                       cont_boundary=bnd, cont_inherit=bool(cfg.fidelity), keep_leaf=cfg.keep_leaf)

    def humans_for(world):
        if cfg.perception == 'camera':
            return VisibleWorld(legal, bank, int(world), T)
        return NominalHumans(legal.tracks, legal.time, T, bank=bank, mode='world:%d' % int(world))

    predicted = None
    map_error = None
    t_map = 0.
    cache = accel.FrozenMapCache() if sb.config.memoize else None
    with accel.active(cache):
        if cfg.map_loop == 'stage2_union':
            # first pass (frozen map, heaviest world) -> one batched Stage 2 call -> one predicted local map
            t1 = time.perf_counter()
            first = [run(c, humans_for(worlds[0]), gmap) for c in candidates]
            trajs = []
            for ro in first:
                P = np.asarray(ro.poses, np.float64)
                cc = getattr(ro, 'cont', None)
                if cc is not None and len(cc):
                    P = np.concatenate([P, np.asarray(cc, np.float64)])
                trajs.append(trajectory_from_poses(P, legal.time))
            try:
                pred = sb.l2.predict(state, trajs, materialize=False, worlds=sb.config.worlds, seed=sb.config.seed)
                predicted = predicted_truth_union(legal.map, [c.map for c in pred.candidates])
            except Exception as exc:                # no Stage 2 answer: frozen map (recorded in the profile)
                predicted = None
                map_error = repr(exc)
            t_map = time.perf_counter()-t1
        out = []
        for j, world in enumerate(worlds):
            ros = []
            for c in candidates:
                hum = humans_for(world)
                if predicted is not None:
                    vmap = legal.map.copy()
                    hum.blocked = _blocked_lookup(predicted)
                    hum.discovery = Discovery(vmap, predicted)
                else:
                    vmap = gmap
                ros.append(run(c, hum, vmap))
            mem = ()
            if members is not None:
                mem = tuple((float(W[m]), _world_people(bank, m, T)) for m in members[j])
            out.append(WorldRollouts(int(world), float(ww[j]), ros, _world_people(bank, world, T), mem))
    prof = dict(total_ms=(time.perf_counter()-t0)*1000, map_ms=t_map*1000, worlds=len(worlds), candidates=len(candidates),
                window=T, map_loop=cfg.map_loop if predicted is not None or cfg.map_loop == 'off' else 'failed',
                map_error=map_error)
    return ClosedLoopResult(out, prof)
