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


@dataclass
class WorldRollouts:
    world: int
    weight: float
    rollouts: list                   # ImaginedRollout per candidate
    people: np.ndarray               # [T, N, 2] the world's people (world x, z), +inf absent


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

    def clearance(self, k, steps):
        """Per world: min body-rectangle clearance of candidate k to the world's people within ``steps``."""
        from eventwm_graph.l3.heuristics import _rect_clearance
        out = []
        for w in self.worlds:
            seq = self._seq(w.rollouts[k])
            n = min(len(seq), len(w.people), int(steps))
            if n <= 0 or not w.people.shape[1]:
                out.append(math.inf)
                continue
            d = _rect_clearance(seq[:n], w.people[:n])
            d = np.where(np.isfinite(d), d, np.inf)
            out.append(float(d.min()) if d.size else math.inf)
        return np.asarray(out)

    def contact(self, k, steps, body=0.2):
        """World-weighted probability that candidate k comes within ``body`` m (body rectangle) of a person."""
        c = self.clearance(k, steps)
        w = np.asarray([x.weight for x in self.worlds])
        return float((w*(c < body)).sum()/max(w.sum(), 1e-9))


def _world_people(bank, world, T, step_s=0.048):
    rel = np.arange(1, T+1)*step_s
    inside = rel[rel <= float(bank.times[-1])+1e-9]
    HP = np.asarray(bank.sandbox_positions(inside), np.float64)[int(world)]
    if len(inside) < T and HP.shape[0] >= 2:
        vel = (HP[-1]-HP[-2])/step_s
        extra = np.arange(1, T-len(inside)+1)*step_s
        HP = np.concatenate([HP, HP[-1][None]+extra[:, None, None]*vel[None]], 0)
    return HP


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
            out.append(WorldRollouts(int(world), float(ww[j]), ros, _world_people(bank, world, T)))
    prof = dict(total_ms=(time.perf_counter()-t0)*1000, map_ms=t_map*1000, worlds=len(worlds), candidates=len(candidates),
                window=T, map_loop=cfg.map_loop if predicted is not None or cfg.map_loop == 'off' else 'failed',
                map_error=map_error)
    return ClosedLoopResult(out, prof)
