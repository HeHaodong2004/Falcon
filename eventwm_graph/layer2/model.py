"""Unified Layer 2 world model: shared human worlds + candidate-specific hypothetical maps.

    l2 = Layer2WorldModel.from_release()                      # verified release checkpoints
    out = l2.predict(state, [tau_1, ..., tau_K])               # one decision state, K candidates

Per decision state, Stage 1 predicts ONE joint human bank (humans are exogenous to the robot
in this simulator, so the bank is shared by every candidate) and Stage 2 predicts, per
candidate trajectory, the hypothetical increment of the legal partial map. Layer 2 then scores
every candidate against the shared bank (robot body envelope vs human disk, 1 m social events,
per-world minima; the L2.5 ``aggregate_joint`` reduction).

Everything a candidate carries is HYPOTHETICAL: predicted FREE/BLOCKED cells are never legal
observations and a predicted future graph is rejected as the input of a later decision state.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Optional, Tuple

import numpy as np
import torch

from eventwm_graph.partial.query import NOMINAL_STEP_SECONDS
from .backends import Stage1HumanBackend, Stage2MapBackend
from .state import DecisionState, normalise_candidates

VERSION = 'layer2_unified_1'
ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT/'release/layer2_v1/RELEASE.json'


def robot_steps(trajectories, start_time, step, max_steps):
    """Sample candidates on the nominal clock ``start_time + (1..T) * step`` (L2.5 convention).

    Returns robots [K,T,4] (x, y, z, yaw; yaw unwrapped then linearly interpolated), lengths [K]
    (steps inside each candidate's duration, capped by the human horizon) and per-candidate
    truncation flags. Steps past a candidate's end repeat its final pose and are masked by length.
    """
    durations = np.array([p[-1, 0]-start_time for p in trajectories], np.float64)
    full = np.floor(durations/step+1e-9).astype(int)
    lengths = np.minimum(full, max_steps)
    T = max(1, int(lengths.max()) if len(lengths) else 1)
    rel = np.arange(1, T+1)*step
    robots = np.zeros((len(trajectories), T, 4), np.float64)
    for k, p in enumerate(trajectories):
        t = np.minimum(start_time+rel, p[-1, 0])
        yaw = np.unwrap(p[:, 4])
        robots[k] = np.stack([np.interp(t, p[:, 0], p[:, 1]), np.interp(t, p[:, 0], p[:, 2]),
                              np.interp(t, p[:, 0], p[:, 3]), np.interp(t, p[:, 0], yaw)], -1)
    return robots, lengths, rel, full > max_steps


@dataclass(frozen=True)
class CandidatePrediction:
    index: int
    query_id: Optional[str]
    trajectory: np.ndarray            # N x 5 [t, x, y, z, yaw] as supplied (absolute time)
    map: dict                         # Stage 2 delta (predict_segments contract); hypothetical
    humans: dict                      # this candidate's reductions against the shared human bank
    metadata: dict = field(default_factory=dict)
    hypothetical: bool = True

    @property
    def duration(self):
        return float(self.trajectory[-1, 0]-self.trajectory[0, 0])

    @property
    def future_graph(self):
        """Complete hypothetical graph at the trajectory end (``hypothetical=True``), or None."""
        return self.map.get('future_partial_graph')

    def summary(self):
        s = self.map['states']; p = self.map['probabilities'].astype(np.float64)
        h = self.humans
        return dict(index=self.index, query_id=self.query_id, duration=self.duration,
                    candidate_cells=int(len(s)), predicted_free=int((s == 1).sum()), predicted_blocked=int((s == 2).sum()),
                    expected_free_uncalibrated=float(p[:, 1].sum()), expected_blocked_uncalibrated=float(p[:, 2].sum()),
                    p_collision=float(h['p_collision']), p_social=float(h['p_social']),
                    expected_social_seconds=float(h['expected_social_duration']),
                    minimum_distance_median=h['minimum_quantiles'][1], human_steps=int(h['steps']),
                    human_horizon_truncated=bool(h['horizon_truncated']))


@dataclass(frozen=True)
class Layer2Prediction:
    timestamp: float                  # decision time t (absolute clock of the frames)
    robot_pose: np.ndarray            # [x, y, z, yaw] at t
    stream: str
    humans: object                    # shared Stage 1 HumanWorlds (times relative to t)
    candidates: Tuple[CandidatePrediction, ...]
    interaction_times: np.ndarray     # relative times (1..T) * 0.048 s used for the candidate reductions
    profile: dict
    provenance: dict
    hypothetical: bool = True

    def humans_at(self, absolute_times):
        """[W,T,N,2] world X/Z of the shared bank at absolute times (absent people = +inf)."""
        rel = np.asarray(absolute_times, np.float64)-self.timestamp
        return self.humans.sandbox_positions(rel)

    def human_futures(self, steps=None, step=NOMINAL_STEP_SECONDS):
        """The shared bank on the planner clock: positions [W, steps+1, N, 2] world X/Z at t + i*step
        (i = 0..steps, absent people = +inf), present [W, N], weights [W], absolute times, track keys.
        ``steps`` defaults to the full human horizon (62 steps = 2.976 s)."""
        n = int(np.floor(float(self.humans.times[-1])/step+1e-9)) if steps is None else int(steps)
        rel = np.arange(n+1)*step
        if rel[-1] > float(self.humans.times[-1])+1e-9:
            raise ValueError('requested steps exceed the human horizon')
        return dict(positions=self.humans.sandbox_positions(rel), present=np.asarray(self.humans.present),
                    weights=np.asarray(self.humans.weights), times=self.timestamp+rel, track_keys=self.humans.track_keys)

    def summaries(self):
        return [c.summary() for c in self.candidates]


class Layer2WorldModel:
    def __init__(self, human_backend, map_backend, *, overlap=True, release=None):
        self.humans = human_backend
        self.maps = map_backend
        self.overlap = bool(overlap)
        self.release = release
        self._pool = ThreadPoolExecutor(1, thread_name_prefix='layer2-stage1') if overlap else None
        self._banks = OrderedDict()     # (id(state), worlds, seed) -> (state, bank): reuse for repeated calls on ONE state object
        self.last_profile = {}

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_release(cls, manifest=RELEASE, *, map_device='cuda:0', human_device=None, lanes=None, overlap=True,
                     torch_threads=1, verify=True):
        """Build both stages from the release manifest (SHA-256 verified)."""
        manifest = Path(manifest)
        rel = json.loads(manifest.read_text())
        if torch_threads is not None:
            torch.set_num_threads(int(torch_threads))
        s1, s2 = rel['stage1'], rel['stage2']
        human = Stage1HumanBackend(ROOT/s1['checkpoint'], device=human_device or s1['device'], worlds=s1['worlds'],
                                   horizon=s1['horizon_seconds'], times=s1['times'],
                                   expected_sha256=s1['checkpoint_sha256'] if verify else None)
        maps = Stage2MapBackend(ROOT/s2['checkpoint'], device=map_device, lanes=s2['lanes'] if lanes is None else lanes,
                                expected_sha256=s2['checkpoint_sha256'] if verify else None)
        return cls(human, maps, overlap=overlap, release=rel)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def reset(self, stream=None):
        """Drop the incremental Stage 1 stream state(s) and the Stage 2 graph/legal-map caches."""
        self.humans.reset(stream)
        self.maps.reset()
        self._banks.clear()

    # ---------------------------------------------------------------- inference
    def _bank(self, state, worlds, seed):
        # The bank is a deterministic function of the (immutable) state, worlds and seed: later calls with the
        # SAME DecisionState object (e.g. further candidate batches of one decision) reuse it.
        key = (id(state), worlds, seed)
        tick = time.perf_counter()
        hit = self._banks.get(key)
        if hit is not None and hit[0] is state:
            self._banks.move_to_end(key)
            return hit[1], (time.perf_counter()-tick)*1000, dict(cached=True)
        with torch.no_grad():
            bank = self.humans.predict(state, worlds=worlds, seed=seed)
        self._banks[key] = (state, bank)
        while len(self._banks) > 2:
            self._banks.popitem(last=False)
        return bank, (time.perf_counter()-tick)*1000, dict(self.humans.last_profile, cached=False)

    def _humans_side(self, state, worlds, seed, robots, lengths, any_candidates):
        bank, human_ms, profile = self._bank(state, worlds, seed)
        i0 = time.perf_counter()
        with np.errstate(invalid='ignore'):      # absent people are +inf in the joint reduction
            red = self.humans.interaction(bank, robots, lengths) if any_candidates else {}
        return bank, human_ms, profile, red, (time.perf_counter()-i0)*1000

    def predict_humans(self, state: DecisionState, worlds=None, seed=None):
        return self._bank(state, worlds, seed)[0]

    def predict_maps(self, state: DecisionState, candidates, materialize=True):
        paths = [c[0] for c in normalise_candidates(candidates, state)]
        return self.maps.predict(state, paths, materialize)

    def predict(self, state: DecisionState, candidates=(), *, materialize=True, worlds=None, seed=None):
        """One decision state, K >= 0 candidate trajectories -> Layer2Prediction."""
        if not isinstance(state, DecisionState):
            raise TypeError('predict needs a DecisionState')
        tick = time.perf_counter()
        cands = normalise_candidates(candidates, state)
        paths = [c[0] for c in cands]
        now = state.timestamp
        step = self.humans.step_seconds
        robots, lengths, rel, truncated = robot_steps(paths, now, step, self.humans.max_steps)
        # Stage 1 bank + the candidate reductions (they need only the bank and the supplied poses) run on the
        # worker thread while Stage 2 runs here; results are identical to the sequential order.
        pending = self._pool.submit(self._humans_side, state, worlds, seed, robots, lengths, bool(paths)) if self._pool is not None else None
        try:
            m0 = time.perf_counter()
            deltas = self.maps.predict(state, paths, materialize)
            map_ms = (time.perf_counter()-m0)*1000
            map_profile = self.maps.last_profile if paths else {}
        finally:
            if pending is not None:
                bank, human_ms, human_profile, red, interaction_ms = pending.result()
        if pending is None:
            bank, human_ms, human_profile, red, interaction_ms = self._humans_side(state, worlds, seed, robots, lengths, bool(paths))
        both = time.perf_counter()
        # shared decision-state consistency
        if abs(float(bank.timestamp)-now) > 1e-7:
            raise RuntimeError('Stage 1 bank clock differs from the decision state')
        for p, d in zip(paths, deltas):
            if abs(float(d['base_timestamp'])-now) > 1e-7 or abs(float(d['timestamp'])-p[-1, 0]) > 1e-7:
                raise RuntimeError('Stage 2 increment clock differs from the decision state / candidate end')
            if d['provenance']['trajectory_sha256'] != hashlib.sha256(p.astype('<f8').tobytes()).hexdigest():
                raise RuntimeError('Stage 2 output order differs from the candidate order')
        i0 = time.perf_counter()
        out = []
        for k, ((p, qid, meta), d) in enumerate(zip(cands, deltas)):
            h = {key: value[k] for key, value in red.items()}
            h.update(steps=int(lengths[k]), horizon_seconds=float(lengths[k]*step), horizon_truncated=bool(truncated[k]))
            out.append(CandidatePrediction(k, qid, p, d, h, meta))
        done = time.perf_counter()
        self.last_profile = dict(candidates=len(paths), total_ms=(done-tick)*1000, stage1_ms=human_ms, stage2_ms=map_ms,
                                 stages_wall_ms=(both-tick)*1000, interaction_ms=interaction_ms, assemble_ms=(done-i0)*1000,
                                 overlapped=pending is not None, stage1=human_profile, stage2=map_profile)
        prov = dict(version=VERSION, hypothetical=True, privileged_runtime_inputs=False,
                    stage1=dict(kind=self.humans.kind, checkpoint_sha256=self.humans.checkpoint_sha256, device=self.humans.device),
                    stage2=dict(kind=self.maps.kind, checkpoint_sha256=self.maps.checkpoint_sha256, device=self.maps.device,
                                lanes=self.maps.lanes, probabilities_calibrated=False),
                    release=None if self.release is None else self.release.get('name'),
                    decision_state=state.identity(), human_horizon_seconds=float(self.humans.horizon),
                    interaction_reduction='l25.numeric.aggregate_joint (robot envelope vs human disk; 1 m social)')
        return Layer2Prediction(now, state.robot_pose, state.stream, bank, tuple(out), rel, self.last_profile, prov)

    def predict_many(self, states, candidates_per_state, **kw):
        """Sequential convenience over several decision states (each with its own candidates)."""
        return [self.predict(s, c, **kw) for s, c in zip(states, candidates_per_state)]
