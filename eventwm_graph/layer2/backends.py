"""Replaceable Stage 1 (humans) and Stage 2 (map increments) backends.

Layer 2 talks to each stage only through these two small adapters, so either stage can be
swapped for another checkpoint or another model family without touching the other one. A
backend must keep the legal-input contract of ``DecisionState`` (no simulator state, future
observation or privileged map) and must not mutate its inputs.

HumanBackend
    ``predict(state, worlds, seed) -> bank`` with ``bank.timestamp == state.timestamp``,
    ``bank.times`` relative seconds, ``bank.sandbox_positions(rel_times) -> [W,T,N,2]`` world
    X/Z with absent people at +inf, ``bank.weights [W]``; and
    ``interaction(bank, robots [K,T,4], lengths [K]) -> dict`` of per-candidate reductions.
MapBackend
    ``predict(state, trajectories, materialize) -> [delta_k]`` with the Stage 2
    ``predict_segments`` delta contract (hypothetical, one per trajectory, same order).
"""
import hashlib
from pathlib import Path

import numpy as np

from eventwm_graph.partial.query import NOMINAL_STEP_SECONDS


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Stage1HumanBackend:
    """Stage 1 ``HumanWorldModel`` (joint IID human worlds, exact count law, legal frames only)."""

    kind = 'stage1_hwm'
    step_seconds = NOMINAL_STEP_SECONDS      # the L2.5 nominal action clock used by the joint reduction

    def __init__(self, checkpoint, device='cpu', worlds=40, horizon=3., times=64, expected_sha256=None):
        from eventwm_graph.stage1_hwm.api import HumanWorldModel
        if expected_sha256 is not None and file_sha256(checkpoint) != expected_sha256:
            raise ValueError('Stage 1 checkpoint differs from the release manifest')
        self.model = HumanWorldModel(checkpoint, device)
        self.checkpoint = str(checkpoint)
        self.checkpoint_sha256 = self.model.checkpoint_sha256
        self.device = str(device)
        self.worlds = int(worlds)
        self.horizon = float(horizon)
        self.times = int(times)

    @property
    def max_steps(self):
        """Robot steps (k * step) that the bank horizon covers."""
        return int(np.floor(self.horizon/self.step_seconds+1e-9))

    def predict(self, state, worlds=None, seed=None):
        return self.model.predict(list(state.frames[:-1]), state.frames[-1], state.partial_graph, horizon=self.horizon,
                                  worlds=self.worlds if worlds is None else int(worlds), seed=seed,
                                  times=np.linspace(0., self.horizon, self.times), stream=state.stream)

    def predict_batch(self, states, worlds=None, seed=None):
        reqs = [dict(history=list(s.frames[:-1]), obs=s.frames[-1], partial_graph=s.partial_graph, stream=s.stream) for s in states]
        return self.model.predict_batch(reqs, horizon=self.horizon, worlds=self.worlds if worlds is None else int(worlds), seed=seed,
                                        times=np.linspace(0., self.horizon, self.times))

    def interaction(self, bank, robots, lengths):
        # bit-identical to bank.evaluate_candidates (l25.numeric.aggregate_joint); numba kernel, releases the GIL
        from .reduction import evaluate_candidates
        return evaluate_candidates(bank, robots, lengths)

    def reset(self, stream=None):
        if stream is None:
            self.model.streams.clear()
        else:
            self.model.streams.pop(stream, None)

    @property
    def last_profile(self):
        return dict(self.model.last_profile)


class Stage2MapBackend:
    """Stage 2 ``LegalVoxelPredictor`` (trajectory-conditioned hypothetical map increments).

    Canonical single-call semantics: each candidate's increment equals a cold
    ``predict_segments(..., trajectories=[tau_k])`` call, whatever else is in the batch and whatever is cached.
    Stage 2 builds its legal voxel map over the default extent (start cell +- 19 m) unless a trajectory's
    crop leaves it; then the extent grows to the union of the call's crops (or a cached larger map is
    reused) and the outputs of v17_a change slightly (~1e-5 relative, rare hard-state flips; found on 6 of
    2,110 development Roots). Candidates inside the default extent are therefore batched together (they
    share the default map: bit-identical to single calls); each candidate outside it runs alone and its
    non-default map is evicted from the cache afterwards.
    """

    kind = 'stage2_claude'

    def __init__(self, checkpoint, device='cuda:0', lanes=3, expected_sha256=None):
        from eventwm_graph.stage2_claude.api import LegalVoxelPredictor
        if expected_sha256 is not None and file_sha256(checkpoint) != expected_sha256:
            raise ValueError('Stage 2 checkpoint differs from the release manifest')
        self.model = LegalVoxelPredictor(checkpoint, device=device, lanes=lanes)
        self.checkpoint = str(checkpoint)
        self.checkpoint_sha256 = self.model.checkpoint_sha256
        self.device = str(device)
        self.lanes = self.model.lanes
        self._profile = {}

    def _call(self, state, trajectories, materialize):
        out = self.model.predict_segments(partial_graph=state.partial_graph, trajectories=trajectories,
                                          range_observations=state.range_packet, track_history=state.track_history,
                                          materialize=materialize)
        for k, v in self.model.last_batch_profile.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self._profile[k] = self._profile.get(k, 0)+v
        return out

    def outside_default_extent(self, state, trajectories):
        """Indices of trajectories whose Stage 2 crop leaves the default legal-map extent."""
        from eventwm_graph.stage2_claude.api import FRAME_MARGIN
        from eventwm_graph.stage2_claude.features import crop_box
        res = state.resolution
        start = trajectories[0][0]
        center = np.floor(np.asarray([start[1], start[3]], np.float64)/res).astype(np.int64)
        margin = int(np.ceil(FRAME_MARGIN/res))
        out = []
        for i, p in enumerate(trajectories):
            lo, hi = crop_box(p, res)
            if np.any(lo < center-margin) or np.any(hi > center+margin):
                out.append(i)
        return out

    def predict(self, state, trajectories, materialize=True):
        self._profile = dict(calls=0, outside_default_extent=0)
        if not trajectories:
            return []
        outside = set(self.outside_default_extent(state, trajectories))
        inside = [i for i in range(len(trajectories)) if i not in outside]
        out = [None]*len(trajectories)
        if inside:
            self._profile['calls'] += 1
            for i, d in zip(inside, self._call(state, [trajectories[i] for i in inside], materialize)):
                out[i] = d
        for i in sorted(outside):
            self._profile['calls'] += 1; self._profile['outside_default_extent'] += 1
            out[i] = self._call(state, [trajectories[i]], materialize)[0]
            key = next(reversed(self.model.map_cache))          # the enlarged map built for this candidate
            self.model.map_cache.pop(key, None)
            self.model.map_quick.pop(key, None)
        return out

    def reset(self):
        self.model.clear_cache()

    @property
    def last_profile(self):
        return dict(self._profile)
