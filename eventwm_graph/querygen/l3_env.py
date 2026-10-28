"""Layer-3 decision environment over the query space (the integration point for Layer 2.5 / Layer 3).

One episode = one Falcon episode under the official rules.  At every decision state the legal query generator
emits K candidate queries (K = 10 quota or K = 6 compact); the policy chooses one; the fixed low-level stack
(planner + executor + dynamics) executes it for one decision interval (<= ``horizon`` native steps, or until the
query completes) in the exact branch simulator -- the Env.step mirror whose human motion is the recorded tape
(Falcon's humans do not react to the robot; bitwise-verified by every authoritative replay).

  obs   : dict(legal=LegalState (pose, goal, legal map, legal tracks), candidates=[Query], index=step,
               features=[K, D] legal sandbox consequence features (``l3_interface.FEATURE_NAMES``) if requested)
  action: index into ``obs['candidates']``
  reward: training signal only (uses the privileged geodesic, as RL rewards may):
          progress (geodesic decrease, m) - time_cost * steps + success_bonus on official success
          - collision_penalty on an official human contact
  done  : official episode end (STOP, official human contact, 500 steps)
  info  : outcome, steps, executed actions, per-decision timing (generator / features / execution ms)

``official_replay()`` re-executes the whole decision sequence with the unchanged ``habitat.Env.step`` and returns
the official measures (success, SPL, PSC, human_collision); it is the evaluation of record.

The policy sees legal information only; nothing privileged is put into ``obs``.
"""
import time

import numpy as np


class QueryEnv:
    def __init__(self, spec, planner='orca', dynamics='native', gen_kw=None, horizon=16, features=False, M=16,
                 feature_steps=48, success_bonus=10., collision_penalty=10., time_cost=0.01):
        from .episode import EpisodeRuntime
        from .generator import QueryGenerator, GeneratorConfig
        from .dynamics import FAMILIES
        self.spec = spec
        self.planner = planner
        self.dyn = FAMILIES[dynamics]
        self.gen = QueryGenerator(self.dyn, GeneratorConfig(**(gen_kw or {})))
        t = time.perf_counter()
        self.rt = EpisodeRuntime(spec, planner, self.dyn, strict=False)     # official rules
        self.rt.sim.check_static = False
        self.setup_s = time.perf_counter()-t
        self.horizon = int(horizon)
        self.features = bool(features)
        self.M, self.feature_steps = int(M), int(feature_steps)
        self.success_bonus, self.collision_penalty, self.time_cost = success_bonus, collision_penalty, time_cost
        self.timing = dict(generator_ms=[], features_ms=[], execute_ms=[], decision_steps=[])
        self.state = None
        self.candidates = None

    # ------------------------------------------------------------------ API
    def reset(self):
        self.state = self.rt.root
        return self._observe()

    def step(self, action):
        from .oracle import horizon_for
        from .native import MAX_EPISODE_STEPS
        q = self.candidates[int(action)]
        st = self.state
        g0 = float(st.geodesic)
        t = time.perf_counter()
        child = self.rt.run(st, q, horizon=min(horizon_for(q, _H(self.horizon)), MAX_EPISODE_STEPS-st.index))
        self.timing['execute_ms'].append(1000*(time.perf_counter()-t))
        self.timing['decision_steps'].append(child.index-st.index)
        self.state = child
        n = child.index-st.index
        g1 = float(child.geodesic) if child.geodesic == child.geodesic else g0
        reward = (g0-g1)-self.time_cost*n
        done = child.terminal
        if child.outcome == 'success':
            reward += self.success_bonus
        elif child.outcome == 'human_collision':
            reward -= self.collision_penalty
        info = dict(outcome=child.outcome, steps=int(child.index), decision=q.label, decision_steps=n)
        obs = None if done else self._observe()
        return obs, float(reward), bool(done), info

    def official_replay(self):
        """Evaluation of record: the decision sequence re-executed with the unchanged Env.step."""
        from .episode import replay
        rep = replay(self.spec, self.planner, self.dyn, self.state.decisions)
        return dict(metrics=rep['metrics'], steps=rep['steps'], actions_identical=rep['actions'] == self.state.actions)

    def close(self):
        self.rt.close()

    # ------------------------------------------------------------- helpers
    def _observe(self):
        st = self.state
        t = time.perf_counter()
        queries, diag = self.gen.propose(st.legal, st.execution)
        self.timing['generator_ms'].append(1000*(time.perf_counter()-t))
        self.candidates = queries
        obs = dict(legal=st.legal, candidates=queries, index=int(st.index), stuck=bool(diag.get('stuck')))
        if self.features:
            from .l3_interface import consequence_features
            from .sandbox import cv_futures
            t = time.perf_counter()
            F = cv_futures(st.legal.tracks, st.legal.time, self.feature_steps, M=self.M, seed=int(st.index))
            X, _ = consequence_features(st.legal, queries, self.planner, self.dyn, F, self.feature_steps)
            self.timing['features_ms'].append(1000*(time.perf_counter()-t))
            obs['features'] = X
        return obs


class _H:
    """Minimal stand-in for OracleConfig in ``horizon_for`` (decision interval)."""
    def __init__(self, horizon):
        self.horizon = horizon


def default_policy(obs):
    """No-selection baseline: keep the running query, else the goal query (first 'continue' or 'goal')."""
    labels = [q.label for q in obs['candidates']]
    for k, q in enumerate(obs['candidates']):
        if q.kind == 'continue':
            return k
    return labels.index('goal') if 'goal' in labels else 0


def benchmark(spec, planner='orca', gen_kw=None, features=False, policy='default', seed=0, max_decisions=200):
    """Run one episode with a fixed policy; return timing and outcome (throughput for RL planning)."""
    rng = np.random.default_rng(seed)
    env = QueryEnv(spec, planner, gen_kw=gen_kw, features=features)
    t0 = time.perf_counter()
    obs = env.reset()
    total, n, info = 0., 0, {}
    while obs is not None and n < max_decisions:
        a = default_policy(obs) if policy == 'default' else int(rng.integers(len(obs['candidates'])))
        obs, r, done, info = env.step(a)
        total += r
        n += 1
        if done:
            break
    wall = time.perf_counter()-t0
    out = dict(key=spec.key, planner=planner, policy=policy, features=features, decisions=n, steps=info.get('steps'),
               outcome=info.get('outcome'), reward=total, wall_s=wall, setup_s=env.setup_s,
               decisions_per_s=n/max(wall, 1e-9), steps_per_s=(info.get('steps') or 0)/max(wall, 1e-9),
               generator_ms_p50=float(np.median(env.timing['generator_ms'])) if env.timing['generator_ms'] else None,
               execute_ms_p50=float(np.median(env.timing['execute_ms'])) if env.timing['execute_ms'] else None,
               features_ms_p50=float(np.median(env.timing['features_ms'])) if env.timing['features_ms'] else None)
    env.close()
    return out


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('keys', nargs='+')
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--planner', default='orca')
    ap.add_argument('--gen', default='{}')
    ap.add_argument('--features', action='store_true')
    ap.add_argument('--policy', default='default', choices=['default', 'random'])
    a = ap.parse_args()
    from .suites import load
    specs = {s.key: s for s in load(a.suite)}
    for k in a.keys:
        print(json.dumps(benchmark(specs[k], a.planner, json.loads(a.gen), a.features, a.policy)), flush=True)


if __name__ == '__main__':
    main()
