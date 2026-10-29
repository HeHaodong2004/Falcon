"""Layer 3 actor: runs episodes with the current shared policy and spools them for the learner.

One actor process = one environment worker of the split-process Sandbox (Layer 2 lives in a server process; the worker
holds the branch simulator, its legal and range sensors, the generator, the imagined rollouts, the Graph Compiler and
a CPU copy of the policy).  Per episode it samples a configuration (planner x dynamics x candidate set) and an episode
from the curriculum, then at every decision:

    candidates -> Sandbox imagination -> compiled consequence graph -> policy (CPU) -> exploration -> execute

and writes the finished episode (packed graphs, actions, behaviour probabilities, reward components, metrics) to
``spool/`` as one compressed file.  Policy weights are reloaded from ``policy.pt`` whenever the learner publishes a new
version (checked at every episode start).

Selectors: ``policy`` (learned, Boltzmann exploration at temperature ``tau`` + epsilon), ``default`` (the no-selection
floor), ``random``, ``greedy`` (hand-written sandbox heuristic baseline, ``heuristics.py``).
"""
import argparse
import json
import math
import os
import pickle
import random
import time
import traceback
import zlib
from pathlib import Path

import numpy as np


class _Ensemble:
    """Several critics of the same layout; heads are averaged (success / collision in probability space)."""

    def __init__(self, nets):
        self.nets = nets

    def __call__(self, b):
        import torch
        outs = [n(b) for n in self.nets]
        avg = {}
        for k in outs[0]:
            if k in ('succ', 'coll'):
                p = torch.stack([torch.sigmoid(o[k]) for o in outs]).mean(0).clamp(1e-6, 1-1e-6)
                avg[k] = torch.log(p/(1-p))
            else:
                avg[k] = torch.stack([o[k] for o in outs]).mean(0)
        return avg

    def eval(self):
        return self


def load_policy(path, device='cpu'):
    import torch
    from .model import ConsequenceGraphNet, ModelConfig, Weights
    ck = torch.load(path, map_location=device)
    cfg = ModelConfig(**ck['model_config'])
    if ck.get('ensemble'):
        nets = []
        for sd in ck['ensemble']:
            n = ConsequenceGraphNet(cfg); n.load_state_dict(sd); n.eval(); nets.append(n)
        return _Ensemble(nets), Weights(**ck['weights']), int(ck.get('version', 0)), ck
    net = ConsequenceGraphNet(cfg)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net, Weights(**ck['weights']), int(ck.get('version', 0)), ck


class PolicyHandle:
    """CPU policy with hot reload.  ``weights``: optional override of the value scalarisation (evaluation only; the
    decomposed heads make it possible without retraining)."""

    def __init__(self, path, weights=None):
        self.path = Path(path) if path else None
        self.override = weights
        self.net = self.w = None
        self.version = -1
        self.mtime = None
        self.meta = {}

    def refresh(self):
        if self.path is None or not self.path.exists():
            return False
        m = self.path.stat().st_mtime
        if m == self.mtime:
            return False
        for _ in range(3):
            try:
                self.net, self.w, self.version, ck = load_policy(self.path)
                if self.override:
                    from dataclasses import replace
                    self.w = replace(self.w, **self.override)
                self.meta = {k: ck.get(k) for k in ('tau', 'epsilon', 'version', 'anchor', 'explore')}
                self.mtime = m
                return True
            except Exception:
                time.sleep(0.5)
        return False

    def scores(self, packed):
        import torch
        from .batch import stack, to_torch
        with torch.no_grad():
            out = self.net(to_torch(stack([packed]), 'cpu'))
            q = self.w.scalarize(out)[0].numpy()
            heads = {k: v[0].numpy() for k, v in out.items()}
        return q, heads


def choose(q, n, tau, eps, rng, anchor=0., default=0):
    """Boltzmann(tau) over the n valid candidates, mixed with epsilon-uniform, and (training only) anchored: with
    probability ``anchor`` the default candidate.  Returns (index, behaviour probability of index)."""
    q = np.asarray(q[:n], np.float64)
    if tau <= 0:
        p = np.zeros(n); p[int(np.argmax(q))] = 1.
    else:
        z = (q-q.max())/tau
        p = np.exp(z); p /= p.sum()
    p = (1-eps)*p+eps/n
    if anchor > 0:
        p = (1-anchor)*p
        p[int(default)] += anchor
    a = int(rng.choice(n, p=p))
    return a, float(p[a])


def cumulants(r):
    """Decomposed per-decision cumulants (network units, batch.CUM order)."""
    from .batch import CUM_SCALE
    c = np.array([float(r['outcome'] == 'success'), float(r['outcome'] == 'human_collision'),
                  r['steps'], r['path'], r['social']], np.float32)
    return c/CUM_SCALE


def run_episode(spec, cfg, sb, selector, policy, rng, tau=0.5, eps=0.03, compiler=None, record=True, max_decisions=400,
                oracle_seq=None, imagination=True, margin=0., anchor=0., near_interval=0, near_m=3.0, scan=False,
                explore=None):
    from .env import L3Episode
    from .batch import pack
    t0 = time.perf_counter()
    ep = L3Episode(spec, cfg, sb if imagination else None, compiler=compiler, imagination=imagination,
                   near_interval=near_interval, near_m=near_m, scan=scan)
    rows = dict(packed=[], actions=[], probs=[], cum=[], labels=[], n=[], default=[], q=[], timing=[], heads=[])
    diverged = False
    try:
        while not ep.over and ep.decisions < max_decisions:
            d = ep.observe()
            n = len(d.queries)
            t = time.perf_counter()
            packed = pack(d.graph)
            qv = heads = None
            if selector == 'policy':
                qv, heads = policy.scores(packed)
                t_eff, a_eff = tau, anchor
                if explore == 'risk' and tau > 0:
                    # explore where it matters: the predicted failure risk of the default sets the temperature and the
                    # default anchor (risky states: broader search; safe states: mostly keep the default)
                    r = 1.-1./(1.+math.exp(-float(heads['succ'][d.default])))
                    t_eff = tau*(0.5+1.5*r)
                    a_eff = min(max(anchor*(1.5-r), 0.), 0.9)
                a, p = choose(qv, n, t_eff, eps, rng, anchor=a_eff, default=d.default)
                if margin > 0 and tau <= 0 and a != d.default and qv[a] < qv[d.default]+margin:
                    a = d.default          # conservative deviation: leave the default only for a clear value gain
            elif selector == 'default':
                a, p = d.default, 1.
            elif selector == 'random':
                a, p = int(rng.integers(n)), 1./n
            elif selector == 'greedy':
                from .heuristics import greedy_index
                a, p = greedy_index(d.graph), 1.
            elif selector == 'progress':
                from .heuristics import progress_index
                a, p = progress_index(d.graph), 1.
            elif selector == 'escape':
                from .heuristics import escape_index
                a, p = escape_index(d.graph), 1.
            elif selector == 'routesafe':
                from .heuristics import routesafe_index
                a, p = routesafe_index(d.graph, d.default), 1.
            elif selector == 'routesafe2':
                from .heuristics import routesafe_progress_index
                a, p = routesafe_progress_index(d.graph, d.default), 1.
            elif selector == 'restart':
                # control experiment: the floor's intent, but a FRESH goal query at every decision instead of continuing
                # the running one (the executor -- route, planner and recovery state -- restarts every interval)
                a, p = (d.labels.index('goal') if 'goal' in d.labels else d.default), 1.
            elif selector == 'imagroll':
                # legal imagined rollout policy (heuristics.imagined_rollout_index; needs --sandbox continuation)
                from .heuristics import imagined_rollout_index
                a, p = imagined_rollout_index(d.result, ep.ep.legal, d.default), 1.
            elif selector == 'routesafe3':
                from .heuristics import routesafe_bank_index
                a, p = routesafe_bank_index(d.graph, d.default), 1.
            elif selector == 'oracle':
                want = oracle_seq[ep.decisions] if ep.decisions < len(oracle_seq) else None
                if want in d.labels:
                    a, p = d.labels.index(want), 1.
                else:
                    a, p, diverged = d.default, 1., True
            else:
                raise ValueError(selector)
            d.timing['policy_ms'] = 1000*(time.perf_counter()-t)
            r = ep.act(a)
            d.timing['env_ms'] = r['env_ms']
            if record:
                rows['packed'].append(packed)
            rows['actions'].append(a); rows['probs'].append(p); rows['cum'].append(cumulants(r))
            rows['labels'].append(d.labels); rows['n'].append(n); rows['default'].append(d.default)
            rows['q'].append(None if qv is None else np.asarray(qv[:n], np.float32))
            rows['heads'].append(None if heads is None else {k: np.asarray(v[:n], np.float32) for k, v in heads.items()})
            rows['timing'].append(d.timing)
        m = ep.metrics()
        m['truncated'] = not ep.over
        m['diverged'] = diverged
        m['decision_list'] = _decisions_json(ep.decision_list())
        m['actions'] = [x if isinstance(x, str) else [float(v) for v in x] for x in ep.ep.state.actions]
        m['wall_s'] = time.perf_counter()-t0
        m['setup_s'] = ep.setup_s
        m['perf'] = {k: float(v) for k, v in ep.ep.perf.items()}
    finally:
        ep.close()
    return m, rows


def _decisions_json(decisions):
    out = []
    for d in decisions:
        q = d['query']
        out.append(dict(q.describe(), cont=d['cont'], start=d['start'], end=d['end'], horizon=d['horizon'],
                        status=d['status'], reason=d['reason'], outcome=d['outcome']))
    return out


def write_episode(spool, name, m, rows, extra):
    tmp = Path(spool)/('.%s.tmp' % name)
    data = dict(metrics=m, rows=rows, **extra)
    tmp.write_bytes(zlib.compress(pickle.dumps(data, protocol=4), 3))
    os.replace(tmp, Path(spool)/('%s.ep' % name))


def read_episode(path):
    return pickle.loads(zlib.decompress(Path(path).read_bytes()))


class Curriculum:
    """Configuration x episode sampler.  ``configs``: list of (ConfigSpec, weight).

    Scene-sticky: a worker draws a scene uniformly, then ``stick`` episodes of that scene (each with a fresh
    configuration draw) before moving on -- habitat keeps the scene's assets loaded, so a same-scene episode starts
    ~4x faster (measured 20 s -> 5 s), and the human tape of a repeated episode is cached."""

    def __init__(self, configs, episodes, seed, stick=3, weights=None):
        self.configs = [c for c, _ in configs]
        w = np.asarray([w for _, w in configs], np.float64)
        self.w = w/w.sum()
        self.by_scene = {}
        for i, e in enumerate(episodes):
            self.by_scene.setdefault((e.dataset, e.scene), []).append(i)
        self.episodes = episodes
        self.scenes = sorted(self.by_scene)
        self.rng = np.random.default_rng(seed)
        self.stick = int(stick)
        self.left = 0
        self.scene = None
        self.ew = weights            # curriculum.EpisodeWeights (None: uniform)

    def sample(self):
        c = self.configs[int(self.rng.choice(len(self.configs), p=self.w))]
        if self.ew is not None:
            self.ew.refresh()
        if self.left <= 0:
            if self.ew is not None:
                sw = np.asarray([self.ew.w[self.by_scene[s]].sum() for s in self.scenes])
                self.scene = self.scenes[int(self.rng.choice(len(self.scenes), p=sw/sw.sum()))]
            else:
                self.scene = self.scenes[int(self.rng.integers(len(self.scenes)))]
            self.left = self.stick
        self.left -= 1
        idx = self.by_scene[self.scene]
        if self.ew is not None:
            pw = self.ew.w[idx]
            return c, self.episodes[idx[int(self.rng.choice(len(idx), p=pw/pw.sum()))]]
        return c, self.episodes[idx[int(self.rng.integers(len(idx)))]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run directory (spool/, policy.pt, actor logs)')
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--server', default=None, help='sandbox server address (split mode); none = in-process sandbox')
    ap.add_argument('--configs', default='orca-native-k10:1')
    ap.add_argument('--suite', default='train')
    ap.add_argument('--episodes', default='', help='explicit comma list of suite indices (default: whole suite)')
    ap.add_argument('--selector', default='policy')
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--eps', type=float, default=None)
    ap.add_argument('--seconds', type=float, default=1e9)
    ap.add_argument('--max-episodes', type=int, default=10**9)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-record', action='store_true')
    ap.add_argument('--sequential', action='store_true', help='run the listed episodes x configs in order once (evaluation)')
    ap.add_argument('--oracle-map', default=None, help='JSON {key|config: [labels]} for --selector oracle (warm-up replays)')
    ap.add_argument('--jobs', default=None, help='JSON list of [suite_key, config] jobs (sequential); overrides suite x configs')
    ap.add_argument('--blind', action='store_true', help='no imagination: no Sandbox, no range frames (floors, no-WM ablation)')
    ap.add_argument('--spool', default='spool', help='output sub-directory (eval actors use eval_spool: never ingested)')
    ap.add_argument('--margin', type=float, default=0., help='greedy evaluation: deviate from the default only if Q gain > margin')
    ap.add_argument('--compiler', default='{}', help='GraphCompiler options (JSON), e.g. {"brake_tail": true}')
    ap.add_argument('--weights', default=None, help='evaluation: override the value scalarisation (JSON of model.Weights fields)')
    ap.add_argument('--near-interval', type=int, default=0, help='decision interval (steps) while a tracked person is within --near-m')
    ap.add_argument('--near-m', type=float, default=3.0)
    ap.add_argument('--sandbox', default='{}', help='SandboxConfig overrides (JSON), e.g. {"continuation": 24}')
    ap.add_argument('--scan', action='store_true', help='Layer 3 information-gathering queries (env.scan_queries)')
    ap.add_argument('--curriculum', default=None, help='episode-weight prior (curriculum.py JSON); the run\'s own logs update it')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    from eventwm_graph.sandbox.core import SandboxConfig
    from .env import ConfigSpec
    from . import suites
    run = Path(a.run)
    spool = run/a.spool; spool.mkdir(parents=True, exist_ok=True)
    logf = open(run/('%s_%02d.log' % ('actor' if a.spool == 'spool' else a.spool, a.worker)), 'a')
    sbcfg = SandboxConfig(materialize=False, **json.loads(a.sandbox))
    if a.blind:
        sb = None
    elif a.server:
        from eventwm_graph.sandbox.service import RemoteSandbox
        sb = RemoteSandbox(a.server, sbcfg)
    else:
        from eventwm_graph.sandbox.core import WorldModelSandbox
        sb = WorldModelSandbox.from_release(sbcfg, map_device='cuda:0')
    configs = []
    for item in a.configs.split(','):
        name, _, w = item.partition(':')
        configs.append((ConfigSpec.parse(name), float(w or 1)))
    specs = suites.load(a.suite) if a.suite in ('train', 'train_all', 'train_querygen', 'dev', 'test', 'test_full') else None
    if specs is None:
        from eventwm_graph.querygen import suites as qs
        specs = qs.load(a.suite)
    if a.episodes:
        specs = [specs[int(i)] for i in a.episodes.split(',')]
    policy = PolicyHandle(run/'policy.pt' if a.selector == 'policy' else None, json.loads(a.weights) if a.weights else None)
    from .compiler import GraphCompiler
    compiler = GraphCompiler(**json.loads(a.compiler))
    rng = np.random.default_rng(1000+a.worker+7919*a.seed)
    ew = None
    if a.curriculum and not a.sequential and a.spool == 'spool':      # training actors only (evaluation stays uniform)
        from .curriculum import EpisodeWeights
        ew = EpisodeWeights(specs, prior=a.curriculum if a.curriculum != 'none' else None, run=run)
    cur = Curriculum(configs, specs, 17+a.worker+7919*a.seed, weights=ew, stick=6 if ew is not None else 3)
    t_start = time.time()
    n_ep = 0
    jobs = None
    oracle_map = json.loads(Path(a.oracle_map).read_text()) if a.oracle_map else {}
    if a.sequential:
        jobs = [(c, s) for s in specs for c, _ in configs]
        if a.jobs:
            from eventwm_graph.querygen.native import EpisodeSpec
            jobs = [(ConfigSpec.parse(c), EpisodeSpec(**sp)) for sp, c in json.loads(Path(a.jobs).read_text())]
        jobs = [j for k, j in enumerate(jobs) if k % int(os.environ.get('L3_NWORKERS', '1')) == a.worker]
    while time.time()-t_start < a.seconds and n_ep < a.max_episodes:
        if jobs is not None:
            if n_ep >= len(jobs):
                break
            cfg, spec = jobs[n_ep]
        else:
            cfg, spec = cur.sample()
        if a.selector == 'policy':
            policy.refresh()
            if policy.net is None:
                time.sleep(5)
                continue
            tau = a.tau if a.tau is not None else float(policy.meta.get('tau') or 0.5)
            eps = a.eps if a.eps is not None else float(policy.meta.get('epsilon') or 0.03)
            anchor = 0. if (a.tau is not None and a.tau <= 0) else float(policy.meta.get('anchor') or 0.)
        else:
            tau, eps, anchor = 0., 0., 0.
        name = ('%s__%s' % (spec.key.replace('/', '_'), cfg.name) if a.sequential else
                '%s__%s__w%02d_s%d_%06d' % (spec.key.replace('/', '_'), cfg.name, a.worker, a.seed, n_ep))
        try:
            if (spool/('%s.ep' % name)).exists():
                n_ep += 1
                continue
            seq = oracle_map.get('%s|%s' % (spec.key, cfg.name)) if a.selector == 'oracle' else None
            m, rows = run_episode(spec, cfg, sb, a.selector, policy, rng, tau=tau, eps=eps, record=not a.no_record,
                                  oracle_seq=seq, imagination=not a.blind, margin=a.margin, compiler=compiler,
                                  anchor=anchor, near_interval=a.near_interval, near_m=a.near_m, scan=a.scan,
                                  explore=policy.meta.get('explore') if a.selector == 'policy' and tau > 0 else None)
            extra = dict(spec=spec.asdict(), config=cfg.name, selector=a.selector, version=policy.version, tau=tau,
                         eps=eps, anchor=anchor, worker=a.worker, compiler=compiler.schema, scan=bool(a.scan))
            write_episode(spool, name, m, rows if not a.no_record else {k: v for k, v in rows.items() if k != 'packed'}, extra)
            logf.write(json.dumps(dict(t=time.time(), name=name, outcome=m['outcome'], decisions=m['decisions'],
                                       steps=m['steps'], wall_s=round(m['wall_s'], 1), setup_s=round(m['setup_s'], 1),
                                       version=policy.version))+'\n')
            logf.flush()
        except Exception as exc:
            logf.write(json.dumps(dict(t=time.time(), name=name, error=repr(exc), tb=traceback.format_exc()[-2000:]))+'\n')
            logf.flush()
            if 'CUDA' in repr(exc) or 'cuda' in repr(exc):
                raise
        n_ep += 1
    logf.close()


if __name__ == '__main__':
    main()
