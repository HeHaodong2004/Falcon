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
import os
import pickle
import random
import time
import traceback
import zlib
from pathlib import Path

import numpy as np


def load_policy(path, device='cpu'):
    import torch
    from .model import ConsequenceGraphNet, ModelConfig, Weights
    ck = torch.load(path, map_location=device)
    cfg = ModelConfig(**ck['model_config'])
    net = ConsequenceGraphNet(cfg)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net, Weights(**ck['weights']), int(ck.get('version', 0)), ck


class PolicyHandle:
    """CPU policy with hot reload."""

    def __init__(self, path):
        self.path = Path(path) if path else None
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
                self.meta = {k: ck.get(k) for k in ('tau', 'epsilon', 'version')}
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


def choose(q, n, tau, eps, rng):
    """Boltzmann(tau) over the n valid candidates, mixed with epsilon-uniform.  Returns (index, prob of index)."""
    q = np.asarray(q[:n], np.float64)
    if tau <= 0:
        p = np.zeros(n); p[int(np.argmax(q))] = 1.
    else:
        z = (q-q.max())/tau
        p = np.exp(z); p /= p.sum()
    p = (1-eps)*p+eps/n
    a = int(rng.choice(n, p=p))
    return a, float(p[a])


def cumulants(r):
    """Decomposed per-decision cumulants (network units, batch.CUM order)."""
    from .batch import CUM_SCALE
    c = np.array([float(r['outcome'] == 'success'), float(r['outcome'] == 'human_collision'),
                  r['steps'], r['path'], r['social']], np.float32)
    return c/CUM_SCALE


def run_episode(spec, cfg, sb, selector, policy, rng, tau=0.5, eps=0.03, compiler=None, record=True, max_decisions=400,
                oracle_seq=None, imagination=True):
    from .env import L3Episode
    from .batch import pack
    t0 = time.perf_counter()
    ep = L3Episode(spec, cfg, sb if imagination else None, compiler=compiler, imagination=imagination)
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
                a, p = choose(qv, n, tau, eps, rng)
            elif selector == 'default':
                a, p = d.default, 1.
            elif selector == 'random':
                a, p = int(rng.integers(n)), 1./n
            elif selector == 'greedy':
                from .heuristics import greedy_index
                a, p = greedy_index(d.graph), 1.
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
    """Configuration x episode sampler.  ``configs``: list of (ConfigSpec, weight)."""

    def __init__(self, configs, episodes, seed):
        self.configs = [c for c, _ in configs]
        w = np.asarray([w for _, w in configs], np.float64)
        self.w = w/w.sum()
        self.episodes = episodes
        self.rng = np.random.default_rng(seed)

    def sample(self):
        c = self.configs[int(self.rng.choice(len(self.configs), p=self.w))]
        e = self.episodes[int(self.rng.integers(len(self.episodes)))]
        return c, e


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
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    from eventwm_graph.sandbox.core import SandboxConfig
    from .env import ConfigSpec
    from . import suites
    run = Path(a.run)
    spool = run/'spool'; spool.mkdir(parents=True, exist_ok=True)
    logf = open(run/('actor_%02d.log' % a.worker), 'a')
    sbcfg = SandboxConfig(materialize=False)
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
    specs = suites.load(a.suite) if a.suite in ('train', 'train_querygen', 'dev', 'test', 'test_full') else None
    if specs is None:
        from eventwm_graph.querygen import suites as qs
        specs = qs.load(a.suite)
    if a.episodes:
        specs = [specs[int(i)] for i in a.episodes.split(',')]
    policy = PolicyHandle(run/'policy.pt' if a.selector == 'policy' else None)
    rng = np.random.default_rng(1000+a.worker+7919*a.seed)
    cur = Curriculum(configs, specs, 17+a.worker+7919*a.seed)
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
        else:
            tau, eps = 0., 0.
        name = ('%s__%s' % (spec.key.replace('/', '_'), cfg.name) if a.sequential else
                '%s__%s__w%02d_s%d_%06d' % (spec.key.replace('/', '_'), cfg.name, a.worker, a.seed, n_ep))
        try:
            if (spool/('%s.ep' % name)).exists():
                n_ep += 1
                continue
            seq = oracle_map.get('%s|%s' % (spec.key, cfg.name)) if a.selector == 'oracle' else None
            m, rows = run_episode(spec, cfg, sb, a.selector, policy, rng, tau=tau, eps=eps, record=not a.no_record,
                                  oracle_seq=seq, imagination=not a.blind)
            extra = dict(spec=spec.asdict(), config=cfg.name, selector=a.selector, version=policy.version, tau=tau,
                         eps=eps, worker=a.worker)
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
