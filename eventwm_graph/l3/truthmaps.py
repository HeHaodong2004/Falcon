"""ANALYSIS ONLY (privileged): per-episode "truth" maps for the progressive-discovery imagination.

Each episode of an evaluation spool (the floor by default) is replayed in querygen's exact branch simulator along its
recorded decisions to the end, and the legal map the robot then holds (everything its sensor certified along the
floor's path) is saved.  The multi-world selector can discover cells from it during imagination
(``L3_MW_TRUTHMAPS=<dir>`` with ``L3_MW_PERCEIVE=1``): an upper-bound proxy for predicted geometry, privileged because
it contains observations the robot has not made yet.

    L3_INHERIT=1 python scripts/l3_run.py eventwm_graph.l3.truthmaps --dir output/eval/floor_dev24/spool \
        --configs orca-native-k10 --out output/truthmaps [--limit 72 --shard i --shards n]
"""
import argparse
import os
from pathlib import Path

import numpy as np


def path_for(out, key, config):
    return Path(out)/('%s__%s.npz' % (key.replace('/', '_'), config))


def save_map(path, g):
    np.savez_compressed(path, state=g.state, height=g.height, origin=np.array([g.ox, g.oz]), res=np.array([g.res]))


def load_map(path):
    from eventwm_graph.querygen.legal import GridMap
    d = np.load(path)
    g = GridMap.__new__(GridMap)
    g.res = float(d['res'][0])
    g.ox, g.oz = int(d['origin'][0]), int(d['origin'][1])
    g.state, g.height = d['state'].copy(), d['height'].copy()
    g.first_seen = np.full(g.state.shape, np.nan, np.float32)
    g.last_seen = np.full(g.state.shape, np.nan, np.float32)
    g.revision, g.timestamp, g.base_height = 0, None, None
    return g


def merged_truth(legal_map, truth):
    """The legal map at t with the truth map's known cells written into its UNKNOWN cells (hypothetical)."""
    g = legal_map.copy()
    ii, jj = np.nonzero(truth.state != 0)
    if not len(ii):
        return g
    keys = np.stack([ii+truth.ox, jj+truth.oz], 1)
    g._ensure(keys)
    a, b = keys[:, 0]-g.ox, keys[:, 1]-g.oz
    unknown = g.state[a, b] == 0
    g.state[a[unknown], b[unknown]] = truth.state[ii[unknown], jj[unknown]]
    g.height[a[unknown], b[unknown]] = truth.height[ii[unknown], jj[unknown]]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='output/eval/floor_dev24/spool')
    ap.add_argument('--out', default='output/truthmaps')
    ap.add_argument('--configs', default='orca-native-k10')
    ap.add_argument('--limit', type=int, default=72)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    os.environ.setdefault('QG_FAST_SENSE', '1')
    from eventwm_graph.querygen import native
    native._install()
    from eventwm_graph.querygen.native import EpisodeSpec
    from eventwm_graph.querygen.episode import EpisodeRuntime
    from eventwm_graph.querygen.dynamics import FAMILIES
    from eventwm_graph.sandbox.episode import generator
    from .actor import read_episode
    from .env import ConfigSpec
    from .cfonline import _query
    from . import sensorcache
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keep = set(a.configs.split(','))
    files = [p for p in sorted(Path(a.dir).glob('*.ep')) if read_episode(p)['config'] in keep][:a.limit]
    for i, p in enumerate(files):
        if i % a.shards != a.shard:
            continue
        d = read_episode(p)
        spec = EpisodeSpec(**d['spec'])
        target = path_for(out, spec.key, d['config'])
        if target.exists():
            continue
        cfg = ConfigSpec.parse(d['config'])
        generator(cfg.gen, cfg.dynamics)            # process-wide executor options before the runtime
        rt = EpisodeRuntime(spec, cfg.planner, FAMILIES[cfg.dynamics], strict=False)
        rt.sim.check_static = False
        sensorcache.attach(rt)
        from eventwm_graph.sandbox.episode import default_query
        goal = default_query()
        st = rt.root
        try:
            for dec in d['metrics']['decision_list']:
                if st.terminal:
                    break
                cont = bool(dec['cont'])
                if cont and (st.execution is None or st.execution.status != 'running'):
                    cont = False                      # replay drifted: keep exploring with the goal query
                    q = goal
                else:
                    q = None if cont else _query(dec)
                try:
                    st = rt.run(st, q, horizon=int(dec['horizon']), cont=cont, geodesic=False)
                except Exception:
                    break
            save_map(target, st.legal.map)
            print('%s known cells %d' % (target.name, int((st.legal.map.state != 0).sum())), flush=True)
        finally:
            try:
                sensorcache.save(rt)
            except Exception:
                pass
            rt.close()


if __name__ == '__main__':
    main()
