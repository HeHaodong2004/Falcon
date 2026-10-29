"""Capability ceiling of the query space per configuration (privileged Oracle; DIAGNOSTIC ONLY, never training data).

The v14 Oracle (expert budget: the evaluation portfolio capped at 1,500 edges) searches the frozen generator's
candidates for a configuration (planner x dynamics x candidate set) on Layer 3 dev episodes.  The share of solved
episodes bounds what any selector over that query space can reach on those episodes (up to the search budget).
Branch-simulator verdicts (the Env.step mirror); no official replay.

    python scripts/l3_run.py eventwm_graph.l3.oracle_ceiling --out output/oracle_ceiling --configs orca-inertia-k10 \
        --limit 24 [--shard i --shards n]
"""
import argparse
import json
import zlib
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--limit', type=int, default=24)
    ap.add_argument('--edges', type=int, default=1500)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a = ap.parse_args()
    from eventwm_graph.querygen import native
    from eventwm_graph.querygen.expert import run_episode, FAST_ORACLE
    from eventwm_graph.sandbox.episode import CONFIGS
    from .env import ConfigSpec
    from . import suites
    native._install()
    specs = suites.load(a.suite)[:a.limit]
    jobs = [(c, s) for s in specs for c in a.configs.split(',')]
    jobs = [j for j in jobs if zlib.crc32(('%s|%s' % (j[1].key, j[0])).encode()) % a.shards == a.shard]
    for cname, spec in jobs:
        cfg = ConfigSpec.parse(cname)
        out = Path(a.out)/cname
        out.mkdir(parents=True, exist_ok=True)
        if (out/('%s.json' % spec.key.replace('/', '_'))).exists():
            continue
        gen_kw = dict(CONFIGS[cfg.gen])
        for k in ('tc_variants', 'th_variants', 'vc_variants', 'xc_variants', 'kturns'):
            if k in gen_kw:
                gen_kw[k] = tuple(gen_kw[k])
        okw = dict(FAST_ORACLE, max_edges=a.edges)
        try:
            r = run_episode(spec, out, cfg.planner, cfg.dynamics, gen_kw, okw, features='none', verify='none')
            print(json.dumps(dict(config=cname, key=spec.key, verdict=r['verdict'], edges=r['search']['edges'],
                                  seconds=r['timing'].get('total'))), flush=True)
        except Exception as exc:
            print(json.dumps(dict(config=cname, key=spec.key, error=repr(exc))), flush=True)


if __name__ == '__main__':
    main()
