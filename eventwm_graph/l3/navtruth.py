"""ANALYSIS ONLY (privileged): complete per-episode truth maps from the scene navmesh, for the SR ceiling of map
discovery inside imagination (``L3_MW_TRUTHMAPS=<dir>`` with ``L3_MW_PERCEIVE=1``; heuristics._truth_rollouts).

``truthmaps.py`` saves the floor run's final legal map, which holds only what the floor robot happened to see.  Here
every navigable cell of the robot's floor (navmesh top-down view at the base height) is FREE, and every non-navigable
cell within ``--band`` m of a navigable one is BLOCKED (the walls and furniture the legal observer's inflated wall
band would certify).  Everything else stays UNKNOWN.  Privileged geometry: never a runtime input.

    python scripts/l3_run.py eventwm_graph.l3.navtruth --suite dev --configs orca-native-k10 --out output/navtruth \
        [--limit 72 --shard i --shards n]
"""
import argparse
import json
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np


def nav_truth(rt, band=0.5):
    """GridMap (legal resolution and key space) of the scene's navmesh around the robot's floor."""
    from scipy import ndimage
    from eventwm_graph.querygen.legal import GridMap, RESOLUTION, FREE, BLOCKED
    task = rt.task
    pf = task.sim.pathfinder
    res = float(RESOLUTION)
    y = float(task.robot.base_pos[1])
    nav = np.asarray(pf.get_topdown_view(res, y), dtype=bool)          # [z, x]
    lb = pf.get_bounds()[0]
    kx0 = math.floor(float(lb[0])/res+0.5)                                # key of top-down column 0 (cell centre)
    kz0 = math.floor(float(lb[2])/res+0.5)
    nav_xz = nav.T                                                       # [x, z] as GridMap.state
    near = ndimage.binary_dilation(nav_xz, iterations=max(1, int(round(band/res))))
    g = GridMap.__new__(GridMap)
    g.res, g.ox, g.oz = res, kx0, kz0
    g.state = np.zeros(nav_xz.shape, np.uint8)
    g.state[nav_xz] = FREE
    g.state[near & ~nav_xz] = BLOCKED
    g.height = np.zeros(nav_xz.shape, np.float32)
    g.first_seen = np.full(nav_xz.shape, np.nan, np.float32)
    g.last_seen = np.full(nav_xz.shape, np.nan, np.float32)
    g.revision, g.timestamp, g.base_height = 0, None, None
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='dev')
    ap.add_argument('--configs', default='orca-native-k10')
    ap.add_argument('--out', default='output/navtruth')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--band', type=float, default=0.5)
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
    from . import suites
    from .env import ConfigSpec
    from .truthmaps import path_for, save_map
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    specs = suites.load(a.suite)
    if a.limit:
        specs = specs[:a.limit]
    for c in a.configs.split(','):
        cfg = ConfigSpec.parse(c)
        for s in specs:
            if zlib.crc32(s.key.encode()) % a.shards != a.shard:
                continue
            target = path_for(out, s.key, cfg.name)
            if target.exists():
                continue
            t0 = time.perf_counter()
            try:
                generator(cfg.gen, cfg.dynamics)
                rt = EpisodeRuntime(EpisodeSpec(**s.asdict()), cfg.planner, FAMILIES[cfg.dynamics], strict=False)
                try:
                    g = nav_truth(rt, a.band)
                finally:
                    rt.close()
                save_map(target, g)
                print(json.dumps(dict(key=s.key, config=cfg.name, free=int((g.state == 1).sum()),
                                      blocked=int((g.state == 2).sum()), seconds=round(time.perf_counter()-t0, 1))),
                      flush=True)
            except Exception as exc:
                print(json.dumps(dict(key=s.key, config=cfg.name, error=repr(exc))), flush=True)


if __name__ == '__main__':
    main()
