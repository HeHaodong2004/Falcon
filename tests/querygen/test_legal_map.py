"""GridMap.apply must reproduce partial.map.PartialGraph.update exactly."""
import numpy as np
from eventwm_graph.querygen.native import EpisodeSpec, NativeTask
from eventwm_graph.querygen.legal import LegalSensor, GridMap, RESOLUTION
from eventwm_graph.partial.map import PartialGraph
from tests.querygen.native_equivalence import random_actions

ok = True
for key, seed in [('hm3d/val/3t8DB4Uzvkt/1', 5), ('hm3d/val/5cdEh9F2hJL/3', 6)]:
    ds, sp, sc, ep = key.split('/')
    t = NativeTask(EpisodeSpec(ds, sp, sc, ep, seed))
    sens = LegalSensor(t)
    row = sens.observe(t.world_time())
    g = GridMap(RESOLUTION, center=(row['robot_pose'][0], row['robot_pose'][2]))
    pg = PartialGraph(RESOLUTION, row['robot_pose'][1])
    revs = []
    for k, a in enumerate([None]+random_actions(np.random.default_rng(seed), 150)):
        if a is not None:
            if t.episode_over:
                break
            t.step(a)
            row = sens.observe(t.world_time())
        g.apply(row); d = pg.update(row)
        revs.append((g.revision, pg.revision))
    cells = {k: (v['state'], v['height']) for k, v in pg.cells.items()}
    mine = {}
    for i, j in np.argwhere(g.state != 0):
        mine[(int(i+g.ox), int(j+g.oz))] = (int(g.state[i, j]), float(g.height[i, j]))
    same = cells == mine and all(a == b for a, b in revs)
    print(key, 'cells', len(cells), 'equal', cells == mine, 'revisions equal', all(a == b for a, b in revs), 'final rev', revs[-1])
    ok &= same
    t.close()
print('LEGAL_MAP_EQUAL' if ok else 'LEGAL_MAP_MISMATCH')
