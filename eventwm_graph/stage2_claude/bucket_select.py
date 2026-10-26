"""Diagnostic (TRAIN validation, CPU): would duration-dependent class thresholds
beat one global operating point? Two-fold cross-validation over Roots."""
import sys
import numpy as np
from .data import split_roots, load_root
from .select import cached_collect, class_map, RB, BB
from eventwm_graph.stage2_segment.evaluate import rates

ckpt = sys.argv[1]
records = cached_collect(ckpt, 'cuda:0', 400)
_, val = split_roots()
val = sorted(val, key=lambda r: r['root_id'])[:400]
EDGES = (.5, 1., 2.)
buckets = []
for row, (scene, root) in zip(val, records):
    d = [w['path'][-1, 0]-w['path'][0, 0] for g in load_root(row) for w in g['windows']]
    assert len(d) == len(root)
    buckets.append(np.searchsorted(EDGES, d))
NB = len(EDGES)+1
grid = [(tf, tb) for tf in np.arange(.25, .65, .025) for tb in np.arange(.3, .7, .025)]
maps = [np.stack([class_map('class', tf, tb) == k for k in range(3)], -1).astype(np.float64) for tf, tb in grid]


def window_cms(idx):
    """Per window, per grid point confusion (G, 3, 3), grouped by (scene, root)."""
    out = []
    for i in idx:
        scene, root = records[i]
        cms = []
        for (h, m), bk in zip(root, buckets[i]):
            c = np.stack([np.einsum('trb,rbk->tk', h, mp) for mp in maps])  # (G,3,3)
            c[:, 1, 0] += m
            cms.append((bk, c))
        out.append((scene, cms))
    return out


def hier(data, choice):
    scenes = {}
    for scene, cms in data:
        if not cms:
            continue
        rc = np.mean([c[choice[bk]] for bk, c in cms], 0)
        scenes.setdefault(scene, []).append(rc)
    m = np.mean([np.mean(v, 0) for v in scenes.values()], 0)
    return rates(m)['label_aware_delta_iou']


def fit(data):
    g0 = max(range(len(grid)), key=lambda j: hier(data, [j]*NB))
    choice = [g0]*NB
    for _ in range(3):
        for b in range(NB):
            best = max(range(len(grid)), key=lambda j: hier(data, choice[:b]+[j]+choice[b+1:]))
            choice[b] = best
    return g0, choice


rng = np.random.default_rng(0)
perm = rng.permutation(len(records))
folds = [perm[:len(perm)//2], perm[len(perm)//2:]]
for k in range(2):
    tr, te = window_cms(folds[k]), window_cms(folds[1-k])
    g0, choice = fit(tr)
    print('fold %d: global %s -> held-out %.4f | per-duration %s -> held-out %.4f' % (
        k, np.round(grid[g0], 3).tolist(), hier(te, [g0]*NB), [np.round(grid[c], 3).tolist() for c in choice], hier(te, choice)), flush=True)
