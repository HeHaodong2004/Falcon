"""Bank imagination dataset tests (``eventwm_graph/l3/bankimag.py``) on banked decision states (``statebank.py``).

    L3_INHERIT=1 python scripts/l3_run.py tests/l3/test_bankimag.py [BANK_DIR] [N_STATES]

* the TRUE-people adapter holds exactly the banked truth, and the CV people are ``CVBank``'s;
* the vectorised geometry equals ``sandbox.imagine.rectangle_clearance`` per person;
* the post-hoc camera view equals the detections of ``perceive.VisibleWorld`` stepped along the same poses;
* imagining a state never mutates its legal state, and is deterministic;
* the harness reproduces the online probe's ``camT`` records exactly (when the bank holds them).
"""
import copy
import glob
import math
import pickle
import sys

import numpy as np

from eventwm_graph.l3 import bankimag as B


def load(d, n):
    paths = sorted(glob.glob(d+'/*.pkl'))
    assert paths, 'no banked states in %s' % d
    step = max(1, len(paths)//n)
    out = []
    for p in paths[::step][:n]:
        item = pickle.load(open(p, 'rb'))
        item['_file'] = p.split('/')[-1]
        out.append(item)
    return out


def test_truth_and_cv_people(items):
    for item in items:
        T = np.asarray(item['truth_people'], np.float64)
        w = B.people_world(item['legal'], B.truth_bank(item))
        assert w.P.shape == (B.W+1, T.shape[1], 2)
        assert np.array_equal(w.P, T[:B.W+1])
        cv = B.cv_bank(item['legal'])
        wc = B.people_world(item['legal'], cv)
        rel = np.arange(1, B.W+1)*0.048
        if wc.P.shape[1]:
            assert np.allclose(wc.P[1:], cv.sandbox_positions(rel)[0])


def test_geometry_matches_rectangle_clearance(items):
    from eventwm_graph.sandbox.imagine import rectangle_clearance
    rng = np.random.default_rng(0)
    poses = np.c_[rng.uniform(-3, 3, 50), np.zeros(50), rng.uniform(-3, 3, 50), rng.uniform(-math.pi, math.pi, 50)]
    people = rng.uniform(-4, 4, (50, 5, 2))
    rc = B.rect_clearances(poses, people)
    cd = B.centre_distances(poses, people)
    for i in range(50):
        for p in range(5):
            assert abs(rc[i, p]-rectangle_clearance(poses[i], people[i, p:p+1])) < 1e-12
            assert abs(cd[i, p]-math.hypot(people[i, p, 0]-poses[i, 0], people[i, p, 1]-poses[i, 2])) < 1e-12


def test_camera_view_matches_visible_world(items):
    sb = B._sandbox()
    for item in items:
        r = B.imagine_state(sb, item, variants=('b',))
        legal = item['legal']
        T = np.asarray(item['truth_people'], np.float64)
        w = B.people_world(legal, B.truth_bank(item))
        ids = list(w.ids)
        for k in range(len(r['labels'])):
            n = int(r['b_n'][k])
            P = np.concatenate([np.asarray(legal.pose, np.float64)[None], r['b_poses'][k, :n].astype(np.float64)])
            vis = B.camera_view(legal, P, T[:n+1])
            for j in range(n+1):
                seen = {t['track_id'] for t in w.tracks_at(j, P[j]) if t['visible_now']}
                assert seen == {ids[p] for p in np.flatnonzero(vis[j])}, (item['_file'], k, j)
            assert np.array_equal(vis, r['b_vis'][k, :n+1])
    sb.close()


def test_legal_state_untouched_and_deterministic(items):
    sb = B._sandbox()
    for item in items[:3]:
        legal = item['legal']
        before = (legal.map.state.copy(), legal.map.revision, np.asarray(legal.pose).copy(), copy.deepcopy(legal.tracks),
                  list(legal.bumps), len(legal.poses), item['execution'].copy() if item['execution'] is not None else None)
        r1 = B.imagine_state(sb, item)
        r2 = B.imagine_state(sb, item)
        assert np.array_equal(legal.map.state, before[0]) and legal.map.revision == before[1]
        assert np.array_equal(legal.pose, before[2]) and legal.tracks == before[3]
        assert list(legal.bumps) == before[4] and len(legal.poses) == before[5]
        for k, v in r1.items():
            if k in ('seconds', 'a_ms', 'b_ms'):
                continue
            if isinstance(v, np.ndarray) and v.dtype.kind == 'f':
                assert np.array_equal(v, r2[k], equal_nan=True), k
            elif isinstance(v, np.ndarray):
                assert np.array_equal(v, r2[k]), k
            else:
                assert v == r2[k], k
    sb.close()


def test_reproduces_online_camT(items):
    have = [it for it in items if 'camT' in it['probe']['cands'][0]['imag']]
    if not have:
        print('   (no camT records in this bank: skipped)')
        return
    sb = B._sandbox()
    for item in have[:3]:
        r = B.imagine_state(sb, item, variants=('b',), physics='')
        for k, c in enumerate(item['probe']['cands']):
            on = c['imag']['camT'][0]
            n = int(r['b_n'][k])
            assert n == int(on['n']) and bool(r['b_stop'][k]) == bool(on['stop']) and r['b_ended'][k] == on['ended']
            for w in B.WINDOWS:
                mine = B._minw(r['b_rc_true'][k:k+1], min(w, n))[0] if n else math.inf
                theirs = math.inf if on['clr'][str(w)] is None else float(on['clr'][str(w)])
                assert (mine == theirs) or abs(mine-theirs) < 1e-4, (item['_file'], c['label'], w, mine, theirs)
    sb.close()


if __name__ == '__main__':
    import os
    from eventwm_graph.querygen.executor import QueryExecution
    assert QueryExecution.INHERIT, 'run with L3_INHERIT=1 (the banked outcomes were labelled with it)'
    os.environ.setdefault('QG_FAST_SENSE', '1')
    d = sys.argv[1] if len(sys.argv) > 1 else 'output/statebank/orca_dev'
    items = load(d, int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn(items)
            print('PASS', name, flush=True)
