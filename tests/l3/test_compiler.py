"""Graph Compiler tests on captured real decision contexts (``eventwm_graph/l3/capture.py``).

    python scripts/l3_run.py tests/l3/test_compiler.py [CAPTURE_DIR]

* layout: every node / edge array matches the declared feature names; values finite;
* agnosticism: renaming the planner / dynamics of every candidate descriptor (identity labels) and the generator's
  execution-style label suffixes changes nothing in the compiled graph;
* hypothetical inputs stay hypothetical: compiling never mutates the legal state (map arrays, pose, tracks);
* the no-imagination compile has the same layout and no imagination-derived values;
* packing keeps the N nearest humans and all candidates.
"""
import copy
import dataclasses
import glob
import pickle
import sys

import numpy as np

from eventwm_graph.sandbox import service  # noqa: F401  (transport reducers for the pickled captures)
from eventwm_graph.l3 import compiler as C
from eventwm_graph.l3.batch import pack, KMAX, NMAX


def load(d):
    out = []
    for f in sorted(glob.glob(d+'/*.pkl')):
        out.append(pickle.loads(open(f, 'rb').read()))
    assert out, 'no captures in %s' % d
    return out


def check_layout(g):
    K = len(g['cand'])
    assert g['ego'].shape == (len(C.EGO),)
    assert g['cand'].shape == (K, len(C.CAND))
    assert g['traj'].shape == (K, C.P, len(C.TRAJ))
    N = len(g['hum'])
    assert g['hum'].shape == (N, len(C.HUM))
    assert g['hfut'].shape == (N, C.P, len(C.HFUT))
    assert g['pair'].shape == (K, C.P, N, len(C.PAIR))
    for k, v in g.items():
        assert np.isfinite(v).all(), k


def test_layout_and_finite(recs):
    for r in recs:
        check_layout(C.compile_record(r))


def test_identity_free(recs):
    for r in recs[:6]:
        g0 = C.compile_record(r)
        r2 = copy.copy(r)
        res = r['result']
        cands = []
        for c in res.candidates:
            d = dict(c.candidate, planner='renamed_planner', dynamics='renamed_dynamics', query_id='x')
            lab = str(d.get('label') or '')
            for suf in ('_th', '_tc', '_vc'):
                lab = lab.replace(suf, '')
            d['label'] = lab if d.get('kind') not in ('goal',) or not lab.startswith('goal') else 'goal'
            cands.append(dataclasses.replace(c, candidate=d))
        r2['result'] = dataclasses.replace(res, candidates=tuple(cands))
        g1 = C.compile_record(r2)
        for k in g0:
            assert np.array_equal(g0[k], g1[k]), k


def test_legal_state_untouched(recs):
    for r in recs[:4]:
        leg = r['legal']
        before = (leg.map.state.copy(), np.asarray(leg.pose).copy(), copy.deepcopy(leg.tracks), len(leg.poses))
        C.compile_record(r)
        assert np.array_equal(before[0], leg.map.state)
        assert np.array_equal(before[1], np.asarray(leg.pose))
        assert before[2] == leg.tracks and before[3] == len(leg.poses)


def test_blind(recs):
    for r in recs[:4]:
        g = C.compile_blind(r['legal'], r['queries'], r.get('diag'), r.get('steps', 0))
        check_layout(g)
        im = [C.CAND.index(c) for grp in C.IMAGINATION_GROUPS for c in C.GROUPS[grp].get('cand', [])]
        assert not g['cand'][:, im].any()
        assert not g['traj'].any() and not g['pair'].any()


def test_pack(recs):
    for r in recs[:4]:
        g = C.compile_record(r)
        p = pack(g)
        assert p['cand_mask'].sum() == min(len(g['cand']), KMAX)
        assert p['hum_mask'].sum() == min(len(g['hum']), NMAX)
        assert np.allclose(p['cand'][:len(g['cand'])].astype(np.float32), g['cand'][:KMAX], atol=2e-2, rtol=2e-3)


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else 'output/captures/c1'
    recs = load(d)
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn(recs)
            print('PASS', name, flush=True)
