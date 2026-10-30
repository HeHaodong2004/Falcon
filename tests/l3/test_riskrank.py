"""Unit tests of the learned listwise ranker option (l3/riskrank.py, l3/riskfit.py; selector option L3_MW_RISK)."""
import os

import numpy as np

from eventwm_graph.l3 import riskrank, riskfit


def test_relevance_grades():
    keys = [(0., 12), (2., -10.0), (2., -10.2), (2., -11.5), (3., -40)]
    rel = riskrank.relevance(keys, bin_m=0.25, levels=4)
    assert rel[0] == 0                       # contact
    assert rel[4] == 6                       # goal reached
    assert rel[1] == 5 and rel[2] == 5       # best running and within 0.25 m of it
    assert rel[3] == 1                       # 1.5 m behind the best running candidate
    assert riskrank.relevance([(0., 3), (0., 9)]) == [0, 0]


def test_selector_margin():
    s = np.array([0.1, 0.5, 0.3])
    assert riskrank.select_from_scores(s, 0, 0.5) == 0          # margin 0.4 <= 0.5: keep the default
    assert riskrank.select_from_scores(s, 0, 0.3) == 1
    assert riskrank.select_from_scores(s, 1, 0.0) == 1          # the default is the argmax
    assert riskfit.choose(s, 2, 0.19) == 1 and riskfit.choose(s, 2, 0.2) == 2


def test_call_env_restores():
    os.environ['L3_MW_GOAL_R'] = '0.19'
    os.environ.pop('L3_MW_PERCEIVE', None)
    with riskrank.call_env():
        assert os.environ.get('L3_MW_GOAL_R') is None
        assert os.environ['L3_MW_PERCEIVE'] == '1' and os.environ['L3_MW_DISCOVER'] == '0'
        assert riskrank.env_matches()
    assert os.environ['L3_MW_GOAL_R'] == '0.19' and 'L3_MW_PERCEIVE' not in os.environ
    assert not riskrank.env_matches()
    del os.environ['L3_MW_GOAL_R']


def test_exact_value_order():
    v = lambda k: riskfit.exact_value(k, 10.0, 64)
    assert v((0., 60)) < v((2., -10.0)) < v((2., -9.0)) < v((3., -50)) < v((3., -10))


def test_within_auc():
    recs = [dict(keys={32: [(0., 5), (2., -3.), (2., -4.)]}), dict(keys={32: [(2., -3.), (2., -4.)]})]
    auc, n = riskfit.within_auc(recs, [np.array([2., 1., 0.]), np.array([0., 1.])], 32)
    assert n == 1 and auc == 1.0


def test_numpy_trees():
    # one tree: categorical split on feature 0 ({1, 3} left), then a numerical split on feature 1 (NaN -> default left)
    dump = dict(tree_info=[dict(tree_structure=dict(
        split_feature=0, decision_type='==', threshold='1||3', default_left=False, missing_type='None',
        left_child=dict(split_feature=1, decision_type='<=', threshold=0.5, default_left=True, missing_type='NaN',
                        left_child=dict(leaf_value=1.0), right_child=dict(leaf_value=2.0)),
        right_child=dict(leaf_value=-1.0)))])
    t = riskrank.NumpyTrees(dump)
    X = np.array([[1, 0.2], [3, 0.9], [3, np.nan], [2, 0.1], [-1, 0.1], [np.nan, 0.1]], float)
    assert t.predict(X).tolist() == [1.0, 2.0, 1.0, -1.0, -1.0, -1.0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name, flush=True)
