"""Calibrated official-contact model (l3/contactmodel.py): features, monotone fit, model file, window choice.

    python scripts/l3_run.py tests/l3/test_contactmodel.py
"""
import json
import math
import os
import tempfile

import numpy as np

from eventwm_graph.l3 import contactmodel as cmod


def _line(n=32, speed=0.1, yaw=0.):
    """Robot moving along world +x (yaw 0: forward = (cos yaw, -sin yaw) = +x)."""
    x = np.arange(1, n+1)*speed
    return np.stack([x, np.zeros(n), np.zeros(n), np.full(n, yaw)], 1)


def test_features_without_people():
    seq = _line()
    for P in (np.zeros((32, 0, 2)), np.full((32, 2, 2), np.nan)):
        f = cmod.geometry_features(seq, P)
        assert f['none'] == 1. and f['rc_min'] == cmod.RC_CAP and f['cd_min'] == cmod.CD_CAP and f['ex'] == 0.
    f = cmod.geometry_features(seq[:0], np.zeros((0, 1, 2)))
    assert f['none'] == 1.


def test_features_geometry():
    seq = _line()
    P = np.zeros((32, 1, 2))
    P[:, 0, 0] = 1.6                      # a person standing on the path, 1.6 m ahead of the start
    f = cmod.geometry_features(seq, P, pose0=np.zeros(4), full=True)
    assert f['none'] == 0. and f['rc_min'] == 0.          # the centre enters the body rectangle
    assert abs(f['cd_min']) < 0.051
    assert 0 <= f['i_ca'] < 32 and f['ex'] > 0.
    assert f['v_rob'] > 1.5 and f['cos_b'] > 0.9          # approached head-on, person ahead
    # rectangle clearance agrees with sandbox.imagine.rectangle_clearance
    from eventwm_graph.sandbox.imagine import rectangle_clearance
    P2 = np.zeros((32, 1, 2)); P2[:, 0, 0] = 1.0; P2[:, 0, 1] = 1.2
    f2 = cmod.geometry_features(seq, P2)
    ref = min(rectangle_clearance(seq[i], P2[i]) for i in range(32))
    assert abs(f2['rc_min']-ref) < 1e-9


def test_monotone_fit_and_probability():
    rng = np.random.default_rng(0)
    n = 3000
    rc = rng.uniform(0., 2., n)
    cd = rc+rng.uniform(0.2, 0.8, n)
    ex = np.exp(-rc/cmod.TAU)*rng.uniform(0.05, 0.5, n)
    p = 1./(1.+np.exp(-(2.-8.*rc)))
    y = rng.random(n) < p
    F = dict(rc_min=rc, cd_min=cd, ex=ex)
    m = cmod.fit_logistic(cmod.design(F), y, 1.)
    for s, c in zip(cmod.SIGNS, m['coef']):
        assert s*c >= -1e-12, (cmod.SIGNS, m['coef'])
    # model file round trip, window choice, probability decreasing with the lateral offset of a passing person
    doc = dict(version=cmod.VERSION, kind='logistic', features=list(cmod.FEATURES),
               windows={str(w): dict(m, threshold=0.2+0.01*i) for i, w in enumerate(cmod.WINDOWS)})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'm.json')
        with open(path, 'w') as fh:
            json.dump(doc, fh)
        cm = cmod.ContactModel(path)
    assert cm.windows == (16, 32, 64)
    assert cm.window(32) == 32 and cm.window(20) == 32 and cm.window(10) == 16 and cm.window(76) == 64
    assert cm.window(76, '16') == 16 and math.isclose(cm.threshold(32), 0.21)
    seq = _line()
    probs = []
    for off in (0., 0.3, 0.6, 1.0, 2.0):
        P = np.zeros((32, 1, 2)); P[:, 0, 0] = 1.6; P[:, 0, 1] = off
        probs.append(cm.prob(cm.features(seq, P), 32))
    assert all(a >= b-1e-12 for a, b in zip(probs, probs[1:])), probs
    assert probs[0] > 0.5 > probs[-1]
    many = cm.prob_many(dict(rc_min=np.array([0., 3.]), cd_min=np.array([0.1, 4.]), ex=np.array([1., 0.])), 32)
    assert many[0] > many[1]


def test_rejects_other_features():
    doc = dict(version=1, kind='logistic', features=['x'], windows={})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'm.json')
        with open(path, 'w') as fh:
            json.dump(doc, fh)
        try:
            cmod.ContactModel(path)
        except ValueError:
            return
    raise AssertionError('a model with other features must be rejected')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name, flush=True)
