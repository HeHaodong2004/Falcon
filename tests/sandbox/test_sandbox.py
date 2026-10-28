"""World Model Sandbox Core tests: real Falcon episodes (unchanged habitat.Env), genuine Layer 2 inference (GPU).

    python scripts/sandbox_run.py tests/sandbox/test_sandbox.py            # all tests (~5 min)
    python scripts/sandbox_run.py tests/sandbox/test_sandbox.py memo       # tests whose name contains 'memo'
"""
import copy
import hashlib
import json
import math

import numpy as np

from eventwm_graph.querygen import suites
from eventwm_graph.querygen.dynamics import FAMILIES, STEP_SECONDS, CTRL_FREQ, dynamics_violations
from eventwm_graph.querygen.executor import Query
from eventwm_graph.layer2.state import DecisionState, LegalityError, RANGE_PROTOCOL
from eventwm_graph.sandbox.context import LegalStream, ContextError, layer2_frame, FRAME_KEYS, CompactUpdates
from eventwm_graph.sandbox.core import WorldModelSandbox, Candidate, SandboxConfig, SandboxError
from eventwm_graph.sandbox.episode import RealEpisode, generator, default_query
from eventwm_graph.sandbox.imagine import NominalHumans, advance, frozen_map
from eventwm_graph.sandbox import accel

EPISODE = ('dev', 7)          # hm3d/val/hkr2MGpHD6B: crowded spawn, up to 6 tracked humans
DECISIONS = 4
_CACHE = {}


def sandbox():
    if 'sb' not in _CACHE:
        _CACHE['sb'] = WorldModelSandbox.from_release(map_device='cuda:0')
    return _CACHE['sb']


def contexts():
    """Real decision contexts along the default policy (fast range sensor), with the raw rows and range frames."""
    if 'ctx' not in _CACHE:
        spec = suites.load(EPISODE[0])[EPISODE[1]]
        gen = generator('k10', 'native')
        ep = RealEpisode(spec, 'orca', 'native', range_sensor='fast')
        rows = []
        ep_rows = [(None, None)]          # the reset observation is already in the stream (not captured as a row)
        def keep(e, row, rf):
            rows.append((row, rf))
        q = default_query()
        out = []
        for d in range(DECISIONS):
            ep.run(q if ep.execution is None or ep.execution.status != 'running' else None, 16, on_step=keep)
            if ep.over:
                break
            state = ep.decision_state()
            queries, _ = gen.propose(ep.legal, ep.execution)
            out.append(dict(state=state, legal=ep.legal.copy(), execution=ep.execution.copy(), level=ep.level,
                            queries=queries, steps=ep.task.steps))
        _CACHE['ctx'] = dict(ep=ep, rows=rows, items=out, gen=gen)
    return _CACHE['ctx']


def candidates(item, planner='orca', dynamics='native', with_continue=True):
    out = []
    for q in item['queries']:
        if q.kind == 'continue':
            if with_continue:
                out.append(Candidate(None, planner, dynamics, execution=item['execution'], level=item['level'], query_id='continue'))
        else:
            out.append(Candidate(q, planner, dynamics, level=item['level'], query_id=q.label))
    return out


def rollouts_equal(a, b):
    return (np.array_equal(a.poses, b.poses) and np.array_equal(a.times, b.times) and a.commands == b.commands
            and a.status == b.status and a.reason == b.reason and a.ended_by == b.ended_by and np.array_equal(a.levels, b.levels))


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- legal context
def test_frames_have_layer2_format():
    c = contexts()
    st = c['items'][-1]['state']
    for f in st.frames:
        assert set(f) == set(FRAME_KEYS)
        su = f['static_updates']
        assert isinstance(su, (list, CompactUpdates)) and all(isinstance(u, dict) for u in list(su)[:5])
        assert len(su) == len(list(su))
        assert len(f['robot_pose']) == 4 and len(f['camera_origin']) == 3 and np.asarray(f['camera_rotation']).shape == (3, 3)
    row, _ = c['rows'][-1]
    fr = layer2_frame(row)
    assert list(fr['static_updates']) == list(row['static_updates'])  # UpdateBlock -> identical reference dicts
    assert fr['static_updates'] == list(row['static_updates']) and fr['static_updates'][3] == list(row['static_updates'])[3]
    assert layer2_frame(row, compact=False)['static_updates'] == list(row['static_updates'])
    assert fr['tracks'] is row['tracks']


def test_legal_graph_equals_reference_partial_graph():
    """The stream's graph == plain PartialGraph.update over the same rows (the function that built the dev data)."""
    from eventwm_graph.partial.map import PartialGraph
    c = contexts()
    ep = c['ep']
    frames = ep.stream.frames
    g = PartialGraph(ep.stream.resolution, float(frames[0]['robot_pose'][1]))
    for f in frames:
        g.update(f)
    ref = g.snapshot()
    got = ep.stream.graph_snapshot()
    assert json.dumps(ref, sort_keys=True) == json.dumps(got, sort_keys=True)
    # the executor's GridMap holds the same certificates (bump-memory discs aside)
    gm = ep.legal.map
    known = {tuple(cell['cell']): cell['state'] for cell in got['cells']}
    mism = 0
    for (kx, kz), s in known.items():
        mism += int(gm.state[kx-gm.ox, kz-gm.oz] != s)
    assert mism <= (len(ep.legal.bumps)*64)


def test_range_packet_alignment_and_rejections():
    c = contexts()
    ep = c['ep']
    st = ep.decision_state()
    p = st.range_packet
    assert p['ranges'].shape == (len(st.frames), 256, 256) and p['ranges'].dtype == np.float32
    assert np.array_equal(p['timestamps'], [f['timestamp'] for f in st.frames])
    row, rf = c['rows'][-1]
    s = LegalStream('t')
    s.observe(c['rows'][0][0], c['rows'][0][1])
    later = c['rows'][1]
    bad = []
    for mutate in ('time', 'origin', 'protocol', 'shape', 'range'):
        r2 = dict(later[1])
        if mutate == 'time':
            r2['timestamp'] = r2['timestamp']+1e-3
        elif mutate == 'origin':
            r2['camera_origin'] = np.asarray(r2['camera_origin'])+1e-3
        elif mutate == 'protocol':
            r2['sensor_protocol'] = 'rendered_depth'
        elif mutate == 'shape':
            r2['range_image'] = r2['range_image'][:128, :128]
        else:
            r2['range_image'] = r2['range_image']*0+11.
        try:
            s.observe(later[0], r2)
            bad.append(mutate)
        except ContextError:
            pass
    assert not bad, bad
    try:
        s.observe(c['rows'][0][0], c['rows'][0][1])         # older timestamp
        raise AssertionError('out-of-order row accepted')
    except ContextError:
        pass
    try:
        s.observe(dict(later[0], predicted=True), later[1])
        raise AssertionError('predicted row accepted')
    except ContextError:
        pass


def test_hypothetical_outputs_are_rejected_as_inputs():
    c = contexts(); sb = sandbox()
    it = c['items'][0]
    res = sb.evaluate(it['state'], it['legal'], candidates(it)[:2])
    fg = res.candidates[0].future_graph
    assert res.hypothetical and all(o.hypothetical for o in res.candidates)
    assert res.candidates[0].map['predicted'] is True
    st = it['state']
    try:
        DecisionState(st.frames, fg, st.range_packet, stream='x')
        raise AssertionError('hypothetical graph accepted')
    except (LegalityError, ValueError):
        pass


# ------------------------------------------------------------------------------------------- imagination
def test_imagination_never_touches_the_legal_state():
    c = contexts(); sb = sandbox()
    it = c['items'][1]
    legal, ex = it['legal'], it['execution']
    g = legal.map
    before = (digest(g.state, g.height, g.first_seen, g.last_seen), legal.pose.tobytes(), json.dumps(legal.tracks, default=str),
              legal.time, len(legal.poses), list(legal.bumps), ex.via_index, ex.steps, ex.status, ex.plan_i)
    sb.evaluate(it['state'], legal, candidates(it))
    after = (digest(g.state, g.height, g.first_seen, g.last_seen), legal.pose.tobytes(), json.dumps(legal.tracks, default=str),
             legal.time, len(legal.poses), list(legal.bumps), ex.via_index, ex.steps, ex.status, ex.plan_i)
    assert before == after
    assert not hasattr(g, '_sandbox_frozen')
    fz = frozen_map(g)
    assert fz is not g and fz.state is g.state and fz._sandbox_frozen


def test_memo_is_exact_across_planners_and_dynamics():
    """Exact per-decision caches: bit-identical rollouts with and without them (heterogeneous families)."""
    c = contexts(); sb = sandbox()
    combos = [('orca', 'native'), ('sfm', 'native'), ('dwa', 'native'), ('orca', 'slowturn'), ('dwa', 'inertia'),
              ('sfm', 'slowturn_inertia')]
    n = 0
    for it in c['items'][:3]:
        humans = NominalHumans(it['legal'].tracks, it['legal'].time, 64, bank=None)
        for planner, dyn in combos:
            gen = generator('k10', dyn)
            qs, _ = gen.propose(it['legal'], None)
            cands = [Candidate(q, planner, dyn) for q in qs if q.kind != 'continue']
            a = sb.imagine_all(it['legal'], cands, humans, memoize=False)
            b = sb.imagine_all(it['legal'], cands, humans, memoize=True)
            b2 = sb.imagine_all(it['legal'], cands, humans, memoize=True)
            for x, y, z in zip(a, b, b2):
                assert rollouts_equal(x, y) and rollouts_equal(y, z), (planner, dyn)
                n += 1
    generator('k10', 'native')
    assert n > 100


def test_exact_helpers_match_the_originals():
    from eventwm_graph.querygen.control import BODY_X, FORWARD_X
    c = contexts()
    g = c['items'][-1]['legal'].map
    rng = np.random.default_rng(5)
    known = np.argwhere(g.state > 0)
    for _ in range(4000):
        i, j = known[rng.integers(len(known))]
        x, z = g.center_xz(i, j)
        P = np.array([x+rng.uniform(-.6, .6), -(z+rng.uniform(-.6, .6))]); psi = float(rng.uniform(-math.pi, math.pi))
        with accel.active(accel.FrozenMapCache()):
            for xr in (BODY_X, FORWARD_X):
                assert np.array_equal(accel._ORIG['body_cells_state'](g, P, psi, 0.1, xr), accel.body_cells_state(g, P, psi, 0.1, xr))
            a = accel._ORIG['blocked_points_near'](g, P, 1.4); b = accel.blocked_points_near(g, P, 1.4)
            assert a.shape == b.shape and np.array_equal(a, b)


def test_dynamics_families_are_respected():
    c = contexts(); sb = sandbox()
    it = c['items'][2]
    humans = NominalHumans(it['legal'].tracks, it['legal'].time, 64, bank=None)
    for name in ('native', 'slowturn', 'inertia', 'slowturn_inertia', 'heavy_brake', 'slowturn_heavy'):
        dyn = FAMILIES[name]
        qs, _ = generator('k10', name).propose(it['legal'], None)
        cands = [Candidate(q, 'orca', name) for q in qs if q.kind != 'continue']
        for r in sb.imagine_all(it['legal'], cands, humans):
            assert r.error is None, r.error
            assert dynamics_violations(dyn, r.commands) == 0
            d = np.diff(r.poses, axis=0)
            step = np.hypot(d[:, 0], d[:, 2])
            dyaw = np.abs(np.arctan2(np.sin(d[:, 3]), np.cos(d[:, 3])))
            assert np.all(step <= dyn.v_max+1e-12) and np.all(dyaw <= dyn.turn_rad+1e-12)
            assert np.all((step < 1e-12) | (dyaw < 1e-12))                 # translate OR turn in place
            moving = step > 1e-12
            fwd = np.c_[np.cos(r.poses[:-1, 3]), -np.sin(r.poses[:-1, 3])]
            along = d[:, 0]*fwd[:, 0]+d[:, 2]*fwd[:, 1]
            assert np.allclose(along[moving], step[moving], atol=1e-12)    # straight ahead (forward = (cos, -sin))
    generator('k10', 'native')


def test_kinematics():
    n = FAMILIES['native']
    p = np.array([1., .2, -3., .3])
    f = advance(p, n, 'forward')
    assert np.allclose(f, [1+25/120*math.cos(.3), .2, -3-25/120*math.sin(.3), .3])
    assert np.allclose(advance(p, n, 'left'), [1, .2, -3, .3+10/120]) and np.allclose(advance(p, n, 'right'), [1, .2, -3, .3-10/120])
    inert = FAMILIES['inertia']
    v = inert.levels[5]
    assert np.allclose(advance(p, inert, (v*CTRL_FREQ, 0.)), [1+v*math.cos(.3), .2, -3-v*math.sin(.3), .3])
    assert np.allclose(advance(p, inert, (0., 0.)), p)


def test_trajectory_and_clock_contract():
    c = contexts(); sb = sandbox()
    it = c['items'][3]
    st = it['state']
    res = sb.evaluate(st, it['legal'], candidates(it))
    for o in res.candidates:
        tr = o.trajectory
        assert np.allclose(tr[0], st.start_pose, rtol=0, atol=1e-12)
        assert np.allclose(tr[:, 0], st.timestamp+np.arange(len(tr))*STEP_SECONDS, rtol=0, atol=1e-9)
        assert abs(o.map['base_timestamp']-st.timestamp) <= 1e-7 and abs(o.map['timestamp']-tr[-1, 0]) <= 1e-7
        assert o.humans['steps'] == min(len(tr)-1, 62)
    assert res.timestamp == st.timestamp and res.layer2.timestamp == st.timestamp
    hf = res.human_futures(16)
    assert hf['positions'].shape[:2] == (40, 17)


def test_batched_equals_individual():
    """One batched evaluation == K single-candidate evaluations on the same decision state."""
    c = contexts(); sb = sandbox()
    it = c['items'][1]
    cands = candidates(it)
    batch = sb.evaluate(it['state'], it['legal'], cands)
    for k, cand in enumerate(cands):
        one = sb.evaluate(it['state'], it['legal'], [cand]).candidates[0]
        b = batch.candidates[k]
        assert rollouts_equal(one.rollout, b.rollout)
        for key in ('candidate_cells', 'probabilities', 'states', 'heights'):
            assert np.array_equal(one.map[key], b.map[key]), key
        for key in ('p_collision', 'p_social', 'expected_social_duration'):
            assert abs(float(one.humans[key])-float(b.humans[key])) <= 1e-12, key
        assert np.allclose(one.humans['minimum'], b.humans['minimum'], rtol=0, atol=1e-12)


def test_one_shared_stage1_bank_per_decision():
    c = contexts(); sb = sandbox()
    it = c['items'][2]
    res = sb.evaluate(it['state'], it['legal'], candidates(it))
    assert res.profile['stage1_ms'] >= 0
    bank = sb.l2.predict_humans(it['state'])
    assert bank is res.humans                                      # cached for this state object: no re-prediction
    assert set(res.nominal.ids) <= set(t['track_id'] for t in it['legal'].tracks)


def test_failure_handling():
    c = contexts(); sb = sandbox()
    it = c['items'][0]
    st, legal = it['state'], it['legal']
    empty = sb.evaluate(st, legal, [])
    assert empty.candidates == () and empty.layer2 is None
    stop = sb.evaluate(st, legal, [Candidate(Query('stop', label='stop'), 'orca', 'native')]).candidates[0]
    assert stop.rollout.ended_by == 'stop' and stop.rollout.steps == 0 and stop.map is not None
    bad = Candidate(Query('goal', label='broken'), 'orca', 'native', planner_kw=dict(no_such_option=1))
    out = sb.evaluate(st, legal, [bad]+candidates(it)[:2])
    assert out.candidates[0].rollout.error is not None and out.candidates[0].rollout.steps == 0 and out.candidates[0].map is not None
    assert all(o.rollout.error is None for o in out.candidates[1:])
    for wrong in (Candidate(None, 'orca', 'native'), Candidate(Query('continue', label='continue'), 'orca', 'native')):
        try:
            sb.evaluate(st, legal, [wrong])
            raise AssertionError('invalid candidate accepted')
        except SandboxError:
            pass
    later = c['items'][1]['legal']
    try:
        sb.evaluate(st, later, candidates(it)[:1])
        raise AssertionError('mismatched legal state accepted')
    except SandboxError:
        pass


def test_maps_off_keeps_risk():
    c = contexts()
    sb = WorldModelSandbox(sandbox().l2, SandboxConfig(maps=False))
    it = c['items'][2]
    res = sb.evaluate(it['state'], it['legal'], candidates(it)[:3])
    assert all(o.map is None and o.humans is not None for o in res.candidates)


# ------------------------------------------------------------------------------------------- fast range sensor
def test_fast_range_sensor_matches_native():
    from eventwm_graph.sandbox.fastrange import compare
    c = contexts()
    ep = c['ep']
    fast = ep.ranges
    f = fast.measure()
    ref = fast.native.measure()
    cmp = compare(f['range_image'], ref['range_image'])
    assert cmp['mask_equal'] and cmp['max_abs'] <= 1e-4, cmp
    assert np.array_equal(f['camera_origin'], ref['camera_origin']) and f['timestamp'] == ref['timestamp']
    assert f['sensor_protocol'] == RANGE_PROTOCOL


if __name__ == '__main__':
    import sys
    import time
    import traceback
    failed = 0
    names = [n for n in sorted(globals()) if n.startswith('test_')]
    if len(sys.argv) > 1:
        names = [n for n in names if any(a in n for a in sys.argv[1:])]
    for name in names:
        tick = time.time()
        try:
            globals()[name]()
            print('PASS %-60s %.1fs' % (name, time.time()-tick), flush=True)
        except Exception:
            failed += 1
            print('FAIL %s\n%s' % (name, traceback.format_exc()), flush=True)
    print('%d passed, %d failed' % (len(names)-failed, failed))
    if 'ctx' in _CACHE:
        _CACHE['ctx']['ep'].close()
    sys.exit(1 if failed else 0)
