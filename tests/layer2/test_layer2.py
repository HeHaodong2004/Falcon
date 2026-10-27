"""Unified Layer 2 integration tests on frozen development Roots (genuine inference, GPU for Stage 2).

    CUDA_VISIBLE_DEVICES=0,1 python scripts/layer2_run.py tests/layer2/test_layer2.py      # all tests
    (pytest also collects them when ``eventwm_graph`` is importable)
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from eventwm_graph.layer2.devdata import dev_rows, DevRoot, reference_records, confusion
from eventwm_graph.layer2.model import Layer2WorldModel, RELEASE, robot_steps
from eventwm_graph.layer2.state import DecisionState, CandidateTrajectory, LegalityError, is_hypothetical_graph
from eventwm_graph.layer2.backends import file_sha256

ROOT = Path(__file__).resolve().parents[2]
_CACHE = {}


def model():
    if 'l2' not in _CACHE:
        _CACHE['l2'] = Layer2WorldModel.from_release(map_device='cuda:0')
    return _CACHE['l2']


def roots(n=3):
    """First development Roots with >= 3 recorded executions and a branch of >= 6 nodes."""
    if 'roots' not in _CACHE:
        out = []
        for row in dev_rows():
            r = DevRoot(row)
            if len(r.root_candidates()) >= 3 and max(len(o['node_ids']) for o in r.outcomes) >= 6:
                out.append(r)
            if len(out) == 3:
                break
        _CACHE['roots'] = out
    return _CACHE['roots'][:n]


def banks_equal(a, b):
    return (a.track_keys == b.track_keys and np.array_equal(a.positions, b.positions) and np.array_equal(a.present, b.present)
            and np.array_equal(a.count_log_probability, b.count_log_probability) and np.array_equal(a.knots, b.knots))


def maps_equal(a, b):
    return all(np.array_equal(a[k], b[k]) for k in ('candidate_cells', 'probabilities', 'states', 'heights'))


# ---------------------------------------------------------------------------- release
def test_release_manifest():
    rel = json.loads(RELEASE.read_text())
    s1 = json.loads((ROOT/'docs/stage1_hwm/SELECTED.json').read_text())
    s2 = json.loads((ROOT/'release/stage2_claude_v17a/manifest.json').read_text())
    assert file_sha256(ROOT/rel['stage1']['checkpoint']) == rel['stage1']['checkpoint_sha256'] == s1['checkpoint_sha256']
    assert file_sha256(ROOT/rel['stage2']['checkpoint']) == rel['stage2']['checkpoint_sha256'] == s2['checkpoint_sha256']
    assert rel['stage1']['worlds'] == s1['worlds'] and rel['stage1']['horizon_seconds'] == s1['horizon_seconds']


# ---------------------------------------------------------------------------- legality / validation (CPU)
def test_decision_state_rejects_inconsistent_or_hypothetical_inputs():
    r = roots(1)[0]
    frames, graph, packet = r.frames(), r.graph(), r.packet()
    DecisionState(frames, graph, packet)

    def raises(exc, fn):
        try:
            fn()
        except exc:
            return
        raise AssertionError('expected %s' % exc.__name__)

    short = {k: (v[:-1] if isinstance(v, np.ndarray) else v) for k, v in packet.items()}
    raises(ValueError, lambda: DecisionState(frames, graph, short))                              # missing range frame
    raises(ValueError, lambda: DecisionState(frames[:-1], graph, {k: (v[:-1] if isinstance(v, np.ndarray) else v)
                                                                   for k, v in packet.items()}))  # graph ahead of frames
    shifted = dict(packet, timestamps=packet['timestamps']+.01)
    raises(ValueError, lambda: DecisionState(frames, graph, shifted))                            # different clocks
    moved = dict(packet, origins=packet['origins']+np.array([.1, 0, 0]))
    raises(ValueError, lambda: DecisionState(frames, graph, moved))                              # different camera poses
    raises(LegalityError, lambda: DecisionState(frames, dict(graph, hypothetical=True), packet))
    fake = copy.deepcopy(graph); fake['cells'].append(dict(fake['cells'][0], cell=[10**6, 10**6], source='predicted_stage2_research'))
    raises(LegalityError, lambda: DecisionState(frames, fake, packet))
    raises(LegalityError, lambda: DecisionState(frames[:-1]+[dict(frames[-1], predicted=True)], graph, packet))
    raises(LegalityError, lambda: DecisionState(frames, graph, dict(packet, sensor_protocol='privileged_depth')))


def test_candidate_validation():
    r = roots(1)[0]
    s = r.decision_state()
    l2 = model()
    p = r.root_candidates()[0][2]
    bad_time = p.copy(); bad_time[:, 0] += .048
    bad_pose = p.copy(); bad_pose[0, 1] += .01
    nonmono = p.copy(); nonmono[2, 0] = nonmono[1, 0]
    for bad in (bad_time, bad_pose, nonmono, p[:, :4]):
        try:
            l2.predict(s, [bad])
        except ValueError:
            continue
        raise AssertionError('invalid candidate accepted')


# ---------------------------------------------------------------------------- contract (GPU)
def test_predict_contract_and_hypothetical_outputs():
    l2 = model()
    r = roots(1)[0]
    s = r.decision_state()
    frames_before = copy.deepcopy(list(s.frames)); graph_before = copy.deepcopy(s.partial_graph)
    packet_before = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.range_packet.items()}
    cands = r.root_candidates()
    out = l2.predict(s, [CandidateTrajectory(c[2], query_id=c[1], metadata=dict(end=c[0])) for c in cands])
    assert out.timestamp == s.timestamp and out.hypothetical and out.provenance['hypothetical']
    assert len(out.candidates) == len(cands)
    for (end, qid, p), c in zip(cands, out.candidates):
        assert c.query_id == qid and c.metadata['end'] == end and np.array_equal(c.trajectory, p)
        assert c.map['provenance']['trajectory_sha256'] == hashlib.sha256(p.astype('<f8').tobytes()).hexdigest()
        assert c.map['predicted'] and c.map['provenance']['hypothetical'] and c.hypothetical
        assert is_hypothetical_graph(c.future_graph)
        assert abs(c.map['timestamp']-p[-1, 0]) < 1e-9 and abs(c.map['base_timestamp']-s.timestamp) < 1e-9
        assert c.humans['steps'] == min(int(np.floor((p[-1, 0]-p[0, 0])/.048+1e-9)), 62)
        assert 0 <= c.humans['p_collision'] <= 1 and 0 <= c.humans['p_social'] <= 1
        try:
            DecisionState(s.frames, c.future_graph, s.range_packet)
            raise AssertionError('hypothetical graph accepted as a legal input')
        except LegalityError:
            pass
        sm = c.summary()
        assert sm['predicted_free'] + sm['predicted_blocked'] == len(c.map['static_updates'])
    # shared bank, relative times, absolute-time accessor
    assert out.humans.timestamp == s.timestamp and out.humans.times[0] == 0. and abs(out.humans.times[-1]-3.) < 1e-12
    pos = out.humans_at(s.timestamp+np.array([0., 1.5]))
    assert pos.shape[:2] == (40, 2)
    # inputs never mutated
    assert list(s.frames) == frames_before and s.partial_graph == graph_before
    assert all(np.array_equal(s.range_packet[k], packet_before[k], equal_nan=True) for k in ('ranges', 'origins', 'rotations', 'timestamps'))
    # K = 0: humans only
    empty = l2.predict(s, [])
    assert empty.candidates == () and banks_equal(empty.humans, out.humans)


def test_shared_decision_state_consistency():
    """Batched == single-candidate calls; overlapped == sequential; bank == standalone Stage 1; reuse is exact."""
    l2 = model()
    seq = Layer2WorldModel(l2.humans, l2.maps, overlap=False, release=l2.release)
    from eventwm_graph.stage1_hwm.api import HumanWorldModel
    alone = HumanWorldModel(str(ROOT/l2.release['stage1']['checkpoint']), 'cpu')
    for r in roots(2):
        s = r.decision_state()
        paths = [c[2] for c in r.root_candidates()][:6]
        l2.reset()
        out = l2.predict(s, paths)
        again = l2.predict(s, paths)                           # same state object: the bank is reused
        assert again.humans is out.humans
        s2 = r.decision_state()                                # new object, same stream: recomputed, identical
        s2b = seq.predict(s2, paths)
        assert banks_equal(s2b.humans, out.humans)
        for k, p in enumerate(paths):
            single = l2.predict(r.decision_state(), [p])
            assert banks_equal(single.humans, out.humans)
            assert maps_equal(single.candidates[0].map, out.candidates[k].map)
            assert maps_equal(s2b.candidates[k].map, out.candidates[k].map)
            for key in ('p_collision', 'p_social', 'expected_social_duration', 'expected_social_severity'):
                # same per-world events; only the BLAS summation order of the weight product depends on K (~1e-17)
                assert abs(single.candidates[0].humans[key]-out.candidates[k].humans[key]) <= 1e-12
            assert np.array_equal(single.candidates[0].humans['collision_world_event'], out.candidates[k].humans['collision_world_event'])
            assert np.array_equal(single.candidates[0].humans['minimum'], out.candidates[k].humans['minimum'])
        ref = alone.predict(list(s.frames[:-1]), s.frames[-1], s.partial_graph, stream='fresh')
        alone.streams.clear()
        assert banks_equal(ref, out.humans)


def test_stage2_predictions_equal_released_reference():
    l2 = model()
    ref = reference_records()
    n = 0
    for r in roots(3):
        cands = r.root_candidates()
        out = l2.predict(r.decision_state(), [c[2] for c in cands])
        for (end, _, _), c in zip(cands, out.candidates):
            x = ref[(r.root_id, end)]
            cm, _ = confusion(c.map, r.target(end))
            sm = c.summary()
            assert (sm['predicted_free'], sm['predicted_blocked'], sm['candidate_cells']) == (x['hard_free'], x['hard_blocked'], x['candidates'])
            assert cm.tolist() == x['cm']
            assert abs(sm['expected_free_uncalibrated']-x['exp_free']) <= 1e-9*max(1., x['exp_free'])
            n += 1
    assert n >= 9


def test_long_paths_keep_single_call_semantics():
    """Candidates whose crop leaves Stage 2's default legal-map extent (start +- 19 m) would change other candidates'
    increments through the shared, enlarged map. Layer 2 output == cold single-candidate Stage 2 calls, any order/cache."""
    l2 = model()
    row = next(r for r in dev_rows() if r['root_id'].startswith('d7ac23c9'))
    r = DevRoot(row)
    s = r.decision_state()
    paths = [c[2] for c in r.root_candidates()]
    assert len(l2.maps.outside_default_extent(s, paths)) >= 3
    l2.reset()
    out = l2.predict(s, paths, materialize=False)
    rev = l2.predict(s, paths[::-1], materialize=False).candidates[::-1]
    for p, a, b in zip(paths, out.candidates, rev):
        l2.maps.model.clear_cache()
        raw = l2.maps.model.predict_segments(partial_graph=s.partial_graph, trajectories=[p], range_observations=s.range_packet,
                                             track_history=s.track_history, materialize=False)[0]
        assert maps_equal(raw, a.map) and maps_equal(raw, b.map)
    l2.reset()


def test_coordinate_and_time_alignment():
    from eventwm_graph.stage1_hwm.legal_state import positives
    l2 = model()
    for r in roots(3):
        s = r.decision_state()
        cands = r.root_candidates()
        out = l2.predict(s, [c[2] for c in cands])
        visible = {str(t['track_id']): np.asarray(t['position'])[[0, 2]] for t in positives(s.frames[-1])}
        for slot, key in enumerate(out.humans.track_keys):
            if key[6:] in visible and key.startswith('track:'):
                assert np.abs(out.humans.positions[:, 0, slot]-visible[key[6:]]).max() < 1e-5
        for c in out.candidates:
            robots, lengths, rel, _ = robot_steps([c.trajectory], s.timestamp, .048, 62)
            for i in range(lengths[0]):
                j = np.flatnonzero(np.abs(c.trajectory[:, 0]-(s.timestamp+rel[i])) < 1e-9)
                if len(j):
                    assert np.abs(robots[0, i, :3]-c.trajectory[j[0], 1:4]).max() < 1e-9
            keys = c.map['candidate_cells']
            if len(keys):     # Stage 2 grid == world X/Z / resolution: the domain lies within 11 m of the path
                centres = (keys.astype(np.float64)+.5)*s.resolution
                d = np.sqrt(((centres[:, None]-c.trajectory[None, :, [1, 3]])**2).sum(-1)).min(1)
                assert d.max() <= 11.+s.resolution


def test_long_candidate_human_horizon_is_flagged():
    l2 = model()
    r = roots(1)[0]
    s = r.decision_state()
    p0 = s.start_pose
    t = np.arange(0, 4.0+1e-9, .048)
    still = np.repeat(p0[None], len(t), 0); still[:, 0] = s.timestamp+t
    out = l2.predict(s, [still])
    c = out.candidates[0]
    assert c.humans['horizon_truncated'] and c.humans['steps'] == 62 and abs(c.humans['horizon_seconds']-2.976) < 1e-9
    assert abs(c.map['timestamp']-(s.timestamp+t[-1])) < 1e-9


def test_incremental_stream_equals_fresh():
    l2 = model()
    from eventwm_graph.stage1_hwm.api import HumanWorldModel
    alone = HumanWorldModel(str(ROOT/l2.release['stage1']['checkpoint']), 'cpu')
    for r in roots(2):
        l2.reset()
        l2.predict(r.decision_state(stream='episode'), [])
        longest = max((o['node_ids'] for o in r.outcomes), key=len)
        for node in longest[1:6:2]:
            sb = r.decision_state(node, stream='episode')
            out = l2.predict(sb, [c[2] for c in r.continuation_candidates(node)])
            ref = alone.predict(list(sb.frames[:-1]), sb.frames[-1], sb.partial_graph, stream='fresh')
            alone.streams.clear()
            assert banks_equal(ref, out.humans)
            assert out.timestamp == float(r.nodes[node]['observation']['timestamp'])


def test_yaw_convention_and_planner_adapters():
    """Recorded motion obeys forward = (cos yaw, -sin yaw) in world x/z (the convention of the frames, the
    joint reduction and the querygen rollouts); rollout poses <-> candidates; bank on the planner clock."""
    from eventwm_graph.layer2.state import trajectory_from_poses
    l2 = model()
    n = 0
    for r in roots(3):
        cands = r.root_candidates()
        for _, _, p in cands:
            for i in range(len(p)-1):
                d = p[i+1, [1, 3]]-p[i, [1, 3]]
                if np.linalg.norm(d) > .01:
                    n += 1
                    assert d/np.linalg.norm(d) @ np.array([np.cos(p[i+1, 4]), -np.sin(p[i+1, 4])]) > .98
        s = r.decision_state()
        rollout = np.repeat(s.robot_pose[None], 11, 0); rollout[:, 3] += np.arange(11)*.0833   # turn in place, 0.48 s
        tau = trajectory_from_poses(rollout, s.timestamp)
        out = l2.predict(s, [tau])
        assert np.allclose(out.candidates[0].trajectory[:, 0]-s.timestamp, np.arange(11)*.048)
        f = out.human_futures(steps=20)
        assert f['positions'].shape[:2] == (40, 21) and np.allclose(f['times'], s.timestamp+np.arange(21)*.048)
        assert np.allclose(f['positions'], out.humans_at(f['times']), rtol=0, atol=1e-9)
    assert n > 10


def test_fast_reduction_is_bit_identical_to_l25_reference():
    """layer2.reduction (numba, GIL-free) == l25.numeric.aggregate_joint via HumanWorlds.evaluate_candidates."""
    from eventwm_graph.layer2 import reduction
    l2 = model()
    checked = 0
    for row in dev_rows()[:40]:
        r = DevRoot(row)
        c = r.root_candidates()
        if not c:
            continue
        s = r.decision_state()
        bank = l2.predict_humans(s)
        still = np.repeat(s.start_pose[None], 70, 0); still[:, 0] = s.timestamp+np.arange(70)*.048
        for paths in ([x[2] for x in c], [c[0][2]], [still]+[x[2] for x in c][:2]):
            robots, lengths, _, _ = robot_steps(paths, s.timestamp, .048, 62)
            with np.errstate(invalid='ignore'):
                a = bank.evaluate_candidates(robots, lengths)
                b = reduction.evaluate_candidates(bank, robots, lengths)
            assert a.keys() == b.keys()
            for k in a:
                assert (a[k] == b[k]) if k == 'minimum_quantiles' else np.array_equal(a[k], b[k]), k
            checked += 1
    assert checked >= 60


class _ConstantVelocityHumans:
    """Minimal stand-in Stage 1 (proves the human backend is replaceable without touching Stage 2)."""
    kind = 'test_constant_velocity'; step_seconds = .048; horizon = 3.; device = 'cpu'; checkpoint_sha256 = None
    max_steps = 62; last_profile = {}

    def predict(self, state, worlds=None, seed=None):
        from eventwm_graph.stage1_hwm.api import HumanWorlds
        from eventwm_graph.stage1_hwm.legal_state import positives
        tr = positives(state.frames[-1])
        times = np.linspace(0, 3., 64); n = len(tr); W = 2
        knots = np.zeros((W, n, 13, 2))
        for i, t in enumerate(tr):
            x = np.asarray(t['position'])[[0, 2]]; v = np.asarray(t['estimated_velocity'])[[0, 2]]
            knots[:, i] = x+np.arange(13)[:, None]*.25*v
        s = np.clip(np.floor(times/.25).astype(int), 0, 11); f = (times-s*.25)/.25
        positions = (knots[:, :, s]+f[None, None, :, None]*(knots[:, :, s+1]-knots[:, :, s])).transpose(0, 2, 1, 3)
        return HumanWorlds(state.timestamp, times, np.full(W, .5), positions, np.ones((W, n), bool),
                           tuple('track:%s' % t['track_id'] for t in tr), np.zeros(n, bool), 0., ('TEST',), 0, None, 'test',
                           np.zeros(24), np.ones(n), knots)

    def interaction(self, bank, robots, lengths):
        return bank.evaluate_candidates(robots, lengths)

    def reset(self, stream=None):
        pass


def test_backends_are_replaceable():
    l2 = model()
    alt = Layer2WorldModel(_ConstantVelocityHumans(), l2.maps, overlap=True)
    r = roots(1)[0]
    s = r.decision_state()
    paths = [c[2] for c in r.root_candidates()]
    out = alt.predict(s, paths)
    ref = l2.predict(s, paths)
    for a, b in zip(out.candidates, ref.candidates):
        assert maps_equal(a.map, b.map)          # Stage 2 unaffected by the human backend
    assert out.provenance['stage1']['kind'] == 'test_constant_velocity'


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
    sys.exit(1 if failed else 0)
