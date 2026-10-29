"""Split-process sandbox tests (``service.py``): a real server process, real episodes (branch simulator, fast range
sensor), genuine Layer 2 inference on GPU0 in both the server and this process (the in-process reference).

    python scripts/sandbox_run.py tests/sandbox/test_service.py            # all tests (~4 min)
"""
import os
import uuid

import numpy as np

from eventwm_graph.querygen import suites
from eventwm_graph.querygen.executor import Query
from eventwm_graph.sandbox.context import LegalStream, ContextError, RESOLUTION
from eventwm_graph.sandbox.core import WorldModelSandbox, Candidate, SandboxConfig, SandboxError
from eventwm_graph.sandbox.episode import BranchEpisode, generator, default_query
from eventwm_graph.sandbox.service import RemoteSandbox, start_server, wait_ready, socket_address

EPISODES = [('dev', 7), ('unseen', 31)]       # dev 7: crowded spawn (up to 6 tracked people)
DECISIONS = 3
_CACHE = {}


def server():
    if 'server' not in _CACHE:
        out = os.path.join('output', 'sandbox_cache', 'test_service')
        os.makedirs(out, exist_ok=True)
        address = 'test-%d-%s' % (os.getpid(), uuid.uuid4().hex[:8])
        ready = os.path.join(out, address+'.ready')
        proc = start_server(address, ready, log=os.path.join(out, address+'.log'))
        wait_ready([ready], [proc])
        _CACHE['server'] = (address, proc, ready)
    return _CACHE['server'][0]


def local():
    if 'local' not in _CACHE:
        _CACHE['local'] = WorldModelSandbox.from_release(map_device='cuda:0')
    return _CACHE['local']


def remote(config=None):
    key = ('remote', None if config is None else repr(config))
    if key not in _CACHE:
        _CACHE[key] = RemoteSandbox(server(), config)
    return _CACHE[key]


class Tee:
    """One episode's observations into a local ``LegalStream`` AND a server-held ``RemoteStream``."""

    def __init__(self, local_stream, remote_stream):
        self.local, self.remote = local_stream, remote_stream

    def observe(self, row, rf):
        self.local.observe(row, rf)
        self.remote.observe(row, rf)

    def decision_state(self, metadata=None):
        return self.local.decision_state(metadata), self.remote.decision_state(metadata)

    def __len__(self):
        return len(self.local)

    def close(self):
        self.remote.close()


def decisions(item, sb_remote, name):
    """Run an episode along the default policy; at each decision yield (local state, remote handle, legal, candidates)."""
    spec = suites.load(item[0])[item[1]]
    gen = generator('k10', 'native')
    tee = Tee(LegalStream(name, RESOLUTION), sb_remote.open_stream(name, unique=False))
    ep = BranchEpisode(spec, 'orca', 'native', range_sensor='fast', stream=tee)
    q = default_query()
    try:
        for _ in range(DECISIONS):
            ep.run(q if ep.execution is None or ep.execution.status != 'running' else None, 16)
            if ep.over:
                break
            st, handle = ep.decision_state()
            queries, _ = gen.propose(ep.legal, ep.execution)
            cands = [Candidate(None, 'orca', 'native', execution=ep.execution, level=ep.level, query_id='continue')
                     if qq.kind == 'continue' else Candidate(qq, 'orca', 'native', level=ep.level, query_id=qq.label)
                     for qq in queries]
            yield st, handle, ep.legal, cands
    finally:
        ep.close()


def assert_same(a, b):
    """Two SandboxResults are identical in every predicted quantity (bit for bit)."""
    assert a.timestamp == b.timestamp and np.array_equal(a.robot_pose, b.robot_pose) and a.stream == b.stream
    assert a.provenance['decision_state'] == b.provenance['decision_state']
    for k in ('positions', 'present', 'weights', 'times', 'knots'):
        assert np.array_equal(getattr(a.humans, k), getattr(b.humans, k)), k
    assert tuple(a.humans.track_keys) == tuple(b.humans.track_keys)
    assert a.nominal.ids == b.nominal.ids and np.array_equal(a.nominal.positions, b.nominal.positions)
    assert len(a.candidates) == len(b.candidates)
    for x, y in zip(a.candidates, b.candidates):
        assert x.candidate == y.candidate
        r, s = x.rollout, y.rollout
        assert np.array_equal(r.poses, s.poses) and np.array_equal(r.times, s.times) and r.commands == s.commands
        assert (r.status, r.reason, r.ended_by, r.error) == (s.status, s.reason, s.ended_by, s.error)
        if x.map is None:
            assert y.map is None
        else:
            assert x.map.keys() == y.map.keys()
            for k in x.map:
                assert same_value(x.map[k], y.map[k]), k
        for k, v in x.humans.items():
            w = y.humans[k]
            assert (np.array_equal(np.asarray(v), np.asarray(w), equal_nan=True) if isinstance(v, np.ndarray)
                    or isinstance(w, np.ndarray) else (v == w or (v != v and w != w))), k
    assert a.summaries() == b.summaries() or all(_same_summary(p, q) for p, q in zip(a.summaries(), b.summaries()))


def same_value(a, b):
    """Content equality for Stage 2 delta values (arrays, increments, graphs, nested containers)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if hasattr(a, 'to_list') and hasattr(b, 'to_list'):                  # stage2_repair SparseIncrement
        return type(a) is type(b) and a.to_list() == b.to_list()
    if hasattr(a, 'to_snapshot') and hasattr(b, 'to_snapshot'):          # stage2_segment NativePartialGraph
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_value(u, v) for u, v in zip(a, b))
    return a == b or (a != a and b != b)


def _same_summary(p, q):
    return p.keys() == q.keys() and all(p[k] == q[k] or (p[k] != p[k] and q[k] != q[k]) for k in p)


# --------------------------------------------------------------------------------------------------------- tests
def test_remote_equals_in_process():
    """Every decision of a real episode: the split sandbox's result equals the in-process sandbox's bit for bit."""
    sb, rs = local(), remote()
    n = 0
    for st, handle, legal, cands in decisions(EPISODES[0], rs, 'exact-%s' % uuid.uuid4().hex[:6]):
        a = sb.evaluate(st, legal, cands)
        b = rs.evaluate(handle, legal, cands)
        assert_same(a, b)
        assert handle.identity() == st.identity() and handle.cells == len(st.partial_graph['cells'])
        n += 1
    assert n == DECISIONS


def test_interleaved_workers_are_isolated():
    """Two workers' episodes interleaved on one server (rotating Layer 2's bank cache, separate Stage 1 streams) give
    the in-process results of each episode."""
    sb = local()
    w1, w2 = remote(), RemoteSandbox(server())
    try:
        g1 = decisions(EPISODES[0], w1, 'iso-a-%s' % uuid.uuid4().hex[:6])
        g2 = decisions(EPISODES[1], w2, 'iso-b-%s' % uuid.uuid4().hex[:6])
        pairs = 0
        for d1, d2 in zip(g1, g2):
            h1 = d1[1]; h2 = d2[1]
            w1.l2.predict_humans(h1); w2.l2.predict_humans(h2)        # both banks built before either prediction
            b1 = w1.evaluate(h1, d1[2], d1[3]); b2 = w2.evaluate(h2, d2[2], d2[3])
            assert_same(sb.evaluate(d1[0], d1[2], d1[3]), b1)
            assert_same(sb.evaluate(d2[0], d2[2], d2[3]), b2)
            pairs += 1
        assert pairs == DECISIONS
    finally:
        w2.close()


def test_maps_off_and_partial_batches():
    """maps=False (risk only) and repeated candidate batches on one decision handle match the in-process sandbox."""
    sb = local()
    rs_off = remote(SandboxConfig(maps=False))
    sb_off = WorldModelSandbox(sb.l2, SandboxConfig(maps=False))
    for st, handle, legal, cands in decisions(EPISODES[1], rs_off, 'off-%s' % uuid.uuid4().hex[:6]):
        assert_same(sb_off.evaluate(st, legal, cands[:3]), rs_off.evaluate(handle, legal, cands[:3]))
        assert_same(sb_off.evaluate(st, legal, cands[3:]), rs_off.evaluate(handle, legal, cands[3:]))
        break


def test_contract_errors():
    rs = remote()
    spec = suites.load(EPISODES[0][0])[EPISODES[0][1]]
    stream = rs.open_stream('errors')
    ep = BranchEpisode(spec, 'orca', 'native', range_sensor='fast', stream=stream)
    stop = [Candidate(Query('stop', label='stop'), 'orca', 'native')]
    try:
        ep.run(default_query(), 16)
        handle, legal_then = ep.decision_state(), ep.legal.copy()
        frame, rf = stream.pending[-1]
        try:                                     # the worker applies LegalStream's contract (clock order) itself
            stream.observe(frame, rf)
            raise AssertionError('out-of-order observation accepted')
        except ContextError:
            pass
        ep.run(None, 4)
        try:                                     # the stream moved past the handle: no silent later state
            rs.evaluate(handle, legal_then, stop)
            raise AssertionError('stale decision handle accepted')
        except SandboxError as exc:
            assert 'moved past' in str(exc)
        later = ep.decision_state()
        res = rs.evaluate(later, ep.legal, stop)
        assert res.candidates[0].rollout.ended_by == 'stop' and res.candidates[0].map is not None
        try:                                     # only RemoteStream handles
            rs.evaluate(object(), ep.legal, stop)
            raise AssertionError('foreign state accepted')
        except SandboxError:
            pass
        other = RemoteSandbox(server())
        try:
            try:                                 # stream names key server-side Stage 1 state: no takeover
                other.open_stream(stream.stream, unique=False)
                raise AssertionError('a stream of another connection was taken over')
            except SandboxError:
                pass
        finally:
            other.close()
    finally:
        ep.close()


def test_streams_are_released():
    rs = RemoteSandbox(server())
    before = rs.server_stats()['streams']
    s = rs.open_stream('release')
    assert rs.server_stats()['streams'] == before+1
    s.close()
    assert rs.server_stats()['streams'] == before
    s2 = rs.open_stream('release-on-disconnect')
    assert rs.server_stats()['streams'] == before+1
    rs.close()                                   # a vanished worker's streams are released by the server
    probe = RemoteSandbox(server())
    import time
    for _ in range(40):
        if probe.server_stats()['streams'] == before:
            break
        time.sleep(0.05)
    assert probe.server_stats()['streams'] == before
    probe.close()


def test_driver_range_backend_is_identical():
    """The torch-free range backend (split-mode workers without a PyTorch CUDA context) gives the torch backend's
    frames bit for bit along a real episode."""
    from eventwm_graph.sandbox.fastrange import FastRangeSensor
    spec = suites.load(EPISODES[0][0])[EPISODES[0][1]]
    ep = BranchEpisode(spec, 'orca', 'native', range_sensor='fast')

    class Both:
        def __init__(self, a, b):
            self.a, self.b, self.frames = a, b, 0

        def measure(self, timestamp=None):
            x, y = self.a.measure(timestamp), self.b.measure(timestamp)
            assert np.array_equal(x['range_image'], y['range_image'], equal_nan=True)
            assert x['fast'] == y['fast'] and np.array_equal(x['camera_origin'], y['camera_origin'])
            self.frames += 1
            return x
    a0 = dict(ep.ranges.stats)                   # the torch sensor already measured the reset frame
    both = Both(ep.ranges, FastRangeSensor(ep.task, ep.sensor.config, backend='driver'))
    ep.ranges = both
    try:
        q = default_query()
        for _ in range(4):
            ep.run(q if ep.execution is None or ep.execution.status != 'running' else None, 16)
            if ep.over:
                break
        assert both.frames >= 32
        assert both.b.stats['native_rays'] == both.a.stats['native_rays']-a0['native_rays'] > 0   # same native recasts
    finally:
        ep.close()


if __name__ == '__main__':
    import sys
    import time
    import traceback
    failed = 0
    names = [n for n in sorted(globals()) if n.startswith('test_')]
    if len(sys.argv) > 1:
        names = [n for n in names if any(a in n for a in sys.argv[1:])]
    try:
        for name in names:
            tick = time.time()
            try:
                globals()[name]()
                print('PASS %-60s %.1fs' % (name, time.time()-tick), flush=True)
            except Exception:
                failed += 1
                print('FAIL %s\n%s' % (name, traceback.format_exc()), flush=True)
    finally:
        if 'server' in _CACHE:
            address, proc, ready = _CACHE['server']
            try:
                RemoteSandbox(address, timeout=10).shutdown_server()
            except Exception:
                pass
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
            if os.path.exists(ready):
                os.unlink(ready)
    print('%d passed, %d failed' % (len(names)-failed, failed))
    sys.exit(1 if failed else 0)
