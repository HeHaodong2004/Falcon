"""Split-process sandbox: GPU world-model servers + CPU environment workers (exact).

The in-process ``WorldModelSandbox`` keeps Layer 2 in every worker (~2.5 GB of GPU memory: Stage 2's captured CUDA
graphs, Stage 1), so GPU memory caps an RL loop at 6-7 workers per GPU while most CPU cores idle -- and the loop is
CPU-bound (environment steps, legal and range sensing, generator, imagined rollouts; Layer 2 is ~80 ms of a ~2.5 s
decision).  Here the GPU-resident part runs in a few SERVER processes and everything else in many WORKERS:

    worker (CPU: simulator, sensors, generator, imagination)          server (GPU: Layer 2)
    RemoteStream.observe(row, range_frame)   buffers legal frames
    RemoteSandbox.evaluate(state, legal, candidates)
      'stage1'  buffered frames + decision metadata  ------------>   context.LegalStream.observe (same code),
                                                                     DecisionState, Stage 1 bank
                <-------------------------------------------------   bank (40 worlds x 3 s), state identity
      nominal humans + imagined rollouts (local)
      'predict' K trajectories  -------------------------------->   Layer 2 predict (Stage 2 + joint risk)
                <-------------------------------------------------   Layer2Prediction (bank re-attached locally)
      SandboxResult: the assembly code of WorldModelSandbox.evaluate (RemoteSandbox subclasses it)

Exactness: the server rebuilds the decision state with ``LegalStream`` from the very frames the worker observed and
makes the same Layer 2 calls on it; the worker runs the same imagination code.  Results equal the in-process sandbox's
bit for bit (``tests/sandbox/test_service.py``).  A server serves its workers one request at a time (one model, one
GPU stream); each worker stream has its own server-side ``LegalStream`` and Stage 1 stream state, so workers never
share context.  Transport: ``multiprocessing.connection`` over Unix sockets (Linux abstract namespace by default).

    python scripts/sandbox_run.py eventwm_graph.sandbox.service serve --address wm0 [--ready FILE]

    sb = RemoteSandbox('wm0', SandboxConfig())
    ep = BranchEpisode(spec, 'orca', 'native', stream=sb.open_stream(spec.key))
    res = sb.evaluate(ep.decision_state(), ep.legal, candidates)      # the same SandboxResult
"""
import argparse
import dataclasses
import json
import itertools
import os
import queue
from pathlib import Path
import threading
import time
import traceback
from concurrent.futures import Future
from types import MappingProxyType, SimpleNamespace

import numpy as np

from eventwm_graph.layer2.state import CLOCK_TOLERANCE
from .context import ContextError, LegalStream, RANGE_SIZE, RESOLUTION, check_observation, layer2_frame
from .core import SandboxConfig, SandboxError, WorldModelSandbox, _humans_only

AUTHKEY = b'wm-sandbox-service-1'
RANGE_KEYS = ('timestamp', 'camera_origin', 'camera_rotation', 'sensor_protocol')
FLUSH_FRAMES = 64          # a worker ships buffered frames at the latest after this many (bounded worker memory)


EXIT_ON_CUDA_FAULT = os.environ.get('SANDBOX_EXIT_ON_CUDA_FAULT', '0') == '1'     # opt-in (Layer 3 fleets)


def cuda_fault(exc):
    """An exception that leaves the process's CUDA context unusable (illegal access, launch failure, ECC)."""
    m = '%s %s' % (type(exc).__name__, exc)
    return 'CUDA error' in m or 'illegal memory access' in m or 'CUBLAS_STATUS' in m or 'cuDNN error' in m


class RemoteError(RuntimeError):
    """An unexpected exception in the sandbox server (its type, message and traceback)."""


def _mappingproxy(d):
    return MappingProxyType(d)


def register_reducers():
    """Transport pickling for Stage 2's read-only outputs: ``mappingproxy`` headers, the immutable graph storage
    (``stage2_segment.native`` ``Table`` / ``NativePartialGraph``) and increments (``stage2_repair`` ``SparseIncrement``)
    are rebuilt through their constructors (read-only again on the worker; equal content)."""
    from multiprocessing.reduction import ForkingPickler
    ForkingPickler.register(MappingProxyType, lambda m: (_mappingproxy, (dict(m),)))
    from eventwm_graph.stage2_segment.native import NativePartialGraph, Table
    ForkingPickler.register(Table, lambda t: (Table, (t.keys, t.states, t.heights, t.first, t.last, t.source_ids, t.sources)))
    ForkingPickler.register(NativePartialGraph, lambda g: (NativePartialGraph, (dict(g._headers), g._table)))
    from eventwm_graph.stage2_repair.increment import SparseIncrement
    ForkingPickler.register(SparseIncrement, lambda u: (SparseIncrement, (u.keys, u.states, u.heights, u.timestamp, u.source)))


register_reducers()


def socket_address(name):
    """A ``multiprocessing.connection`` AF_UNIX address: a path, or ``name`` in Linux's abstract namespace."""
    name = str(name)
    return name if ('/' in name or name.startswith('\0')) else '\0wm-sandbox.'+name


# ====================================================================== server
class SandboxServer:
    """Layer 2 + one ``LegalStream`` per open worker stream; requests are served one at a time."""

    def __init__(self, layer2):
        self.l2 = layer2
        self.streams = {}           # name -> LegalStream
        self.owner = {}             # name -> connection id
        self.current = {}           # name -> (DecisionState, worlds, seed, bank) of its latest 'stage1'
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.requests = queue.Queue()   # served by ONE thread (``run``): the thread that built the model owns its CUDA
                                        # context (Stage 2's driver-API module loads fail on other threads)
        self.stats = dict(requests=0, busy_s=0., observe_s=0., stage1_s=0., predict_s=0., frames=0, decisions=0,
                          clients=0, started=time.time())

    # ------------------------------------------------------------------ operations
    def _stream(self, name, conn_id):
        s = self.streams.get(name)
        if s is None or self.owner[name] != conn_id:
            raise SandboxError('stream %r is not open on this connection' % (name,))
        return s

    def op_open(self, conn_id, name, resolution=RESOLUTION, range_size=RANGE_SIZE, global_graph=False, compact=True):
        if name in self.streams and self.owner[name] != conn_id:
            raise SandboxError('stream %r is open on another connection (stream names key Stage 1 state)' % (name,))
        self.streams[name] = LegalStream(name, resolution, range_size, global_graph=global_graph, compact=compact)
        self.owner[name] = conn_id
        self.current.pop(name, None)
        self.l2.humans.reset(name)
        return name

    def op_observe(self, conn_id, name, pairs):
        s = self._stream(name, conn_id)
        t = time.perf_counter()
        for frame, rf in pairs:
            s.observe(frame, rf)
        self.stats['frames'] += len(pairs)
        self.stats['observe_s'] += time.perf_counter()-t
        return len(s)

    def op_stage1(self, conn_id, name, pairs, frames, metadata, worlds, seed):
        s = self._stream(name, conn_id)
        self.op_observe(conn_id, name, pairs)
        if len(s) != frames:
            raise SandboxError('the decision state is at frame %d but the server stream holds %d' % (frames, len(s)))
        t = time.perf_counter()
        state = s.decision_state(metadata)
        bank = self.l2.predict_humans(state, worlds=worlds, seed=seed)
        self.current[name] = (state, worlds, seed, bank)
        self.stats['stage1_s'] += time.perf_counter()-t
        return dict(bank=bank, identity=state.identity(), timestamp=state.timestamp, robot_pose=state.robot_pose,
                    frames=len(state.frames), cells=len(state.partial_graph['cells']))

    def op_predict(self, conn_id, name, frames, trajectories, materialize, maps, worlds, seed):
        self._stream(name, conn_id)
        cur = self.current.get(name)
        if cur is None or len(cur[0].frames) != frames or cur[1] != worlds or cur[2] != seed:
            raise SandboxError("'predict' needs the 'stage1' call of the same decision state")
        state, _, _, bank = cur
        t = time.perf_counter()
        # Layer 2 reuses a decision's bank for the same state object; its two-entry cache may have rotated through other
        # workers' decisions since 'stage1'.  The bank is a deterministic function of the state: re-inserting it only
        # saves the recomputation.
        self.l2._banks[(id(state), worlds, seed)] = (state, bank)
        if maps:
            pred = self.l2.predict(state, trajectories, materialize=materialize, worlds=worlds, seed=seed)
        else:
            pred = _humans_only(self.l2, state, trajectories, SimpleNamespace(worlds=worlds, seed=seed))
        self.stats['predict_s'] += time.perf_counter()-t
        self.stats['decisions'] += 1
        return dict(prediction=dataclasses.replace(pred, humans=None), profile=dict(self.l2.last_profile))   # the worker holds the bank

    def op_close(self, conn_id, name):
        if self.owner.get(name) == conn_id:
            self.streams.pop(name, None); self.owner.pop(name, None); self.current.pop(name, None)
            self.l2.humans.reset(name)
        return True

    def op_stats(self, conn_id):
        out = dict(self.stats, streams=len(self.streams), pid=os.getpid(), now=time.time())
        try:
            import torch
            out.update(gpu_reserved_gb=torch.cuda.memory_reserved()/1e9, gpu_max_reserved_gb=torch.cuda.max_memory_reserved()/1e9)
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------ connections
    def dispatch(self, conn_id, op, kw):
        fn = getattr(self, 'op_'+str(op), None)
        if fn is None:
            raise SandboxError('unknown operation %r' % (op,))
        with self.lock:
            t = time.perf_counter()
            try:
                return fn(conn_id, **kw)
            finally:
                self.stats['requests'] += 1
                self.stats['busy_s'] += time.perf_counter()-t

    def submit(self, conn_id, op, kw):
        fut = Future()
        self.requests.put((fut, conn_id, op, kw))
        return fut.result()

    def run(self):
        """Serve queued requests in arrival order until shutdown (call from the thread that built the model)."""
        while not self.stopping.is_set():
            try:
                fut, conn_id, op, kw = self.requests.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                fut.set_result(self.dispatch(conn_id, op, kw))
            except BaseException as exc:
                fut.set_exception(exc)

    def client_loop(self, conn, conn_id):
        with self.lock:
            self.stats['clients'] += 1
        try:
            while not self.stopping.is_set():
                try:
                    op, kw = conn.recv()
                except (EOFError, OSError):
                    break
                if op == 'shutdown':
                    conn.send(('ok', True))
                    self.stopping.set()
                    break
                fatal = False
                try:
                    reply = ('ok', self.submit(conn_id, op, kw))
                except Exception as exc:
                    reply = ('error', type(exc).__name__, str(exc), traceback.format_exc())
                    fatal = cuda_fault(exc)
                try:
                    conn.send(reply)
                except Exception as exc:                     # pickling failed before anything was sent
                    conn.send(('error', type(exc).__name__, 'reply not transferable: %s' % exc, traceback.format_exc()))
                if fatal and EXIT_ON_CUDA_FAULT:
                    # the CUDA context is unusable after an illegal access: exit so the fleet restarts the server
                    print('CUDA fault: server exits (%s)' % str(reply[2])[:200], flush=True)
                    os._exit(17)
        finally:
            if not self.stopping.is_set():                   # a vanished worker's streams are released
                for name in [n for n, c in list(self.owner.items()) if c == conn_id]:
                    try:
                        self.submit(conn_id, 'close', dict(name=name))
                    except Exception:
                        pass
            conn.close()


def serve(address, map_device='cuda:0', lanes=None, ready=None, authkey=AUTHKEY, torch_threads=1):
    """Run a sandbox server until a client sends 'shutdown' (blocking)."""
    from multiprocessing.connection import Listener
    from eventwm_graph.layer2.model import Layer2WorldModel
    from .episode import freeze_gc
    t0 = time.perf_counter()
    l2 = Layer2WorldModel.from_release(map_device=map_device, lanes=lanes, torch_threads=torch_threads)
    server = SandboxServer(l2)
    listener = Listener(socket_address(address), 'AF_UNIX', authkey=authkey)
    freeze_gc()                                              # the model's objects are permanent: no GC scans of them
    if ready:
        s2 = (l2.release or {}).get('stage2', {})            # which Stage 2 checkpoint serves (L3_S2_CKPT override)
        Path(ready).write_text(json.dumps(dict(pid=os.getpid(), address=str(address), build_s=time.perf_counter()-t0,
                                               stage2=dict(checkpoint=s2.get('checkpoint'),
                                                           sha256=l2.maps.checkpoint_sha256,
                                                           override=s2.get('override')))))
    ids = itertools.count()

    def accept():
        while not server.stopping.is_set():
            try:
                conn = listener.accept()
            except Exception:
                if server.stopping.is_set():
                    break
                continue                                     # a failed handshake does not stop the server
            threading.Thread(target=server.client_loop, args=(conn, next(ids)), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    server.run()
    listener.close()
    l2.close()
    return server.stats


# ====================================================================== worker side
class Connection:
    """One worker's connection to a server (thread-safe request/reply)."""

    def __init__(self, address, authkey=AUTHKEY, timeout=300.):
        from multiprocessing.connection import Client
        t0 = time.time()
        while True:
            try:
                self.conn = Client(socket_address(address), 'AF_UNIX', authkey=authkey)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.time()-t0 > timeout:
                    raise
                time.sleep(0.25)
        self.address = str(address)
        self.lock = threading.Lock()
        self.calls = 0
        self.wait_s = 0.

    def call(self, op, **kw):
        with self.lock:
            t = time.perf_counter()
            self.conn.send((op, kw))
            reply = self.conn.recv()
            self.wait_s += time.perf_counter()-t
            self.calls += 1
        if reply[0] == 'ok':
            return reply[1]
        _, name, msg, tb = reply
        if name == 'ContextError':
            raise ContextError(msg)
        if name == 'SandboxError':
            raise SandboxError(msg)
        raise RemoteError('%s: %s\n--- server traceback ---\n%s' % (name, msg, tb))

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


class RemoteStream:
    """Worker-side proxy of a server-held ``LegalStream``: checks each observation with the same contract and buffers
    it; the frames travel with the next decision ('stage1') or after ``FLUSH_FRAMES``."""

    def __init__(self, conn, name, resolution=RESOLUTION, range_size=RANGE_SIZE, global_graph=False, compact=True):
        self.conn = conn
        self.stream = str(name)
        self.size = int(range_size)
        self.compact = bool(compact)
        self.pending = []
        self.n = 0
        self.last = None
        self.closed = False
        conn.call('open', name=self.stream, resolution=float(resolution), range_size=self.size,
                  global_graph=bool(global_graph), compact=self.compact)

    def __len__(self):
        return self.n

    @property
    def timestamp(self):
        return None if self.last is None else float(self.last['timestamp'])

    def observe(self, row, range_frame):
        frame = layer2_frame(row, self.compact)
        img, o, r = check_observation(frame, range_frame, self.size, self.timestamp)
        rf = {k: range_frame[k] for k in RANGE_KEYS}
        rf['range_image'] = np.array(img, np.float32, copy=True)     # sensors may reuse their output buffer
        self.pending.append((frame, rf))
        self.n += 1
        self.last = frame
        if len(self.pending) >= FLUSH_FRAMES:
            self.flush()
        return frame

    def take(self):
        p, self.pending = self.pending, []
        return p

    def flush(self):
        if self.pending:
            self.conn.call('observe', name=self.stream, pairs=self.take())

    def decision_state(self, metadata=None):
        if self.last is None:
            raise ContextError('no observation yet')
        return RemoteDecision(self, self.last, metadata, self.n)

    def close(self):
        if not self.closed:
            self.closed = True
            self.pending = []
            try:
                self.conn.call('close', name=self.stream)
            except Exception:
                pass


class RemoteDecision:
    """Worker-side handle of the decision state the server builds from the same observations (time, pose, stream;
    the content identity once the server has built it)."""

    def __init__(self, source, frame, metadata, frames):
        self.source = source
        self.stream = source.stream
        self.metadata = dict(metadata or {})
        self.frames = int(frames)
        self.timestamp = float(frame['timestamp'])
        self.robot_pose = np.asarray(frame['robot_pose'], np.float64)
        self.cells = None
        self._identity = None
        self._bank = None

    @property
    def start_pose(self):
        return np.r_[self.timestamp, self.robot_pose]

    def identity(self):
        if self._identity is None:
            raise SandboxError('the server has not built this decision state yet')
        return self._identity


class RemoteLayer2:
    """The Layer 2 calls of ``WorldModelSandbox.evaluate``, answered by a server."""

    def __init__(self, conn):
        self.conn = conn
        self.last_profile = {}

    def predict_humans(self, state, worlds=None, seed=None):
        if not isinstance(state, RemoteDecision):
            raise SandboxError('a remote sandbox evaluates RemoteStream decision states')
        key = (worlds, seed)
        if state._bank is not None and state._bank[0] == key:
            return state._bank[1]
        if state.source.n != state.frames:
            raise SandboxError('the stream has moved past this decision state (%d -> %d frames)' % (state.frames, state.source.n))
        r = self.conn.call('stage1', name=state.stream, pairs=state.source.take(), frames=state.frames,
                           metadata=state.metadata, worlds=worlds, seed=seed)
        if abs(float(r['timestamp'])-state.timestamp) > CLOCK_TOLERANCE or not np.array_equal(r['robot_pose'], state.robot_pose):
            raise SandboxError('the server decision state differs from the worker handle')
        state._identity, state.cells, state._bank = r['identity'], r['cells'], (key, r['bank'])
        return r['bank']

    def predict(self, state, candidates=(), *, materialize=True, worlds=None, seed=None, maps=True):
        bank = self.predict_humans(state, worlds, seed)
        r = self.conn.call('predict', name=state.stream, frames=state.frames, trajectories=list(candidates),
                           materialize=bool(materialize), maps=bool(maps), worlds=worlds, seed=seed)
        self.last_profile = r['profile']
        return dataclasses.replace(r['prediction'], humans=bank)

    def reset(self, stream=None):
        pass                                                 # server-side state is per stream (RemoteStream.close)

    def close(self):
        self.conn.close()


class RemoteSandbox(WorldModelSandbox):
    """``WorldModelSandbox`` whose Layer 2 runs in a server process: same ``evaluate`` and results."""

    def __init__(self, address, config=None, authkey=AUTHKEY, timeout=300.):
        config = config or SandboxConfig()
        if config.prefetch_maps:
            raise SandboxError('prefetch_maps is an in-process option')
        self.conn = Connection(address, authkey, timeout)
        super().__init__(RemoteLayer2(self.conn), config)
        self._names = itertools.count()

    def open_stream(self, name, unique=True, resolution=RESOLUTION, range_size=RANGE_SIZE, global_graph=False,
                    compact=True):
        """A new server-held legal stream for one episode.  ``unique`` suffixes the worker's pid and a counter (stream
        names key the server's Stage 1 state; workers sharing a server may run the same episode)."""
        if unique:
            name = '%s#%d.%d' % (name, os.getpid(), next(self._names))
        return RemoteStream(self.conn, name, resolution, range_size, global_graph, compact)

    @staticmethod
    def check_context(state, legal):
        if not isinstance(state, RemoteDecision):
            raise SandboxError('a remote sandbox evaluates RemoteStream decision states')
        if abs(float(legal.time)-state.timestamp) > CLOCK_TOLERANCE:
            raise SandboxError('executor legal state (t=%.6f) and decision state (t=%.6f) differ in time'
                               % (float(legal.time), state.timestamp))
        if not np.allclose(np.asarray(legal.pose, np.float64), state.robot_pose, rtol=0, atol=1e-7):
            raise SandboxError('executor legal state and decision state differ in the robot pose')
        if getattr(legal.map, 'timestamp', None) is not None and abs(float(legal.map.timestamp)-state.timestamp) > CLOCK_TOLERANCE:
            raise SandboxError('executor legal map is not at the decision time')

    def _layer2_predict(self, state, trajs):
        cfg = self.config
        return self.l2.predict(state, trajs, materialize=cfg.materialize, worlds=cfg.worlds, seed=cfg.seed, maps=cfg.maps)

    def server_stats(self):
        return self.conn.call('stats')

    def shutdown_server(self):
        self.conn.conn.send(('shutdown', {}))
        return self.conn.conn.recv()

    def close(self):
        self.conn.close()


def start_server(address, ready, log=None, map_device='cuda:0', lanes=None, env=None):
    """Launch ``serve`` in a new process (``scripts/sandbox_run.py`` entry); returns the Popen.  Wait with
    ``wait_ready``."""
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, str(root/'scripts/sandbox_run.py'), 'eventwm_graph.sandbox.service', 'serve', '--address', str(address),
           '--ready', str(ready), '--map-device', map_device]
    if lanes is not None:
        cmd += ['--lanes', str(lanes)]
    out = open(log, 'w') if log else subprocess.DEVNULL
    return subprocess.Popen(cmd, cwd=str(root), stdout=out, stderr=subprocess.STDOUT, env=env)


def wait_ready(ready_files, procs=(), timeout=600.):
    t0 = time.time()
    while not all(Path(f).exists() for f in ready_files):
        for p in procs:
            if p.poll() is not None:
                raise RuntimeError('a sandbox server exited during start-up (code %s)' % p.returncode)
        if time.time()-t0 > timeout:
            raise TimeoutError('sandbox servers not ready after %.0f s' % timeout)
        time.sleep(0.5)
    return [json.loads(Path(f).read_text()) for f in ready_files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['serve'])
    ap.add_argument('--address', required=True)
    ap.add_argument('--ready', default=None)
    ap.add_argument('--map-device', default='cuda:0')
    ap.add_argument('--lanes', type=int, default=None)
    a = ap.parse_args()
    # serve from the importable module, not this __main__ copy: pickled helpers must resolve on the workers
    from eventwm_graph.sandbox import service
    stats = service.serve(a.address, a.map_device, a.lanes, a.ready)
    print(json.dumps({k: v for k, v in stats.items()}, default=float))


if __name__ == '__main__':
    main()
