"""The measured COMBINED job pool, separated from fixed benchmark inputs."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Client


class ProcessProtocol:
    def __init__(self, command, log_path, cwd, pids):
        self.pids = pids
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        self.proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True)
        pids.add(self.proc.pid)
        self.lines = queue.Queue()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open('a', buffering=1)

        def reader():
            try:
                for line in self.proc.stdout:
                    if line.startswith(('__READY__', '__RESULT__')):
                        self.lines.put(line)
                    else:
                        log.write(line)
            finally:
                log.close()
                self.lines.put(None)
        threading.Thread(target=reader, daemon=True).start()
        try:
            line = self.lines.get(timeout=120)
            if not line or not line.startswith('__READY__'):
                raise RuntimeError('Worker initialization failed: ' + str(log_path))
            self.ready = json.loads(line[len('__READY__'):])
        except BaseException:
            self.close()
            raise

    def request(self, job):
        self.proc.stdin.write(json.dumps(job) + '\n')
        self.proc.stdin.flush()
        line = self.lines.get(timeout=300)
        if not line or not line.startswith('__RESULT__'):
            raise RuntimeError('Worker exited without a result; inspect persistent logs')
        return json.loads(line[len('__RESULT__'):])

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 15)
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, 9)
                    self.proc.wait()
        self.pids.discard(self.proc.pid)


class Worker(ProcessProtocol):
    def __init__(self, index, repo, output, session_dir, pids, checkpoints, started):
        self.index, self.last_scene = index, None
        self.checkpoints, self.started = checkpoints, started
        began = time.perf_counter()
        command = [sys.executable, str(repo/'scripts/dag_pool_worker.py'),
                   '--source', str(repo), '--mode', 'COMBO_NO_RENDERER',
                   '--checkpoint', '--exact-root-gate', '--collection-output', str(output)]
        super().__init__(command, session_dir/f'worker_{index:02d}.log', repo, pids)
        self.ready_seconds = time.perf_counter() - began

    def execute(self, job):
        began = time.perf_counter()
        if job['worker_id'] == 'root':
            result = self.request(job)
        else:
            cp = self.checkpoints[job['root_id']]
            with Client(cp['socket'], family='AF_UNIX') as conn:
                conn.send(dict(job=job))
                if not conn.poll(30):
                    raise TimeoutError('Checkpoint branch startup timeout')
                ready = conn.recv()
                if not ready.get('started'):
                    raise RuntimeError('Checkpoint refused branch')
                self.pids.add(ready['pid'])
                try:
                    if not conn.poll(300):
                        raise TimeoutError('Checkpoint branch execution timeout')
                    result = conn.recv()['result']
                finally:
                    self.pids.discard(ready['pid'])
            result['fork_ready'] = ready
        result.update(dispatch_wall=time.perf_counter()-began,
                      finished_at=time.perf_counter()-self.started, slot=self.index)
        self.last_scene = job['scene']
        return result


class Validator(ProcessProtocol):
    def __init__(self, repo, output, session_dir, pids):
        super().__init__([sys.executable, str(repo/'scripts/eventwm_validate_staged.py'),
                         '--collection-output', str(output)],
                        session_dir/'validation_worker.log', repo, pids)

    def execute(self, path):
        reply = self.request(dict(output=str(path)))
        if reply['error'] is not None:
            raise RuntimeError('Frozen validation failed: ' + repr(reply['error']))
        return reply


def stop_checkpoint(cp, pids):
    with Client(cp['socket'], family='AF_UNIX') as conn:
        conn.send(dict(command='stop'))
        if not conn.poll(35) or not conn.recv().get('stopped'):
            raise RuntimeError('Checkpoint shutdown failed')
    pids.discard(cp['checkpoint_pid'])
