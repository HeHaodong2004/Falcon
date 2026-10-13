"""Linux COW root continuation, with fresh independently forked branch worlds.

The root owner is frozen BEFORE root-only static certification. Each independent
branch process inherits the entire native/Python world and the canonical runtime
continuation. No state setter, clock adjustment, tolerance or physics reset.
"""
import gc
import hashlib
import json
import os
import pathlib
import random
import select
import signal
import sys
import time
from multiprocessing.connection import Listener

import numpy as np
import torch


def rng_hash():
    return hashlib.sha256(repr((random.getstate(), np.random.get_state(),
                               torch.get_rng_state().tolist())).encode()).hexdigest()


def install_checkpoint(rt, current, metrics, counts, phase):
    def checkpoint(args, attempt, env, root_state, history, control_times):
        start = time.perf_counter()
        if root_state.get('root_checkpoint_not_supported'):
            raise RuntimeError('Checkpoint unsupported')
        tasks = sorted(p.name for p in pathlib.Path('/proc/self/task').iterdir())
        assert len(tasks) == 1, ('Fork requires single-threaded snapshot owner', tasks)
        assert env.sim.renderer is None, 'Cannot fork a renderer/EGL context'
        assert not torch.cuda.is_initialized(), 'Cannot fork initialized CUDA'
        assert torch.get_num_threads() == 1
        fds = []
        for p in pathlib.Path('/proc/self/fd').iterdir():
            try:
                target = os.readlink(p)
            except FileNotFoundError:
                continue
            assert '/dev/nvidia' not in target and '/dev/dri' not in target, target
            fds.append(dict(fd=p.name, target=target))
        maps = pathlib.Path('/proc/self/maps').read_text()
        writable_shared = [line for line in maps.splitlines() if line.split()[1].endswith('s') and 'w' in line.split()[1]]
        assert not writable_shared, ('Mutable shared mappings cannot form a COW checkpoint', writable_shared)
        cfg = env._config.environment
        # Frozen config is 10,000,000 s, not disabled. Preserve the timer/value.
        # Bounded jobs are <=300 s and owners live <=3600 s, far below that limit.
        assert cfg.max_episode_seconds == 0 or cfg.max_episode_seconds-env._elapsed_seconds > 7200, 'Wall deadline too near for this bounded checkpoint proof'
        identity = dict(root_id=current['job']['root_id'], checkpoint_pid=None,
                        parent_worker_pid=os.getpid(), thread_ids=tasks,
                        torch_threads=torch.get_num_threads(), renderer=False,
                        cuda_initialized=False, rng_hash=rng_hash(),
                        sim_time=float(env.sim.get_world_time()),
                        elapsed_steps=env._elapsed_steps,
                        task_steps=env.task._cur_episode_step,
                        wall_episode_start=env._episode_start_time,
                        wall_episode_limit=cfg.max_episode_seconds,
                        source='complete process continuation before root query generation',
                        writable_shared_maps=writable_shared)
        checkpoint_dir = args.output/'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        identity['root_state_hash'] = rt.digest(root_state)
        identity['prefix_history_hash'] = rt.digest(history)
        identity['control_times'] = control_times
        sock = '/tmp/ewdag_%s_%s.sock' % (os.getpid(), current['job']['root_id'][-12:])
        assert not pathlib.Path(sock).exists()
        listener = Listener(sock, family='AF_UNIX')
        os.chmod(sock, 0o600)
        python_rng = random.getstate()
        sys.stdout.flush();sys.stderr.flush()
        pid = os.fork()
        if pid:
            # Closing the copied Listener object may unlink the child's endpoint;
            # close only this process's file descriptor, not Listener.close().
            listener._listener._socket.close()
            listener._listener._unlink.cancel()
            identity['checkpoint_pid'] = pid
            identity['socket'] = sock
            identity['capture_seconds'] = time.perf_counter()-start
            current['checkpoint'] = identity
            metrics['checkpoint_capture'] += identity['capture_seconds']
            rt.write(checkpoint_dir/'manifest.json', identity)
            return attempt

        # CPython's at-fork callback reseeds random; science requires exact state.
        random.setstate(python_rng)
        assert rng_hash() == identity['rng_hash']
        os.setsid()
        # Detach inherited worker protocol pipes. Logging is isolated per anchor.
        log = os.open(checkpoint_dir/'anchor.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(log, 1);os.dup2(log, 2);os.close(log)
        devnull = os.open('/dev/null', os.O_RDONLY);os.dup2(devnull, 0);os.close(devnull)
        children = set()
        owner_started = time.monotonic()
        try:
            while True:
                assert time.monotonic()-owner_started < 3600, 'Bounded checkpoint owner lifetime exceeded'
                for child in list(children):
                    done, status = os.waitpid(child, os.WNOHANG)
                    if done:
                        children.remove(child)
                        if status:
                            rt.write(checkpoint_dir/('child_%s_exit.json' % child), dict(pid=child,status=status))
                # select prevents idle zombies and permits bounded shutdown.
                readable, _, _ = select.select([listener._listener._socket], [], [], .2)
                if not readable:
                    continue
                conn = listener.accept()
                message = conn.recv()
                if message.get('command') == 'stop':
                    # Result delivery precedes os._exit by a few instructions.
                    # Drain exited children; never mutate/step the checkpoint.
                    deadline=time.monotonic()+30
                    while children:
                        for child in list(children):
                            done,status=os.waitpid(child,os.WNOHANG)
                            if done:
                                children.remove(child)
                                assert status == 0, ('Branch child failed',child,status)
                        assert time.monotonic()<deadline, 'Timed out draining branch children'
                        if children:time.sleep(.01)
                    conn.send(dict(stopped=True, pid=os.getpid()))
                    conn.close();break
                job = message['job']
                assert cfg.max_episode_seconds == 0 or cfg.max_episode_seconds-env._elapsed_seconds > 600, 'Insufficient original wall deadline for a bounded branch'
                base_output = args.output.resolve()
                destination = pathlib.Path(job['output']).resolve()
                assert destination == base_output or job.get('checkpoint_root_output') == str(base_output)
                allowed_output = current.get('collection_output') or '/home/haodong/workspace/projects/eventwm-falcon/output/execution_dag_exploration'
                assert pathlib.Path(allowed_output) in destination.parents
                assert job['root_id'] == current['job']['root_id']
                assert job['worker_id'] != 'root' and job['query_id']
                # Owner never executes a branch or simulates/observes while idle.
                assert rng_hash() == identity['rng_hash'], 'Owner RNG contamination'
                assert float(env.sim.get_world_time()) == identity['sim_time'], 'Owner clock contamination'
                before_fork = time.perf_counter()
                python_rng = random.getstate()
                sys.stdout.flush();sys.stderr.flush()
                child = os.fork()
                if child:
                    children.add(child);conn.close();continue
                random.setstate(python_rng)
                assert rng_hash() == identity['rng_hash']
                listener._listener._socket.close()
                current['job'].clear();current['job'].update(job)
                args.output = destination
                args.worker_id = job['worker_id']
                args.query_id = job['query_id']
                args.limit_steps = job.get('limit_steps')
                branch_attempt = args.output/'audits/workers'/args.worker_id
                branch_attempt.mkdir(parents=True, exist_ok=True)
                # Preserve baseline audit/config/episode bytes at branch path.
                for p in attempt.iterdir():
                    if p.is_file() and p.name != 'branch.json':
                        (branch_attempt/p.name).write_bytes(p.read_bytes())
                current['is_branch_child'] = True
                current['branch_connection'] = conn
                current['branch_start'] = time.perf_counter()
                current['branch_cpu_start'] = time.process_time()
                current['branch_fork_seconds'] = current['branch_start']-before_fork
                metrics.clear();counts.clear()
                metrics['checkpoint_restore_fork'] = current['branch_fork_seconds']
                phase[0] = 'prefix'  # unchanged second-root capture switches to rollout
                conn.send(dict(started=True, pid=os.getpid(),
                               fork_seconds=current['branch_fork_seconds'],
                               rng_hash=rng_hash(), sim_time=float(env.sim.get_world_time())))
                return branch_attempt
        except BaseException as exc:
            rt.write(checkpoint_dir/'ANCHOR_STOP.json', dict(type=type(exc).__name__,reason=str(exc),pid=os.getpid()))
            for child in children:
                try:os.kill(child, signal.SIGTERM)
                except ProcessLookupError:pass
            raise
        finally:
            # Branch children return above and must NOT enter anchor cleanup.
            if not current.get('is_branch_child'):
                listener.close()
                try:os.unlink(sock)
                except FileNotFoundError:pass
                os._exit(0)
    rt._exact_root_checkpoint_hook = checkpoint
