"""Same native Worker.execute protocol, with indexed-catalogue bootstrap."""
import sys
import time
from eventwm_execution_pool import ProcessProtocol, Worker, Validator


class PipelineWorker(Worker):
    def __init__(self, index, repo, output, session_dir, pids, checkpoints, started):
        self.index, self.last_scene = index, None
        self.checkpoints, self.started = checkpoints, started
        began = time.perf_counter()
        command = [sys.executable, str(repo/'scripts/eventwm_pipeline_worker.py'),
                   '--source', str(repo), '--mode', 'COMBO_NO_RENDERER',
                   '--checkpoint', '--exact-root-gate', '--collection-output', str(output)]
        ProcessProtocol.__init__(self, command, session_dir/f'worker_{index:02d}.log', repo, pids)
        self.ready_seconds = time.perf_counter()-began


class PipelineValidator(Validator):
    def __init__(self, repo, output, session_dir, pids):
        ProcessProtocol.__init__(self,
            [sys.executable, str(repo/'scripts/eventwm_pipeline_validator.py'), '--collection-output', str(output)],
            session_dir/'validation_worker.log', repo, pids)
