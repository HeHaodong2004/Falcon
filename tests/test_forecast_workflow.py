import json
from pathlib import Path
import tempfile
import unittest
from eventforecastwm.workflow import retry_ready,capacity_available


class TestRecoveryDispatch(unittest.TestCase):
    def test_two_cpu_workers_do_not_relax_gpu_ownership(self):
        cfg=dict(cpu_jobs=2,jobs_per_gpu=1)
        self.assertTrue(capacity_available(None,{None:1,0:1},cfg))
        self.assertFalse(capacity_available(None,{None:2},cfg))
        self.assertFalse(capacity_available(0,{None:1,0:1},cfg))
        self.assertTrue(capacity_available(1,{None:2,0:1},cfg))
        self.assertFalse(capacity_available(None,{None:1},{}))
        with self.assertRaises(ValueError):capacity_available(None,{},dict(cpu_jobs=3))

    def test_retries_preserve_required_recovery_span(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);config=run/'config.json';config.write_text(json.dumps(dict(run=str(run))))
            job=dict(config=str(config));self.assertTrue(retry_ready(job,100.))
            (run/'launcher_exit.json').write_text(json.dumps(dict(timestamp=100.,code=1)))
            self.assertFalse(retry_ready(job,120.));self.assertFalse(retry_ready(job,164.))
            self.assertTrue(retry_ready(job,165.))
