"""Isolated runner for the query-generator / oracle research package.

Installs ``eventwm_graph`` as a namespace (skipping its heavy ``__init__``),
pins single-threaded numerical libraries and the verified native ray-batch
build, then runs ``python scripts/querygen_run.py <module> [args...]``.
"""
import os
from pathlib import Path
import runpy
import sys
import types

BASE = Path(__file__).resolve().parents[1]
os.chdir(BASE)  # Habitat resolves data/ and configs relative to the checkout.
sys.path.insert(0, str(BASE))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('EVENTWM_NATIVE_BATCH_BUILD',
                      '/home/haodong/workspace/projects/eventwm-falcon/output/'
                      'stage01_acceleration_20261021/native_batch/build')
os.environ.setdefault('MAGNUM_LOG', 'quiet')
os.environ.setdefault('HABITAT_SIM_LOG', 'quiet')
ns = types.ModuleType('eventwm_graph')
ns.__path__ = [str(BASE / 'eventwm_graph')]
sys.modules['eventwm_graph'] = ns
module = sys.argv.pop(1)
sys.argv[0] = module
code = 0
try:
    runpy.run_module(module, run_name='__main__', alter_sys=True)
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
except BaseException:
    import traceback
    traceback.print_exc()
    code = 1
finally:
    sys.stdout.flush(); sys.stderr.flush()
    # Habitat/Magnum static-plugin teardown aborts at interpreter exit
    # (malloc_consolidate) after all work completes; skip finalizers.
    os._exit(code)
