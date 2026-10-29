"""Layer 3 runner (same environment setup as scripts/sandbox_run.py): ``eventwm_graph`` as a namespace (no heavy package import), single-threaded BLAS, the
verified native ray-batch build, physical GPU0 for Layer 2 (the conda env exports CUDA_VISIBLE_DEVICES=1, which is
another session's GPU), Habitat data paths relative to this checkout.

    python scripts/l3_run.py eventwm_graph.l3.<module> ...
    python scripts/l3_run.py tests/l3/test_compiler.py
"""
import os
from pathlib import Path
import runpy
import sys
import types

BASE = Path(__file__).resolve().parents[1]
os.chdir(BASE)                       # Habitat resolves data/ and configs relative to the checkout
sys.path.insert(0, str(BASE))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
os.environ.setdefault('NUMBA_CACHE_DIR', str(BASE/'output/jit_cache'))
os.environ.setdefault('EVENTWM_NATIVE_BATCH_BUILD',
                      '/home/haodong/workspace/projects/eventwm-falcon/output/stage01_acceleration_20261021/native_batch/build')
os.environ.setdefault('MAGNUM_LOG', 'quiet')
os.environ.setdefault('HABITAT_SIM_LOG', 'quiet')
if os.environ.get('CUDA_VISIBLE_DEVICES', '1') == '1' and not os.environ.get('SANDBOX_KEEP_CUDA_ENV'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'      # physical GPU0 = cuda:0 (see docs/layer2/RESULTS.md section 3)
DEPS = '/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023/output/l2_two_stage_20261023/deps'
if os.path.isdir(DEPS):
    sys.path.append(DEPS)            # orjson for the frozen Layer 2 development data (tests only)
ns = types.ModuleType('eventwm_graph')
ns.__path__ = [str(BASE/'eventwm_graph')]
sys.modules['eventwm_graph'] = ns
module = sys.argv.pop(1)
sys.argv[0] = module
code = 0
try:
    if module.endswith('.py'):
        runpy.run_path(module, run_name='__main__')
    else:
        runpy.run_module(module, run_name='__main__', alter_sys=True)
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
except BaseException:
    import traceback
    traceback.print_exc()
    code = 1
finally:
    sys.stdout.flush(); sys.stderr.flush()
    # Habitat/Magnum static-plugin teardown aborts at interpreter exit after all work completes; skip finalizers
    os._exit(code)
