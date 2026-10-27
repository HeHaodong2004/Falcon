"""Layer 2 runner: imports ``eventwm_graph`` as a namespace (no Habitat import), single-threaded BLAS.

    python scripts/layer2_run.py eventwm_graph.layer2.benchmark --device cuda:0
    python scripts/layer2_run.py tests/layer2/test_layer2.py
"""
import os
from pathlib import Path
import runpy
import sys
import types

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
os.environ.setdefault('NUMBA_CACHE_DIR', str(BASE / 'output/layer2/jit_cache'))
# orjson (fast reader of the frozen development Roots) lives in the legacy dependency directory
DEPS = '/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023/output/l2_two_stage_20261023/deps'
if os.path.isdir(DEPS):
    sys.path.insert(0, DEPS)
ns = types.ModuleType('eventwm_graph')
ns.__path__ = [str(BASE / 'eventwm_graph')]
sys.modules['eventwm_graph'] = ns
module = sys.argv.pop(1)
sys.argv[0] = module
if module.endswith('.py'):
    runpy.run_path(module, run_name='__main__')
else:
    runpy.run_module(module, run_name='__main__', alter_sys=True)
