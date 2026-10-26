"""Isolated Stage 1 research runner; never imports or starts the simulator.

Imports ``eventwm_graph`` as a namespace so the heavy Stage 0 package
``__init__`` (Habitat geometry) is not loaded. Single-threaded BLAS by default.
"""
import os
from pathlib import Path
import runpy
import sys
import types

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
LEGACY = '/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023'
os.environ.setdefault('EVENTWM_TWO_STAGE_BASE', str(BASE / 'output/s1base'))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_CACHE_DIR', str(BASE / 'output/stage1_hwm/jit_cache'))
sys.path.insert(0, LEGACY + '/output/l2_two_stage_20261023/deps')
ns = types.ModuleType('eventwm_graph')
ns.__path__ = [str(BASE / 'eventwm_graph')]
sys.modules['eventwm_graph'] = ns
module = sys.argv.pop(1)
sys.argv[0] = module
if module.endswith('.py'):
    runpy.run_path(module, run_name='__main__')
else:
    runpy.run_module(module, run_name='__main__', alter_sys=True)
