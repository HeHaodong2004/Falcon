"""Isolated research runner; never imports or starts the simulator."""
import os
from pathlib import Path
import runpy
import sys
import types

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
os.environ['EVENTWM_STAGE2_BASE'] = '/home/haodong/workspace/projects/eventwm-falcon-l2-two-stage-20261023'
os.environ['EVENTWM_TWO_STAGE_BASE'] = os.environ['EVENTWM_STAGE2_BASE']
os.environ['EVENTWM_STAGE2_RESEARCH_BASE'] = str(BASE)
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('NUMBA_NUM_THREADS','2')
os.environ.setdefault('NUMBA_THREADING_LAYER','workqueue')
os.environ.setdefault('NUMBA_CACHE_DIR', str(BASE / 'output/stage2_segment_20261024/jit_cache'))
sys.path.insert(0, str(Path(os.environ['EVENTWM_STAGE2_BASE']) / 'output/l2_two_stage_20261023/deps'))
ns = types.ModuleType('eventwm_graph')
ns.__path__ = [str(BASE / 'eventwm_graph')]
sys.modules['eventwm_graph'] = ns
module = sys.argv.pop(1)
sys.argv[0] = module
runpy.run_module(module, run_name='__main__', alter_sys=True)
