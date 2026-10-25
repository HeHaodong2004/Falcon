"""Research runner with a process-start witness for complete startup timing."""
import os
from pathlib import Path
import runpy
import time

os.environ['EVENTWM_STAGE2_PROCESS_START']=str(time.perf_counter())
runpy.run_path(str(Path(__file__).with_name('stage2_segment_run.py')),run_name='__main__')
