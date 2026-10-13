"""Run the frozen native worker with only indexed catalogue I/O installed."""
import runpy
import sys
from pathlib import Path

from collector import runtime, storage
from eventwm_300k_storage import compress_prefix
storage.write_prefix_truth=compress_prefix
import controlled.run_case as controlled
from eventwm_pipeline_io import install_indexed_reads

output = Path(sys.argv[sys.argv.index('--collection-output')+1]).resolve()
install_indexed_reads(runtime, controlled, {}, output)
runpy.run_path(str(Path(__file__).with_name('dag_pool_worker.py')), run_name='__main__')
