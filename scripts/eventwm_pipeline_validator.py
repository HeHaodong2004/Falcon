"""Frozen labels/audits with indexed catalogue I/O only."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import eventwm_collect as collector
from collector.core import read
from eventwm_pipeline_io import install_indexed_reads

output = Path(sys.argv[sys.argv.index('--collection-output')+1]).resolve()
install_indexed_reads(collector, SimpleNamespace(read=read), {}, output)
runpy.run_path(str(Path(__file__).with_name('eventwm_validate_staged.py')), run_name='__main__')
