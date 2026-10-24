"""Run every inherited L2 regression directory; fail if a suite is empty."""
import json
from pathlib import Path
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest

start=time.perf_counter();results=[]
for path in sorted(Path("tests").glob("l2*")):
    if not path.is_dir():continue
    suite=unittest.TestLoader().discover(str(path),pattern="test*.py")
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.testsRun>0,path
    results.append(dict(suite=str(path),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
Path("output/l2_final_800k_20261023/regressions.json").write_text(json.dumps(dict(suites=results,seconds=time.perf_counter()-start),indent=2))
raise SystemExit(int(any(r["failures"] or r["errors"] for r in results)))
