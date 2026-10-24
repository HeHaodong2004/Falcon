"""Run the preserved 322 cases/oracles plus decision-contract regressions."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import l2_partial_structural_validation as preserved
from eventwm_graph.l2_partial.repair_profile import _publish,_metadata,_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=(preserved.REPO/args.output).resolve()
    if not out.is_relative_to(preserved.REPO/'output/l2_decision_ready_20261021') or out.exists():raise ValueError('new decision research output required')
    out.parent.mkdir(parents=True,exist_ok=True)
    before=_sources();start=time.perf_counter();commands=[];oracles=None;failure=None
    try:
        commands=preserved.test_commands(out)
        command=[sys.executable,'-m','unittest','discover','-s','tests/l2_decision_ready','-v']
        result=subprocess.run(command,cwd=preserved.REPO,text=True,capture_output=True)
        log=out.with_name(out.stem+'_decision_tests.log');log.write_text(result.stdout+result.stderr)
        counts=re.findall(r'Ran (\d+) tests?',result.stdout+result.stderr)
        commands.append({'command':command,'returncode':result.returncode,'tests':int(counts[-1]) if counts else None,
                         'log':str(log.relative_to(preserved.REPO)),'log_sha256':preserved.sha(log)})
        if result.returncode:raise RuntimeError('decision regression failed')
        oracles=preserved.oracles()
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    after=_sources()
    value={'schema':'l2_decision_validation_1','commands':commands,'total_tests':sum(c['tests'] or 0 for c in commands),
           'preserved_oracles':oracles,'source_before':before,'source_after':after,'sources_unchanged':before==after,
           'metadata':_metadata(),'failure':failure,'elapsed_s':time.perf_counter()-start,
           'script_sha256':preserved.sha(__file__),'statistical_fitting':False,'new_truth_payloads_opened':False}
    _publish(out,value);print(json.dumps({'output':str(out),'total':value['total_tests'],'failure':failure}),flush=True)
    if failure or before!=after:raise RuntimeError('validation failed or source changed; artifact preserved')


if __name__=='__main__':main()
