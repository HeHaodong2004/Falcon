"""Finish all frozen controls and the one final trained candidate automatically."""
import json
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from eventwm_graph.l2_final.common import OUT,dump


def ready(path):
    try:
        obj=json.loads(path.read_text())
        return obj.get('sources_stable',False) and obj.get('inputs_stable',False)
    except (OSError,ValueError):return False


def run(*arguments):
    subprocess.run([sys.executable,'-u',*arguments],cwd=BASE,check=True)


def main():
    b=OUT/'evaluation_arm_original_repaired'
    guards=[b/f'evaluation_shard_{a:04d}_{z:04d}/execution_guard.json' for a,z in ((0,1055),(1055,2110))]
    while not all(ready(p) for p in guards):time.sleep(1)
    if not (b/'evaluation/merged_integrity.json').exists():run('-m','eventwm_graph.l2_final.evaluate','merge','--arm','B')
    while not (OUT/'training/training.json').exists():time.sleep(1)
    public_guard=OUT/'evaluation_arm_original_public/evaluation_shard_0000_0528/execution_guard.json'
    while not ready(public_guard):time.sleep(1)
    for arm in ('C','C_public'):run('-m','eventwm_graph.l2_final.evaluate','setup','--arm',arm)
    workers=[]
    for arm,start,stop in (('C',0,1055),('C',1055,2110),('C_public',0,528)):
        log=(OUT/'logs'/f'{arm}_{start}_{stop}.log').open('x')
        command=[sys.executable,'-u','-m','eventwm_graph.l2_final.evaluate','run','--arm',arm,'--start',str(start),'--stop',str(stop)]
        workers.append((subprocess.Popen(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT),log))
    # Public path finishes first. The other two evaluation workers plus the
    # three-worker training census leave one CPU slot for ancillary evidence.
    public,public_log=workers[-1];code=public.wait();public_log.close()
    if code:raise RuntimeError('public operational evaluation failed; preserve its ledger')
    run('-m','eventwm_graph.l2_final.benchmark','--arm','C')
    from eventwm_graph.l2_final.evaluate import module
    module().physical('new')
    for process,log in workers[:-1]:
        code=process.wait();log.close()
        if code:raise RuntimeError('full final evaluation failed; preserve its ledger')
    run('-m','eventwm_graph.l2_final.evaluate','merge','--arm','C')
    run('-m','eventwm_graph.l2_final.metrics')
    dump(OUT/'evaluation_campaign_complete.json',dict(B_full_roots=2110,C_full_roots=2110,public_roots_per_variant=528,
         complete=True,selection='only prescribed development ES during fitting; no post-evaluation fitting',protected_data_opened=False))


if __name__=='__main__':main()
