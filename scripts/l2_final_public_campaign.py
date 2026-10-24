"""Verify restored public initialization; retain rejected causal-prefix evidence."""
import os,subprocess,sys,json,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from eventwm_graph.l2_final.common import OUT,dump

def main():
    workers=[]
    for arm in ('B_public_final','C_public_final'):
        subprocess.run([sys.executable,'-m','eventwm_graph.l2_final.evaluate','setup','--arm',arm],cwd=BASE,check=True)
        log=(OUT/'logs'/(arm+'.log')).open('x')
        command=[sys.executable,'-u','-m','eventwm_graph.l2_final.evaluate','run','--arm',arm,'--start','0','--stop','528']
        workers.append((subprocess.Popen(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT),log))
    for p,log in workers:
        code=p.wait();log.close()
        if code:raise RuntimeError('restored public evaluation failed; preserve ledger')
    dump(OUT/'public_final_complete.json',dict(complete=True,roots_per_variant=528,variants=['B_public_final','C_public_final'],
        reason='revert native dense support and latency regression; no changes to learned parameters'))

if __name__=='__main__':main()
