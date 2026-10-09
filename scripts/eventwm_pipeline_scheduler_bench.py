#!/usr/bin/env python3
"""Synthetic tick CPU scaling only. These are not native collection speeds."""
import argparse,json,time
from eventwm_pipeline_index import ActiveIndex

def bench(n):
    roots={str(i):dict(root_id=str(i),_pipeline_ordinal=i,state='completed',attempts=[]) for i in range(n)}
    for i in range(n,n+72):roots[str(i)]=dict(root_id=str(i),_pipeline_ordinal=i,state='pending',attempts=[])
    history=list(roots.values());index=ActiveIndex(roots,lambda r:False)
    def old_tick():
        retry=[r for r in history if r['state']=='failed']
        ready=[r for r in history if r['state']=='pending']
        pending=sum(r['state']=='pending' for r in history)
        prep=next((r for r in history if r['state']=='preparing'),None)
        unlaunched=any(r['state'] in ('pending','preparing') for r in history)
        return len(retry)+len(ready)+pending+bool(prep)+unlaunched
    def new_tick():return len(index.members['retry'])+index.count('pending')+bool(index.first('pending'))+bool(index.reserved_preparation(set()))
    out={'historical_proposals':n,'active':72}
    for name,fn in [('before',old_tick),('after',new_tick)]:
        ticks=100 if name=='before' else 100000;t=time.perf_counter();cpu=time.process_time()
        for _ in range(ticks):fn()
        wall=time.perf_counter()-t;c=time.process_time()-cpu
        out[name]=dict(ticks=ticks,wall_seconds=wall,cpu_seconds=c,iterations_per_second=ticks/wall,cpu_seconds_per_tick=c/ticks)
    out['scope']='Synthetic metadata scheduling only; no Habitat and no retained q/h.'
    return out
if __name__=='__main__':print(json.dumps([bench(25000),bench(250000)],indent=2))
