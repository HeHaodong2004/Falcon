"""Audit fixed-follower labels in immutable eligible root payloads only."""
import collections
import json
import time
import numpy as np
from eventwm_graph.l2_partial.data import read
from .data import OUT,SNAP


def main():
    m=json.loads((SNAP/'manifest.json').read_text());counts=collections.defaultdict(collections.Counter)
    durations=collections.defaultdict(list);issues=[];started=time.perf_counter()
    for i,row in enumerate(m['rows']):
        outcomes=read(SNAP/'roots'/row['root_id']/'outcomes.json.gz');c=counts[row['split']]
        if len(outcomes)!=row['queries']:issues.append(dict(root_id=row['root_id'],error='query count mismatch'))
        for outcome in outcomes:
            s=outcome['summary'];c['query_outcomes']+=1
            for key in ('human_collision','static_obstacle_collision','any_psc_violation','psc_valid','query_complete'):
                c[key+'_available']+=isinstance(s.get(key),bool)
                c[key+'_positive']+=s.get(key) is True
            for flag,timekey in (('human_collision','first_collision_time_s'),('any_psc_violation','first_psc_violation_time_s')):
                if s.get(flag):
                    c[timekey+'_available_positive']+=isinstance(s.get(timekey),(int,float))
            c['termination:'+str(s.get('termination'))]+=1
            durations[row['split']].append(s['elapsed_s'])
        if (i+1)%2000==0:print('label audit',i+1,round(time.perf_counter()-started,1),flush=True)
    result=dict(source_statistics=m['source_statistics'],eligible={k:dict(v) for k,v in counts.items()},
        executed_duration_s={k:dict(n=len(v),quantile_levels=[0,.05,.5,.9,.95,.99,1],
            quantiles=np.quantile(v,[0,.05,.5,.9,.95,.99,1]).tolist()) for k,v in durations.items()},
        issues=issues,seconds=time.perf_counter()-started,
        scope='fixed immutable Stage1 follower outcomes; protected payloads not opened; labels are offline secondary outputs, no action selection')
    (OUT/'label_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
