"""Root-independent Monte Carlo validation of the frozen same-target contrast."""
from dataclasses import fields
import json
import fcntl
import time
import numpy as np
from eventwm_graph.l2_partial.data import read
from eventwm_graph.l2_partial.state import Hypothesis
from eventwm_graph.l2_partial.repair_evaluation import DiagnosticHypothesis
from .data import OUT,SNAP
from .evaluate import (model_set,restore,source_maps,visible_ids,truth_at,forecast_score,
    carried_maps,count_scores,summarize)
from .seed_sensitivity import root_seed


def main():
    lock=(OUT/'conditioning_mc.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('fixed conditioning validation already running; no duplicate',flush=True);return
    if (OUT/'conditioning_root_keyed_107.json').exists():
        print('fixed conditioning validation already sealed; no repeat',flush=True);return
    wm=model_set()['D_trained_graph'];original=json.loads((OUT/'evaluation/conditioning.json').read_text())
    byroot={}
    for row in original:byroot.setdefault(row['root_id'],[]).append(row)
    records=[];failures=[];started=time.perf_counter()
    for i,c in enumerate(np.load(OUT/'dev.npy',allow_pickle=True).tolist()):
        if c['root_id'] not in byroot:continue
        try:
            folder=SNAP/'roots'/c['root_id'];legal=read(folder/'legal_root.json.gz')
            queries=read(folder/'queries.json.gz');outcomes=read(folder/'outcomes.json.gz');nodes=read(folder/'future_observations.json.gz')
            qi=max(range(len(queries)),key=lambda j:(len(queries[j]['compiled']['actions']),queries[j]['query_id']))
            node_ids=outcomes[qi]['node_ids'];ob=nodes[node_ids[min(9,len(node_ids)-1)]]['observation'];t=float(ob['timestamp'])
            b=restore(wm,c,legal['partial_graph'])
            with np.load(folder/'privileged_future.npz') as truth_file:
                times=truth_file['timestamps'].copy();truth=truth_file['positions'][...,[0,2]].copy()
            origin_maps=source_maps(b.hypotheses,b.timestamp,truth[0]);ids=visible_ids(legal,truth[0])
            tagged=[DiagnosticHypothesis(**{f.name:getattr(h,f.name) for f in fields(Hypothesis)},diagnostic_origin=j) for j,h in enumerate(b.hypotheses)]
            tagged_b=b.fork(hypotheses=tagged)
            if tagged_b.fingerprint()!=b.fingerprint():raise AssertionError('offline origin tags changed physical state')
            posterior=wm.observe(tagged_b,ob);prior=wm.observe(tagged_b,{**ob,'human_sensor_available':False})
            for old_record in byroot[c['root_id']]:
                T=b.timestamp+old_record['target_horizon'];target=truth_at(times,truth,T);scores=[]
                for state in (prior,posterior):
                    future=wm.predict_shared(state,T-t,16,root_seed(107,c['root_id']),times=[0.,T-t])
                    scores.append({**forecast_score(future,1,target,truth_at(times,truth,t),ids,
                        ancestry=carried_maps(future,origin_maps)),**count_scores(state,len(target))})
                a,z=scores;row={**old_record,'prior':a,'posterior':z,'draw_seed':root_seed(107,c['root_id'])}
                for key in ('population_es3','tracked_es','count_nll','gospa'):
                    row[key+'_gain']=a[key]-z[key] if a.get(key) is not None and z.get(key) is not None else None
                row['coverage90_change']=z['coverage90']-a['coverage90'] if a['coverage90'] is not None and z['coverage90'] is not None else None
                records.append(row)
        except Exception as e:failures.append(dict(root_id=c['root_id'],error=repr(e)))
        if (i+1)%100==0:print('conditional root MC',i+1,'pairs',len(records),'seconds',round(time.perf_counter()-started,1),flush=True)
    result=summarize(dict(forecasts=[],paths=[],filtering=[],conditioning=records,observations=[],failures=[]))
    output=dict(records=records,conditioning=result['conditioning'],failures=failures,
        original_pairs=len(original),pairs=len(records),seconds=time.perf_counter()-started,
        scope='post-fit numerical validation; same fixed checkpoint/observations/targets; root-keyed107 seed shared only within each paired root; no selection')
    (OUT/'conditioning_root_keyed_107.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output['conditioning']['2.0']),flush=True)


if __name__=='__main__':main()
