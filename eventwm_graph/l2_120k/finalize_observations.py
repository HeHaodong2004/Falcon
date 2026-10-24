"""Audit availability and clocks; correct scores, never observations or predictions.

The already-running evaluator predates the unavailable-prefix guard. Preserve its
raw rows and audit every canonical legal prefix before applying that guard.
"""
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from eventwm_graph.l2_partial.data import read
from eventwm_graph.l2_partial.snapshot import sha
from .data import OUT,SNAP
from .evaluate import summarize,model_set,restore,HORIZONS
from .observations import predict_observations


def main():
    destination=OUT/'evaluation';path=destination/'observations.json'
    original=destination/'observations_before_availability_audit.json'
    if not original.exists():shutil.copyfile(path,original)
    summary_original=destination/'summary_before_availability_audit.json'
    if not summary_original.exists():shutil.copyfile(destination/'summary.json',summary_original)
    rows=json.loads(original.read_text());byroot={};frames=unavailable=prefixes=0;started=time.perf_counter()
    negative_planned={r['root_id'] for r in rows if r['kind']=='planned_query_complete_any' and not r['observed']}
    direct_none={};wm=None
    for c in np.load(OUT/'dev.npy',allow_pickle=True):
        folder=SNAP/'roots'/c['root_id'];qs=read(folder/'queries.json.gz')
        outcomes=read(folder/'outcomes.json.gz');nodes=read(folder/'future_observations.json.gz')
        qi=max(range(len(qs)),key=lambda i:(len(qs[i]['compiled']['actions']),qs[i]['query_id']))
        obs=[nodes[j]['observation'] for j in outcomes[qi]['node_ids']
            if nodes[j]['observation']['timestamp']<=c['timestamp']+3.+1e-8]
        available=[r.get('observation_valid',True) and r.get('human_sensor_available',True) for r in obs]
        length=min(62,len(qs[qi]['compiled']['actions']))
        bad=sum(not v for v in available[:length]);byroot[c['root_id']]=bad
        frames+=len(available);unavailable+=sum(not v for v in available);prefixes+=bad>0
        if c['root_id'] in negative_planned and not bad:
            if wm is None:wm=model_set()['D_trained_graph']
            legal=read(folder/'legal_root.json.gz');b=restore(wm,c,legal['partial_graph'])
            clock=np.unique(np.r_[0.,np.arange(1,63)*.048,HORIZONS,3.])
            future=wm.predict_shared(b,3.,16,41,times=clock)
            prediction=predict_observations(wm,b,qs[qi],future)
            direct_none[c['root_id']]=prediction['none_probability']
    corrected=[];removed=[];cancellation_corrections=[]
    for row in rows:
        if row['kind']=='planned_query_complete_any' and byroot[row['root_id']]:
            removed.append(row);corrected.append({k:v for k,v in row.items() if k in ('root_id','scene_id')})
            corrected[-1].update(kind='planned_query_complete_unavailable',status='unsupported incomplete sensor availability; no negative evidence')
        else:
            row=dict(row)
            if not row.get('observed',True) and row['kind'] in ('recorded_pose_prefix_any','planned_query_complete_any'):
                old=row['nll']
                if row['kind']=='recorded_pose_prefix_any':
                    # No event and first detection never occurring are identical.
                    row['nll']=row['first_detection_nll']
                    row['log_none_probability']=-row['nll'] if row['nll'] is not None else None
                else:
                    probability=direct_none[row['root_id']];row['none_probability']=probability
                    row['nll']=-math.log(probability) if probability else None
                row['infinite_nll']=row['nll'] is None
                if old is None and row['nll'] is not None:
                    cancellation_corrections.append(dict(root_id=row['root_id'],kind=row['kind'],
                        rounded_any_probability=row['probability'],old_nll=None,corrected_nll=row['nll']))
            corrected.append(row)
    path.write_text(json.dumps(corrected,indent=2))
    raw={k:json.loads((destination/(k+'.json')).read_text()) for k in ('forecasts','paths','filtering','conditioning','observations','failures')}
    current=json.loads((destination/'summary.json').read_text());new=summarize(raw)
    current.update(new);(destination/'summary.json').write_text(json.dumps(current,indent=2))
    errors=np.asarray([r['native_clock_error_s'] for r in corrected if r['kind']=='planned_query_endpoint'])
    audit=dict(available_frames=frames-unavailable,unavailable_frames=unavailable,total_recorded_frames=frames,
        unavailable_prefixes=prefixes,removed_complete_prefix_scores=len(removed),
        planned_negative_prefixes_replayed=len(direct_none),cancellation_corrections=cancellation_corrections,
        native_clock_error_s=dict(records=len(errors),mean=float(errors.mean()) if len(errors) else None,
            abs_p50=float(np.quantile(abs(errors),.5)) if len(errors) else None,
            abs_p95=float(np.quantile(abs(errors),.95)) if len(errors) else None,
            abs_max=float(abs(errors).max()) if len(errors) else None),
        seconds=time.perf_counter()-started,
        original_observations_sha256=sha(original),final_observations_sha256=sha(path),
        original_summary_sha256=sha(summary_original),final_summary_sha256=sha(destination/'summary.json'),
        original_evaluator_sha256=json.loads((OUT/'training/configuration.json').read_text())['sources']['eventwm_graph/l2_120k/evaluate.py'],
        final_scoring_sources={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('evaluate.py'))},
        scope='all2110 canonical dev prefixes; actual timestamps authoritative; nominal planned exposure scores remain qualified, no unavailable frame treated as negative')
    (OUT/'observation_availability_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
