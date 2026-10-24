"""Score public fixed-query first-detection probabilities on complete exposures."""
import json
import math
import numpy as np
from eventwm_graph.l2_partial.data import read
from eventwm_graph.l2_partial.repair_evaluation import scene_ci
from eventwm_graph.l2_partial.snapshot import sha
from .data import OUT,SNAP
from .evaluate import model_set,restore,HORIZONS
from .observations import predict_observations


def main():
    directory=OUT/'evaluation';rawpath=directory/'observations.json';raw=json.loads(rawpath.read_text())
    complete={r['root_id']:r for r in raw if r['kind']=='planned_query_complete_any'}
    endpoints={}
    for r in raw:
        if r['kind']=='planned_query_endpoint':endpoints.setdefault(r['root_id'],[]).append(r)
    audit=json.loads((OUT/'observation_availability_audit.json').read_text())
    if audit['unavailable_frames']:raise ValueError('this postprocessor requires explicit per-endpoint availability for unavailable campaigns')
    wm=model_set()['D_trained_graph'];records=[];failures=[]
    for i,c in enumerate(np.load(OUT/'dev.npy',allow_pickle=True)):
        if c['root_id'] not in complete:continue
        try:
            folder=SNAP/'roots'/c['root_id'];legal=read(folder/'legal_root.json.gz');qs=read(folder/'queries.json.gz')
            q=max(qs,key=lambda q:(len(q['compiled']['actions']),q['query_id']))
            b=restore(wm,c,legal['partial_graph']);clock=np.unique(np.r_[0.,np.arange(1,63)*.048,HORIZONS,3.])
            future=wm.predict_shared(b,3.,16,41,times=clock);prediction=predict_observations(wm,b,q,future)
            labels=sorted(endpoints[c['root_id']],key=lambda r:r['time'])
            assert len(labels)==len(prediction['first_detection_mass'])
            assert abs(prediction['any_probability']-complete[c['root_id']]['probability'])<1e-12
            detected=[j for j,r in enumerate(labels) if r['observed']]
            likelihood=prediction['first_detection_mass'][detected[0]] if detected else prediction['none_probability']
            nll=-math.log(likelihood) if likelihood else None
            error=abs(sum(prediction['first_detection_mass'])+prediction['none_probability']-1.)
            row=complete[c['root_id']];row['first_detection_nll']=nll
            row['infinite_first_detection_nll']=likelihood==0;row['first_distribution_normalization_error']=error
            records.append(dict(root_id=c['root_id'],scene_id=c['scene_id'],first_detection_nll=nll,
                first_positive_index=detected[0] if detected else None,normalization_error=error,infinite=likelihood==0))
        except Exception as e:failures.append(dict(root_id=c['root_id'],error=repr(e)))
        if (i+1)%250==0:print('planned first detection',i+1,'complete scored',len(records),flush=True)
    summarypath=directory/'summary.json';before=sha(summarypath);summary=json.loads(summarypath.read_text())
    score=scene_ci(records,'first_detection_nll');infinite=sum(r['infinite'] for r in records)
    if infinite:score=dict(status='INFINITE_OR_ZERO_SUPPORT',infinite_records=infinite,finite_support_summary=score)
    target=summary['observations']['planned_query_complete_any'];target['first_detection_nll']=score
    target['first_detection_nll_infinite_records']=infinite
    rawpath.write_text(json.dumps(raw,indent=2));summarypath.write_text(json.dumps(summary,indent=2))
    report=dict(records=records,failures=failures,expected_complete_prefixes=len(complete),scored=len(records),
        first_detection_nll=score,summary_before_sha256=before,summary_after_sha256=sha(summarypath),
        observation_rows_after_sha256=sha(rawpath),checkpoint_sha256=sha(OUT/'training/checkpoint.pt'),
        scope='public planned-pose first-time law against recorded legal frames; complete available prefixes only; original global41 finite-workload scope')
    (OUT/'planned_first_detection.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('records','failures')}),flush=True)


if __name__=='__main__':main()
