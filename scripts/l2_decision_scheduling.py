"""Shared-forecast base/selected-branch scheduling and dense profile attribution."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import l2_partial_structural_benchmark as baseline
from eventwm_graph.l2_partial.planning import packet_from_batch
from eventwm_graph.l2_partial.repair_profile import _publish,_sources,_sha,_percentiles,_rss_kib


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    output=(baseline.REPO/a.output).resolve()
    if not output.is_relative_to(baseline.REPO/'output/l2_decision_ready_20261021') or output.exists():raise ValueError('new decision output required')
    source=_sources();before=baseline.collector_metadata();models,_=baseline.make_models();m=models['structural_bound'];old=models['old_reference']
    rows=baseline.selected_rows(12,baseline.REPO/'docs/l2/partial_graph_wm_structural_20261021/artifacts/raw/batch_cohort.json')
    records=[]
    for row in rows:
        previous,actual,work,_,_,_=baseline._causal_input(old,baseline.SNAPSHOT,row)
        incoming,_=baseline.convert_incoming(previous,m,actual,old);b=m.observe(incoming,actual)
        for rep in range(5):
            start=time.perf_counter();f=m.predict_shared(b,3.,16,27+rep);forecast_ms=(time.perf_counter()-start)*1000.
            results=[]
            for selected in ((),(0,3,8,15)):
                start=time.perf_counter();batch=m.evaluate_queries(b,work,f,budget_ms=1e12,refined_query_indices=selected)
                packet=packet_from_batch(b,work,batch,pointgoal=[3.,0.,0.]);elapsed=(time.perf_counter()-start)*1000.
                results.append(batch)
                records.append({'root_id':row['root_id'],'scene_id':row['scene_id'],'repetition':rep,
                    'worlds':16,'candidates':24,'selected_indices':selected,'forecast_ms':forecast_ms,
                    'evaluate_packet_ms':elapsed,'timings_ms':batch.timings_ms,'packet_ms':packet.timings_ms['packet'],
                    'successors':sum(len(c.observation_branches) for c in packet.candidates),
                    'shared_reference':packet.shared_future.reference})
            for left,right in zip(results[0].predictions,results[1].predictions):
                np.testing.assert_array_equal(left.world_collision_probability,right.world_collision_probability)
                np.testing.assert_array_equal(left.world_social_probability,right.world_social_probability)
    profiles=[]
    for n,cert in ((6,False),(12,False),(6,True),(12,True)):
        previous,actual,queries=baseline.preserved_fixture(old,n,cert)
        b,_=baseline.convert_incoming(previous,m,actual,old)
        def call():
            post,batch=m.online_call(b,actual,queries,budget_ms=None)
            return packet_from_batch(post,queries,batch,pointgoal=[3.,0.,0.])
        call();profile=cProfile.Profile();profile.enable();packet=call();profile.disable()
        name=output.with_name(output.stem+'_dense_'+str(n)+'_'+str(int(cert)))
        profile.dump_stats(str(name)+'.prof')
        with Path(str(name)+'.txt').open('x') as stream:pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(50)
        profiles.append({'tracked':n,'certificate':cert,'profile':str(name.relative_to(baseline.REPO))+'.prof',
            'text':str(name.relative_to(baseline.REPO))+'.txt','profile_sha256':_sha(str(name)+'.prof'),
            'actual_filter_unresolved_mass':packet.uncertainty_diagnostics['actual_filter_unresolved_mass']})
    value={'schema':'l2_scheduling_profile_1','source_hashes':source,'sources_unchanged':source==_sources(),
        'script_sha256':_sha(__file__),'records':records,'profiles':profiles,
        'summary':{('base' if not chosen else 'selected_four'):{
            'evaluate_packet_ms':_percentiles([r['evaluate_packet_ms'] for r in records if bool(r['selected_indices'])==chosen]),
            'branch_ms':_percentiles([r['timings_ms']['branches'] for r in records if bool(r['selected_indices'])==chosen]),
            'packet_ms':_percentiles([r['packet_ms'] for r in records if bool(r['selected_indices'])==chosen])} for chosen in (False,True)},
        'shared_forecast_ms':_percentiles([r['forecast_ms'] for r in records if not r['selected_indices']]),
        'collector_before':before,'collector_after':baseline.collector_metadata(),'rss_kib':_rss_kib(),
        'selection':'fixed external row indices for cost attribution, no value-of-computation policy',
        'semantics':'same shared future for base and selected passes; selected pass re-evaluates base hazards; profile times instrumented'}
    _publish(output,value);print(json.dumps({'summary':value['summary'],'sources_unchanged':value['sources_unchanged']}),flush=True)


if __name__=='__main__':main()
