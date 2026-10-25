"""Paired complete-API thread comparison, preserving saved workloads."""
import json
import time
import numpy as np
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.benchmark import inputs
from eventwm_graph.stage2_segment.common import OUT,dump,snapshot


def main():
    dest=OUT/'feature_thread_pilot';dest.mkdir(exist_ok=False)
    models={n:SegmentPredictor(OUT/'training/pilot_ray/last.pt',execution='optimized',future_format='native',device='cuda:1',feature_threads=n) for n in (1,2)}
    workload=inputs('single');examples=workload['representative'][:6]+workload['dense'][:2];records=[];holders={}
    for index,ex in enumerate(examples):
        for model in models.values():model.predict(partial_graph=ex['graph'],trajectory=ex['trajectory'])
        for condition in ('map_cold','retained_cache'):
            for repeat in range(8):
                for threads in ((1,2) if (index+repeat)%2 else (2,1)):
                    m=models[threads]
                    if condition=='map_cold':m.clear_cache()
                    cpu=time.process_time();tick=time.perf_counter();holders[threads]=m.predict(partial_graph=ex['graph'],trajectory=ex['trajectory']);end=time.perf_counter()
                    records.append(dict(threads=threads,condition=condition,root_id=ex['root_id'],seconds=end-tick,cpu_seconds=time.process_time()-cpu,profile=dict(m.last_profile)))
        a=models[1].predict(partial_graph=ex['graph'],trajectory=ex['trajectory']);b=models[2].predict(partial_graph=ex['graph'],trajectory=ex['trajectory'])
        for field in ('candidate_cells','probabilities','states','heights'):np.testing.assert_array_equal(a[field],b[field])
        assert a['future_partial_graph'].to_snapshot()==b['future_partial_graph'].to_snapshot()
    summary={}
    for threads in models:
        summary[str(threads)]={}
        for condition in ('map_cold','retained_cache'):
            rs=[r for r in records if r['threads']==threads and r['condition']==condition]
            summary[str(threads)][condition]=dict(p50_ms=float(np.median([r['seconds'] for r in rs]))*1000,p95_ms=float(np.quantile([r['seconds'] for r in rs],.95))*1000,
                mean_cpu_utilization_percent=100*sum(r['cpu_seconds'] for r in rs)/sum(r['seconds'] for r in rs),
                trajectory_ms=1000*float(np.mean([r['profile']['trajectory_seconds'] for r in rs])),map_ms=1000*float(np.mean([r['profile']['map_seconds'] for r in rs])))
    dump(dest/'results.json',dict(summary=summary,calls=records,source=snapshot(),scientific_outputs_exact=True,background='full extraction14workers; exploratory paired timing'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
