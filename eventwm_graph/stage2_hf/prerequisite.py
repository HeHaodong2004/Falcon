"""Bounded full-call profile and exact original candidate-domain coverage."""
import json,time,resource,gc
import numpy as np
import torch
from .common import OUT,OLD,CKPT,CKPT_SHA,read,save,sha,source
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.benchmark import inputs
def main():
 dest=OUT/'prerequisite';dest.mkdir(parents=True,exist_ok=False)
 save(dest/'protocol.json',dict(source=source(),checkpoint_sha256=sha(CKPT),inputs='Original frozen real representative/dense single inputs; first12percohort,5rounds,cold/hot. No competing fitting/extraction.',coverage='Original full fitting/Devaudit plus matchedprofilecandidatecells; no alternativehardfilterproposed',timer='Complete predict including actualfuturegraph; wire excluded; internalCUDA enqueue stages are not independent GPU event times.',gc=gc.isenabled()))
 model=SegmentPredictor(CKPT,execution='optimized',future_format='native',device='cuda:1',feature_threads=1)
 rows=[];startup=[]
 for label in ('representative','dense'):
  for ex in inputs('single')[label][:12]:
   t=time.perf_counter();d=model.predict(partial_graph=ex['graph'],trajectory=ex['trajectory']);startup.append(time.perf_counter()-t)
   for cold in (True,False):
    for repeat in range(5):
     if cold:model.clear_cache()
     cpu=time.process_time();t=time.perf_counter();d=model.predict(partial_graph=ex['graph'],trajectory=ex['trajectory']);end=time.perf_counter()
     rows.append(dict(workload=label,cold=cold,root_id=ex['root_id'],repeat=repeat,seconds=end-t,cpu_seconds=time.process_time()-cpu,profile=dict(model.last_profile),candidates=len(d['candidate_cells']),updates=len(d['static_updates']),future_cells=len(d['future_partial_graph']['cells']),resolution=ex['graph']['resolution'],poses=len(ex['trajectory']),known_cells=len(ex['graph']['cells'])))
 summaries={}
 for label in ('representative','dense'):
  for cold in (True,False):
   group=[r for r in rows if r['workload']==label and r['cold']==cold];v=np.array([r['seconds']for r in group])*1000
   summaries[label+('_cold'if cold else'_retained')]=dict(p50_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),p99_ms=float(np.quantile(v,.99)),mean_cpu_ms=float(np.mean([r['cpu_seconds']for r in group]))*1000,
    component_means_ms={k:float(np.mean([r['profile'][k]for r in group if r['profile'].get(k)is not None]))*1000 for k in set().union(*(r['profile']for r in group))if k.endswith('_seconds')and any(r['profile'].get(k)is not None for r in group)})
 old=read(OLD/'output/stage2_segment_20261024/prepared/full_mixed_ray/manifest.json');dev=read(OLD/'output/stage2_segment_20261024/evaluation/final_segment_dev/metrics.json')
 save(dest/'profile.json',dict(summary=summaries,calls=rows,startup_seconds=startup,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cuda_peak_allocated=torch.cuda.max_memory_allocated('cuda:1'),cuda_peak_reserved=torch.cuda.max_memory_reserved('cuda:1')))
 save(dest/'coverage.json',dict(training_roots=old['roots'],segments=old['counts']['candidate_segments'],training_audit_counts=old['counts'],manifest_sha256=sha(OLD/'output/stage2_segment_20261024/prepared/full_mixed_ray/manifest.json'),development_metrics_sha256=sha(OLD/'output/stage2_segment_20261024/evaluation/final_segment_dev/metrics.json'),scope='Reuses exact previouslyverified originalcandidatecoverage audit; allnewrepresentations retain originaldomain and must check everytarget againstit during extraction/evaluation. Coarser CNN grids change representation only, never candidate or target identities. No hard geometry masks.'))
 print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
