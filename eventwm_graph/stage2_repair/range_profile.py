"""Diagnostic complete-API cost with synthetic residual weights, not accuracy."""
import cProfile
import io
from pathlib import Path
import pstats
import time
import numpy as np
import torch
from .diagnose import OUT,save
from .range_api import RangeEvidencePredictor
from .range_evidence import EvidenceNet
from .range_benchmark import workloads


def run():
    torch.set_num_threads(1);point=OUT/'training/point_residual_full/epoch06.pt';model=EvidenceNet();model.initialize(point);cp=OUT/'training/range_profile.pt';torch.save(dict(schema='legal_range_evidence16_1',config=dict(point_checkpoint=str(point)),model=model.state_dict(),threshold=.2),cp);predictor=RangeEvidencePredictor(cp,device='cuda:1');ex,packet,tracks=workloads()[0];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
    def call(k):return predictor.predict_segments(partial_graph=ex['graph'],trajectories=paths[:k],range_observations=packet,track_history=tracks)
    call(6);rows=[]
    for k in [1,4,6]:
        for cache in ['cold','warm']:
            if cache=='cold':predictor.clear_cache()
            else:call(k)
            tick=time.perf_counter();call(k);torch.cuda.synchronize(predictor.device);rows.append(dict(k=k,cache=cache,ms=(time.perf_counter()-tick)*1000,profile=predictor.last_batch_profile));print(rows[-1],flush=True)
    prof=cProfile.Profile();prof.enable();call(6);prof.disable();stream=io.StringIO();pstats.Stats(prof,stream=stream).sort_stats('cumulative').print_stats(35);(OUT/'range_profile.txt').write_text(stream.getvalue());save(OUT/'range_profile.json',dict(role='Timing diagnosis, synthetic zero residual weights, NOT acceptance or accuracy',rows=rows))


if __name__=='__main__':run()
