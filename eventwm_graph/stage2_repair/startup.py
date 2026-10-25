"""Fresh-process import, model initialization and first complete K6 timing."""
import argparse
import os
import time
import numpy as np
import torch
from .diagnose import OUT,CKPT,save,sha
from .predictor import make_predictor
from eventwm_graph.stage2_hf.baseline import ThresholdBaseline
from eventwm_graph.stage2_segment.benchmark import inputs


def main(kind,checkpoint,device='cuda:0',name='startup',global_features=False):
    imported=time.perf_counter();process_start=float(os.environ['EVENTWM_STAGE2_PROCESS_START'])
    t=time.perf_counter();ex=inputs('multi')[0];g=ex['graph'];paths=[np.asarray(v['trajectory'],np.float64)for v in ex['paths'][:6]];input_seconds=time.perf_counter()-t
    t=time.perf_counter();m=make_predictor(checkpoint,device=device,global_features=global_features,increment_format='columns')if kind=='candidate'else ThresholdBaseline(CKPT,threshold=.2,execution='optimized',future_format='native',device=device,feature_threads=1);init_seconds=time.perf_counter()-t
    torch.cuda.synchronize(device);t=time.perf_counter();d=m.predict_segments(partial_graph=g,trajectories=paths);torch.cuda.synchronize(device);call_seconds=time.perf_counter()-t
    save(OUT/('startup_'+name+'_'+kind+'.json'),dict(kind=kind,checkpoint_sha256=sha(checkpoint if kind=='candidate'else CKPT),device=device,
        process_import_seconds=imported-process_start,fixture_read_seconds=input_seconds,model_initialization_seconds=init_seconds,
        first_complete_K6_seconds=call_seconds,total_since_runner_start_seconds=time.perf_counter()-process_start,
        candidates=[len(v['candidate_cells'])for v in d],materialized_future_graphs=len(d),serialization_included=False,
        numba_cache=os.environ.get('NUMBA_CACHE_DIR'),scope='Fresh interpreter; explicitly empty per-process Numba cache when requested; CUDA driver initialization/model loading/first complete graph call separate. Input fixture read is startup-only, excluded from initialized-call timer.'))
    print(kind,'import',imported-process_start,'initialize',init_seconds,'first_K6',call_seconds,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['baseline','candidate'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--name',default='startup');p.add_argument('--global-features',action='store_true');main(**vars(p.parse_args()))
