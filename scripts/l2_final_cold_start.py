"""One fresh-process/fresh-Numba-cache complete call, separate from steady P95."""
import time
START=time.perf_counter()
import importlib.util,json,resource,sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
import torch
torch.set_num_threads(1)
from eventwm_graph.l2_final.common import OUT,OLD,sha,dump
from eventwm_graph.l2_intermittent.model import IntermittentWorldModel
from eventwm_graph.l2_intermittent.common import read,canonical
path=BASE/'docs/l2/intermittent_wm_20261022/reviews/reviewer_c_benchmark.py'
spec=importlib.util.spec_from_file_location('cold_benchmark',path);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
import_s=time.perf_counter()-START
tick=time.perf_counter();model=IntermittentWorldModel.load(OUT/'training/checkpoint.pt');load_ms=(time.perf_counter()-tick)*1000
root='7419795f11b29b176cc887b88fd48cf12e6c8a8ae46240696940ef8b310a0fa9';folder=OLD/'snapshot/roots'/root
legal=read(folder/'legal_root.json.gz');query,_=canonical(folder)
tick=time.perf_counter();prior=model.initialize(legal['history'][:-1],bench.prefix_snapshot(legal));init_ms=(time.perf_counter()-tick)*1000
meta=dict(root_id=root,scene_id='hm3d_JFgrz9MNz4b',tags=['first60'],history_length=len(legal['history']),tracked_max=max(c.n for c in prior.components))
tick=time.perf_counter();call=bench.one_call(model,prior,legal['history'][-1],query,meta,'fresh_process_and_numba_cache',16,'primary',0);wall_ms=(time.perf_counter()-tick)*1000
dump(OUT/'fresh_process_startup.json',dict(import_seconds=import_s,model_load_ms=load_ms,public_prefix_initialization_ms=init_ms,first_call_including_readouts_and_analytic_control_ms=wall_ms,
     complete_call=call,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model_version=model.model_version,checkpoint_sha256=sha(OUT/'training/checkpoint.pt'),
     scope='one descriptive representative root, fresh process and empty NUMBA_CACHE_DIR; first call wall includes separate propagation control and readouts; not the warm/cold population P95'))
assert call['complete'],call
