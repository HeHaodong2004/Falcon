import cProfile
import io
import pstats
import time
import numpy as np
import torch
from .diagnose import OUT,save
from .range_radix_api import RadixRangePredictor
from .range_benchmark import workloads


def run():
    torch.set_num_threads(1);m=RadixRangePredictor(OUT/'training/range_evidence64/selected.pt',device='cuda:1');ex,packet,tracks=workloads()[20];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
    def call():return m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
    call();prof=cProfile.Profile();prof.enable();tick=time.perf_counter();call();torch.cuda.synchronize(m.device);warm=(time.perf_counter()-tick)*1000;prof.disable();stream=io.StringIO();pstats.Stats(prof,stream=stream).sort_stats('cumulative').print_stats(40);(OUT/'radix_profile.txt').write_text(stream.getvalue());warm_profile=m.last_batch_profile;m.clear_cache();tick=time.perf_counter();call();torch.cuda.synchronize(m.device);cold=(time.perf_counter()-tick)*1000;result=dict(group=20,cold_ms=cold,warm_ms=warm,warm_profile=warm_profile,cold_profile=m.last_batch_profile,mesh_prepare_seconds=m.active_mesh.prepare_seconds,radix_build_seconds=m.active_mesh.radix_build_seconds);save(OUT/'radix_profile.json',result);print(result,flush=True)


if __name__=='__main__':run()
