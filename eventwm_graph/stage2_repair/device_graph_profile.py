"""Locate remaining complete-API host cost; GPU forward time is separate."""
import cProfile
import io
import pstats
import time
import numpy as np
import torch
from .diagnose import OUT,save
from .range_benchmark import workloads
from .range_gpu_graph import DeviceGraphSpatialPredictor as ExactFastSpatialPredictor


def run():
    torch.set_num_threads(1);m=ExactFastSpatialPredictor(OUT/'training/support_pretrained256_control/selected.pt',device='cuda:1');m.config['observed_obstacle_schema']='capped_2m_complete';ex,packet,tracks=workloads()[20];paths=[np.asarray(p['trajectory'],np.float64)for p in ex['paths'][:6]]
    def call():return m.predict_segments(partial_graph=ex['graph'],trajectories=paths,range_observations=packet,track_history=tracks)
    call();records=[];events=[];original=m.model.probabilities
    def probabilities(*args,**kwargs):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();result=original(*args,**kwargs);b.record();events.append((a,b));return result
    with torch.cuda.device(m.device):
        m.model.probabilities=probabilities
        for cache in ['warm','cold']:
            if cache=='cold':m.clear_cache()
            prof=cProfile.Profile();events.clear();prof.enable();tick=time.perf_counter();call();torch.cuda.synchronize(m.device);elapsed=time.perf_counter()-tick;prof.disable()
            stream=io.StringIO();pstats.Stats(prof,stream=stream).sort_stats('cumulative').print_stats(50);(OUT/('cpu_cost_device_graph_%s.txt'%cache)).write_text(stream.getvalue())
            records.append(dict(cache=cache,complete_ms=elapsed*1000,neural_forward_gpu_ms=sum(a.elapsed_time(b)for a,b in events),profile=m.last_batch_profile));print(records[-1],flush=True)
    save(OUT/'cpu_cost_device_graph.json',records)


if __name__=='__main__':run()
