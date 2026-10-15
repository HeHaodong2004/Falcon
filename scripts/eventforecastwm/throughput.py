"""Bounded training-throughput measurement on existing frozen legal windows."""
import argparse
import json
from pathlib import Path
import time
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.model_posterior import PosteriorForecastWorldModel
from eventforecastwm.train import load_roots,prepare_roots,prepared_batch,loss
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.guard import Guard


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--prepared-cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);guard=Guard(out.parent)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for throughput measurement')
    roots=load_roots(a.manifest,read(a.protocol)['subsets']['full'][:128],guard)
    template=PosteriorForecastWorldModel(noise='integrated');prepare(template,roots,a.manifest,a.prepared_cache,prepare_roots,guard)
    results=[]
    for size in (32,64,128):
        guard.check();torch.manual_seed(20261014)
        model=PosteriorForecastWorldModel(noise='integrated').to(a.device).train()
        batch=prepared_batch([r['windows'][0] for r in roots[:size]],a.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        torch.cuda.reset_peak_memory_stats(a.device);times=[]
        for step in range(15):
            guard.check();torch.cuda.synchronize(a.device);start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True);value,_=loss(model,batch);value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize(a.device)
            if step>=3:times.append(time.perf_counter()-start)
        results.append(dict(batch_size=size,measured_steps=len(times),seconds_per_step=sum(times)/len(times),
            root_presentations_per_second=size*len(times)/sum(times),peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device),
            padded_humans=batch['base'].humans.shape[1],history_steps=batch['base'].humans.shape[2]))
        del optimizer,model,batch;torch.cuda.empty_cache()
    save_json(out,dict(results=results,source_sha256=sha256(__file__),manifest_sha256=sha256(a.manifest),
        purpose='throughput only; same frozen root prefix, different padding maxima; no forecast performance claim',
        temporary_optimization_discarded=True))
    print(json.dumps(results))


if __name__=='__main__':main()
