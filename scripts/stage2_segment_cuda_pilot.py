"""F32 transfer-inclusive neural pilot on available workstation GPUs."""
import copy
import json
import time
import numpy as np
import torch
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.benchmark import inputs
from eventwm_graph.stage2_segment.features import SegmentFeatures
from eventwm_graph.stage2_segment.fused import transform
from eventwm_graph.stage2_segment.common import OUT,dump,snapshot


def main():
    dest=OUT/'cuda_execution_pilot';dest.mkdir(exist_ok=False)
    dump(dest/'protocol.json',dict(source=snapshot(),purpose='bounded execution-only F32GPU transfer+network+return pilot; no dev labels',
        precision='float32,TF32disabled',variants=['cpu8192','cuda1_pageable','cuda1_pinned'],repeats=20,
        gate='scientific F32equivalence2e-6 remains unchanged; only promote after full end-to-end proof/timing; no GPU-only forward speed claim'))
    torch.backends.cuda.matmul.allow_tf32=False
    model=SegmentPredictor(OUT/'training/pilot_ray/last.pt',execution='optimized',future_format='native')
    cpu=model.net;device=torch.device('cuda:1');tick=time.perf_counter();gpu=copy.deepcopy(cpu).to(device);torch.cuda.synchronize(device)
    initialization=time.perf_counter()-tick;records=[];errors=[];holders={};rng=np.random.default_rng(2431)
    for ex in inputs('single')['representative'][:12]:
        c=SegmentFeatures(ex['graph'],ex['trajectory'],model.prepare_graph(ex['graph']));keys=c.candidates('both');x,h=transform(c,keys);tensor=torch.from_numpy(x)
        pinned=torch.empty(x.shape,dtype=torch.float32,pin_memory=True)
        def call(name):
            if name=='cpu8192':return torch.cat([cpu(tensor[i:i+8192]) for i in range(0,len(x),8192)],0)
            if name=='cuda1_pinned':pinned.copy_(tensor);data=pinned.to(device,non_blocking=True)
            else:data=tensor.to(device)
            return gpu(data).cpu()
        with torch.inference_mode():
            reference=call('cpu8192');p=torch.softmax(reference[:,:3],-1).numpy()
            for name in ('cpu8192','cuda1_pageable','cuda1_pinned'):
                y=call(name);q=torch.softmax(y[:,:3],-1).numpy()
                errors.append(dict(model=name,root_id=ex['root_id'],cells=len(x),max_probability_error=float(np.max(abs(p-q))),
                    mean_probability_error=float(np.mean(abs(p-q))),hard_state_differences=int(np.count_nonzero(p.argmax(-1)!=q.argmax(-1))),
                    max_height_error=float(np.max(abs(reference[:,3].numpy()-y[:,3].numpy())))))
            for repeat in range(20):
                for name in rng.permutation(['cpu8192','cuda1_pageable','cuda1_pinned']):
                    tick=time.perf_counter();holders[name]=call(name);end=time.perf_counter()
                    records.append(dict(model=name,root_id=ex['root_id'],cells=len(x),seconds=end-tick))
    summary={name:dict(p50_ms=float(np.median([r['seconds'] for r in records if r['model']==name]))*1000,
        p95_ms=float(np.quantile([r['seconds'] for r in records if r['model']==name],.95))*1000) for name in ('cpu8192','cuda1_pageable','cuda1_pinned')}
    dump(dest/'results.json',dict(summary=summary,errors=errors,calls=records,device=torch.cuda.get_device_name(device),initialization_seconds=initialization,
        max_cuda_allocated=torch.cuda.max_memory_allocated(device),max_cuda_reserved=torch.cuda.max_memory_reserved(device),background_contention='full extraction14workers and baseline evaluation2workers'))
    print(json.dumps(dict(summary=summary,max_probability_error=max(r['max_probability_error'] for r in errors),max_height_error=max(r['max_height_error'] for r in errors))),flush=True)


if __name__=='__main__':main()
