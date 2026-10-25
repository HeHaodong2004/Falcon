"""Bounded execution microstudy; no checkpoint fitting or dev labels."""
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.benchmark import inputs
from eventwm_graph.stage2_segment.features import SegmentFeatures
from eventwm_graph.stage2_segment.fused import transform
from eventwm_graph.stage2_segment.common import OUT,dump,snapshot


def main():
    dest=OUT/'neural_execution_pilot';dest.mkdir(exist_ok=False)
    dump(dest/'protocol.json',dict(source=snapshot(),scope='execution-only fitting-checkpoint microstudy; saved legal inputs, no development targets read',
        modes=['reference_fp32','script_fp32','optimized_script_fp32','dynamic_int8'],batch_sizes=[8192,65536],repeats=20,
        selection='FP32 promotes only under2e-6probability/height tolerance with hard-state equivalence. Int8 is a scientifically distinct precision variant and requires a separate predeclared fitting-only proper-score/fullAPI quality gate; never claim2e-6equivalence.',
        contention='full extraction14workers plus baseline evaluation2workers; exploratory only'))
    model=SegmentPredictor(OUT/'training/pilot_ray/last.pt',execution='optimized',future_format='native')
    net=model.net;example=torch.zeros((8,88))
    scripted=torch.jit.freeze(torch.jit.trace(copy.deepcopy(net),example).eval())
    optimized=torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(copy.deepcopy(net),example).eval()))
    quantized=torch.quantization.quantize_dynamic(copy.deepcopy(net),{nn.Linear},dtype=torch.qint8).eval()
    modes=dict(reference_fp32=net,script_fp32=scripted,optimized_script_fp32=optimized,dynamic_int8=quantized)
    examples=inputs('single')['representative'][:6];records=[];errors=[];holders={}
    rng=np.random.default_rng(2430)
    for index,ex in enumerate(examples):
        context=SegmentFeatures(ex['graph'],ex['trajectory'],model.prepare_graph(ex['graph']));keys=context.candidates('both');x,h=transform(context,keys);tensor=torch.from_numpy(x)
        with torch.inference_mode():
            reference=net(tensor).numpy();p=torch.softmax(torch.from_numpy(reference[:,:3]),-1).numpy()
            for name,variant in modes.items():
                y=variant(tensor).numpy();q=torch.softmax(torch.from_numpy(y[:,:3]),-1).numpy()
                errors.append(dict(model=name,root_id=ex['root_id'],cells=len(x),max_probability_error=float(np.max(abs(p-q))),
                    mean_probability_error=float(np.mean(abs(p-q))),hard_state_differences=int(np.count_nonzero(p.argmax(-1)!=q.argmax(-1))),
                    max_height_error=float(np.max(abs(reference[:,3]-y[:,3])))))
            for batch in (8192,65536):
                for variant in modes.values():
                    for start in range(0,len(x),batch):variant(tensor[start:start+batch])
                for repeat in range(20):
                    for name in rng.permutation(list(modes)):
                        tick=time.perf_counter();outputs=[]
                        for start in range(0,len(x),batch):outputs.append(modes[name](tensor[start:start+batch]))
                        holders[name]=outputs;end=time.perf_counter()
                        records.append(dict(model=name,batch=batch,root_id=ex['root_id'],cells=len(x),seconds=end-tick))
    summary={}
    for name in modes:
        summary[name]={str(batch):dict(p50_ms=float(np.median([r['seconds'] for r in records if r['model']==name and r['batch']==batch]))*1000,
            p95_ms=float(np.quantile([r['seconds'] for r in records if r['model']==name and r['batch']==batch],.95))*1000)
            for batch in (8192,65536)}
    dump(dest/'results.json',dict(summary=summary,errors=errors,calls=records,quantized_backend=torch.backends.quantized.engine,
        script_graph=str(optimized.graph),torch_config=torch.__config__.show()))
    print(json.dumps(dict(summary=summary,errors=errors)),flush=True)


if __name__=='__main__':main()
