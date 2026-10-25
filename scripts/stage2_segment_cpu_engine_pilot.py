"""Bounded CPU execution experiment; neural timings are NOT complete latency."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.special import expit
from eventwm_graph.stage2_segment.api import SegmentPredictor
from eventwm_graph.stage2_segment.benchmark import inputs
from eventwm_graph.stage2_segment.common import OUT,dump,sha,snapshot
from eventwm_graph.stage2_segment.cuda import critical_blocks


def main():
    sys.path.insert(0,str(OUT/'runtime_deps/ort119'))
    import onnx,onnxruntime as ort
    dest=OUT/'cpu_engine_pilot';dest.mkdir(exist_ok=False)
    checkpoint=OUT/'training/pilot_ray/last.pt';model=SegmentPredictor(checkpoint)
    network=model.net;weights=[(network[i].weight.detach().numpy(),network[i].bias.detach().numpy()) for i in (0,2,4)]
    export=dest/'active88.onnx'
    torch.onnx.export(network,torch.zeros(8192,88),export,opset_version=14,input_names=['features'],output_names=['values'],
        dynamic_axes={'features':{0:'cells'},'values':{0:'cells'}})
    sessions={}
    for threads in (1,2):
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for key in ('session.intra_op.allow_spinning','session.inter_op.allow_spinning'):options.add_session_config_entry(key,'0')
        sessions[threads]=ort.InferenceSession(str(export),options,providers=['CPUExecutionProvider'])
    def reference(x):
        values=np.empty((len(x),4),np.float32)
        with torch.inference_mode():
            for lo in range(0,len(x),8192):values[lo:lo+8192]=network(torch.from_numpy(x[lo:lo+8192])).numpy()
        return values
    def run(mode,x):
        if mode=='torch1':return reference(x),0
        if mode=='numpy1':
            values=np.empty((len(x),4),np.float32)
            for lo in range(0,len(x),8192):
                hidden=x[lo:lo+8192]
                for index,(weight,bias) in enumerate(weights):
                    hidden=hidden@weight.T;hidden+=bias
                    if index<2:hidden*=expit(hidden)
                values[lo:lo+8192]=hidden
        else:
            threads=int(mode[3]);session=sessions[threads]
            if mode.endswith('full'):values=session.run(None,{'features':x})[0]
            else:
                values=np.empty((len(x),4),np.float32)
                for lo in range(0,len(x),8192):values[lo:lo+8192]=session.run(None,{'features':x[lo:lo+8192]})[0]
        blocks=critical_blocks(values)
        with torch.inference_mode():
            for block in blocks:
                lo=int(block)*8192;hi=min(lo+8192,len(x));values[lo:hi]=network(torch.from_numpy(x[lo:hi])).numpy()
        return values,len(blocks)
    modes=['torch1','numpy1','ort1_full','ort1_chunk','ort2_full','ort2_chunk'];records=[];equivalence=[]
    for index,ex in enumerate(inputs('single')['representative'][:6]):
        context=model.feature_class(ex['graph'],ex['trajectory'],model.prepare_graph(ex['graph']));keys=context.candidates('both')
        x,_=context.transform(keys,'both');x=np.ascontiguousarray(x[:,16:104]);expected=reference(x)
        for mode in modes:
            actual,count=run(mode,x)
            ep=torch.softmax(torch.from_numpy(expected[:,:3]),-1).numpy();ap=torch.softmax(torch.from_numpy(actual[:,:3]),-1).numpy()
            np.testing.assert_allclose(ap,ep,rtol=0,atol=2e-6);np.testing.assert_array_equal(ap.argmax(-1),ep.argmax(-1))
            np.testing.assert_allclose(actual[:,3],expected[:,3],rtol=0,atol=2e-6)
            equivalence.append(dict(mode=mode,root_id=ex['root_id'],cells=len(x),probability_max_error=float(np.max(abs(ap-ep))),
                height_residual_max_error=float(np.max(abs(actual[:,3]-expected[:,3]))),exact_states=True,near_tie_reference_blocks=count))
        for repeat in range(8):
            order=modes if (index+repeat)%2 else modes[::-1]
            for mode in order:
                cpu=time.process_time();tick=time.perf_counter();holder,count=run(mode,x);wall=time.perf_counter()-tick
                records.append(dict(mode=mode,root_id=ex['root_id'],cells=len(x),seconds=wall,cpu_seconds=time.process_time()-cpu,near_tie_reference_blocks=count))
    summary={m:dict(p50_ms=float(np.median([r['seconds'] for r in records if r['mode']==m]))*1000,
        p95_ms=float(np.quantile([r['seconds'] for r in records if r['mode']==m],.95))*1000) for m in modes}
    dump(dest/'results.json',dict(summary=summary,rows=records,equivalence=equivalence,source=snapshot(),checkpoint_sha256=sha(checkpoint),onnx_sha256=sha(export),
        versions=dict(onnx=onnx.__version__,onnxruntime=ort.__version__,torch=torch.__version__,numpy=np.__version__),
        wheels={p.name:sha(p) for p in (OUT/'runtime_wheels').glob('*.whl')},providers=['CPUExecutionProvider'],
        scope='NEURAL EXECUTION ONLY, initialized, accuracyguardincluded; NOT complete inference or acceptance; fullfitGPU0/oneCPU active',
        dependencies='isolated runtime_deps/ort119; original environment unchanged',reference_block_size=8192,near_tie_logit_margin=1e-4))
    print(summary,flush=True)


if __name__=='__main__':main()
