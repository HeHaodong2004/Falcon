"""Exploratory matched real K6 neural-layout cost; NOT complete inference."""
import copy,time
import numpy as np
import torch
from .common import OUT,read,save,source
from .spatial import SpatialNet,build,spatial_probabilities
from .spatial_cuda import QueryKernel,path_arrays
from eventwm_graph.stage2_segment.benchmark import inputs

def main():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device='cuda:0';g=inputs('multi')[0];graph=g['graph'];paths=[np.array(p['trajectory'])for p in g['paths'][:6]];keys=np.array([c['cell']for c in graph['cells']],np.int32);states=np.array([c['state']for c in graph['cells']],np.uint8);height=np.array([c['height']for c in graph['cells']],np.float32);geo=[graph['resolution'],graph['height'],graph['clearance']]
 ds=[build(keys,states,height,p,geo,make_query=False,make_candidate=False)for p in paths];h=max(int(d['shape'][0])for d in ds);w=max(int(d['shape'][1])for d in ds);maps=np.zeros((6,6,h//2,w//2),np.float32);fine=np.zeros((6,3,h,w),np.float32)
 for i,d in enumerate(ds):a,b=map(int,d['shape']);maps[i,:,:a//2,:b//2]=d['maps'];fine[i,:,:a,:b]=d['fine']
 mt=torch.from_numpy(maps).to(device);ft=torch.from_numpy(fine).to(device);pa=path_arrays(paths,[d['lo']for d in ds],[d['shape']for d in ds],[d['resolution']for d in ds]);t={k:torch.from_numpy(v).to(device)for k,v in pa.items()};kernel=QueryKernel(device);q=kernel(mt,t['path'],t['pre'],t['globals'],t['spec']);saved=torch.load(OUT/'training/spatial_bce/epoch04.pt',map_location='cpu');base=SpatialNet(**saved['model_config']).eval();base.load_state_dict(saved['model']);models={};data={};ref=None
 for precision in ('float32','float16'):
  for layout,fmt in [('nchw',torch.contiguous_format),('channels_last',torch.channels_last)]:
   dtype=torch.float32 if precision=='float32'else torch.float16;name=precision+'_'+layout;models[name]=copy.deepcopy(base).to(device=device,dtype=dtype,memory_format=fmt);data[name]=[x.to(dtype=dtype).contiguous(memory_format=fmt)for x in (mt,q,ft)]
 rows=[];differences={}
 with torch.inference_mode():
  for name,m in models.items():
   for _ in range(3):out=m(*data[name])
   pred=spatial_probabilities(out.float()).cpu().numpy()
   if ref is None:ref=pred
   differences[name]=float(np.max(abs(pred-ref)))
  for rep in range(20):
   for name in list(models)[::1 if rep%2 else -1]:
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record(torch.cuda.current_stream(device));start=time.perf_counter();out=models[name](*data[name]);b.record(torch.cuda.current_stream(device));b.synchronize();rows.append(dict(name=name,wall_ms=(time.perf_counter()-start)*1000,gpu_stream_ms=a.elapsed_time(b)))
 result=dict(source=source(),scope='Neural-only exploratory K6forward, realpaths but competing fullcacheextraction and bothCNNtraining jobs. NOT an endtoend latency gate.',tf32=False,supersedes='layout_probe.json had defaultCuDNN TF32allowed; preserved, notaccepted asstrictnumerical comparison',shape=[6,h,w],probability_difference_from_f32_nchw=differences,summary={name:dict(p50_gpu_ms=float(np.median([r['gpu_stream_ms']for r in rows if r['name']==name])),p95_gpu_ms=float(np.quantile([r['gpu_stream_ms']for r in rows if r['name']==name],.95)))for name in models},calls=rows);save(OUT/'layout_probe_v2.json',result);print(result['summary'],differences,flush=True)
if __name__=='__main__':main()
