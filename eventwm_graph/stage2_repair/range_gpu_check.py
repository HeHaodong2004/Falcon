"""Matched CPU/GPU observed-only range reconstruction and first-hit check."""
import ctypes as ct
import numpy as np
import torch
from .diagnose import OUT,read,save
from .observed_range_mesh import ObservedRangeMesh
from .range_gpu import GpuObservedRangeMesh


def run():
    torch.set_num_threads(1);reports=[]
    for path in sorted((OUT/'range_replay').glob('*.npz'))[:4]:
        with np.load(path)as z:a={k:z[k]for k in z.files}
        kwargs=dict(ranges=a['depth'],origins=a['origins'],rotations=a['rotations'],timestamps=a['timestamps'],now=a['timestamps'][-1],resolution=.1016884210526,device='cuda:1')
        cpu=ObservedRangeMesh(**kwargs);gpu=GpuObservedRangeMesh(**kwargs)
        rng=np.random.default_rng(2971);n=10000
        origins=np.repeat(a['origins'][-1:],n,axis=0).astype(np.float32);directions=rng.normal(size=(n,3)).astype(np.float32);directions/=np.linalg.norm(directions,axis=1)[:,None];limits=np.full(n,10.,np.float32)
        results=[]
        for mesh in [cpu,gpu]:
            k=mesh.kernel;out=torch.empty(n,device=k.device);k.module.launch('test_rays',n,[k.tensor(origins),k.tensor(directions),k.tensor(limits),ct.c_int(n),out,*k.mesh]);results.append(out.cpu().numpy())
        difference=abs(results[0]-results[1]);report=dict(root_id=path.stem,cpu_triangles=cpu.triangles,gpu_triangles=gpu.triangles,cpu_prepare_s=cpu.prepare_seconds,gpu_prepare_s=gpu.prepare_seconds,rays=n,hit_difference_over_1mm=int((difference>.001).sum()),max_difference_m=float(difference.max()),mean_difference_m=float(difference.mean()))
        reports.append(report);print(report,flush=True)
        # Startup is separate from steady map construction.
        repeats=[]
        for _ in range(3):repeats.append(GpuObservedRangeMesh(**kwargs).prepare_seconds)
        report['gpu_repeated_prepare_s']=repeats
    save(OUT/'range_gpu_equivalence.json',reports)


if __name__=='__main__':run()
