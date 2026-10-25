"""Real measured-range mesh reduction: first-hit differences and build cost."""
import ctypes as ct
import numpy as np
import torch
from .diagnose import OUT,save
from .range_planar import PlanarRangeMesh
from .range_gpu import GpuObservedRangeMesh


def run():
    torch.set_num_threads(1);reports=[]
    for path in sorted((OUT/'range_replay').glob('*.npz'))[:4]:
        with np.load(path)as z:a={k:z[k]for k in z.files}
        kw=dict(ranges=a['depth'],origins=a['origins'],rotations=a['rotations'],timestamps=a['timestamps'],now=a['timestamps'][-1],resolution=.1016884210526,device='cuda:1');full=GpuObservedRangeMesh(**kw);reduced=PlanarRangeMesh(**kw);rng=np.random.default_rng(2971);n=10000;o=np.repeat(a['origins'][-1:],n,0).astype(np.float32);d=rng.normal(size=(n,3)).astype(np.float32);d/=np.linalg.norm(d,axis=1)[:,None];results=[]
        for mesh in [full,reduced]:
            k=mesh.kernel;out=torch.empty(n,device=k.device);k.module.launch('test_rays',n,[k.tensor(o),k.tensor(d),k.tensor(np.full(n,10.,np.float32)),ct.c_int(n),out,*k.mesh]);results.append(out.cpu().numpy())
        difference=abs(results[0]-results[1]);rec=dict(root_id=path.stem,full_triangles=full.triangles,planar_triangles=reduced.triangles,full_build_s=full.prepare_seconds,planar_build_s=reduced.prepare_seconds,rays=n,over_1mm=int((difference>.001).sum()),max_difference_m=float(difference.max()),mean_difference_m=float(difference.mean()),repeated_planar_build_s=[PlanarRangeMesh(**kw).prepare_seconds for _ in range(3)]);reports.append(rec);print(rec,flush=True)
    save(OUT/'planar_real_ray_check.json',reports)


if __name__=='__main__':run()
