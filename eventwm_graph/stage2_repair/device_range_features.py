"""Same16 range features; query keys/floors remain device-resident."""
from collections import OrderedDict
import ctypes as ct
import numpy as np
import torch
from .legal_camera import current_extrinsic,base_rotations
from .gpu_obstacle_distance import SOURCE as GATHER
from .cuda_module import CudaModule


def extract(predictor,prepared,contexts,batch):
    kernel=predictor.range_kernel;module=kernel.module;mesh=predictor.active_mesh;module.active_mesh=mesh;device=predictor.device;r=prepared.resolution;paths=[c.path for c in contexts];low=batch['low'];nx,nz=map(int,batch['shape']);area=batch['area'];uk=batch['union'];ff=batch['floors'];nu=len(uk);keys=batch['keys'];n=len(keys)
    def tensor(a,dtype=None):return torch.from_numpy(np.ascontiguousarray(a,dtype=dtype)).to(device,non_blocking=True)
    views=[];parameters=[];dedup={};steps=[];initial=[];interval=[];when=[];durations=[]
    for p in paths:
        translation,extrinsic=current_extrinsic(p[:1],predictor.current_origin,predictor.current_rotation);base=base_rotations(p[:,4]);origins=p[:,1:4]+base@translation;rots=base@extrinsic;cams=np.concatenate([origins,rots.reshape(-1,9),p[:,2:3]],1);params,_=kernel.camera_parameters(p,translation,extrinsic,r);sequence=[]
        for c,pr in zip(cams,params):
            identity=c.tobytes()
            if identity not in dedup:dedup[identity]=len(views);views.append(c);parameters.append(pr)
            sequence.append(dedup[identity])
        initial.append(sequence[0]);steps.append(np.asarray(sequence[1:],np.int32));duration=max(p[-1,0]-p[0,0],1.e-4);durations.append(duration);interval.extend(np.diff(p[:,0]));when.extend((p[1:,0]-p[0,0])/duration)
    cameras=np.asarray(views,np.float64);nv=len(cameras)
    if nv*area*4>512*1024*1024:raise ValueError('device query range grid exceeds512MiB')
    cam=tensor(cameras);step=tensor(np.concatenate(steps));offset=tensor(np.r_[0,np.cumsum([len(s)for s in steps])].astype(np.int32));init=tensor(initial,np.int32);param=tensor(parameters,np.float64);interval_t=tensor(interval,np.float64);when_t=tensor(when,np.float64);duration_t=tensor(durations,np.float64)
    if not hasattr(predictor,'pipeline_body_module'):predictor.pipeline_body_module=CudaModule(GATHER,device)
    if not hasattr(mesh,'pipeline_body_cache'):mesh.pipeline_body_cache=OrderedDict()
    padding=int(np.ceil(2./r))+2;body_low=low-padding;body_shape=batch['shape']+2*padding;body_key=tuple(body_low)+tuple(body_shape)+(float(paths[0][0,2]),)
    if body_key not in mesh.pipeline_body_cache:
        bx,bz=map(int,body_shape);field=torch.zeros((bx,bz),device=device,dtype=torch.int32);mesh.builder.launch('observed_obstacles',mesh.triangles,[mesh.kernel.static['triangles'],ct.c_int(mesh.triangles),ct.c_float(paths[0][0,2]+.2),ct.c_float(paths[0][0,2]+1.4),ct.c_float(r),ct.c_int(int(body_low[0])),ct.c_int(int(body_low[1])),ct.c_int(bx),ct.c_int(bz),field])
        _,_,squared=predictor.edt.transform((field>0).to(torch.uint8),torch.empty_like(field,dtype=torch.float32),r,nearest_height=False,channel_count=1);mesh.pipeline_body_cache[body_key]=squared[0]
        while len(mesh.pipeline_body_cache)>8:mesh.pipeline_body_cache.popitem(last=False)
    obstacle=torch.empty(nu,device=device);predictor.pipeline_body_module.launch('gather_obstacle',nu,[mesh.pipeline_body_cache[body_key],uk,ct.c_int(nu),ct.c_int(int(body_low[0])),ct.c_int(int(body_low[1])),ct.c_int(int(body_shape[1])),ct.c_double(r),obstacle])
    if not hasattr(mesh,'pipeline_support_cache'):mesh.pipeline_support_cache=OrderedDict()
    union_positions=np.unique(np.concatenate(batch['positions']),axis=0);support_key=(prepared.identity,union_positions.tobytes(),float(paths[0][0,2]),nu)
    if support_key not in mesh.pipeline_support_cache:
        support=torch.empty(nu,device=device);difference=torch.empty_like(support);module.launch('square_support',nu,[uk,ff,ct.c_int(nu),ct.c_double(r),support,difference,*mesh.kernel.mesh]);mesh.pipeline_support_cache[support_key]=(support,difference)
        while len(mesh.pipeline_support_cache)>4:mesh.pipeline_support_cache.popitem(last=False)
    support,difference=mesh.pipeline_support_cache[support_key];frame=torch.zeros((nv,area),dtype=torch.int32,device=device);out=torch.empty((n,16),device=device)
    module.launch('generated_views',nu*nv,[uk,ff,ct.c_int(nu),cam,ct.c_int(nv),batch['unknown'],ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(nx),ct.c_int(nz),ct.c_double(r),frame,*mesh.kernel.mesh])
    module.launch('wall_views',nv*105,[cam,ct.c_int(nv),batch['unknown'],ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(nx),ct.c_int(nz),ct.c_double(r),frame,*mesh.kernel.mesh])
    module.launch('pack_evidence',n,[frame,ct.c_int(area),keys,batch['ids'],batch['cells'],batch['queries'],ct.c_int(n),step,offset,init,param,interval_t,when_t,duration_t,ff,support,difference,obstacle,ct.c_double(r),ct.c_int(int(low[0])),ct.c_int(int(low[1])),ct.c_int(nz),out])
    return out
