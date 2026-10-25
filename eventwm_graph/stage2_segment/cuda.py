"""Optional F32 neural/F64 legal-line execution on an explicitly chosen GPU.

No labels or hidden map enter this service. Transfers/synchronization occur
inside prediction. Line probes near integer boundaries use the exact CPU
reference; this is numerical preservation, never candidate pruning.
"""
from collections import OrderedDict
import copy
import time
import numpy as np
import torch
from numba import njit
from .features import legal_lines_reference
from eventwm_graph.stage2_quality.features import inside_descriptors


def key_array(keys):
    keys=np.asarray(keys)
    if keys.ndim!=2 or keys.shape[1]!=2 or keys.dtype.kind not in ('i','u'):raise ValueError('CUDA keys must be integer[N,2]')
    if keys.dtype!=np.dtype(np.int32):
        if np.any(keys<np.iinfo(np.int32).min) or np.any(keys>np.iinfo(np.int32).max):raise ValueError('CUDA keys exceed int32 geometry')
    return np.ascontiguousarray(keys,dtype=np.int32)


@njit(cache=True)
def trajectory_constants(path):
    cosine=np.cos(path[:,4]);sine=np.sin(path[:,4]);duration=path[-1,0]-path[0,0];den=max(duration,1e-4)
    dt=np.empty(len(path),np.float64);when=np.empty(len(path),np.float64)
    rotation=0.;stationary=0.;unwrapped=0.;ylo=0.;yhi=0.
    for j in range(len(path)):
        dt[j]=path[j,0]-path[j-1,0] if j else 0.;when[j]=(path[j,0]-path[0,0])/den
        if j:
            dy=np.arctan2(np.sin(path[j,4]-path[j-1,4]),np.cos(path[j,4]-path[j-1,4]))
            rotation+=abs(dy);unwrapped+=dy;ylo=min(ylo,unwrapped);yhi=max(yhi,unwrapped)
            if (path[j,1]-path[j-1,1])**2+(path[j,3]-path[j-1,3])**2<1e-8:stationary+=dt[j]
    return cosine,sine,dt,when,np.array([rotation/(2*np.pi),(yhi-ylo)/(2*np.pi),stationary/den,(len(path)-1)/128.],np.float64)


@njit(cache=True)
def critical_blocks(values):
    blocks=np.zeros((len(values)+8191)//8192,np.uint8)
    for i in range(len(values)):
        a=values[i,0];b=values[i,1];c=values[i,2];m=max(a,b,c)
        if int(m-a<1e-4)+int(m-b<1e-4)+int(m-c<1e-4)>=2:blocks[i//8192]=1
    return np.flatnonzero(blocks)


class CudaExecution:
    def __init__(self,net,device):
        self.device=torch.device(device)
        if self.device.type!='cuda':raise ValueError('CUDA device required')
        torch.backends.cuda.matmul.allow_tf32=False
        self.reference=copy.deepcopy(net).eval()
        self.net=copy.deepcopy(net).to(self.device).eval()
        torch.cuda.synchronize(self.device)
        properties=torch.cuda.get_device_properties(self.device)
        self.device_info=dict(name=properties.name,compute_capability=[properties.major,properties.minor],total_memory_bytes=properties.total_memory,torch_cuda_runtime=torch.version.cuda)
        self.fractions=torch.from_numpy(np.arange(1,17,dtype=np.float64)/17.).to(self.device)
        self.last_profile={}
        from .cuda_kernel import FeatureKernel
        self.feature_kernel=FeatureKernel(self.device)

    def launch_features(self,context,keys,end_only=False):
        keys=key_array(keys)
        prepared=context.prepared
        if not hasattr(prepared,'cuda_grids'):prepared.cuda_grids=OrderedDict()
        cache=prepared.cuda_grids;key=(str(self.device),*map(int,context.lo),*map(int,context.shape))
        if key in cache:grid=cache[key];cache.move_to_end(key)
        else:
            grid=torch.from_numpy(context.states).to(self.device);cache[key]=grid
            while len(cache)>4:cache.popitem(last=False)
        cosine,sine,dt,when,globals=trajectory_constants(context.path)
        def tensor(value):return torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
        n=len(keys);view=torch.empty((n,20),device=self.device,dtype=torch.float32)
        lines=torch.empty((n,4 if end_only else 12),device=self.device,dtype=torch.float32)
        vb=torch.empty(n,device=self.device,dtype=torch.uint8);lb=torch.empty_like(vb)
        scale=max(float(np.max(abs(keys.astype(np.int64)))) if n else 0.,float(np.max(abs(context.path[:,[1,3]])))/context.resolution,1.)
        buffers=[tensor(keys),tensor(context.path),tensor(np.column_stack((cosine,sine,dt,when))),tensor(globals),grid,view,lines,vb,lb]
        self.feature_kernel.launch(buffers,[n,len(context.path),*map(int,grid.shape),*map(int,context.lo),context.resolution,64*np.finfo(np.float64).eps*(2*scale+1),int(end_only)])
        self.last_profile['cuda_feature_kernel_identity']=self.feature_kernel.identity
        return (lines,lb),(view,vb)

    def transform_device(self,context,keys,end_only=False):
        """Keep the full feature matrix on-device; return owned CPU heights."""
        keys=key_array(keys)
        begin=time.perf_counter();line_task,view_task=self.launch_features(context,keys,end_only)
        lines,lb=line_task;view,vb=view_task;prepared=context.prepared
        prepared.map_features(np.empty((0,2),np.int32));map_lo,free,blocked=prepared.integrals
        key=(str(self.device),*map(int,context.lo),*map(int,context.shape))
        if not hasattr(prepared,'cuda_geometry'):prepared.cuda_geometry=OrderedDict()
        cache=prepared.cuda_geometry
        if key in cache:geometry=cache[key];cache.move_to_end(key)
        else:
            geometry=[torch.from_numpy(np.ascontiguousarray(a)).to(self.device) for a in (context.base_height,*context.dist,free,blocked)]
            cache[key]=geometry
            while len(cache)>4:cache.popitem(last=False)
        grid=prepared.cuda_grids[key]
        length=float(np.linalg.norm(np.diff(context.xy,axis=0),axis=1).sum())
        constants=np.array([context.duration/5,length/5,context.resolution,context.path[-1,2]-context.path[0,2],np.cos(context.path[-1,4]-context.path[0,4]),np.sin(context.path[-1,4]-context.path[0,4])],np.float32)
        if not np.isfinite(constants).all():raise ValueError('nonfinite legal segment constants')
        n=len(keys);width=4 if end_only else 12
        x=torch.empty((n,76+width),device=self.device,dtype=torch.float32);baseline=torch.empty(n,device=self.device,dtype=torch.float32)
        buffers=[torch.from_numpy(keys).to(self.device),view,lines,grid,*geometry,torch.from_numpy(constants).to(self.device),x,baseline]
        uploaded=time.perf_counter()
        self.feature_kernel.launch_assembly(buffers,[n,*map(int,grid.shape),*map(int,context.lo),free.shape[0]-1,free.shape[1]-1,*map(int,map_lo),width],
            [context.resolution,*map(float,context.xy[0]),*map(float,context.xy[-1]),context.path[0,2]])
        # Only numerical boundary rows visit the CPU reference. No cells leave
        # the candidate domain and no observations or neural outputs are pruned.
        vcritical=np.flatnonzero(vb.cpu().numpy());lcritical=np.flatnonzero(lb.cpu().numpy())
        if len(vcritical):
            exact=inside_descriptors(keys[vcritical],context.path,context.resolution)[1]
            x[torch.from_numpy(vcritical).to(self.device),31:51]=torch.from_numpy(exact).to(self.device)
        if len(lcritical):
            exact=legal_lines_reference(keys[lcritical],context.path[-1:] if end_only else context.path,context.resolution,context.lo,context.states)
            if end_only:exact=exact[:,8:12]
            x[torch.from_numpy(lcritical).to(self.device),76:]=torch.from_numpy(np.ascontiguousarray(exact)).to(self.device)
        self.last_profile.update(ray_boundary_fallback_rows=len(lcritical),opportunity_boundary_fallback_rows=len(vcritical))
        corrected=time.perf_counter()
        if not bool(torch.isfinite(x).all().item()):raise ValueError('nonfinite legal features')
        heights=baseline.cpu().numpy();end=time.perf_counter()
        context.last_transform_profile=dict(feature_upload_launch_seconds=uploaded-begin,device_feature_and_boundary_seconds=corrected-uploaded,
            feature_finite_and_height_return_seconds=end-corrected,trajectory_seconds=None,map_seconds=None,legal_line_seconds=None)
        return x,heights

    def launch_lines(self,context,keys,end_only=False):
        if len(keys)>65536:
            return [self.launch_lines(context,keys[i:i+65536],end_only) for i in range(0,len(keys),65536)]
        prepared=context.prepared
        if not hasattr(prepared,'cuda_grids'):prepared.cuda_grids=OrderedDict()
        cache=prepared.cuda_grids;key=(str(self.device),*map(int,context.lo),*map(int,context.shape))
        if key in cache:
            grid=cache[key];cache.move_to_end(key)
        else:
            grid=torch.from_numpy(context.states).to(self.device);cache[key]=grid
            while len(cache)>4:cache.popitem(last=False)
        path=context.path;ids=[len(path)-1] if end_only else [min(1,len(path)-1),len(path)//2,len(path)-1]
        unique,inverse=np.unique(path[ids][:,[1,3]],axis=0,return_inverse=True)
        poses=torch.from_numpy(np.ascontiguousarray(unique)).to(self.device)
        targets=(torch.from_numpy(keys).to(self.device).double()+.5)*context.resolution
        # Separate Tensor operations retain the reference F64 rounding order.
        points=poses[None,:,None,:]+(targets[:,None,None,:]-poses[None,:,None,:])*self.fractions[None,None,:,None]
        coordinates=points/context.resolution
        lo=torch.from_numpy(context.lo.astype(np.int64)).to(self.device)
        ij=torch.floor(coordinates).long()-lo
        inside=(ij[...,0]>=0)&(ij[...,0]<grid.shape[0])&(ij[...,1]>=0)&(ij[...,1]<grid.shape[1])
        states=grid[ij[...,0].clamp(0,grid.shape[0]-1),ij[...,1].clamp(0,grid.shape[1]-1)]
        states=torch.where(inside,states,torch.zeros((),dtype=torch.uint8,device=self.device))
        free=(states==1).sum(-1).double()/16.;blocked=(states==2).sum(-1).double()/16.
        first_unknown=torch.where(states==0,self.fractions[None,None,:],torch.ones((),dtype=torch.float64,device=self.device)).amin(-1)
        first_blocked=torch.where(states==2,self.fractions[None,None,:],torch.ones((),dtype=torch.float64,device=self.device)).amin(-1)
        out=torch.stack((free,blocked,first_unknown,first_blocked),-1).float()
        out=out[:,torch.from_numpy(inverse).to(self.device),:].reshape(len(keys),-1)
        scale=max(float(np.max(abs(keys.astype(np.int64)))) if len(keys) else 0.,float(np.max(abs(path[:,[1,3]])))/context.resolution,1.)
        boundary=(abs(coordinates-torch.round(coordinates))<=64*np.finfo(np.float64).eps*(2*scale+1)).reshape(len(keys),-1).any(-1)
        return out,boundary

    def finish_lines(self,task,context,keys,end_only=False):
        if isinstance(task,list):
            values=[];count=0
            for i,item in enumerate(task):
                values.append(self.finish_lines(item,context,keys[i*65536:(i+1)*65536],end_only));count+=self.last_profile.get('ray_boundary_fallback_rows',0)
            self.last_profile['ray_boundary_fallback_rows']=count
            return np.concatenate(values)
        out,boundary=task;values=out.cpu().numpy();critical=np.flatnonzero(boundary.cpu().numpy())
        if len(critical):
            exact=legal_lines_reference(keys[critical],context.path[-1:] if end_only else context.path,context.resolution,context.lo,context.states)
            values[critical]=exact[:,8:12] if end_only else exact
        self.last_profile['ray_boundary_fallback_rows']=len(critical)
        return values

    def launch_opportunity(self,context,keys):
        if len(keys)>32768:return [self.launch_opportunity(context,keys[i:i+32768]) for i in range(0,len(keys),32768)]
        if len(context.path)<2:return None
        path=context.path;count=len(path)-1;den=max(context.duration,1e-4)
        cosine,sine,dt,when,global_values=trajectory_constants(path)
        def tensor(value):return torch.from_numpy(np.ascontiguousarray(value)).to(self.device)
        centers=(tensor(keys).float()+.5)*np.float32(context.resolution)
        poses=tensor(path[:,[1,3]])
        # Reference minimum-distance feature uses F32 arithmetic then F64sqrt.
        d32=centers[:,None,:]-poses.float()[None,:,:]
        min_distance=(d32.square().sum(-1).amin(-1)).double().sqrt()/10.
        dx=centers[:,0,None].double()-poses[None,1:,0];dz=centers[:,1,None].double()-poses[None,1:,1]
        distance=(dx.square()+dz.square()).sqrt()
        facing=((dx*tensor(cosine[1:])[None,:]-dz*tensor(sine[1:])[None,:])/distance.clamp_min(1e-4)).clamp(-1.,1.)
        within=distance<=10.;m0=within&(facing>=0.);m5=within&(facing>=.5);m8=within&(facing>=.866025403784)
        positive=facing.clamp_min(0.);weighted=positive*(1.-distance/11.).clamp_min(0.)
        dtt=tensor(dt[1:])[None,:];whent=tensor(when[1:])[None,:]
        values=[min_distance,torch.where(within,facing,-1.).amax(-1),weighted.amax(-1),
            torch.where(m0,distance,11.).amin(-1)/10.,torch.where(m5,distance,11.).amin(-1)/10.,torch.where(m8,distance,11.).amin(-1)/10.,
            m0.sum(-1).double()/count,m5.sum(-1).double()/count,m8.sum(-1).double()/count,
            (m0*dtt).sum(-1)/den,(m5*dtt).sum(-1)/den,(m8*dtt).sum(-1)/den,
            torch.where(m5,whent,1.).amin(-1),torch.where(m5,whent,0.).amax(-1),facing.amax(-1),weighted.sum(-1)/count]
        for value in global_values:values.append(torch.full_like(min_distance,float(value)))
        out=torch.stack(values,-1).float()
        numerical=((abs(distance-10.)<=1e-8)|(abs(facing)<=1e-8)|(abs(facing-.5)<=1e-8)|(abs(facing-.866025403784)<=1e-8)).any(-1)
        return out,numerical

    def finish_opportunity(self,task,context,keys):
        if task is None:
            self.last_profile['opportunity_boundary_fallback_rows']=len(keys)
            return inside_descriptors(keys,context.path,context.resolution)[1]
        if isinstance(task,list):
            out=[];count=0
            for i,item in enumerate(task):
                out.append(self.finish_opportunity(item,context,keys[i*32768:(i+1)*32768]));count+=self.last_profile.get('opportunity_boundary_fallback_rows',0)
            self.last_profile['opportunity_boundary_fallback_rows']=count
            return np.concatenate(out)
        output,mask=task;values=output.cpu().numpy();critical=np.flatnonzero(mask.cpu().numpy())
        if len(critical):values[critical]=inside_descriptors(keys[critical],context.path,context.resolution)[1]
        self.last_profile['opportunity_boundary_fallback_rows']=len(critical)
        return values

    def forward(self,x,context=None,keys=None):
        start=time.perf_counter()
        with torch.inference_mode(),torch.cuda.amp.autocast(enabled=False),torch.cpu.amp.autocast(enabled=False):
            if len(x)<=65536:values=self.net(x if isinstance(x,torch.Tensor) else torch.from_numpy(x).to(self.device)).cpu().numpy()
            else:
                values=np.empty((len(x),4),np.float32)
                for lo in range(0,len(x),65536):
                    data=x[lo:lo+65536];values[lo:lo+65536]=self.net(data if isinstance(data,torch.Tensor) else torch.from_numpy(data).to(self.device)).cpu().numpy()
            returned=time.perf_counter()
            # Correct only full reference-sized blocks that contain near ties.
            blocks=critical_blocks(values)
            for block in blocks:
                lo=int(block)*8192;hi=min(lo+8192,len(x));data=x[lo:hi]
                if context is not None:
                    from .fused import transform
                    # Exact CPU packing preserves every reference feature;
                    # retain the original forward block shape for near ties.
                    data,_=transform(context,keys[lo:hi],end_only=x.shape[1]==80)
                values[lo:hi]=self.reference(data.cpu() if isinstance(data,torch.Tensor) else torch.from_numpy(data)).numpy()
        self.last_profile.update(neural_transfer_forward_return_seconds=returned-start,near_tie_reference_blocks=len(blocks),
            near_tie_reference_seconds=time.perf_counter()-returned,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        return values
