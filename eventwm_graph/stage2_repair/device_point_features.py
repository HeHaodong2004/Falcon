"""Same88 scientific features on device-resident, original ordered query keys."""
import time
import numpy as np
import torch
from eventwm_graph.stage2_segment.cuda import trajectory_constants
from eventwm_graph.stage2_quality.features import inside_descriptors
from eventwm_graph.stage2_segment.features import legal_lines_reference


def extract(engine,prepared,contexts,batch):
    device=engine.device;n=len(batch['keys']);state,height,df,db,free,blocked=batch['fields'];shape=tuple(state.shape);lo=prepared.global_lo
    x=torch.empty((n,88),device=device);baseline=torch.empty(n,device=device);view=torch.empty((n,20),device=device);lines=torch.empty((n,12),device=device);vb=torch.empty(n,dtype=torch.uint8,device=device);lb=torch.empty_like(vb);tasks=[]
    def tensor(a,dtype=None):return torch.from_numpy(np.ascontiguousarray(a,dtype=dtype)).to(device,non_blocking=True)
    for i,c in enumerate(contexts):
        start,end=map(int,batch['offsets'][i:i+2]);count=end-start;p=c.path;cosine,sine,dt,when,globals_=trajectory_constants(p);pre=tensor(np.column_stack([cosine,sine,dt,when]));path=tensor(p);global_values=tensor(globals_)
        length=float(np.linalg.norm(np.diff(c.xy,axis=0),axis=1).sum());constants=tensor(np.array([c.duration/5,length/5,c.resolution,p[-1,2]-p[0,2],np.cos(p[-1,4]-p[0,4]),np.sin(p[-1,4]-p[0,4])],np.float32))
        # A conservative coordinate bound may flag extra numerical-boundary
        # rows, which still use the independent original CPU reference.
        bounds=np.r_[c.candidate_lo,c.candidate_lo+c.candidate_shape];scale=max(float(np.abs(bounds.astype(np.int64)).max()),float(np.max(abs(p[:,[1,3]])))/c.resolution,1.)
        buffers=[batch['keys'][start:end],path,pre,global_values,state,view[start:end],lines[start:end],vb[start:end],lb[start:end]]
        scalars=[count,len(p),*map(int,shape),*map(int,lo),c.resolution,64*np.finfo(np.float64).eps*(2*scale+1),0]
        assembly=[batch['keys'][start:end],view[start:end],lines[start:end],state,height,df,db,free,blocked,constants,x[start:end],baseline[start:end]]
        integers=[count,*map(int,shape),*map(int,lo),shape[0],shape[1],*map(int,lo),12];floats=[c.resolution,*map(float,c.xy[0]),*map(float,c.xy[-1]),p[0,2]];tasks.append((buffers,scalars,assembly,integers,floats))
    for buffers,scalars,assembly,integers,floats in tasks:
        if scalars[0]:engine.feature_kernel.launch(buffers,scalars);engine.feature_kernel.launch_assembly(assembly,integers,floats)
    # One boundary/finite check; keys and heights otherwise stay on device.
    finite=torch.isfinite(x).all().to(torch.uint8);flags=torch.cat([vb,lb,finite[None]]).cpu().numpy()
    if not flags[-1]:raise ValueError('nonfinite device query features')
    fallbacks=[]
    for i,c in enumerate(contexts):
        start,end=map(int,batch['offsets'][i:i+2]);vc=np.flatnonzero(flags[start:end]);lc=np.flatnonzero(flags[n+start:n+end])
        if len(vc):
            rows=tensor(vc+start,np.int64);keys=batch['keys'][rows].cpu().numpy();x[rows,31:51]=tensor(inside_descriptors(keys,c.path,c.resolution)[1])
        if len(lc):
            rows=tensor(lc+start,np.int64);keys=batch['keys'][rows].cpu().numpy();x[rows,76:]=tensor(legal_lines_reference(keys,c.path,c.resolution,c.lo,c.states))
        fallbacks.append(dict(opportunity_rows=len(vc),line_rows=len(lc)))
    return x,baseline,fallbacks
