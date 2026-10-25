"""Same scientific CUDA features with shared synchronization across queries."""
from collections import OrderedDict
import time
import numpy as np
import torch
from eventwm_graph.stage2_segment.cuda import trajectory_constants,key_array
from eventwm_graph.stage2_quality.features import inside_descriptors
from eventwm_graph.stage2_segment.features import legal_lines_reference


def transform_batch(engine,contexts,keysets):
    """Launch all exact kernels, then transfer boundary flags and heights once.

    Numerical boundary rows still use the original CPU reference. No feature
    arithmetic, candidate domain or precision is approximated.
    """
    device=engine.device;tasks=[];tick=time.perf_counter()
    for context,keys in zip(contexts,keysets):
        keys=key_array(keys);prepared=context.prepared
        def tensor(a):return torch.from_numpy(np.ascontiguousarray(a)).to(device)
        key=(str(device),*map(int,context.lo),*map(int,context.shape))
        for attr in ['cuda_grids','cuda_geometry']:
            if not hasattr(prepared,attr):setattr(prepared,attr,OrderedDict())
        if key not in prepared.cuda_grids:prepared.cuda_grids[key]=tensor(context.states)
        prepared.map_features(np.empty((0,2),np.int32));map_lo,free,blocked=prepared.integrals
        if key not in prepared.cuda_geometry:prepared.cuda_geometry[key]=[tensor(a)for a in (context.base_height,*context.dist,free,blocked)]
        for cache in [prepared.cuda_grids,prepared.cuda_geometry]:
            cache.move_to_end(key)
            while len(cache)>8:cache.popitem(last=False)
        grid=prepared.cuda_grids[key];geometry=prepared.cuda_geometry[key]
        cosine,sine,dt,when,globals_=trajectory_constants(context.path);n=len(keys)
        kt=tensor(keys);view=torch.empty((n,20),device=device);lines=torch.empty((n,12),device=device);vb=torch.empty(n,device=device,dtype=torch.uint8);lb=torch.empty_like(vb)
        scale=max(float(np.max(abs(keys.astype(np.int64))))if n else 0.,float(np.max(abs(context.path[:,[1,3]])))/context.resolution,1.)
        buffers=[kt,tensor(context.path),tensor(np.column_stack((cosine,sine,dt,when))),tensor(globals_),grid,view,lines,vb,lb]
        if n:engine.feature_kernel.launch(buffers,[n,len(context.path),*map(int,grid.shape),*map(int,context.lo),context.resolution,64*np.finfo(np.float64).eps*(2*scale+1),0])
        length=float(np.linalg.norm(np.diff(context.xy,axis=0),axis=1).sum());p=context.path
        constants=np.array([context.duration/5,length/5,context.resolution,p[-1,2]-p[0,2],np.cos(p[-1,4]-p[0,4]),np.sin(p[-1,4]-p[0,4])],np.float32)
        if not np.isfinite(constants).all():raise ValueError('nonfinite legal segment constants')
        x=torch.empty((n,88),device=device);baseline=torch.empty(n,device=device)
        assembly=[kt,view,lines,grid,*geometry,tensor(constants),x,baseline]
        if n:engine.feature_kernel.launch_assembly(assembly,[n,*map(int,grid.shape),*map(int,context.lo),free.shape[0]-1,free.shape[1]-1,*map(int,map_lo),12],
            [context.resolution,*map(float,context.xy[0]),*map(float,context.xy[-1]),p[0,2]])
        tasks.append((x,baseline,vb,lb,buffers,assembly))
    launched=time.perf_counter();flags=torch.cat([v for task in tasks for v in task[2:4]]).cpu().numpy()if tasks else np.empty(0,np.uint8)
    at=0;fallbacks=[]
    for context,keys,task in zip(contexts,keysets,tasks):
        x,baseline,vb,lb,_,_=task;n=len(keys);vc=np.flatnonzero(flags[at:at+n]);lc=np.flatnonzero(flags[at+n:at+2*n]);at+=2*n
        if len(vc):x[tensor(vc),31:51]=tensor(inside_descriptors(keys[vc],context.path,context.resolution)[1])
        if len(lc):x[tensor(lc),76:]=tensor(legal_lines_reference(keys[lc],context.path,context.resolution,context.lo,context.states))
        fallbacks.append(dict(opportunity_rows=len(vc),line_rows=len(lc)))
    if tasks and not bool(torch.stack([torch.isfinite(t[0]).all()for t in tasks]).all().item()):raise ValueError('nonfinite legal features')
    heights=torch.cat([t[1]for t in tasks]).cpu().numpy()if tasks else np.empty(0,np.float32);at=0;baselines=[]
    for keys in keysets:baselines.append(heights[at:at+len(keys)]);at+=len(keys)
    return [t[0]for t in tasks],baselines,dict(launch_seconds=launched-tick,total_seconds=time.perf_counter()-tick,boundary_fallbacks=fallbacks)
