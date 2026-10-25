"""Execution-only packing of the same active scientific features."""
import math
import time
import numpy as np
from numba import njit,prange
from eventwm_graph.stage2_quality.features import inside_descriptors
from .features import legal_lines


@njit(cache=True,parallel=True)
def segment_descriptors(keys,path,resolution):
    """Exact descriptor arithmetic; reuse native float32 positions' distances.

    Arbitrary hypothetical float64 planner positions retain the separate
    float32 candidate-distance pass. Every supplied time and heading is kept.
    """
    out=np.zeros((len(keys),20),np.float32)
    cosine=np.cos(path[:,4]);sine=np.sin(path[:,4]);duration=path[-1,0]-path[0,0]
    rotation=0.;stationary=0.;unwrapped=0.;ylo=0.;yhi=0.;native=True
    times=np.empty(len(path),np.float64);fractions=np.empty(len(path),np.float64)
    for j in range(len(path)):
        if float(np.float32(path[j,1]))!=path[j,1] or float(np.float32(path[j,3]))!=path[j,3]:native=False
        times[j]=path[j,0]-path[j-1,0] if j else 0.
        fractions[j]=(path[j,0]-path[0,0])/max(duration,1e-4)
        if j:
            dy=math.atan2(math.sin(path[j,4]-path[j-1,4]),math.cos(path[j,4]-path[j-1,4]))
            rotation+=abs(dy);unwrapped+=dy;ylo=min(ylo,unwrapped);yhi=max(yhi,unwrapped)
            if (path[j,1]-path[j-1,1])**2+(path[j,3]-path[j-1,3])**2<1e-8:stationary+=times[j]
    for i in prange(len(keys)):
        x=np.float32(np.float32(keys[i,0])+.5)*np.float32(resolution)
        z=np.float32(np.float32(keys[i,1])+.5)*np.float32(resolution)
        mind=1e30
        if native:
            dx=np.float32(float(x)-path[0,1]);dz=np.float32(float(z)-path[0,3]);mind=np.float64(dx*dx+dz*dz)
        else:
            for j in range(len(path)):
                if j>0 and np.float32(path[j,1])==np.float32(path[j-1,1]) and np.float32(path[j,3])==np.float32(path[j-1,3]):continue
                dx=float(x)-float(np.float32(path[j,1]));dz=float(z)-float(np.float32(path[j,3]));mind=min(mind,dx*dx+dz*dz)
        best=-1.;weighted=0.;min0=11.;min5=11.;min8=11.;first=1.;last=0.;s=0.;maxall=-1.
        c0=0.;c5=0.;c8=0.;t0=0.;t5=0.;t8=0.;rel_x=0.;rel_z=0.;dist=0.;w=0.
        for j in range(1,len(path)):
            if j==1 or path[j,1]!=path[j-1,1] or path[j,3]!=path[j-1,3]:
                rel_x=float(x)-path[j,1];rel_z=float(z)-path[j,3];d2=rel_x*rel_x+rel_z*rel_z;dist=math.sqrt(d2)
                w=max(0.,1.-dist/11.)
                if native:
                    dx32=np.float32(rel_x);dz32=np.float32(rel_z)
                    mind=min(mind,np.float64(dx32*dx32+dz32*dz32))
            facing=max(-1.,min(1.,(rel_x*cosine[j]-rel_z*sine[j])/max(dist,1e-4)))
            maxall=max(maxall,facing);s+=max(0.,facing)*w;weighted=max(weighted,max(0.,facing)*w)
            if dist<=10.:
                best=max(best,facing);dt=times[j]
                if facing>=0.:c0+=1.;t0+=dt;min0=min(min0,dist)
                if facing>=.5:c5+=1.;t5+=dt;min5=min(min5,dist);first=min(first,fractions[j]);last=max(last,fractions[j])
                if facing>=.866025403784:c8+=1.;t8+=dt;min8=min(min8,dist)
        count=max(1,len(path)-1);den=max(duration,1e-4)
        out[i,0]=math.sqrt(mind)/10.;out[i,1]=best;out[i,2]=weighted
        out[i,3]=min0/10.;out[i,4]=min5/10.;out[i,5]=min8/10.;out[i,6]=c0/count;out[i,7]=c5/count;out[i,8]=c8/count
        out[i,9]=t0/den;out[i,10]=t5/den;out[i,11]=t8/den;out[i,12]=first;out[i,13]=last;out[i,14]=maxall;out[i,15]=s/count
        out[i,16]=rotation/(2*math.pi);out[i,17]=(yhi-ylo)/(2*math.pi);out[i,18]=stationary/den;out[i,19]=(len(path)-1)/128.
    return out


@njit(cache=True)
def candidate_lines(keys, path, resolution, lo, states):
    """Same16probes for candidate-domain keys inside the convex scratch crop.

    Supplied poses and candidate cell centers lie strictly inside this crop;
    every point on their line is therefore inside. Positive crop coordinates
    permit truncation, with exact reference arithmetic at grid boundaries.
    """
    out = np.empty((len(keys),12),np.float32)
    indices = (min(1,len(path)-1),len(path)//2,len(path)-1)
    starts = np.empty((3,4),np.float64); duplicates = np.full(3,-1,np.int64)
    for ray in range(3):
        j=indices[ray];sx=path[j,1];sz=path[j,3]
        starts[ray,0]=sx;starts[ray,1]=sz;starts[ray,2]=sx/resolution-lo[0];starts[ray,3]=sz/resolution-lo[1]
        for previous in range(ray):
            k=indices[previous]
            if path[k,1]==sx and path[k,3]==sz:
                duplicates[ray]=previous;break
    for i in range(len(keys)):
        for ray in range(3):
            duplicate=duplicates[ray]
            if duplicate>=0:
                for col in range(4):out[i,ray*4+col]=out[i,duplicate*4+col]
                continue
            sx=starts[ray,0];sz=starts[ray,1];gx=starts[ray,2];gz=starts[ray,3]
            dx=keys[i,0]-lo[0]+.5-gx;dz=keys[i,1]-lo[1]+.5-gz
            bx=64*2.220446049250313e-16*(abs(sx/resolution)+abs(keys[i,0])+1)
            bz=64*2.220446049250313e-16*(abs(sz/resolution)+abs(keys[i,1])+1)
            wx=(keys[i,0]+.5)*resolution;wz=(keys[i,1]+.5)*resolution
            free=0;blocked=0;first_unknown=17;first_blocked=17
            for step in range(1,17):
                fraction=step/17.;px=gx+dx*fraction;pz=gz+dz*fraction
                ix=int(px);iz=int(pz);fx=px-ix;fz=pz-iz
                if fx<=bx or fx>=1.-bx:ix=int(math.floor((sx+(wx-sx)*fraction)/resolution))-lo[0]
                if fz<=bz or fz>=1.-bz:iz=int(math.floor((sz+(wz-sz)*fraction)/resolution))-lo[1]
                state=states[ix,iz]
                free+=int(state==1);blocked+=int(state==2)
                if state==0 and first_unknown==17:first_unknown=step
                if state==2 and first_blocked==17:first_blocked=step
            out[i,ray*4]=free/16.;out[i,ray*4+1]=blocked/16.
            out[i,ray*4+2]=first_unknown/17.;out[i,ray*4+3]=first_blocked/17.
    return out


@njit(cache=True)
def candidate_lines_vector(keys,path,resolution,lo,states):
    """Fixed-length integer reductions and rare exact boundary slow path."""
    out=np.empty((len(keys),12),np.float32);indices=(min(1,len(path)-1),len(path)//2,len(path)-1)
    starts=np.empty((3,4),np.float64);duplicates=np.full(3,-1,np.int64)
    ixbuf=np.empty(16,np.int64);izbuf=np.empty(16,np.int64)
    for ray in range(3):
        j=indices[ray];sx=path[j,1];sz=path[j,3]
        starts[ray,0]=sx;starts[ray,1]=sz;starts[ray,2]=sx/resolution-lo[0];starts[ray,3]=sz/resolution-lo[1]
        for previous in range(ray):
            k=indices[previous]
            if path[k,1]==sx and path[k,3]==sz:duplicates[ray]=previous;break
    for i in range(len(keys)):
        for ray in range(3):
            duplicate=duplicates[ray]
            if duplicate>=0:
                for col in range(4):out[i,ray*4+col]=out[i,duplicate*4+col]
                continue
            sx=starts[ray,0];sz=starts[ray,1];gx=starts[ray,2];gz=starts[ray,3]
            dx=keys[i,0]-lo[0]+.5-gx;dz=keys[i,1]-lo[1]+.5-gz
            bx=64*2.220446049250313e-16*(abs(sx/resolution)+abs(keys[i,0])+1)
            bz=64*2.220446049250313e-16*(abs(sz/resolution)+abs(keys[i,1])+1)
            boundary=False
            for index in range(16):
                fraction=(index+1)/17.;px=gx+dx*fraction;pz=gz+dz*fraction
                ix=int(px);iz=int(pz);fx=px-ix;fz=pz-iz;ixbuf[index]=ix;izbuf[index]=iz
                boundary|=(fx<=bx) or (fx>=1.-bx) or (fz<=bz) or (fz>=1.-bz)
            if boundary:
                wx=(keys[i,0]+.5)*resolution;wz=(keys[i,1]+.5)*resolution
                for index in range(16):
                    fraction=(index+1)/17.
                    ixbuf[index]=int(math.floor((sx+(wx-sx)*fraction)/resolution))-lo[0]
                    izbuf[index]=int(math.floor((sz+(wz-sz)*fraction)/resolution))-lo[1]
            free=0;blocked=0;first_unknown=17;first_blocked=17
            for index in range(16):
                state=states[ixbuf[index],izbuf[index]]
                free+=int(state==1);blocked+=int(state==2)
                first_unknown=min(first_unknown,index+1 if state==0 else 17)
                first_blocked=min(first_blocked,index+1 if state==2 else 17)
            out[i,ray*4]=free/16.;out[i,ray*4+1]=blocked/16.
            out[i,ray*4+2]=first_unknown/17.;out[i,ray*4+3]=first_blocked/17.
    return out


@njit(cache=True,parallel=True)
def assemble(keys, crop_lo, states, heights, d_free, d_blocked, resolution,
             xy0, xy1, robot_height, constants, view, lines, map_lo, free, blocked,spatial_only=False):
    n = len(keys); out = np.empty((n, 76+lines.shape[1]), np.float32); baseline = np.empty(n, np.float32)
    nx = free.shape[0]-1; nz = free.shape[1]-1
    for i in prange(n):
        x = keys[i, 0]-crop_lo[0]; z = keys[i, 1]-crop_lo[1]
        col = 0
        for dx in (-3, 0, 3):
            for dz in (-3, 0, 3):
                a = max(0, min(states.shape[0]-1, x+dx)); b = max(0, min(states.shape[1]-1, z+dz))
                state = states[a, b]
                out[i, col] = state == 1; out[i, col+9] = state == 2; col += 1
        out[i, 18] = d_free[x, z]/np.float32(10.)
        out[i, 19] = d_blocked[x, z]/np.float32(10.)
        cx = np.float32(np.float32(keys[i, 0])+np.float32(.5))*np.float32(resolution)
        cz = np.float32(np.float32(keys[i, 1])+np.float32(.5))*np.float32(resolution)
        out[i, 20] = (cx-xy0[0])/np.float32(10.); out[i, 21] = (cz-xy0[1])/np.float32(10.)
        out[i, 22] = (cx-xy1[0])/np.float32(10.); out[i, 23] = (cz-xy1[1])/np.float32(10.)
        baseline[i] = heights[x, z]
        out[i, 24] = baseline[i]-np.float32(robot_height)
        for j in range(6):
            out[i, 25+j] = constants[j]
        if not spatial_only:
            for j in range(20):out[i,31+j]=view[i,j]
        x = keys[i, 0]-map_lo[0]; z = keys[i, 1]-map_lo[1]; col = 51
        for radius in (1, 4, 12):
            a = max(0, min(nx, x-radius)); b = max(0, min(nx, x+radius+1))
            c = max(0, min(nz, z-radius)); d = max(0, min(nz, z+radius+1)); area = float((2*radius+1)**2)
            f = (free[b, d]-free[a, d]-free[b, c]+free[a, c])/area
            h = (blocked[b, d]-blocked[a, d]-blocked[b, c]+blocked[a, c])/area
            out[i, col] = f; out[i, col+1] = h; out[i, col+2] = max(0., 1.-f-h); col += 3
        for radius in (4, 12):
            for sx, sz in ((-1,-1),(-1,1),(1,-1),(1,1)):
                xa = x-radius if sx<0 else x; xb = x+1 if sx<0 else x+radius+1
                za = z-radius if sz<0 else z; zb = z+1 if sz<0 else z+radius+1
                a = max(0, min(nx, xa)); b = max(0, min(nx, xb)); c = max(0, min(nz, za)); d = max(0, min(nz, zb)); area = float((radius+1)**2)
                out[i, col] = (free[b,d]-free[a,d]-free[b,c]+free[a,c])/area
                out[i, col+1] = (blocked[b,d]-blocked[a,d]-blocked[b,c]+blocked[a,c])/area; col += 2
        if not spatial_only:
            for j in range(lines.shape[1]):out[i,76+j]=lines[i,j]
    return out, baseline


def transform(context, keys, end_only=False,cuda=None,device_output=False):
    if cuda is not None and len(keys) and device_output:return cuda.transform_device(context,keys,end_only)
    start = time.perf_counter(); path = context.path
    task=None;opportunity=None
    if cuda is not None and len(keys):task,opportunity=cuda.launch_features(context,keys,end_only)
    launched=time.perf_counter()
    if task is not None:
        context.prepared.map_features(np.empty((0,2),np.int32))
        integral=time.perf_counter()
        length=float(np.linalg.norm(np.diff(context.xy,axis=0),axis=1).sum())
        constants=np.array([context.duration/5,length/5,context.resolution,path[-1,2]-path[0,2],np.cos(path[-1,4]-path[0,4]),np.sin(path[-1,4]-path[0,4])],np.float32)
        width=4 if end_only else 12
        x,baseline=assemble(keys,context.lo,context.states,context.base_height,context.dist[0],context.dist[1],context.resolution,
            context.xy[0],context.xy[-1],path[0,2],constants,np.empty((0,20),np.float32),np.empty((0,width),np.float32),*context.prepared.integrals,True)
        spatial=time.perf_counter();view=cuda.finish_opportunity(opportunity,context,keys);trajectory=time.perf_counter()
        lines=cuda.finish_lines(task,context,keys,end_only);rays=time.perf_counter()
        x[:,31:51]=view;x[:,76:]=lines
        joined=time.perf_counter()
        if not np.isfinite(x).all():raise ValueError('nonfinite legal features')
        end=time.perf_counter();context.last_transform_profile=dict(legal_line_launch_seconds=launched-start,map_seconds=(spatial-launched)+(end-rays),
            trajectory_seconds=trajectory-spatial,legal_line_seconds=rays-trajectory,map_integral_seconds=integral-launched,
            spatial_assembly_seconds=spatial-integral,feature_join_seconds=joined-rays,finite_validation_seconds=end-joined)
        return x,baseline
    view = segment_descriptors(keys, path, context.resolution); trajectory = time.perf_counter()
    if task is not None:lines=cuda.finish_lines(task,context,keys,end_only)
    else:
        lines = candidate_lines_vector(keys, path[-1:] if end_only else path, context.resolution, context.lo, context.states.astype(np.int32))
        if end_only:lines=lines[:,8:12]
    rays = time.perf_counter()
    context.prepared.map_features(np.empty((0, 2), np.int32))
    length = float(np.linalg.norm(np.diff(context.xy, axis=0), axis=1).sum())
    constants = np.array([context.duration/5, length/5, context.resolution, path[-1,2]-path[0,2],
                          np.cos(path[-1,4]-path[0,4]), np.sin(path[-1,4]-path[0,4])], np.float32)
    x, baseline = assemble(keys, context.lo, context.states, context.base_height, context.dist[0], context.dist[1],
        context.resolution, context.xy[0], context.xy[-1], path[0,2], constants, view, lines, *context.prepared.integrals)
    end = time.perf_counter()
    context.last_transform_profile = dict(trajectory_seconds=trajectory-launched,legal_line_launch_seconds=launched-start,legal_line_seconds=rays-trajectory,map_seconds=end-rays)
    if not np.isfinite(x).all():raise ValueError('nonfinite legal features')
    return x, baseline


@njit(cache=True)
def probabilities(values, baseline, out, height, begin):
    critical=np.empty(len(values),np.int64);count=0
    for i in range(len(values)):
        a = values[i, 0]; b = values[i, 1]; c = values[i, 2]; m = max(a, b, c)
        ea = np.float32(math.exp(a-m)); eb = np.float32(math.exp(b-m)); ec = np.float32(math.exp(c-m))
        denom = np.float32(np.float32(ea+eb)+ec)
        out[begin+i, 0] = ea/denom; out[begin+i, 1] = eb/denom; out[begin+i, 2] = ec/denom
        height[begin+i] = baseline[begin+i]+values[i, 3]
        if int(m-a<1e-5)+int(m-b<1e-5)+int(m-c<1e-5)>=2:
            critical[count]=i;count+=1
    return critical[:count]
