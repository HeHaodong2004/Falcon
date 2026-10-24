"""Bounded native-action beam, JIT numeric kernels only; no policy learning."""
import math
import numpy as np
from numba import njit
from .geometry import swept_free


@njit(cache=True)
def greedy_proposal(start,target,terminal_yaw,terminal,people,grid,heights,origin,resolution,
                    horizon,step,turn,dt):
    """Preserve a fully checked turn-then-forward candidate outside the beam.

    This prevents beam pruning from replacing a simple route rejoin with a
    large-radius arc. It is accepted only if the entire available plan is safe.
    """
    path=np.zeros((horizon+1,4));path[0]=start;actions=np.zeros(horizon,np.int64)
    length=0
    for depth in range(horizon):
        a=path[depth];dx=target[0]-a[0];dz=target[2]-a[2]
        arrived=math.hypot(dx,dz)<=.155
        desired=terminal_yaw if arrived else math.atan2(-dz,dx)
        angle=math.atan2(math.sin(desired-a[3]),math.cos(desired-a[3]))
        if arrived and (not terminal or abs(angle)<=turn/2+1e-5):break
        action=(1 if angle>0 else 2) if abs(angle)>turn/2+1e-5 else 0
        b=a.copy()
        if action==0:
            b[0]+=step*math.cos(a[3]);b[2]-=step*math.sin(a[3])
            ix=int(math.floor((b[0]-origin[0])/resolution));iz=int(math.floor((b[2]-origin[1])/resolution))
            if 0<=ix<grid.shape[0] and 0<=iz<grid.shape[1] and grid[ix,iz]:b[1]=heights[ix,iz]
        else:b[3]=math.atan2(math.sin(a[3]+(turn if action==1 else -turn)),math.cos(a[3]+(turn if action==1 else -turn)))
        if not swept_free(a,b,grid,heights,origin,resolution):return np.empty(0,np.int64),np.empty((0,4))
        safe,_,clearance=human_cost(a,b,depth,people,dt)
        if not safe or clearance<.10:return np.empty(0,np.int64),np.empty((0,4))
        path[depth+1]=b;actions[depth]=action;length+=1
    return actions[:length],path[:length+1]


@njit(cache=True)
def path_score(path,actions,target,terminal_yaw,terminal,preferred):
    score=0.
    for i in range(1,len(path)):
        p=path[i];dx=target[0]-p[0];dz=target[2]-p[2];distance=math.hypot(dx,dz)
        desired=terminal_yaw if terminal and distance<=.155 else math.atan2(-dz,dx)
        angle=abs(math.atan2(math.sin(desired-p[3]),math.cos(desired-p[3])))
        score+=.015+(.004 if actions[i-1]>0 else 0.)+.08*distance+.03*angle
        if i>1 and actions[i-1]>0 and actions[i-2]>0 and actions[i-1]!=actions[i-2]:score+=.06
    if len(actions) and preferred>0 and actions[0]>0 and actions[0]!=preferred:score+=.12
    return score+distance+(.55 if terminal else .45)*angle


@njit(cache=True)
def maneuver_proposals(start,target,terminal_yaw,terminal,people,grid,heights,origin,resolution,
                       horizon,step,turn,dt,preferred):
    """Preserve meaningful turn/advance/rejoin maneuvers before beam pruning.

    All are sequences of actual native actions, each checked over its sweep.
    A longer safe sequence dominates a beam dying before the requested horizon.
    """
    best_score=1e30;best_actions=np.empty(0,np.int64);best_path=np.empty((0,4));checks=0
    for sign in (1,2):
        for turns in (4,8,12,16,20,24):
            for advances in (4,8,12):
                path=np.zeros((horizon+1,4));path[0]=start;actions=np.zeros(horizon,np.int64)
                valid=True;length=0
                for depth in range(horizon):
                    a=path[depth];dx=target[0]-a[0];dz=target[2]-a[2];arrived=math.hypot(dx,dz)<=.155
                    desired=terminal_yaw if arrived else math.atan2(-dz,dx)
                    angle=math.atan2(math.sin(desired-a[3]),math.cos(desired-a[3]))
                    if arrived and (not terminal or abs(angle)<=turn/2+1e-5):break
                    if depth<turns:action=sign
                    elif depth<turns+advances:action=0
                    else:action=(1 if angle>0 else 2) if abs(angle)>turn/2+1e-5 else 0
                    b=a.copy()
                    if action==0:
                        b[0]+=step*math.cos(a[3]);b[2]-=step*math.sin(a[3])
                        ix=int(math.floor((b[0]-origin[0])/resolution));iz=int(math.floor((b[2]-origin[1])/resolution))
                        if 0<=ix<grid.shape[0] and 0<=iz<grid.shape[1] and grid[ix,iz]:b[1]=heights[ix,iz]
                    else:b[3]=math.atan2(math.sin(a[3]+(turn if action==1 else -turn)),math.cos(a[3]+(turn if action==1 else -turn)))
                    checks+=1
                    if not swept_free(a,b,grid,heights,origin,resolution):valid=False;break
                    safe,_,_=human_cost(a,b,depth,people,dt)
                    if not safe:valid=False;break
                    path[depth+1]=b;actions[depth]=action;length+=1
                if valid and length:
                    score=path_score(path[:length+1],actions[:length],target,terminal_yaw,terminal,preferred)
                    if score<best_score:
                        best_score=score;best_actions=actions[:length].copy();best_path=path[:length+1].copy()
    return best_actions,best_path,best_score,checks


@njit(cache=True)
def human_cost(a,b,depth,people,dt):
    """Time-aligned swept rectangle/disk check with relative substep motion.

    people rows: X,Z,VX,VZ,age,velocity_valid. Centers are exact at last_seen.
    Current exact observations are never replaced by learned predictions.
    Prediction uncertainty is a declared heuristic, not a confidence guarantee.
    """
    minimum=100.;social=0.
    angle=math.atan2(math.sin(b[3]-a[3]),math.cos(b[3]-a[3]))
    for j in range(len(people)):
        h=people[j]
        for k in range(5):
            u=k/4.; t=(depth+u)*dt; age=h[4]+t
            x=a[0]+u*(b[0]-a[0]);z=a[2]+u*(b[2]-a[2]);yaw=a[3]+u*angle
            hx=h[0]+h[2]*age; hz=h[1]+h[3]*age
            dx=hx-x; dz=hz-z;c=math.cos(yaw);s=math.sin(yaw)
            lx=c*dx-s*dz;lz=s*dx+c*dz
            ex=max(-.646495-lx,0.,lx-.584202);ez=max(-.242846-lz,0.,lz-.238225)
            # Subinterval cover: relative translation plus body angular sweep.
            cover=(math.hypot(b[0]-a[0],b[2]-a[2])+math.hypot(h[2],h[3])*dt+.68*abs(angle))/8
            uncertainty=.03+min(.28,.10*age+.12*age*age)
            if h[5]<.5: uncertainty+=min(.32,3.*age)
            clearance=math.hypot(ex,ez)-.616-uncertainty-cover
            minimum=min(minimum,clearance)
            if clearance<=0.: return False,0.,minimum
            # Euclidean center <1m is a conservative PSC planning surrogate;
            # authoritative geodesic/fallback PSC is evaluated only by native metrics.
            d=math.hypot(dx,dz)
            social+=max(0.,1.-d)/5.
    return True,social,minimum


@njit(cache=True)
def beam_search(start,target,terminal_yaw,terminal,people,grid,heights,origin,resolution,
                horizon,width,step,turn,dt,preferred,forecast,forecast_times,now,forecast_weight):
    poses=np.zeros((horizon+1,width,4));poses[0,0]=start
    parents=np.zeros((horizon+1,width),np.int64);actions=np.zeros((horizon+1,width),np.int64)
    costs=np.zeros(width); first=np.zeros(width,np.int64);n=1
    expanded=0;static_rejected=0;human_rejected=0;deepest=0;best=0
    for depth in range(horizon):
        capacity=width*3
        cp=np.zeros((capacity,4));cs=np.full(capacity,1e30);cc=np.zeros(capacity)
        ca=np.zeros(capacity,np.int64);ci=np.zeros(capacity,np.int64);cf=np.zeros(capacity,np.int64)
        count=0
        for i in range(n):
            for action in range(3):
                expanded+=1;a=poses[depth,i];b=a.copy()
                if action==0:
                    b[0]+=step*math.cos(a[3]);b[2]-=step*math.sin(a[3])
                    ix=int(math.floor((b[0]-origin[0])/resolution));iz=int(math.floor((b[2]-origin[1])/resolution))
                    if 0<=ix<grid.shape[0] and 0<=iz<grid.shape[1] and grid[ix,iz]: b[1]=heights[ix,iz]
                else:
                    angle=a[3]+(turn if action==1 else -turn);b[3]=math.atan2(math.sin(angle),math.cos(angle))
                if not swept_free(a,b,grid,heights,origin,resolution): static_rejected+=1;continue
                admissible,social,clearance=human_cost(a,b,depth,people,dt)
                if not admissible: human_rejected+=1;continue
                f=action if depth==0 else first[i]
                running=costs[i]+.015+(0.004 if action!=0 else 0.)+.3*social
                # Discourage one-step inverse turning loops, while allowing turn-to-rejoin.
                if depth>0 and action>0 and actions[depth,i]>0 and action!=actions[depth,i]: running+=.06
                if depth==0 and preferred>0 and action>0 and action!=preferred: running+=.12
                dx=target[0]-b[0];dz=target[2]-b[2];dist=math.hypot(dx,dz)
                desired=terminal_yaw if terminal and dist<.155 else math.atan2(-dz,dx)
                error=abs(math.atan2(math.sin(desired-b[3]),math.cos(desired-b[3])))
                # Stage cost prevents receding-horizon procrastination: delaying
                # the same progress until the final beam step is strictly worse.
                running+=.08*dist+.03*error
                value=running+dist+(.55 if terminal else .45)*error
                # Shared joint-future marginal proximity affects ranking only.
                # Never clears the mandatory observation-based collision screen.
                if len(forecast_times)>0 and forecast_weight>0:
                    stamp=now+(depth+1)*dt
                    k=np.searchsorted(forecast_times,stamp)
                    if k<len(forecast_times):
                        risk=0.;valid=0
                        for j in range(forecast.shape[1]):
                            x=forecast[k,j,0];z=forecast[k,j,1]
                            if math.isfinite(x) and math.isfinite(z):
                                valid+=1;d=math.hypot(x-b[0],z-b[2]);risk+=max(0.,1.5-d)
                        if valid: value+=forecast_weight*risk/valid
                duplicate=-1
                for j in range(count):
                    if cf[j]==f and abs(cp[j,0]-b[0])<.015 and abs(cp[j,2]-b[2])<.015 and abs(cp[j,3]-b[3])<.01:
                        duplicate=j;break
                if duplicate>=0 and cs[duplicate]<=value: continue
                j=duplicate if duplicate>=0 else count
                cp[j]=b;cs[j]=value;cc[j]=running;ca[j]=action;ci[j]=i;cf[j]=f
                if duplicate<0: count+=1
        if count==0: break
        # Preserve first-action diversity; bounded width, deterministic tie order.
        order=np.argsort(cs[:count]); selected=np.zeros(width,np.int64);used=np.zeros(count,np.bool_)
        m=0
        for f in range(3):
            quota=0
            for j in order:
                if cf[j]==f and quota<max(1,width//4) and m<width:
                    selected[m]=j;used[j]=True;m+=1;quota+=1
        for j in order:
            if m>=width: break
            if not used[j]: selected[m]=j;m+=1
        best_score=1e30
        for i in range(m):
            j=selected[i];poses[depth+1,i]=cp[j];parents[depth+1,i]=ci[j];actions[depth+1,i]=ca[j]
            costs[i]=cc[j];first[i]=cf[j]
            if cs[j]<best_score: best_score=cs[j];best=i
        n=m;deepest=depth+1
        # A plan may end on reaching its local objective. Requiring a full
        # horizon after arrival forces fictional loitering and creates cycles.
        reached=False; reached_score=1e30
        for i in range(m):
            p=poses[depth+1,i]
            distance=math.hypot(p[0]-target[0],p[2]-target[2])
            angle=abs(math.atan2(math.sin(p[3]-terminal_yaw),math.cos(p[3]-terminal_yaw)))
            if distance<=.155 and (not terminal or angle<=turn/2+1e-5) and costs[i]<reached_score:
                reached=True;reached_score=costs[i];best=i
        if reached: break
    if deepest==0: return np.empty(0,np.int64),np.empty((0,4)),expanded,static_rejected,human_rejected,0
    sequence=np.zeros(deepest,np.int64);path=np.zeros((deepest+1,4));index=best
    for d in range(deepest,0,-1):
        path[d]=poses[d,index];sequence[d-1]=actions[d,index];index=parents[d,index]
    path[0]=start
    return sequence,path,expanded,static_rejected,human_rejected,deepest
