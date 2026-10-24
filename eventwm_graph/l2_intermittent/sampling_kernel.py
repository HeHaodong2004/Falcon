"""Fused single-thread path arithmetic with retained NumPy Generator state."""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def fused_paths(m,intent,q,st,nxt,counter,valid,rel,start,law,event_next,rng,cv_diffusion):
    W,N=q.shape;T=len(rel)
    pos=np.full((W,T,N,2),np.nan);vel=np.full((W,T,N,2),np.nan)
    stops=np.zeros((W,T,N),np.bool_);next_series=np.zeros((W,T,N));counter_series=np.zeros((W,T,N),np.int64)
    cursors=np.zeros((W,N),np.int64)
    for ti in range(T):
        t=start+rel[ti];previous=start+rel[ti-1] if ti else start
        clock=np.full((W,N),previous)
        while ti:
            any_active=False
            for w in range(W):
                for i in range(N):
                    if valid[w,i] and clock[w,i]<t-1e-12:any_active=True
            if not any_active:break
            normals=rng.normal(0.,1.,size=(W,N,2,2));changes=np.zeros((W,N),np.bool_)
            for w in range(W):
                for i in range(N):
                    active=valid[w,i] and clock[w,i]<t-1e-12
                    u=min(nxt[w,i],t) if cv_diffusion<0 else t
                    dt=max(0.,u-clock[w,i]) if active else 0.
                    if cv_diffusion<0:
                        k=law[3] if st[w,i] else law[2]
                        e=math.exp(-k*dt);a=-math.expm1(-k*dt)/k
                        vv=-math.expm1(-2*k*dt)/(2*k);pv=(-math.expm1(-k*dt)/k)**2/2
                        zeta=k*dt;pp=(dt-2*a+vv)/(k*k)
                        if zeta<1e-3:pp=dt**3*(1/3-zeta/4+7*zeta*zeta/60-zeta**3/24)
                        scale=math.sqrt(law[4] if st[w,i] else q[w,i])
                    else:
                        e=1.;a=dt;vv=dt;pv=dt**2/2;pp=dt**3/3;scale=math.sqrt(cv_diffusion)
                    lv=math.sqrt(max(vv,0.));lp=pv/lv if lv>0 else 0.;lpp=math.sqrt(max(pp-lp*lp,0.))
                    for axis in range(2):
                        target=intent[w,i,axis] if cv_diffusion<0 and not st[w,i] else 0.
                        noise_p=lp*normals[w,i,0,axis]+lpp*normals[w,i,1,axis]
                        noise_v=lv*normals[w,i,0,axis];v=m[w,i,2+axis]
                        m[w,i,axis]+=a*v+(dt-a)*target+scale*noise_p
                        m[w,i,2+axis]=e*v+(1-e)*target+scale*noise_v
                    if active:clock[w,i]=u
                    change=active and nxt[w,i]<=t+1e-12 and cv_diffusion<0
                    changes[w,i]=change
                    if change:st[w,i]=not st[w,i]
            for w in range(W):
                for i in range(N):
                    if not changes[w,i]:continue
                    variance=law[5]**2 if st[w,i] else law[6]*q[w,i]
                    normal=rng.normal(0.,1.,size=2)
                    for axis in range(2):
                        m[w,i,2+axis]=(0. if st[w,i] else intent[w,i,axis])+normal[axis]*math.sqrt(variance)
                    counter[w,i]+=1;nxt[w,i]=event_next[w,i,cursors[w,i]];cursors[w,i]+=1
        for w in range(W):
            for i in range(N):
                if valid[w,i]:
                    pos[w,ti,i]=m[w,i,:2];vel[w,ti,i]=m[w,i,2:];stops[w,ti,i]=st[w,i]
                if cv_diffusion>=0:nxt[w,i]=t+1e10
                next_series[w,ti,i]=nxt[w,i];counter_series[w,ti,i]=counter[w,i]
    return pos,vel,stops,next_series,counter_series
