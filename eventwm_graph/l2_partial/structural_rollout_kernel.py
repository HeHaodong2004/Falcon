"""Single-thread canonical integration; random draws remain in NumPy order.

This compiles only state arithmetic. It does not change the transition kernel,
world count, canonical reset trials, timestamps or random number stream.
"""
import math
import threading
import numpy as np
from numba import njit
from .probability import transition_matrices
from .repair_model import NATIVE_DT
from .structural_budget import check_budget


@njit(cache=True,nogil=True)
def _integrate(initial,target,dt,a,decay,noise,extra,mask,jump,retention):
    S,W,N,_=noise.shape
    path=np.empty((S+1,W,N,4),np.float64)
    path[0]=initial
    for s in range(S):
        for w in range(W):
            for n in range(N):
                vx=path[s,w,n,2];vz=path[s,w,n,3]
                if mask[s,w,n]:
                    vx=retention*vx+jump[s,w,n,0]
                    vz=retention*vz+jump[s,w,n,1]
                aa=a[s,w,n];ee=decay[s,w,n]
                path[s+1,w,n,0]=(path[s,w,n,0]+aa*vx)+(dt[s]-aa)*target[w,n,0]+noise[s,w,n,0]+extra[s,w,n,0]
                path[s+1,w,n,1]=(path[s,w,n,1]+aa*vz)+(dt[s]-aa)*target[w,n,1]+noise[s,w,n,1]+extra[s,w,n,1]
                path[s+1,w,n,2]=ee*vx+(1-ee)*target[w,n,0]+noise[s,w,n,2]
                path[s+1,w,n,3]=ee*vz+(1-ee)*target[w,n,1]+noise[s,w,n,3]
    return path


_WARM=False
_LOCK=threading.Lock()


def warmup_rollout_kernel():
    global _WARM
    with _LOCK:
        if not _WARM:
            x=np.zeros((1,1,4));target=np.zeros((1,1,2));noise=np.zeros((1,1,1,4));extra=np.zeros((1,1,1,2))
            _integrate(x,target,np.array([.048]),np.zeros((1,1,1)),np.ones((1,1,1)),noise,extra,np.zeros((1,1,1),np.bool_),extra,1.)
            _WARM=True


def rollout(model,means,targets,groups,valid,rel,start,rng,reset_already):
    current=float(start);spec=[];outputs=[0]
    for relative in rel[1:]:
        stop=float(start+relative)
        while current<stop-1e-10:
            at_grid=abs(current/NATIVE_DT-round(current/NATIVE_DT))<1e-7
            boundary=(math.floor((current+1e-9)/NATIVE_DT)+1)*NATIVE_DT
            nxt=min(stop,boundary)
            spec.append((nxt-current,at_grid and not reset_already))
            current=nxt;reset_already=abs(current-boundary)>1e-9
        outputs.append(len(spec))
    S=len(spec);W,N,_=means.shape
    noise=np.zeros((S,W,N,4));extra=np.zeros((S,W,N,2));mask=np.zeros((S,W,N),bool);jump=np.zeros((S,W,N,2))
    aa=np.zeros((S,W,N));decay=np.ones((S,W,N));dt=np.array([x[0] for x in spec]);factors={}
    probability=-math.expm1(-model.maneuver_rate*NATIVE_DT)
    for s,(step,reset) in enumerate(spec):
        check_budget('forecast:process_noise_preparation')
        if reset:
            mask[s]=rng.random((W,N))<probability
            jump[s]=rng.normal(0.,model.reset_std,size=(W,N,2))
        for rate,select,target,sqrt_diff in groups:
            key=(round(step,12),rate)
            if key not in factors:
                F,Q,a,e=transition_matrices(step,rate)
                factors[key]=(np.linalg.cholesky(Q+np.eye(4)*1e-16),a,e)
            L,a,e=factors[key]
            noise[s].reshape(-1,4)[select]=(rng.normal(size=(len(select),4))@L.T)*sqrt_diff
            aa[s].ravel()[select]=a;decay[s].ravel()[select]=e
            if rate>0:
                extra[s].reshape(-1,2)[select]=rng.normal(size=(len(select),2))*np.sqrt(.0025*step)
    path=_integrate(means,targets,dt,aa,decay,noise,extra,mask,jump,model.maneuver_retention)
    selected=path[np.asarray(outputs)]
    positions=np.where(valid[None,...,None],selected[...,:2],np.nan).transpose(1,0,2,3)
    velocities=np.where(valid[None,...,None],selected[...,2:],np.nan).transpose(1,0,2,3)
    until=(math.floor((current-1e-9)/NATIVE_DT)+1)*NATIVE_DT
    return positions,velocities,path[-1],until
