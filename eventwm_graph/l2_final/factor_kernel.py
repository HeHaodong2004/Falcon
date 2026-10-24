"""Compiled arithmetic for the existing offline CTMC gradient factor sampler.

Same Gaussian initialization, NumPy Generator, array draw order and CTMC law.
No online inference module imports this training-only optimization.
"""
import math
import numpy as np
from numba import njit
from eventwm_graph.l2_120k.learning import psd_root


@njit(cache=True)
def evolve_factors(base,nxt,stopped,times,law,rng):
    length=len(base);h=len(times);noise=np.zeros_like(base);coef=np.zeros((length,2));clock=np.zeros(length)
    out_base=np.zeros((length,h,2));out_noise=np.zeros_like(out_base);out_coef=np.zeros((length,h))
    for ti in range(h):
        end=times[ti]
        while True:
            active=False
            for i in range(length):
                if clock[i]<end-1e-12:active=True
            if not active:break
            z=rng.normal(0.,1.,size=(length,2,2));changed=np.zeros(length,np.bool_)
            for i in range(length):
                valid=clock[i]<end-1e-12;until=min(nxt[i],end);dt=max(0.,until-clock[i]) if valid else 0.
                k=law[3] if stopped[i] else law[2];e=math.exp(-k*dt);a=-math.expm1(-k*dt)/k
                vv=-math.expm1(-2*k*dt)/(2*k);pv=(1-e)**2/(2*k*k);pp=(dt-2*a+vv)/(k*k)
                if k*dt<1e-3:pp=dt**3*(1/3-k*dt/4+7*(k*dt)**2/60)
                lv=math.sqrt(max(vv,0.));lp=pv/lv if lv>0 else 0.;lpp=math.sqrt(max(pp-lp*lp,0.))
                for axis in range(2):
                    base[i,axis]+=a*base[i,2+axis];base[i,2+axis]*=e
                    noise[i,axis]+=a*noise[i,2+axis];noise[i,2+axis]*=e
                    incp=lp*z[i,0,axis]+lpp*z[i,1,axis];incv=lv*z[i,0,axis]
                    if stopped[i]:
                        base[i,axis]+=incp*math.sqrt(law[4]);base[i,2+axis]+=incv*math.sqrt(law[4])
                    else:noise[i,axis]+=incp;noise[i,2+axis]+=incv
                coef[i,0]+=a*coef[i,1];coef[i,1]*=e
                if not stopped[i]:coef[i,0]+=dt-a;coef[i,1]+=1-e
                if valid:clock[i]=until
                if valid and nxt[i]<=end+1e-12:changed[i]=True;stopped[i]=not stopped[i]
            bs=rng.normal(0.,1.,size=(length,2))*law[5]
            ns=rng.normal(0.,1.,size=(length,2))*math.sqrt(law[6])
            for i in range(length):
                wait=rng.exponential(1/(law[1] if stopped[i] else law[0]))
                if not changed[i]:continue
                for axis in range(2):
                    base[i,2+axis]=bs[i,axis] if stopped[i] else 0.
                    noise[i,2+axis]=0. if stopped[i] else ns[i,axis]
                coef[i,1]=0. if stopped[i] else 1.;nxt[i]=clock[i]+wait
        for i in range(length):
            out_base[i,ti]=base[i,:2];out_noise[i,ti]=noise[i,:2];out_coef[i,ti]=coef[i,0]
    return out_base,out_noise,out_coef


def sample_factors(mean,cov,times,law,rng,samples,mixture=None):
    n=len(mean);shape=(n,3,2,samples)
    init=mean[:,None,None,None,:]+np.einsum('nij,nkasj->nkasi',psd_root(cov),rng.normal(size=shape+(4,)))
    if mixture is not None:
        gm,gc,gp=mixture
        choice=(rng.random(shape+(1,))>np.cumsum(gp,axis=-1)[:,None,None,None,:]).sum(-1)
        idx=np.arange(n)[:,None,None,None]
        init=gm[idx,choice]+np.einsum('nkasij,nkasj->nkasi',psd_root(gc)[idx,choice],rng.normal(size=shape+(4,)))
    stopped=np.broadcast_to(np.array([False,True])[None,None,:,None],shape).copy()
    init[:,:,1,:,2:]=rng.normal(size=(n,3,samples,2))*law['stop_reset_std']
    nxt=rng.exponential(1/np.where(stopped,law['restart_rate'],law['stop_rate']))
    values=np.array([law[k] for k in ('stop_rate','restart_rate','move_relaxation','stop_relaxation','stop_diffusion','stop_reset_std','restart_variance_factor')])
    result=evolve_factors(init.reshape(-1,4).copy(),nxt.ravel(),stopped.ravel(),np.asarray(times,float),values,rng)
    return result[0].reshape(shape+(len(times),2)),result[1].reshape(shape+(len(times),2)),result[2].reshape(shape+(len(times),))
