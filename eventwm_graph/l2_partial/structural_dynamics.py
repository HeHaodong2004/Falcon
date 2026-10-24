"""The preserved canonical transition law, without redundant hidden-state draws."""
import math
import numpy as np
from .repair_model import NATIVE_DT
from .maneuver import sample_resets
from .probability import transition_matrices


def symmetric_gaussian_sample(mean,covariance,rng):
    """Use the unique symmetric PSD root, not a LAPACK eigenvector basis.

    Both factors sample the same Gaussian. U sqrt(D) changes a common seeded
    world when an eigenvector sign flips; U sqrt(D) U.T does not. No speed or
    variance clipping is introduced (negative roundoff eigenvalues use zero).
    """
    mean=np.asarray(mean,float);covariance=np.asarray(covariance,float)
    eigenvalues,U=np.linalg.eigh(covariance)
    if np.any(eigenvalues < -1e-8):raise ValueError('indefinite state covariance')
    root=(U*np.sqrt(np.maximum(eigenvalues,0))[...,None,:])@np.swapaxes(U,-1,-2)
    return mean+np.einsum('nij,nj->ni',root,rng.normal(size=mean.shape))


def sample_hidden_states(mean,covariance,count,rng,exchangeable=True):
    from .hidden import _gaussians
    mean,covariance=_gaussians(mean,covariance)
    if exchangeable:
        if len(mean)!=1:raise ValueError('exchangeable hidden prior needs one Gaussian')
        mean=np.broadcast_to(mean,(count,4));covariance=np.broadcast_to(covariance,(count,4,4))
    elif len(mean)!=count:raise ValueError('private hidden count mismatch')
    return symmetric_gaussian_sample(mean,covariance,rng)

def sample_maneuver_paths(self,worlds,rel,start,rng,n):
    # One physical-state tensor for tracked AND private hidden people.
    # Their targets/rates remain distinct; every latent person is retained.
    W=len(worlds)
    counts=[int(round(h.expected_hidden)) if h.hidden_mean is not None else 0 for h in worlds]
    N=max((h.n+k for h,k in zip(worlds,counts)),default=0)
    means=np.zeros((W,N,4));targets=np.zeros((W,N,2));diff=np.zeros((W,N));rates=np.zeros((W,N));valid=np.zeros((W,N),bool)
    for w,(h,k) in enumerate(zip(worlds,counts)):
        means[w,:h.n]=h.mean;targets[w,:h.n]=h.intent;diff[w,:h.n]=h.diffusion;rates[w,:h.n]=h.relaxation
        if k:
            if not h.hidden_exchangeable and h.hidden_maneuver_mean is None and not np.any(h.hidden_covariance):
                # This is an already sampled private state, not a new draw.
                # Drawing zero-noise normals here advances the RNG stream and
                # changes common future worlds at every action boundary.
                hidden=h.hidden_mean.copy()
            elif h.hidden_maneuver_mean is not None:
                if h.hidden_exchangeable:
                    gm=np.repeat(h.hidden_maneuver_mean,k,axis=0);gc=np.repeat(h.hidden_maneuver_covariance,k,axis=0)
                    gp=np.repeat(h.hidden_maneuver_probability,k,axis=0)
                else:gm,gc,gp=h.hidden_maneuver_mean,h.hidden_maneuver_covariance,h.hidden_maneuver_probability
                choice=(np.cumsum(gp,axis=1)<rng.random((k,1))).sum(1);ii=np.arange(k)
                hidden=sample_hidden_states(gm[ii,choice],gc[ii,choice],k,rng,False)
            else:hidden=sample_hidden_states(h.hidden_mean,h.hidden_covariance,k,rng,h.hidden_exchangeable)
            means[w,h.n:h.n+k]=hidden;diff[w,h.n:h.n+k]=.25;rates[w,h.n:h.n+k]=.7
        valid[w,:h.n+k]=True
    positions=np.full((W,len(rel),N,2),np.nan);velocities=positions.copy()
    positions[:,0]=np.where(valid[...,None],means[:,:,:2],np.nan);velocities[:,0]=np.where(valid[...,None],means[:,:,2:],np.nan)
    flat_target=targets.reshape(-1,2);flat_diff=diff.ravel();groups=[]
    for rate in np.unique(rates[valid]):
        select=np.flatnonzero((valid&(rates==rate)).ravel())
        groups.append((float(rate),select,flat_target[select],np.sqrt(flat_diff[select,None])))
    if self.optimized:
        from .structural_rollout_kernel import rollout
        positions,velocities,means,until=rollout(self,means,targets,groups,valid,rel,start,rng,
            all(h.maneuver_until>start+1e-9 for h in worlds))
    else:
        factors={};current=float(start);reset_already=all(h.maneuver_until>start+1e-9 for h in worlds)
        for si,relative in enumerate(rel[1:],1):
            stop=float(start+relative)
            while current<stop-1e-10:
                at_grid=abs(current/NATIVE_DT-round(current/NATIVE_DT))<1e-7
                if at_grid and not reset_already:
                    means,_=sample_resets(means,rng,self.maneuver_rate,self.reset_std,retention=self.maneuver_retention)
                boundary=(math.floor((current+1e-9)/NATIVE_DT)+1)*NATIVE_DT
                nxt=min(stop,boundary);dt=nxt-current;flat=means.reshape(-1,4)
                for rate,select,target,sqrt_diff in groups:
                    key=(round(dt,12),rate)
                    if key not in factors:
                        F,Q,a,e=transition_matrices(dt,rate);factors[key]=(F,np.linalg.cholesky(Q+np.eye(4)*1e-16),a,e)
                    F,L,a,e=factors[key];x=flat[select]@F.T
                    x[:,:2]+=(dt-a)*target;x[:,2:]+=(1-e)*target
                    x+=(rng.normal(size=x.shape)@L.T)*sqrt_diff
                    if rate>0:x[:,:2]+=rng.normal(size=x[:,:2].shape)*np.sqrt(.0025*dt)
                    flat[select]=x
                current=nxt;reset_already=abs(current-boundary)>1e-9
            positions[:,si]=np.where(valid[...,None],means[:,:,:2],np.nan)
            velocities[:,si]=np.where(valid[...,None],means[:,:,2:],np.nan)
        until=(math.floor((current-1e-9)/NATIVE_DT)+1)*NATIVE_DT
    updated=[]
    for w,(h,k) in enumerate(zip(worlds,counts)):
        changes={'mean':means[w,:h.n].copy(),'maneuver_mean':None,'maneuver_covariance':None,'maneuver_probability':None,'maneuver_until':until}
        if h.hidden_mean is not None:changes.update(hidden_mean=means[w,h.n:h.n+k].copy(),hidden_covariance=np.zeros((k,4,4)),hidden_exchangeable=False,
            hidden_maneuver_mean=None,hidden_maneuver_covariance=None,hidden_maneuver_probability=None)
        updated.append(h.fork(**changes))
    return positions,velocities,updated

