"""Explicit Gaussian disintegration and integrated Ornstein–Uhlenbeck law."""
import math
import numpy as np
from .state import Hypothesis, normalized

def transition_matrices(dt,k,q=1.):
    if dt<0:raise ValueError('time reversal')
    if k<1e-8:
        e=1.;a=dt; qpp=dt**3/3;qpv=dt**2/2;qvv=dt
    else:
        e=math.exp(-k*dt);a=-math.expm1(-k*dt)/k
        qpp=(dt-2*a+(-math.expm1(-2*k*dt))/(2*k))/(k*k)
        qpv=(1-e)**2/(2*k*k);qvv=(-math.expm1(-2*k*dt))/(2*k)
    F=np.eye(4);F[:2,2:]=a*np.eye(2);F[2:,2:]=e*np.eye(2)
    Q=np.zeros((4,4));Q[:2,:2]=max(0,qpp)*np.eye(2);Q[:2,2:]=Q[2:,:2]=qpv*np.eye(2);Q[2:,2:]=qvv*np.eye(2)
    return F,Q*q,a,e

def advance(h,dt):
    F,Q,a,e=transition_matrices(dt,h.relaxation)
    target=h.intent if h.intent is not None else h.mean[:,2:]
    mean=h.mean@F.T;mean[:,:2]+=(dt-a)*target;mean[:,2:]+=(1-e)*target
    scale=h.diffusion if h.diffusion is not None else np.full(h.n,.16)
    cov=F@h.covariance@F.T+scale[:,None,None]*Q
    if h.relaxation>0:cov[:,:2,:2]+=np.eye(2)[None]*(.0025*dt)
    return h.fork(mean=mean,covariance=cov)

def condition_position(mean,covariance,position):
    """p(v|p=z), with an exact Dirac position and stable predictive density."""
    z=np.asarray(position);P=covariance[:2,:2];d=z-mean[:2]
    # Singular Diracs have a separate exact support rule, not a fake noisy sensor.
    if np.max(np.abs(P))<1e-12:
        if np.linalg.norm(d)>1e-8:return None,-np.inf
        return (mean.copy(),covariance.copy()),0.
    # A two-dimensional analytic inverse avoids a LAPACK dispatch per candidate.
    # The same 1e-12 numerical regularization and Gaussian law are retained.
    aa=P[0,0]+1e-12;bb=P[0,1];cc=P[1,0];dd=P[1,1]+1e-12
    determinant=aa*dd-bb*cc
    if determinant<=0:return None,-np.inf
    ld=math.log(determinant);inv=np.array([[dd,-bb],[-cc,aa]])/determinant
    gain=covariance[:, :2]@inv
    m=mean+gain@d;c=covariance-gain@covariance[:2,:]
    m[:2]=z;c[:2,:]=0;c[:,:2]=0;c=(c+c.T)/2
    return (m,c),float(-math.log(2*math.pi)-.5*(ld+d@inv@d))

def consume_hidden(pmf):
    counts=np.arange(len(pmf));den=pmf@counts
    if den<=0:return None,0.
    posterior=np.zeros_like(pmf);posterior[:-1]=pmf[1:]*counts[1:]/den
    return posterior,float(den)

def weighted_condition(hypotheses,likelihood):
    """Exact finite-mixture Bayes update for a declared coarsened event."""
    ws=np.array([h.weight for h in hypotheses])*np.asarray(likelihood)
    p=float(ws.sum())
    return ([h.fork(weight=float(w/p)) for h,w in zip(hypotheses,ws) if w>0] if p else []),p

def condition_empty_quadrature(hypotheses,boxes,unresolved_mass,worlds=16,points=256):
    """Deterministic Gaussian quadrature for certified-empty rectangles.

    This is an approximate truncated joint distribution, not an unchanged
    Gaussian falsely labeled empty. Every retained support point obeys the
    certificate; conditional velocity covariance is integrated analytically.
    Evidence error against Gaussian-CDF integration is returned explicitly.
    """
    from scipy.special import ndtri
    from scipy.stats import qmc,multivariate_normal
    candidates=[];analytic=0.;empirical=0.
    for h in hypotheses:
        if h.n==0:
            candidates.append((h.weight,h));analytic+=h.weight;empirical+=h.weight;continue
        positions=h.mean[:,:2];P=h.covariance[:,:2,:2]
        if np.max(np.abs(P))<1e-12:
            valid=all(not np.any(np.all((positions>=lo)&(positions<=hi),axis=1)) for lo,hi in boxes)
            if valid:candidates.append((h.weight,h));analytic+=h.weight;empirical+=h.weight
            continue
        normal=ndtri(np.clip(qmc.Sobol(h.n*2,scramble=True,seed=73).random_base2(int(np.log2(points))),1e-10,1-1e-10)).reshape(points,h.n,2)
        eig,U=np.linalg.eigh(P);sample=positions+np.einsum('nij,pnj->pni',U,normal*np.sqrt(np.maximum(eig,0))[None])
        valid=np.ones(points,bool)
        for lo,hi in boxes:valid &= ~np.any(np.all((sample>=lo)&(sample<=hi),axis=-1),axis=1)
        mass=h.weight*valid.mean();empirical+=mass
        # Analytic probability for one rectangle, conditional independent slots.
        if len(boxes)==1:
            lo,hi=map(np.asarray,boxes[0]);empty=1.
            for m,p in zip(positions,P):
                dist=multivariate_normal(mean=m,cov=p+np.eye(2)*1e-14,allow_singular=True)
                inside=dist.cdf(hi)-dist.cdf([lo[0],hi[1]])-dist.cdf([hi[0],lo[1]])+dist.cdf(lo)
                empty*=max(0.,min(1.,1-inside))
            analytic+=h.weight*empty
        else:analytic+=mass  # no false claim of exact union-box CDF
        ids=np.flatnonzero(valid)
        if not len(ids):continue
        # Compress the truncated quadrature by deterministic equal-mass strata.
        chosen=ids[np.minimum(len(ids)-1,((np.arange(min(worlds,len(ids)))+.5)*len(ids)/min(worlds,len(ids))).astype(int))]
        for index in chosen:
            mean=h.mean.copy();cov=h.covariance.copy()
            for i in range(h.n):(mean[i],cov[i]),_=condition_position(h.mean[i],h.covariance[i],sample[index,i])
            candidates.append((mass/len(chosen),h.fork(mean=mean,covariance=cov)))
    # The unresolved channel declares P(empty)=0.5 per certificate; it is a
    # contamination observation law, not an unproved posterior bound.
    residual=unresolved_mass*.5**len(boxes);total=empirical+residual
    if total<=0:return [h.fork(weight=0.) for h in hypotheses],1.,{'unsupported_empty':True}
    # Keep all bounded per-hypothesis quadrature components, preserving mass.
    hs=[h.fork(weight=w/total) for w,h in candidates]
    return hs,residual/total,{'empty_evidence_quadrature':empirical,'empty_evidence_cdf':analytic,
                            'empty_evidence_absolute_error':abs(empirical-analytic),'empty_quadrature_points':points}
