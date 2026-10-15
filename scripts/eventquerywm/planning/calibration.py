"""Small monotone CV probability mapping, fitted on optimization roots only."""
import hashlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def root_folds(metadata, folds=5, seed=20261010):
    if folds < 2:raise ValueError('Need at least two internal folds')
    if any(m.get('role') != 'train' or m.get('split') != 'train' for m in metadata):
        raise ValueError('Calibration fitting is optimization TRAIN only')
    roots=sorted({m['root_id'] for m in metadata},key=lambda r:hashlib.sha256(f'{seed}/{r}'.encode()).hexdigest())
    if len(roots) < folds:raise ValueError('Too few calibration roots')
    assignment={r:i%folds for i,r in enumerate(roots)}
    return np.array([assignment[m['root_id']] for m in metadata]),assignment


def fit_monotone(exposure, collision):
    x=np.log1p(np.asarray(exposure,dtype=float));y=np.asarray(collision,dtype=float)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isin(y,[0,1]).all() or not len(y):
        raise ValueError('Invalid calibration data')
    if np.any(x<0):raise ValueError('Negative exposure')
    def objective(w):
        z=w[0]+w[1]*x;p=expit(z)
        loss=np.mean(np.logaddexp(0,z)-y*z)+1e-4*w[1]**2
        gradient=np.array([(p-y).mean(),np.mean((p-y)*x)+2e-4*w[1]])
        return loss,gradient
    prior=np.clip(y.mean(),1e-6,1-1e-6)
    result=minimize(objective,[np.log(prior/(1-prior)),1.],jac=True,method='L-BFGS-B',
                    bounds=[(-20,20),(0,100)],options=dict(maxiter=100,ftol=1e-12))
    if not result.success:raise RuntimeError('Calibration optimization failed: '+result.message)
    return dict(intercept=float(result.x[0]),slope=float(result.x[1]),transform='log1p',ridge=1e-4)


def probability(exposure, calibration):
    x=np.asarray(exposure,dtype=float)
    if not np.isfinite(x).all() or (x<0).any():raise ValueError('Invalid exposure')
    return expit(calibration['intercept']+calibration['slope']*np.log1p(x))


def cross_fit(exposure, y, metadata, folds=5, seed=20261010):
    assignment,byroot=root_folds(metadata,folds,seed);oof=np.zeros(len(y));fits=[]
    for f in range(folds):
        train=assignment!=f;test=~train
        fitted=fit_monotone(np.asarray(exposure)[train],np.asarray(y)[train])
        oof[test]=probability(np.asarray(exposure)[test],fitted)
        fits.append(dict(fold=f,fit=fitted,fit_roots=sorted({m['root_id'] for i,m in enumerate(metadata) if train[i]}),
                         assessment_roots=sorted({m['root_id'] for i,m in enumerate(metadata) if test[i]})))
    return oof,dict(final=fit_monotone(exposure,y),folds=fits,root_assignment=byroot,seed=seed,
        caveat='Targets/roots disjoint for CV calibration folds. Learned backbone already saw optimization roots; their in-sample metrics are not generalization estimates.')
