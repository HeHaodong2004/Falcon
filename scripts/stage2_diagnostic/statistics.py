"""Transparent histogram rank metrics, with explicit finite-bin error bounds."""
import numpy as np
from .common import EDGES

def rank_metrics(negative,positive):
    n=np.asarray(negative,np.float64);p=np.asarray(positive,np.float64)
    N=n.sum();P=p.sum()
    if not N or not P:return dict(prevalence=float(P/max(N+P,1e-20)),auroc=None,auprc_ap=None)
    below=np.cumsum(n)-n
    auc=float(np.sum(p*(below+.5*n))/(P*N))
    tie=float(np.sum(p*n)/(P*N))
    pd=p[::-1];nd=n[::-1];tp=np.cumsum(pd);fp=np.cumsum(nd)
    precision=tp/np.maximum(tp+fp,1e-30);recall=tp/P
    ap=float(np.sum((pd/P)*precision))
    beforep=tp-pd;beforen=fp-nd
    lower=beforep/np.maximum(beforep+beforen+nd,1e-30)
    upper=tp/np.maximum(tp+beforen,1e-30)
    return dict(prevalence=float(P/(P+N)),auroc=auc,
        auroc_binning_bounds=[max(0.,auc-.5*tie),min(1.,auc+.5*tie)],
        auprc_ap=ap,auprc_binning_bounds=[float(np.sum((pd/P)*lower)),float(np.sum((pd/P)*upper))],
        prevalence_auprc_baseline=float(P/(P+N)),method='noninterpolated AP from highresolution bins; NOT trapezoidal PR area')

def quantiles(histogram,qs=(0,.01,.05,.25,.5,.75,.95,.99,1)):
    h=np.asarray(histogram,np.float64);nonzero=np.flatnonzero(h)
    if not len(nonzero):return {}
    c=np.cumsum(h);total=c[-1];result={}
    for q in qs:
        i=nonzero[0] if q==0 else nonzero[-1] if q==1 else min(int(np.searchsorted(c,q*total)),len(h)-1)
        result[str(q)]={'estimate':float((EDGES[i]+EDGES[i+1])/2),'interval':[float(EDGES[i]),float(EDGES[i+1])]}
    return result
