"""Declared proper scores and diagnostic distances over complete populations."""
import math
import numpy as np
from scipy.spatial.distance import cdist


def population_distance(a,b,bandwidth=3.):
    """Hilbert norm of Gaussian-kernel counting-measure + count embeddings.

    Includes every human, including never-localized sampled states. This is not
    GOSPA used as an ES distance. Scores have embedding units, not meters. No
    claim of numerical equivalence to an unrelated historical ES3 is made.
    """
    a=np.asarray(a,float).reshape(-1,2);b=np.asarray(b,float).reshape(-1,2)
    aa=np.exp(-cdist(a,a,'sqeuclidean')/(2*bandwidth**2)).sum()
    bb=np.exp(-cdist(b,b,'sqeuclidean')/(2*bandwidth**2)).sum()
    ab=np.exp(-cdist(a,b,'sqeuclidean')/(2*bandwidth**2)).sum()
    return math.sqrt(max(0.,aa+bb-2*ab+(len(a)-len(b))**2))


def population_energy(samples,truth,dimension=2,return_parts=False):
    """U-statistic ES for iid samples of the normalized continuous model law."""
    W=len(samples)
    if W<2:raise ValueError('unbiased energy score requires >=2 iid worlds')
    sets=[np.asarray(p).reshape(-1,dimension) for p in [*samples,truth]]
    counts=np.array([len(p) for p in sets]);points=np.concatenate(sets)
    membership=np.repeat(np.eye(W+1),counts,axis=1)
    kernel=np.exp(-cdist(points,points,'sqeuclidean')/18.)
    gram=membership@kernel@membership.T
    diagonal=np.diag(gram)
    distance=np.sqrt(np.maximum(0.,diagonal[:,None]+diagonal[None,:]-2*gram+(counts[:,None]-counts[None,:])**2))
    first=float(distance[:W,W].mean());pair=float(distance[:W,:W].sum()/(W*(W-1)))
    result=first-.5*pair
    return dict(score=result,mean_truth_distance=first,mean_pairwise_distance=pair) if return_parts else result


def vector_energy(samples,truth):
    x=np.asarray(samples,float).reshape(len(samples),-1);y=np.asarray(truth,float).ravel()
    if len(x)<2:raise ValueError('at least2 samples required')
    return float(np.linalg.norm(x-y,axis=-1).mean()-cdist(x,x).sum()/(2*len(x)*(len(x)-1)))


def count_pmf(belief):
    size=max(h.n+len(h.hidden_pmf) for h in belief.hypotheses)
    p=np.zeros(size)
    for h in belief.hypotheses:p[h.n:h.n+len(h.hidden_pmf)]+=h.weight*h.hidden_pmf
    if abs(p.sum()-1)>1e-8:raise ValueError('a point count score requires a normalized law')
    return p


def count_scores(belief,count):
    p=count_pmf(belief);prob=float(p[count]) if count<len(p) else 0.
    return dict(true_count=int(count),count_mae=abs(float(p@np.arange(len(p)))-count),
        count_nll=-math.log(prob) if prob else None,count_probability=prob,
        infinite_count_nll=prob==0.,predicted_count=float(p@np.arange(len(p))))
