"""Pure NumPy paired diagnostics; no model/data loading or state mutation.

Scores use float64 arithmetic on saved float32 probabilities for stable paired
subtraction. This changes no probability, clipping convention or decision.
"""
import numpy as np


def probability_check(p):
    p=np.asarray(p)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('invalid probabilities')
    if np.any(p<0) or np.any(p>1) or np.any(abs(p.sum(1,dtype=np.float64)-1)>1e-6):raise ValueError('probability bounds/normalization')


def counts(y):
    y=np.asarray(y)
    return [int((y>0).sum()),int((y==1).sum()),int((y==2).sum())]


def expected(p):
    # Preserve the global float32 revelation-sum contract, then accumulate F64.
    return [float((p[:,1]+p[:,2]).sum(dtype=np.float64)),float(p[:,1].sum(dtype=np.float64)),float(p[:,2].sum(dtype=np.float64))]


def cm(y,state):
    return np.bincount(3*np.asarray(y,dtype=np.int64)+np.asarray(state,dtype=np.int64),minlength=9).reshape(3,3)


def loss_arrays(y,p):
    q=np.asarray(p,np.float64); y=np.asarray(y,np.int64)
    ce=-np.log(np.clip(q[np.arange(len(y)),y],1e-9,1))
    error=q.copy();error[np.arange(len(y)),y]-=1
    return np.column_stack((ce,np.square(error).sum(1)))


def sensitivity(a,b):
    if not len(a):return None
    a=np.asarray(a);b=np.asarray(b)
    delta=abs(a.astype(np.float64)-b.astype(np.float64));qdiff=abs((a[:,1]+a[:,2]).astype(np.float64)-(b[:,1]+b[:,2]).astype(np.float64))
    hard_a=a.argmax(1);hard_b=b.argmax(1)
    correct=(hard_a==hard_b)&(hard_a>0);union=int((hard_a>0).sum()+(hard_b>0).sum()-correct.sum())
    soft_min=np.minimum(a[:,1:],b[:,1:]).sum(dtype=np.float64);soft_max=np.maximum(a[:,1:],b[:,1:]).sum(dtype=np.float64)
    qa=(a[:,1]+a[:,2]).astype(np.float64);qb=(b[:,1]+b[:,2]).astype(np.float64)
    return dict(cells=len(a),mean_absolute_probability_change=delta.mean(0).tolist(),
        mean_total_variation=float(.5*delta.sum(1).mean()),mean_absolute_revelation_change=float(qdiff.mean()),
        absolute_revelation_change_quantiles={str(t):float(np.quantile(qdiff,t)) for t in (0,.25,.5,.75,.95,.99,1)},
        hard_labeled_set_overlap=float(correct.sum()/union) if union else 0.,
        soft_labeled_minmax_overlap=float(soft_min/soft_max) if soft_max else 0.,
        soft_revelation_minmax_overlap=float(np.minimum(qa,qb).sum()/np.maximum(qa,qb).sum()) if np.maximum(qa,qb).sum() else 0.,
        expected_counts_a=expected(a),expected_counts_b=expected(b))


def paired_scores(y_a,y_b,p_a,p_b):
    n=len(y_a)
    if not (len(y_b)==len(p_a)==len(p_b)==n):raise ValueError('unaligned pair arrays')
    if not n:return dict(cells=0,own_scores=None,cross_scores=None,gain=None,own_confusion=np.zeros((3,3)).tolist(),cross_confusion=np.zeros((3,3)).tolist())
    aa=loss_arrays(y_a,p_a);bb=loss_arrays(y_b,p_b);ab=loss_arrays(y_b,p_a);ba=loss_arrays(y_a,p_b)
    own=.5*(aa.mean(0)+bb.mean(0));cross=.5*(ab.mean(0)+ba.mean(0))
    state_a=p_a.argmax(1);state_b=p_b.argmax(1)
    return dict(cells=n,true_counts_a=counts(y_a),true_counts_b=counts(y_b),
        expected_counts_a=expected(p_a),expected_counts_b=expected(p_b),
        own_scores=own.tolist(),cross_scores=cross.tolist(),gain=(cross-own).tolist(),
        own_confusion=(.5*(cm(y_a,state_a)+cm(y_b,state_b))).tolist(),
        cross_confusion=(.5*(cm(y_b,state_a)+cm(y_a,state_b))).tolist(),
        sensitivity=sensitivity(p_a,p_b))


def key_codes(keys):
    values=np.asarray(keys,np.int64)
    return ((values[:,0]+(1<<31)).astype(np.uint64)<<np.uint64(32)) | (values[:,1]+(1<<31)).astype(np.uint64)


def project_targets(keys,a,b):
    """Build scoring-only truth extension; never pass extras to inference."""
    code=key_codes(keys);ys=[];outside=[]
    if len(code)>1 and np.any(code[1:]<=code[:-1]):raise ValueError('union keys must be unique row-major')
    for target in (a,b):
        y=np.zeros(len(keys),np.uint8);tk=np.asarray(list(target),np.int32).reshape(-1,2)
        tc=key_codes(tk);at=np.searchsorted(code,tc);valid=at<len(keys)
        if len(keys):valid&=code[np.minimum(at,len(keys)-1)]==tc
        values=list(target.values());y[at[valid]]=[values[i][0] for i in np.flatnonzero(valid)]
        ys.append(y);outside.append([tuple(k) for k in tk[~valid]])
    extra=sorted(set(outside[0])|set(outside[1]));extras=np.asarray(extra,np.int32).reshape(-1,2)
    tails=[np.array([target.get(k,(0,0))[0] for k in extra],np.uint8) for target in (a,b)]
    return ys,extras,tails,[len(v) for v in outside]


def union_predictions(keys,left_keys,left_p,right_keys,right_p):
    code=key_codes(keys);result=[];supports=[]
    for own,p in ((left_keys,left_p),(right_keys,right_p)):
        at=np.searchsorted(code,key_codes(own));np.testing.assert_array_equal(keys[at],own)
        full=np.zeros((len(keys),3),np.float32);full[:,0]=1;full[at]=p
        mask=np.zeros(len(keys),bool);mask[at]=True;result.append(full);supports.append(mask)
    return result,supports


def compare_probabilities(a,b):
    if not len(a):return dict(cells=0,max_probability_difference=0.,mean_probability_difference=0.,hard_label_disagreements=0)
    d=abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(cells=len(a),max_probability_difference=float(d.max()),mean_probability_difference=float(d.mean()),
                hard_label_disagreements=int((a.argmax(1)!=b.argmax(1)).sum()),rows_above_2e6=int((d.max(1)>2e-6).sum()))


def spatial_regions(keys,path_a,path_b,resolution):
    centers=((keys.astype(np.float32)+np.float32(.5))*np.float32(resolution)).astype(np.float64)
    outputs=[]
    for path in (path_a,path_b):
        distance=np.full(len(keys),np.inf);opportunity=np.zeros(len(keys),bool)
        for row in path[1:]:
            delta=centers-row[[1,3]];d2=np.square(delta).sum(1);distance=np.minimum(distance,d2)
            forward=delta[:,0]*np.cos(row[4])-delta[:,1]*np.sin(row[4])
            opportunity|=(d2<=100)&(forward>=.5*np.maximum(np.sqrt(d2),1e-8))
        outputs.append((distance<=4,opportunity))
    return {name:(outputs[0][i]&~outputs[1][i],outputs[1][i]&~outputs[0][i])
            for i,name in enumerate(('exclusive_2m_near_path','exclusive_10m_60degree_heading_opportunity'))}


def spatial_shifts(regions,p_a,p_b,y_a,y_b):
    qa=(p_a[:,1]+p_a[:,2]).astype(np.float64);qb=(p_b[:,1]+p_b[:,2]).astype(np.float64)
    truth=(y_a>0).astype(np.float64)-(y_b>0).astype(np.float64);delta=qa-qb;result={}
    for name,(a,b) in regions.items():
        n=int(a.sum()+b.sum());d=float(delta[a].sum()-delta[b].sum());t=float(truth[a].sum()-truth[b].sum())
        result[name]=dict(cells_a=int(a.sum()),cells_b=int(b.sum()),
            pair_mean_own_minus_cross_expected_count=.5*d,pair_mean_own_minus_cross_true_count=.5*t,
            per_cell_mean_expected_contrast=d/n if n else None,per_cell_mean_true_contrast=t/n if n else None)
    return result


def check_math():
    """Small exact/algebraic checks; no source records or neural inference."""
    y_a=np.array([1,1,0,0],np.uint8);y_b=np.array([0,0,1,1],np.uint8)
    q=np.tile(np.array([.1,.7,.2],np.float32),(4,1))
    direct=paired_scores(y_a,y_b,q,q);np.testing.assert_allclose(direct['gain'],[0,0],atol=1e-14)
    pa=q.copy();pb=q.copy();pa[2:]=[1,0,0];pb[:2]=[1,0,0]
    gated=paired_scores(y_a,y_b,pa,pb);assert min(gated['gain'])>0
    onea=np.eye(3)[y_a];oneb=np.eye(3)[y_b]
    analytic=np.sum((pa.astype(np.float64)-pb)*(onea-oneb),axis=1).mean()
    np.testing.assert_allclose(gated['gain'][1],analytic,atol=1e-14)
    keys=np.array([[0,0],[1,0],[2,0],[3,0]],np.int32)
    ys,extra,tails,missing=project_targets(keys,{(0,0):(1,0.),(9,9):(2,0.)},{(2,0):(1,0.)})
    assert missing==[1,0] and extra.tolist()==[[9,9]] and tails[0].tolist()==[2]
    np.testing.assert_array_equal(ys[0],[1,0,0,0]);np.testing.assert_array_equal(ys[1],[0,0,1,0])
    u=np.array([[1,0,0]],np.float32)
    ext=paired_scores(np.r_[y_a,2],np.r_[y_b,0],np.r_[pa,u],np.r_[pb,u])
    np.testing.assert_allclose(ext['gain'],np.asarray(gated['gain'])*4/5,atol=1e-14)
    for p in (q,pa,pb):probability_check(p)
    ce,br=loss_arrays(np.array([1]),u)[0];np.testing.assert_allclose([ce,br],[-np.log(1e-9),2])
    return dict(status='PASS_PURE_NUMPY_PAIR_ALGEBRA',candidate_gating_only_gain=gated['gain'],fixed_network_identical_gain=direct['gain'])
