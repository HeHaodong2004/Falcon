"""Frozen identities and evaluation helpers; no training or production mutation."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/stage2_diagnostic_20261025'
EVALUATION = ROOT / 'output/stage2_segment_20261024/evaluation/final_segment_dev'
CHECKPOINT = ROOT / 'output/stage2_segment_20261024/training/final/last.pt'
CHECKPOINT_SHA = '1ba6844d8b7eb336c46b41125eefae16e413c4ba408da9b0a1dc47ebcef66bfd'
THRESHOLDS = np.array([.01,.02,.05,.10,.15,.20,.30,.40,.50,.60,.70,.80,.90], np.float64)
# Linear plus logarithmic bin edges preserve detail near the dominant zero mass.
# AUC/AP are explicitly histogram estimates, with conservative binning bounds.
EDGES = np.unique(np.r_[0., np.linspace(0.,1.,65537), np.geomspace(1e-10,1.,65537)])

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def stable(value):
    return int(hashlib.sha256(('stage2-diagnostic-2525:'+str(value)).encode()).hexdigest()[:16],16)

def make_model(device='cuda:1'):
    from eventwm_graph.stage2_segment.api import SegmentPredictor
    if digest(CHECKPOINT)!=CHECKPOINT_SHA: raise ValueError('checkpoint identity changed')
    return SegmentPredictor(CHECKPOINT,execution='optimized',future_format='native',device=device,cache_size=2,feature_threads=1)

def labels(keys,target):
    """Exact row-major key membership; never provide target to model inference."""
    keys=np.asarray(keys,np.int32)
    y=np.zeros(len(keys),np.uint8)
    if not target:return y
    truth=np.asarray(list(target),np.int32).reshape(-1,2)
    if not len(keys):raise ValueError('positive target outside empty domain')
    lo=np.minimum(keys.min(0),truth.min(0)); hi=np.maximum(keys.max(0),truth.max(0))
    width=np.int64(hi[1])-lo[1]+1
    code=(keys[:,0].astype(np.int64)-lo[0])*width+keys[:,1]-lo[1]
    tc=(truth[:,0].astype(np.int64)-lo[0])*width+truth[:,1]-lo[1]
    if np.any(np.diff(code)<=0):raise ValueError('keys are not unique row-major cells')
    position=np.searchsorted(code,tc); safe=np.minimum(position,len(keys)-1)
    if np.any(position>=len(keys)) or np.any(code[safe]!=tc):raise ValueError('positive outside frozen candidate universe')
    y[position]=np.array([v[0] for v in target.values()],np.uint8)
    return y

def fixed_probabilities(model,graph,path,keys,bounds=None):
    """Network-only intervention: fixed keys AND map crop, supplied real path.

    This bypasses candidate selection deliberately. Extended-domain values are
    diagnostic extrapolation, never claimed to be the deployed predictor.
    """
    import torch
    from eventwm_graph.stage2_segment.fused import transform,probabilities
    context=model.feature_class(graph,path,model.prepare_graph(graph))
    if not len(keys):return np.empty((0,3),np.float32),np.empty(0,np.float32),context
    if bounds is None:
        lo=np.minimum(context.lo,keys.min(0)-16)
        hi=np.maximum(context.lo+context.shape,keys.max(0)+17)
        bounds=(lo,hi-lo)
    lo,shape=bounds
    context.lo=np.asarray(lo,np.int32).copy();context.shape=np.asarray(shape,np.int32).copy()
    if np.any(context.shape<=0) or int(np.prod(context.shape.astype(np.int64)))>4000000:
        raise ValueError('fixed diagnostic crop is invalid or exceeds fourmillioncells')
    if np.any(keys<context.lo) or np.any(keys>=context.lo+context.shape):
        raise ValueError('fixed candidate outside sharedcrop')
    context.states,context.base_height,context.dist=context.prepared.crop(context.lo,context.shape)
    context._candidate_keys=np.asarray(keys,np.int32)
    x,baseline=transform(context,keys,cuda=model.cuda,device_output=model.cuda is not None)
    p=np.empty((len(keys),3),np.float32);height=np.empty(len(keys),np.float32)
    with torch.inference_mode():
        if model.cuda is not None and len(keys):
            values=model.cuda.forward(x,context,keys);critical=probabilities(values,baseline,p,height,0)
            if len(critical):p[critical]=torch.softmax(torch.from_numpy(values[critical,:3]),-1).numpy()
        else:
            for begin in range(0,len(keys),8192):
                end=min(begin+8192,len(keys));values=model.net(torch.from_numpy(x[begin:end]))
                critical=probabilities(values.numpy(),baseline,p,height,begin)
                if len(critical):p[begin+critical]=torch.softmax(values[critical,:3],-1).numpy()
    if len(path)==1:p.fill(0);p[:,0]=1
    return p,height,context

def confusion(y,pred):
    return np.bincount(3*np.asarray(y,dtype=np.int64)+np.asarray(pred,dtype=np.int64),minlength=9).reshape(3,3)

def extended_rates(cm):
    from eventwm_graph.stage2_segment.evaluate import rates
    cm=np.asarray(cm,np.float64); r=rates(cm)
    tp=cm[1:,1:].sum(); fp=cm[0,1:].sum(); fn=cm[1:,0].sum()
    actual_new=cm[1:].sum(); predicted_free=cm[:,1].sum()
    r.update(revelation_precision=float(tp/max(tp+fp,1e-20)),revelation_recall=float(tp/max(tp+fn,1e-20)),
        revelation_f1=float(2*tp/max(2*tp+fp+fn,1e-20)),
        false_revelation_rate=float(fp/max(cm[0].sum(),1e-20)),
        wrong_free_blocked_rate=float((cm[1,2]+cm[2,1])/max(actual_new,1e-20)),
        false_free_unknown_count=float(cm[0,1]),false_free_blocked_count=float(cm[2,1]),
        false_free_unknown_fraction=float(cm[0,1]/max(predicted_free,1e-20)),
        false_free_blocked_fraction=float(cm[2,1]/max(predicted_free,1e-20)),
        false_free_on_actual_blocked_rate=float(cm[2,1]/max(cm[2].sum(),1e-20)))
    return r
