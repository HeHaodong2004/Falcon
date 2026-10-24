"""Offline optimized/reference association comparisons on immutable dev data."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import numpy as np
from .common import BASE, OUT, OLD, SNAP, read, canonical, sha, dump, seed_for
from .model import IntermittentWorldModel
from .baselines import MemoryCV
from eventwm_graph.l2_120k.evaluate import restore
from eventwm_graph.l2_120k.model import load_model


def reference_types():
    # The historical algorithm is compiled under its original runtime path.
    # Both arms deliberately share model-version RNG keys and CURRENT float64
    # dynamics, isolating association arithmetic from the numeric precision fix.
    name='eventwm_graph.l2_intermittent._association_reference'
    module=types.ModuleType(name);module.__file__=str(BASE/'eventwm_graph/l2_intermittent/model.py')
    sys.modules[name]=module
    exec(compile((OUT/'reference_runtime/model.py').read_bytes(),module.__file__,'exec'),module.__dict__)
    # Receipt-age composition and numeric precision are independent repairs;
    # apply the same propagation entry to isolate association optimization.
    module.IntermittentWorldModel.propagate=IntermittentWorldModel.propagate
    name='eventwm_graph.l2_intermittent._cv_reference'
    cv=types.ModuleType(name);cv.__file__=str(BASE/'eventwm_graph/l2_intermittent/baselines.py');sys.modules[name]=cv
    text=(OUT/'reference_runtime/baselines.py').read_text().replace('from .model import','from ._association_reference import')
    exec(compile(text,cv.__file__,'exec'),cv.__dict__)
    return module.IntermittentWorldModel,cv.MemoryCV


def compare(left,right):
    if len(left.components)!=len(right.components):raise AssertionError('carrier count changed')
    error=0.;exact=True
    for a,b in zip(left.components,right.components):
        for key,v in vars(a).items():
            w=getattr(b,key)
            if isinstance(v,np.ndarray):
                if v.shape!=w.shape:raise AssertionError('shape changed: '+key)
                exact &= np.array_equal(v,w)
                if v.size:error=max(error,float(np.max(abs(v.astype(float)-w.astype(float)))))
            elif isinstance(v,(int,float,bool)):
                error=max(error,abs(float(v)-float(w)));exact &= v==w
            elif v!=w:raise AssertionError('field changed: '+key)
    return error,bool(exact)


def run(limit=24):
    import torch
    torch.set_num_threads(1)
    ref_cls,ref_cv=reference_types()
    new=IntermittentWorldModel.load(OUT/'training/checkpoint.pt');old=ref_cls.load(OUT/'training/checkpoint.pt')
    cv=MemoryCV(.16,law=dict(new.law));oldcv=ref_cv(.16,law=dict(new.law))
    oldcv.model_version=cv.model_version
    assert old.model_version==new.model_version
    sources={name:sha(BASE/'eventwm_graph/l2_intermittent'/name) for name in ('model.py','dynamics.py','neural.py','numerics.py')}
    rows=list(np.load(OLD/'dev.npy',allow_pickle=True))
    # Fixed first metadata roots plus dense/high-history stress, no score choice.
    chosen=list(range(min(limit,len(rows))))
    chosen += sorted(range(len(rows)),key=lambda i:(-max((h.n for h in rows[i]['filtered_hypotheses']),default=0),rows[i]['root_id']))[:8]
    chosen=list(dict.fromkeys(chosen));records=[];errors=[];start=time.perf_counter();bootstrap=load_model()
    for ri in chosen:
        row=rows[ri];folder=SNAP/'roots'/row['root_id'];legal=read(folder/'legal_root.json.gz');_,obs=canonical(folder)
        init=restore(bootstrap,row,legal['partial_graph'])
        for tag,a,b in [('stop_move',new,old),('memory_cv',cv,oldcv)]:
            aa=a.from_legacy(init,seed_for(row['root_id']));bb=b.from_legacy(init,seed_for(row['root_id']))
            for ob in obs[:6]:
                try:
                    av=bv=None;ae=be=None
                    t=time.perf_counter()
                    try:av=a.observe(aa,ob)
                    except Exception as exc:ae=repr(exc)
                    newms=(time.perf_counter()-t)*1000;t=time.perf_counter()
                    try:bv=b.observe(bb,ob)
                    except Exception as exc:be=repr(exc)
                    oldms=(time.perf_counter()-t)*1000
                    if ae is not None or be is not None:
                        if ae!=be:raise AssertionError('reference and optimized errors differ: '+repr((ae,be)))
                        records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],model=tag,time=ob['timestamp'],max_error=0.,bitwise=True,matched_rejection=ae,new_ms=newms,reference_ms=oldms))
                        break
                    aa,bb=av,bv
                    err,exact=compare(aa,bb)
                    da=aa.diagnostics;db=bb.diagnostics
                    assoc_error=max((abs(x[1]-y[1]) for x,y in zip(da.get('association_assignments',[]),db.get('association_assignments',[]))),default=0.)
                    records.append(dict(root_id=row['root_id'],scene_id=row['scene_id'],model=tag,time=ob['timestamp'],max_error=err,bitwise=exact,association_mass_error=assoc_error,new_ms=newms,reference_ms=oldms))
                    if err>1e-10 or assoc_error>1e-10:raise AssertionError('law changed')
                except Exception as exc:
                    errors.append(dict(root_id=row['root_id'],model=tag,error=repr(exc)));break
    result=dict(records=records,errors=errors,passed=not errors,max_error=max((r['max_error'] for r in records),default=0.),
        all_bitwise=all(r['bitwise'] for r in records),seconds=time.perf_counter()-start,
        reference_sha256=sha(OUT/'reference_runtime/model.py'),optimized_sha256=sha(BASE/'eventwm_graph/l2_intermittent/model.py'),
        sources=sources,source_stable=all(sha(BASE/'eventwm_graph/l2_intermittent'/name)==value for name,value in sources.items()),
        semantics='same version/RNG keys, receipt composition and float64 evolution in both arms; validates association/likelihood/beam/miss calculations only')
    path=OUT/'association_reference.json'
    if path.exists():raise ValueError('preserve prior comparisons')
    dump(path,result);print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=24);run(p.parse_args().limit)
