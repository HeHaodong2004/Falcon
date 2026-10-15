"""Secondary root-balanced evaluation of every supported development window.

The primary checkpoint-selection protocol remains the frozen root-origin bank.
This diagnostic exposes longer histories and later actual-arrival origins. A
root is still one statistical unit, regardless of its number of windows.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model,extra_metrics,scene_interval
from eventforecastwm.train import load_roots,prepare_roots,evaluate
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.benchmark_reuse import CooperativeGuard as Guard


def aggregate(rows,keys):
    by_root=defaultdict(list)
    for row in rows:by_root[row['root_id']].append(row)
    result=[]
    for rid,values in by_root.items():
        scenes={r['scene'] for r in values}
        if len(scenes)!=1:raise ValueError('Root crosses scenes')
        root=dict(root_id=rid,scene=next(iter(scenes)),windows=len(values))
        for key in keys:
            supported=[r[key] for r in values if r.get(key) is not None]
            if supported:root[key]=float(np.mean(supported));root[key+'_windows']=len(supported)
        result.append(root)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--prepared-cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--geometry-manifest');a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);guard=Guard(out.parent)
    model=load_model(a.checkpoint).eval();checkpoint=torch.load(a.checkpoint,map_location='cpu')
    ids=read(a.protocol)['development']
    if set(ids)&set(checkpoint.get('fit_root_ids',[])):raise ValueError('Development entered calibration')
    roots=load_roots(a.manifest,ids,guard)
    if a.geometry_manifest:
        from eventforecastwm.geometry_data import attach_geometry
        attach_geometry(roots,a.geometry_manifest,guard)
    prepare(model,roots,a.manifest,a.prepared_cache,prepare_roots,guard)
    # The historical evaluator has no guard argument. Bound explicit stop and
    # resource-check latency at each batch without changing its numerical RNG.
    from eventforecastwm import train
    original_batch=train.prepared_batch
    def checked_batch(*args,**kwargs):
        guard.check();return original_batch(*args,**kwargs)
    train.prepared_batch=checked_batch
    flat=[dict(root_id=r['root_id'],scene=r['scene'],windows=[w],window_index=i)
          for r in roots for i,w in enumerate(r['windows'])]
    # Every input is the actually available history at that window's origin;
    # future target tensors remain exclusively in the metric implementation.
    with torch.no_grad():
        scores=evaluate(model,flat,'cpu',horizons=(.25,1.,3.,5.),draws=48)
        extras=extra_metrics(model,flat,checkpoint['anonymous_occupancy_prior'],'cpu',horizons=(.25,1.,3.,5.),guard=guard)
    summaries={}
    trajectory_keys=('energy','nll','mean_ade','mean_fde','map_ade','map_fde')
    occupancy_keys=('brier','nll','average_precision','positive_nll','ece10')
    for h,score in scores.items():
        rows=score['records'];extra=extras[h]['records']
        for group in (rows,extra):
            for r,w in zip(group,flat):r['window_index']=w['window_index'];r['origin_time']=w['windows'][0]['metadata']['timestamp']
        root_rows=aggregate(rows,trajectory_keys)
        occ_rows=aggregate([dict(root_id=r['root_id'],scene=r['scene'],**r.get('occupancy',{})) for r in extra],occupancy_keys)
        def summarize(records,keys):
            return {key:dict(roots=sum(key in r for r in records),windows=sum(r.get(key+'_windows',0) for r in records),
                mean=float(np.mean([r[key] for r in records if key in r])) if any(key in r for r in records) else None,
                scene_interval=scene_interval(records,key)) for key in keys}
        summaries[h]=dict(trajectory=summarize(root_rows,trajectory_keys),occupancy=summarize(occ_rows,occupancy_keys),
            root_trajectory_records=root_rows,root_occupancy_records=occ_rows,
            identity_window_agents=sum(r['agents'] for r in rows),physical_windows=sum(r['physical_supported'] for r in rows))
    save_json(out,dict(checkpoint_sha256=sha256(a.checkpoint),manifest_sha256=sha256(a.manifest),protocol_sha256=sha256(a.protocol),
        source_sha256=sha256(__file__),development_roots=len(roots),available_windows=len(flat),horizons=summaries,
        window_scores=scores,window_extras=extras,
        estimand='Each metric averages valid windows inside each root, then gives supported roots equal weight. Window/agent occurrences are correlated, not independent trajectories.',
        selection='Secondary train-internal diagnostic only; does not replace the primary root-origin checkpoint-selection protocol',
        horizon5_status='Only supported complete targets; extrapolation beyond3s training'))
    print({'output':str(out),'roots':len(roots),'windows':len(flat)})


if __name__=='__main__':main()
