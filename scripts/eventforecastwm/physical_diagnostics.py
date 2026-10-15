"""Matched trajectory support, local-map proxies and joint proximity diagnostics.

The eroded robot-floor navmesh is not a human contact oracle. Out-of-grid points
are explicitly unsupported, and target violations are reported for comparison.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model,scene_interval
from eventforecastwm.geometry_data import attach_geometry
from eventforecastwm.train import load_roots,prepare_roots,prepared_batch,encode
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.sampling import joint_samples
from eventforecastwm.benchmark_reuse import CooperativeGuard as Guard


def diagnostics(paths,valid,geometry,support):
    # paths: samples,time,agents,xy; a sample is a coherent joint scene path.
    paths=paths[:,:,valid];s,t,n,_=paths.shape
    if not n:return {}
    speed=((paths[:,1:]-paths[:,:-1])/.25).norm(dim=-1)
    acceleration=((paths[:,2:]-2*paths[:,1:-1]+paths[:,:-2])/.25**2).norm(dim=-1)
    result=dict(speed_p95=float(torch.quantile(speed.flatten(),.95)),
        speed_gt3=float((speed>3.).float().mean()),
        acceleration_p95=float(torch.quantile(acceleration.flatten(),.95)))
    inside=(paths.abs()<=7.75).all(-1)&bool(support.all())
    result['map_supported_fraction']=float(inside.float().mean())
    if inside.any():
        grid=paths.reshape(1,-1,1,2)/8.
        walk=F.grid_sample(geometry[None,1:2],grid,align_corners=False).reshape(s,t,n)
        result['map_nonwalkable_proxy_fraction']=float((walk[inside]<.5).float().mean())
    if n>1:
        i,j=torch.triu_indices(n,n,offset=1)
        distance=(paths[:,:,i]-paths[:,:,j]).norm(dim=-1)
        result['pair_center_distance_lt04_fraction']=float((distance<.4).float().mean())
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--geometry-manifest',required=True)
    p.add_argument('--prepared-cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--condition-on-map',action='store_true',help='Only for a map-conditioned training recipe');a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);guard=Guard(out.parent)
    model=load_model(a.checkpoint).eval();roots=load_roots(a.manifest,read(a.protocol)['development'],guard)
    attach_geometry(roots,a.geometry_manifest,guard);prepare(model,roots,a.manifest,a.prepared_cache,prepare_roots,guard)
    rows=[];generator=torch.Generator().manual_seed(91324)
    for root in roots:
        guard.check();w=root['windows'][0];batch=prepared_batch([w],'cpu')
        # Only map-trained models may condition on the geometry sidecar.
        # Geometry is still used as an independent diagnostic for every model.
        if not a.condition_on_map:
            batch['geometry']=torch.zeros_like(batch['geometry']);batch['geometry_support']=torch.zeros_like(batch['geometry_support'])
        f=model.imagine(encode(model,batch),horizon_s=3.);valid=batch['targets']['track_valid'][0,:12].all(0)
        if not valid.any():continue
        paths,_=joint_samples(f,16,generator);truth=batch['targets']['track_positions'][0,:12][None]
        mean=(f.positions*f.weights[:,:,None,None,None]).sum(1)
        row=dict(root_id=root['root_id'],scene=root['scene'],agents=int(valid.sum()))
        for name,prediction in [('target',truth),('mean',mean),('sample',paths[0])]:
            for key,value in diagnostics(prediction,valid,w['geometry'],w['geometry_support']).items():row[name+'_'+key]=value
        rows.append(row)
    keys=sorted(set().union(*(r.keys() for r in rows))-{'root_id','scene','agents'})
    result=dict(checkpoint_sha256=sha256(a.checkpoint),manifest_sha256=sha256(a.manifest),
        geometry_manifest_sha256=sha256(a.geometry_manifest),condition_on_map=a.condition_on_map,
        roots=len(rows),draws=16,records=rows,
        metrics={k:float(np.mean([r[k] for r in rows if k in r])) for k in keys},
        metric_roots={k:sum(k in r for r in rows) for k in keys},
        scene_intervals={k:scene_interval(rows,k) for k in keys},source_sha256=sha256(__file__),
        limitations='3s complete identity support; navmesh is eroded and at robot height. Off-grid points are unsupported, not violations. Proximity is a body-center diagnostic, not a native collision label. Sample finite differences include readout noise.')
    save_json(out,result);print(json.dumps(result['metrics']))


if __name__=='__main__':main()
