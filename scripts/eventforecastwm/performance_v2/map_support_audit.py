"""Target-only reference for interpreting coarse eroded-navmesh diagnostics.

No forecasts, fitting or selection occur here. A nonwalkable raster cell is not
an exact human collision label; target trajectories establish its reference rate.
"""
import argparse
import json
from pathlib import Path
import time
import torch

from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import scene_interval
from eventforecastwm.benchmark_reuse import CooperativeGuard
from eventforecastwm.data import batch_windows
from eventforecastwm.geometry_data import attach_geometry
from eventforecastwm.train import load_roots
from eventforecastwm.window_evaluate import aggregate
from .physics_diagnostics import map_values


def run(cfg):
    out=Path(cfg['run']);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed audit is immutable')
    start=time.time();guard=CooperativeGuard(out);torch.set_num_threads(1)
    roots=load_roots(cfg['manifest'],read(cfg['protocol'])['development'],guard)
    attach_geometry(roots,cfg['geometry_manifest'],guard)
    windows=[(r,j,w) for r in roots for j,w in enumerate(r['windows'])];rows=[]
    for offset in range(0,len(windows),32):
        guard.check();selected=windows[offset:offset+32];batch=batch_windows([w for _,_,w in selected])
        for horizon in (1.,2.,3.):
            length=round(4*horizon);exposure=batch['targets']['physical_exposure'][:,:length].all(1)
            for population,prefix in [('tracked','track'),('all_humans','all_human')]:
                points=batch['targets'][prefix+'_positions'][:,:length]
                complete=batch['targets'][prefix+'_valid'][:,:length].all(1)&exposure[:,None]
                inside,known,walk,_=map_values(points,batch['geometry'],batch['geometry_support'])
                for i,(root,j,_) in enumerate(selected):
                    valid=complete[i][None].expand(length,-1)
                    if not valid.any():continue
                    support=valid&known[i]
                    row=dict(root_id=root['root_id'],scene=root['scene'],window=j,horizon=horizon,population=population,
                        agents=int(complete[i].sum()),known_map_fraction=float(known[i][valid].float().mean()),
                        outside_grid=float((~inside[i][valid]).float().mean()))
                    if support.any():row['nonwalkable_navmesh_proxy']=float((~walk[i][support]).float().mean())
                    rows.append(row)
    keys=('known_map_fraction','outside_grid','nonwalkable_navmesh_proxy');summary={}
    for windows_population in ('origin','all_windows'):
        summary[windows_population]={}
        for population in ('tracked','all_humans'):
            summary[windows_population][population]={}
            for horizon in (1.,2.,3.):
                guard.check();selected=[r for r in rows if r['population']==population and r['horizon']==horizon
                    and (windows_population=='all_windows' or r['window']==0)]
                values=aggregate(selected,keys)
                summary[windows_population][population][str(horizon)]={key:scene_interval(values,key) for key in keys}
    result=dict(status='target_map_reference_complete',runtime_s=time.time()-start,config=cfg,
        source_sha256=sha256(__file__),map_lookup_sha256=sha256(Path(__file__).with_name('physics_diagnostics.py')),
        identities={k+'_sha256':sha256(cfg[k]) for k in ('manifest','protocol','geometry_manifest')},
        records=rows,summary=summary,confirmation=False,
        semantics='Complete target trajectories under complete native scene exposure. Tick/agent means inside window, windows inside root. Coarse eroded-navmesh proxy at legal robot height, not human contacts. Forecast tail diagnostics include unsupported legal slots and thus have a different population; do not interpret their difference from this target reference as a paired violation excess.')
    save_json(out/'result.json',result);print(json.dumps(dict(status=result['status'],runtime_s=result['runtime_s'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);run(read(parser.parse_args().config))
