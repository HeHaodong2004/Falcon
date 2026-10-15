"""Compact legal static-navmesh features, independent of all dynamic truth.

The mesh is already eroded for a particular navigation body. These features
are geometry covariates and a walkability proxy, not exact human contact labels.
Only legal ego frame/height and the known static mesh enter this adapter.
"""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import habitat_sim
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.guard import Guard


@lru_cache(maxsize=8)
def pathfinder(path):
    pf=habitat_sim.PathFinder()
    if not pf.load_nav_mesh(path):raise ValueError('Static navmesh failed to load: '+path)
    return pf


def extract_frame(mesh,origin,yaw):
    pf=pathfinder(str(mesh));axis=np.linspace(-8+.25,8-.25,32)
    x,z=np.meshgrid(axis,axis);local=np.stack([x,z],-1)
    rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    world=local@rotation+np.asarray(origin)[[0,2]]
    features=np.zeros((2,32,32),np.float32)
    for j in range(32):
        for i in range(32):
            p=np.array([world[j,i,0],origin[1],world[j,i,1]],np.float32)
            walkable=pf.is_navigable(p,.5)
            features[1,j,i]=float(walkable)
            if walkable:features[0,j,i]=min(2.,pf.distance_to_closest_obstacle(p,2.))/2.
    return torch.from_numpy(features)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();manifest=Path(a.manifest);out=Path(a.output)
    if (out/'manifest.json').exists():raise FileExistsError('Completed geometry cache is immutable')
    out.mkdir(parents=True,exist_ok=True);guard=Guard(out)
    inventory=read('/tmp/eventwm_v4_300k/scene_inventory.json')
    scenes={s['physical_scene_id']:s['scene']['navmesh'] for s in inventory['scenes'] if s.get('compatible')}
    source=read(manifest);rows=source['roots'][:a.limit or None];entries=[];started=time.time()
    for i,row in enumerate(rows):
        guard.check()
        if (out/'STOP_RESEARCH').exists():raise RuntimeError('Research-only geometry stop requested')
        source_path=manifest.parent/row['file']
        if sha256(source_path)!=row['sha256']:raise ValueError('Physical cache payload changed')
        root=torch.load(source_path,map_location='cpu')
        mesh=Path(scenes[root['scene']]) if root['scene'] in scenes else None
        supported=mesh is not None and mesh.is_file()
        dest=out/(row['root_id']+'.pt')
        config=dict(root_sha256=row['sha256'],mesh=str(mesh) if mesh else None,
            mesh_sha256=sha256(mesh) if supported else None,extractor_sha256=sha256(__file__),supported=supported)
        if dest.exists():
            value=torch.load(dest,map_location='cpu')
            if value['config']!=config:raise ValueError('Geometry cache changed')
        else:
            grids=[extract_frame(mesh,w['metadata']['origin'],w['metadata']['yaw']) if supported
                   else torch.zeros(2,32,32) for w in root['windows']]
            value=dict(config=config,features=torch.stack(grids),
                support=torch.full((len(grids),32,32),supported,dtype=torch.bool))
            torch.save(value,dest.with_suffix('.tmp'));os.replace(dest.with_suffix('.tmp'),dest)
        entries.append(dict(root_id=row['root_id'],file=dest.name,sha256=sha256(dest),**config))
        save_json(out/'progress.json',dict(roots=i+1,total=len(rows),seconds=time.time()-started))
        if i%32==0:print(json.dumps(dict(roots=i+1,total=len(rows),seconds=time.time()-started)),flush=True)
    save_json(out/'manifest.json',dict(roots=entries,source_manifest_sha256=sha256(manifest),
        supported_roots=sum(r['supported'] for r in entries),unsupported_roots=sum(not r['supported'] for r in entries),
        semantics='known eroded-static-navmesh clearance/2 and walkability; no dynamic truth'))


if __name__=='__main__':main()
