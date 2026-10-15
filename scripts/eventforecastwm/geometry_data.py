"""Attach hash-verified known-map covariates without changing physical targets."""
from pathlib import Path
import torch
from eventbeliefwm.data import read,sha256


def attach_geometry(roots,manifest,guard=None):
    path=Path(manifest);entries={r['root_id']:r for r in read(path)['roots']}
    for root in roots:
        if guard:guard.check()
        row=entries[root['root_id']];payload=path.parent/row['file']
        if sha256(payload)!=row['sha256']:raise ValueError('Known-map payload changed')
        value=torch.load(payload,map_location='cpu')
        if len(value['features'])!=len(root['windows']):raise ValueError('Map/window identity mismatch')
        for i,w in enumerate(root['windows']):
            w['geometry']=value['features'][i];w['geometry_support']=value['support'][i]
    return roots
