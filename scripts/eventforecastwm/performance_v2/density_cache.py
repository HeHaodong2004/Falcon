"""Verified, bounded-chunk recovery for frozen probability-fitting inputs."""
from pathlib import Path
import torch

from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.agent_scale_calibration import pack_supported
from .occupancy_prior import atomic_save


def materialize(directory,identity,ids,build,guard,chunk_size=64):
    """Preserve root/window order and tensor values across interruption.

    ``build`` returns already supported rows and their root identities. No
    observations or targets are changed here. A chunk without its receipt was
    never committed and is recomputed; a committed chunk must pass its hash.
    """
    directory=Path(directory);parts=directory/'density_parts';parts.mkdir(exist_ok=True)
    cached=[];root_ids=[]
    for start in range(0,len(ids),chunk_size):
        guard.check();part=parts/f'{start:06d}.pt';receipt=part.with_suffix('.json')
        selected=ids[start:start+chunk_size]
        if receipt.exists():
            if not part.exists() or sha256(part)!=read(receipt)['sha256']:
                raise ValueError('Committed density chunk bytes changed')
            saved=torch.load(part,map_location='cpu')
            if saved['identity']!=identity or saved['fit_root_ids']!=selected:
                raise ValueError('Density chunk identity changed')
            tensors,rows=saved['tensors'],saved['root_ids']
        else:
            batches,rows=build(selected)
            tensors=pack_supported(batches) if rows else None
            if tensors is not None and len(rows)!=len(tensors[0]):
                raise ValueError('Density rows and root identities disagree')
            atomic_save(dict(identity=identity,fit_root_ids=selected,tensors=tensors,root_ids=rows),part)
            save_json(receipt,dict(sha256=sha256(part),roots=len(selected),supported_windows=len(rows)))
        if set(rows)-set(selected):raise ValueError('Density chunk has an unexpected root')
        if tensors is not None:cached.append(tensors)
        root_ids.extend(rows)
        save_json(directory/'progress.json',dict(status='building_fixed_density_cache',
            roots=min(start+chunk_size,len(ids)),total_roots=len(ids),supported_windows=len(root_ids)))
    return pack_supported(cached),root_ids
