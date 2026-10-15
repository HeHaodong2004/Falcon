"""Self-contained model configuration, optimizer and RNG resume state."""
from dataclasses import asdict
import random
import numpy as np
import torch
from .model import FactorizedEventWM, V2Config
from .data import torch_load
import json
from pathlib import Path


def resolve_checkpoint(path):
    path = Path(path)
    if path.suffix == '.json':
        selection = json.loads(path.read_text())
        resolved = (path.parent/selection['checkpoint']).resolve()
        if path.parent.resolve() not in resolved.parents:
            raise ValueError('Checkpoint selection must resolve within its run directory')
        return resolved
    return path


def save_checkpoint(path, model, optimizer, epoch, step, generator, provenance, training_state=None):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(version='eventwm-v2-checkpoint-v1', config=asdict(model.config), model=model.state_dict(),
        optimizer=optimizer.state_dict(), epoch=epoch, step=step, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
        numpy_rng=np.random.get_state(), python_rng=random.getstate(), shuffle_rng=generator.get_state(),
        provenance=provenance, training_state=training_state), temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch_load(resolve_checkpoint(path))
    if checkpoint.get('version') != 'eventwm-v2-checkpoint-v1':
        raise ValueError('Expected a V2 checkpoint')
    model = FactorizedEventWM(V2Config(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, checkpoint


def restore_rng(checkpoint, generator):
    torch.set_rng_state(checkpoint['torch_rng'])
    if checkpoint['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    np.random.set_state(checkpoint['numpy_rng'])
    random.setstate(checkpoint['python_rng'])
    generator.set_state(checkpoint['shuffle_rng'])
