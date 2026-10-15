"""Package a fitted legal-context regime gate with its exact physical parent."""
import argparse
from pathlib import Path
import subprocess
import torch

from eventbeliefwm.data import sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model
from .regime import RegimeWorldModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',required=True);p.add_argument('--gate',required=True)
    p.add_argument('--output',required=True);p.add_argument('--stop-weight',type=float,default=.5,choices=(.5,1.))
    a=p.parse_args();out=Path(a.output)
    if out.exists() or out.with_suffix('.json').exists():raise FileExistsError('Immutable composition already exists')
    parent=torch.load(a.physical,map_location='cpu');gate=torch.load(a.gate,map_location='cpu')
    if gate['identity']['checkpoint_sha256']!=sha256(a.physical):raise ValueError('Gate was fitted to another distribution')
    if not set(gate['fit_root_ids'])<=set(parent['fit_root_ids']):raise ValueError('Gate fitting expanded the physical membership')
    physical=load_model(a.physical).eval();model=RegimeWorldModel(physical.recipe,stop_weight=a.stop_weight)
    model.physical.load_state_dict(physical.state_dict());model.stop_head.load_state_dict(gate['head']);model.eval()
    identity=dict(physical=str(Path(a.physical).resolve()),physical_sha256=sha256(a.physical),
        gate=str(Path(a.gate).resolve()),gate_sha256=sha256(a.gate),stop_weight=a.stop_weight,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sha256(__file__),regime_sha256=sha256(Path(__file__).with_name('regime.py')),
        new_optimizer_steps=0,final_release=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(recipe=model.recipe,model=model.state_dict(),fit_root_ids=parent['fit_root_ids'],
        anonymous_occupancy_prior=parent['anonymous_occupancy_prior'],calibration=parent.get('calibration'),
        regime_calibration=gate['receipt'],composition=identity),out)
    save_json(out.with_suffix('.json'),dict(status='regime_components_assembled',identity=identity,
        checkpoint_sha256=sha256(out),parameter_counts=model.parameter_counts()))


if __name__=='__main__':main()
