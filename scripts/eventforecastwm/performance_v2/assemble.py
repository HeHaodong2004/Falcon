"""Package frozen physical and anonymous components without further fitting."""
import argparse
from pathlib import Path
import subprocess
import torch
from eventbeliefwm.data import sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model
from eventforecastwm.performance_v2.hybrid import assemble


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--physical',required=True);parser.add_argument('--prior',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--strength',type=float,default=128.,choices=(128.,512.))
    args=parser.parse_args();out=Path(args.output)
    if out.exists() or out.with_suffix('.json').exists():raise FileExistsError('Assembled artifacts are immutable')
    physical=load_model(args.physical).eval();parent=torch.load(args.physical,map_location='cpu')
    tables=torch.load(args.prior,map_location='cpu')
    if parent['fit_root_ids']!=tables['fit_root_ids']:raise ValueError('Physical and occupancy fitting roots differ')
    model=assemble(physical,tables,args.strength).eval();out.parent.mkdir(parents=True,exist_ok=True)
    identity=dict(physical_checkpoint=str(Path(args.physical).resolve()),physical_sha256=sha256(args.physical),
        anonymous_prior=str(Path(args.prior).resolve()),anonymous_prior_sha256=sha256(args.prior),strength=args.strength,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sha256(__file__),hybrid_sha256=sha256(Path(__file__).with_name('hybrid.py')),
        fitting_performed=False,selection_population='development only; this is not a final release decision')
    torch.save(dict(recipe=model.recipe,model=model.state_dict(),fit_root_ids=parent['fit_root_ids'],
        anonymous_occupancy_prior=tables['prior'],calibration=parent.get('calibration'),
        regime_calibration=parent.get('regime_calibration'),composition=identity,
        physical_source_parent=args.physical,anonymous_distribution='Static-geometry-conditioned marginal cell probabilities; no joint anonymous paths'),out)
    save_json(out.with_suffix('.json'),dict(status='components_assembled',identity=identity,
        checkpoint_sha256=sha256(out),parameter_counts=model.parameter_counts()))


if __name__=='__main__':main()
