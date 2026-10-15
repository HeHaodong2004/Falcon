"""Compare historical C256/C4096 weights using matched legal inputs/targets."""
import argparse
import json
from pathlib import Path
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model
from eventforecastwm.train import load_roots,prepare_roots,evaluate
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.guard import Guard


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--old-run',required=True);p.add_argument('--old-cache',required=True)
    p.add_argument('--prepared-cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);guard=Guard(out.parent);protocol=read(a.protocol)
    olddir=Path(a.old_cache);oldrows={r['root_id']:r for r in read(olddir/'manifest.json')['roots']}
    groups={name:load_roots(a.manifest,ids,guard) for name,ids in
        [('original_training',protocol['previous_pilot_roots']),('new_development',protocol['development'])]}
    mismatches=[];human_mismatches=[];robot_difference_channels=set()
    for root in groups['original_training']:
        guard.check();r=oldrows[root['root_id']];f=olddir/r['file']
        if sha256(f)!=r['sha256']:raise ValueError('Historical cache hash mismatch')
        old=torch.load(f,map_location='cpu');new=root['windows'][0]
        human_equal=torch.equal(old['base'].humans,new['base'].humans)
        if not human_equal:human_mismatches.append(root['root_id'])
        if old['base'].robot.shape!=new['base'].robot.shape:raise ValueError('Historical ego history shape mismatch')
        delta=(old['base'].robot!=new['base'].robot).any(0)
        robot_difference_channels.update(delta.nonzero().flatten().tolist())
        equal=human_equal and not delta.any()
        equal=equal and torch.equal(old['geometry'],new['geometry']) and torch.equal(old['geometry_support'],new['geometry_support'])
        if not equal:mismatches.append(root['root_id'])
    models={name:load_model(Path(a.old_run)/filename).eval() for name,filename in [('C256','best.pt'),('C4096','last.pt')]}
    all_roots=sum(groups.values(),[]);prepare(next(iter(models.values())),all_roots,a.manifest,a.prepared_cache,prepare_roots,guard)
    results={}
    for name,model in models.items():
        results[name]={}
        for group,roots in groups.items():
            guard.check();results[name][group]=evaluate(model,roots,'cpu',horizons=(1.,3.),draws=48)
    result=dict(results=results,original_training_input_mismatches=mismatches,
        original_training_human_input_mismatches=human_mismatches,
        robot_difference_channels=sorted(robot_difference_channels),
        ego_difference_semantics='Channel10 is the previous angular command; the exogenous physical encoder never reads it. All human features and timestamp channels are checked separately.',
        checkpoint_hashes={name:sha256(Path(a.old_run)/filename) for name,filename in [('C256','best.pt'),('C4096','last.pt')]},
        protocol_sha256=sha256(a.protocol),manifest_sha256=sha256(a.manifest),source_sha256=sha256(__file__),
        limitations='Checkpoint comparison, not a randomized loss/LR ablation. Historical fit roots and new dev roots have different distributions. Same current support masks and calibrated-time inputs used for both checkpoints. No new uncertainty calibration.')
    save_json(out,result)
    print(json.dumps({name:{g:{h:s['metrics'] for h,s in scores.items()} for g,scores in groups.items()} for name,groups in results.items()}))


if __name__=='__main__':main()
