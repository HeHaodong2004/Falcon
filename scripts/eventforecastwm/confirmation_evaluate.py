"""Evaluate a precommitted checkpoint panel on fixed reserved memberships.

No calibration, optimization, checkpoint search, or membership selection occurs
here. Partial panels can resume only with identical model and source hashes.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventbeliefwm.metrics import paired_root_interval
from eventforecastwm.benchmark import load_model,extra_metrics,scene_interval
from eventforecastwm.distribution import enable_evaluation_dispatch
from eventforecastwm.train import load_roots,prepare_roots,evaluate
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.benchmark_reuse import CooperativeGuard as Guard
from eventforecastwm.uncertainty import assess as joint_uncertainty


def main():
    p=argparse.ArgumentParser();p.add_argument('--decision',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--membership',required=True);p.add_argument('--prepared-cache',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed confirmation panel is immutable')
    decision=read(a.decision);membership=read(a.membership);manifest=read(a.manifest)
    horizons=tuple(float(h) for h in decision.get('horizons',(1.,3.,5.)))
    if not horizons or len(set(horizons))!=len(horizons) or any(h<=0 or h>5 or abs(round(4*h)-4*h)>1e-7 for h in horizons):
        raise ValueError('Confirmation horizons must be distinct supported canonical ticks through five seconds')
    if (decision['membership_sha256']!=sha256(a.membership) or decision['manifest_sha256']!=sha256(a.manifest) or
        manifest['membership_sha256']!=sha256(a.membership)):raise ValueError('Confirmation membership/decision mismatch')
    ids={r for bank in membership['banks'].values() for r in bank}
    if ids!={r['root_id'] for r in manifest['roots']}:raise ValueError('Incomplete reserved cache')
    for spec in decision['models'].values():
        if sha256(spec['checkpoint'])!=spec['sha256']:raise ValueError('Selected checkpoint changed')
    source=Path(__file__).parent
    identity=dict(decision_sha256=sha256(a.decision),membership_sha256=sha256(a.membership),manifest_sha256=sha256(a.manifest),
        sources={str(p.relative_to(source)):sha256(p) for p in source.rglob('*.py')},selection_frozen=True)
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Exact confirmation recovery identity changed')
    save_json(out/'identity.json',identity);guard=Guard(out);enable_evaluation_dispatch();started=time.time();results={}
    for name,spec in decision['models'].items():
        if (out/'STOP_RESEARCH').exists():raise RuntimeError('Research-only confirmation stop')
        guard.check();torch.manual_seed(20261014);np.random.seed(20261014)
        model=load_model(spec['checkpoint']).eval();checkpoint=torch.load(spec['checkpoint'],map_location='cpu')
        if ids&set(checkpoint.get('fit_root_ids',[])):raise ValueError('Reserved identities entered calibration')
        prior=checkpoint.get('anonymous_occupancy_prior')
        if prior is None or len(prior)<20:raise ValueError('Training-only calibrated occupancy prior required')
        for bank,rids in membership['banks'].items():
            dest=out/name/(bank+'.json')
            if dest.exists():
                result=read(dest)
                if result['panel_identity_sha256']!=sha256(out/'identity.json'):raise ValueError('Partial result identity mismatch')
            else:
                roots=load_roots(a.manifest,rids,guard)
                if spec.get('geometry_manifest'):
                    from eventforecastwm.geometry_data import attach_geometry
                    if sha256(spec['geometry_manifest'])!=spec['geometry_manifest_sha256']:raise ValueError('Geometry cache changed')
                    attach_geometry(roots,spec['geometry_manifest'],guard)
                cache=Path(a.prepared_cache)/('motion_'+format(model.core.config.motion_window_s,'.6g').replace('.','p'))
                prepare(model,roots,a.manifest,cache,prepare_roots,guard)
                guard.check();scores=evaluate(model,roots,'cpu',horizons=horizons,draws=48)
                extras=extra_metrics(model,roots,prior,'cpu',horizons=horizons,guard=guard)
                joint=joint_uncertainty(model,roots,draws=64,guard=guard);joint['confirmation']=True
                intervals={h:{k:dict(root=paired_root_interval([r[k] for r in v['records'] if k in r]),scene=scene_interval(v['records'],k))
                    for k in ('energy','nll','mean_ade','mean_fde')} for h,v in scores.items()}
                result=dict(schema_version=3,protocol_sha256=sha256(a.membership),manifest_sha256=sha256(a.manifest),
                    checkpoint=spec['checkpoint'],checkpoint_sha256=spec['sha256'],recipe=model.recipe,
                    selection_data='reserved confirmation; panel frozen before any reserved forecast scores',
                    horizon5_status='extrapolation beyond3s training; complete supported targets only',
                    scores=scores,extras=extras,intervals=intervals,joint_uncertainty=joint,calibration=checkpoint.get('calibration'),
                    panel_identity_sha256=sha256(out/'identity.json'),bank=bank)
                save_json(dest,result);del roots
            results.setdefault(name,{})[bank]=dict(path=str(dest),sha256=sha256(dest),
                horizons={h:dict(roots=v['trajectory_roots'],physical_roots=v['physical_roots'],metrics=v['metrics']) for h,v in result['scores'].items()})
            save_json(out/'progress.json',dict(completed_panels=sum(len(v) for v in results.values()),
                target_panels=len(decision['models'])*len(membership['banks']),seconds=time.time()-started))
        del model,checkpoint
    save_json(out/'result.json',dict(models=results,identity_sha256=sha256(out/'identity.json'),seconds=time.time()-started))
    print(json.dumps(dict(output=str(out/'result.json'),models=list(results))))


if __name__=='__main__':main()
