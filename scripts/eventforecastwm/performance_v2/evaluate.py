"""Fixed-budget development evaluation, without calibration or model fitting."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from eventbeliefwm.data import read, sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.benchmark import load_model, extra_metrics, scene_interval
from eventforecastwm.benchmark_reuse import CooperativeGuard
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.train import load_roots, prepare_roots, evaluate
from eventforecastwm.uncertainty import assess
from eventforecastwm.window_evaluate import aggregate


def run(cfg):
    out=Path(cfg['run']);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed evaluation is immutable')
    guard=CooperativeGuard(out);torch.set_num_threads(1);torch.manual_seed(20261014);np.random.seed(20261014)
    start=time.time();protocol=read(cfg['protocol']);state=torch.load(cfg['checkpoint'],map_location='cpu')
    if set(protocol['development']) & set(state.get('fit_root_ids',[])):raise ValueError('Evaluation leakage')
    identity=dict(config=cfg,checkpoint_sha256=sha256(cfg['checkpoint']),manifest_sha256=sha256(cfg['manifest']),
        protocol_sha256=sha256(cfg['protocol']),source_sha256=sha256(__file__))
    save_json(out/'identity.json',identity)
    model=load_model(cfg['checkpoint']).eval()
    roots=load_roots(cfg['manifest'],protocol['development'],guard)
    if cfg.get('geometry_manifest'):
        from eventforecastwm.geometry_data import attach_geometry
        attach_geometry(roots,cfg['geometry_manifest'],guard)
        identity['geometry_manifest_sha256']=sha256(cfg['geometry_manifest'])
    prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],prepare_roots,guard)
    # Bound cooperative stop / resource check latency without changing RNG order.
    from eventforecastwm import train
    original=train.prepared_batch
    def checked(*args,**kwargs):guard.check();return original(*args,**kwargs)
    train.prepared_batch=checked
    prior=state.get('anonymous_occupancy_prior')
    if prior is None:
        prior_state=torch.load(cfg['prior_checkpoint'],map_location='cpu')
        prior=prior_state['anonymous_occupancy_prior']
        identity['prior_checkpoint_sha256']=sha256(cfg['prior_checkpoint'])
    save_json(out/'identity.json',identity)
    horizons=tuple(cfg.get('horizons',[1.,2.,3.]))
    scores=evaluate(model,roots,'cpu',horizons=horizons,draws=48)
    extras=extra_metrics(model,roots,prior,'cpu',horizons=horizons,guard=guard)
    uncertainty=assess(model,roots,64,guard) if cfg.get('uncertainty',True) else None
    all_windows=None
    if cfg.get('all_windows',True):
        flat=[dict(root_id=r['root_id'],scene=r['scene'],windows=[w],window_index=j)
              for r in roots for j,w in enumerate(r['windows'])]
        window_scores=evaluate(model,flat,'cpu',horizons=(1.,3.),draws=48)
        summary={}
        for h,value in window_scores.items():
            for row,w in zip(value['records'],flat):row['window_index']=w['window_index']
            keys=('energy','nll','mean_ade','mean_fde')
            root_rows=aggregate(value['records'],keys)
            summary[h]=dict(records=root_rows,metrics={key:scene_interval(root_rows,key) for key in keys})
        all_windows=dict(summary=summary,scores=window_scores,estimand='Mean supported windows within root, then equal supported root weights')
    result=dict(status='evaluation_complete',identity=identity,recipe=model.recipe,scores=scores,extras=extras,
        uncertainty=uncertainty,all_windows=all_windows,runtime_s=time.time()-start,
        primary_rng='unchanged v1 benchmark; 48 paths per stratified bank, minADE/FDE K=6, joint density K=64',
        selection_data='training-internal development only; no confirmation data read')
    save_json(out/'result.json',result)
    print(json.dumps(dict(status=result['status'],runtime_s=result['runtime_s'],scores={h:v['metrics'] for h,v in scores.items()},
        occupancy={h:v['occupancy'] for h,v in extras.items()},uncertainty=uncertainty['metrics'] if uncertainty else None)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(read(p.parse_args().config))


if __name__=='__main__':main()
