"""Exact trajectory-marginal occupancy auxiliary loss on supported identities.

The auxiliary population marginalizes to fully supervised tracks. It does not
call unobserved/unsupported people empty and does not alter inference support.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from eventbeliefwm.data import read,sha256
from eventbeliefwm.metrics import mean_valid
from eventforecastwm import train
from eventforecastwm.model import energy_score,trajectory_log_probability
from eventforecastwm.performance_v2.diagnose import point_grid
from eventforecastwm.readouts import tracked_occupancy


def loss(model,batch,objective='distribution',nll_weight=.05,horizon=3.,occupancy_weight=2.):
    if objective!='distribution':raise ValueError('Controlled occupancy loss requires the mixed proper objective')
    forecast=model.imagine(train.encode(model,batch),horizon_s=horizon)
    length=forecast.positions.shape[2]
    truth=batch['targets']['track_positions'][:,:length];mask=batch['targets']['track_valid'][:,:length]
    es,supported=energy_score(forecast,truth,mask,draws=max(1,6//forecast.weights.shape[1]))
    lp,count=trajectory_log_probability(forecast,truth,mask)
    energy=mean_valid(es,supported);nll=mean_valid(-lp/count.clamp_min(1),count>0)
    complete=mask.all(1)&forecast.valid
    probability=tracked_occupancy(replace(forecast,valid=complete)).clamp(1e-6,1-1e-6)
    target=point_grid(truth,complete[:,None].expand_as(mask))
    occupancy=mean_valid(F.binary_cross_entropy(probability,target,reduction='none').mean((1,2,3)),complete.any(1))
    value=energy+nll_weight*nll+occupancy_weight*occupancy
    return value,dict(energy=float(energy.detach()),nll=float(nll.detach()),occupancy_bce=float(occupancy.detach()),
        occupancy_weight=occupancy_weight,loss=float(value.detach()),supported_roots=int(supported.sum()),
        occupancy_roots=int(complete.any(1).sum()),
        effective_modes=float((-(forecast.weights*forecast.weights.clamp_min(1e-12).log()).sum(1)).exp().mean().detach()))


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--config',required=True)
    args,_=p.parse_known_args();cfg=read(args.config);weight=cfg['occupancy_weight']
    train.loss=lambda *args,**kwargs:loss(*args,**kwargs,occupancy_weight=weight)
    original=train.fingerprints
    def fingerprints():
        result=original();result['performance_v2/train_occupancy.py']=sha256(__file__)
        result['performance_v2/diagnose.py']=sha256(Path(__file__).with_name('diagnose.py'))
        result['readouts.py']=sha256(Path(__file__).parents[1]/'readouts.py')
        return result
    train.fingerprints=fingerprints
    if cfg['model']['variant']=='posterior_v1':
        from eventforecastwm.train_posterior import main as run
    else:
        from eventforecastwm.performance_v2.train import main as run
    run()


if __name__=='__main__':main()
