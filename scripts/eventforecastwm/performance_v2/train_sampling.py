"""Importance-corrected sampling of actual older-observation windows.

Roots remain uniformly shuffled. An existing window containing a legal track
at least one second old receives four sampling slots; other windows receive
one. No histories are edited. Inverse selection probabilities correct the
objective to the original uniform-window distribution within each root.
"""
import argparse
from pathlib import Path

import torch

from eventbeliefwm.data import read, sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm import train
from eventforecastwm.prepared_cache import prepare
from eventforecastwm.model_posterior import PosteriorForecastWorldModel


def sampling_slots(windows):
    multiplicities=[]
    for window in windows:
        prepared=window['_prepared']
        older=prepared.valid & (-prepared.last_observed_times >= 1.)
        multiplicities.append(4 if older.any() else 1)
    total=sum(multiplicities); count=len(windows); slots=[]
    for window,multiplicity in zip(windows,multiplicities):
        value=dict(window, _sampling_importance=total/(count*multiplicity))
        slots.extend([value]*multiplicity)
    return slots,multiplicities


def weighted_loss(model,batch,objective='distribution',nll_weight=.05,horizon=3.):
    if objective!='distribution':raise ValueError('Sampling control requires the proper distribution objective')
    forecast=model.imagine(train.encode(model,batch),horizon_s=horizon);length=forecast.positions.shape[2]
    truth=batch['targets']['track_positions'][:,:length];mask=batch['targets']['track_valid'][:,:length]
    es,supported=train.energy_score(forecast,truth,mask,draws=max(1,6//forecast.weights.shape[1]))
    lp,count=train.trajectory_log_probability(forecast,truth,mask)
    importance=batch['sampling_importance']
    # Self-normalization matches the original valid-root conditional mean;
    # its finite-minibatch ratio bias is explicit in the experiment receipt.
    mean=lambda values,valid:(values*importance*valid).sum()/(importance*valid).sum().clamp_min(1e-12)
    energy=mean(es,supported);nll=mean(-lp/count.clamp_min(1),count>0);value=energy+nll_weight*nll
    return value,dict(energy=float(energy.detach()),nll=float(nll.detach()),loss=float(value.detach()),
        supported_roots=int(supported.sum()),importance_min=float(importance.min()),importance_max=float(importance.max()),
        effective_modes=float((-(forecast.weights*forecast.weights.clamp_min(1e-12).log()).sum(1)).exp().mean().detach()))


def install(cfg):
    protocol=read(cfg['protocol']);fitting=set(protocol['subsets'][str(cfg['scale'])])
    original_prepare=train.prepare_roots;original_batch=train.prepared_batch;original_fingerprints=train.fingerprints
    def cached_prepare(model,roots,guard=None):
        prepare(model,roots,cfg['manifest'],cfg['prepared_cache'],original_prepare,guard)
        receipts=[]
        for root in roots:
            if root['root_id'] not in fitting:continue
            count=len(root['windows']);root['windows'],multiplicities=sampling_slots(root['windows'])
            receipts.append(dict(root_id=root['root_id'],actual_windows=count,sampling_slots=len(root['windows']),
                                 multiplicities=multiplicities))
        save_json(Path(cfg['run'])/'sampling.json',dict(roots=receipts,
            actual_windows=sum(r['actual_windows'] for r in receipts),
            weighted_slots=sum(r['sampling_slots'] for r in receipts),
            criterion='Any ever-seen legal track age >=1s; four copies versus one; no target-dependent selection',
            correction='Inverse q/p weights, self-normalized over supported roots in each minibatch',
            finite_batch_ratio_bias=True, data_usage_fit_windows_means_sampling_slots=True))
    def prepared_batch(windows,device):
        result=original_batch(windows,device)
        result['sampling_importance']=torch.tensor([w.get('_sampling_importance',1.) for w in windows],device=device)
        return result
    def fingerprints():
        result=original_fingerprints();result['performance_v2/train_sampling.py']=sha256(__file__)
        for name in ('prepared_cache.py','model_posterior.py'):
            result[name]=sha256(Path(train.__file__).parent/name)
        return result
    train.prepare_roots=cached_prepare;train.prepared_batch=prepared_batch;train.fingerprints=fingerprints
    train.ForecastWorldModel=PosteriorForecastWorldModel;train.loss=weighted_loss


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    install(read(args.config));train.train(args.config)


if __name__=='__main__':main()
