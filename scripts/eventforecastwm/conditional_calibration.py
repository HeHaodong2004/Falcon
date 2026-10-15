"""Fit 63 conditional variance-probability parameters on training roots only.

The physical mean, global noise law, and marginal variance categories stay
frozen. Exact cached component likelihoods make this bounded experiment cheap.
It uses the same complete root-origin support as other noise calibrations.
"""
from pathlib import Path
import torch
from eventbeliefwm.data import sha256
from eventforecastwm import benchmark,benchmark_cached
from eventforecastwm.model_conditional_scale import ConditionalScaleWorldModel,component_log_densities
from eventforecastwm.train import prepared_batch,encode
from eventforecastwm.observations import protect_observe


def profile(model,roots,device,guard=None,objective_kind='nll'):
    if objective_kind!='nll':raise ValueError('Discrete probabilities require exact NLL gradients')
    features=[];terms=[];counts=[]
    with torch.no_grad():
        for start in range(0,len(roots),32):
            if guard:guard.check()
            batch=prepared_batch([r['windows'][0] for r in roots[start:start+32]],device)
            f=model.imagine(encode(model,batch),horizon_s=3.)
            raw=batch['targets']['track_valid'][:,:12];mask=raw.all(1)[:,None].expand_as(raw)
            density,count=component_log_densities(f,batch['targets']['track_positions'][:,:12],mask);keep=count>0
            features.append(f.legal_context[keep]);counts.append(count[keep])
            terms.append((density+f.weights.clamp_min(1e-12).log()[...,None])[keep])
    features=torch.cat(features);terms=torch.cat(terms);counts=torch.cat(counts)
    if not len(counts):raise ValueError('No supported calibration roots')
    for p in model.parameters():p.requires_grad_(False)
    for p in model.scale_context.parameters():p.requires_grad_(True)
    optimizer=torch.optim.Adam(model.scale_context.parameters(),lr=.03)
    curve=[];steps=512
    for step in range(steps):
        if guard:guard.check()
        logp=(model.scale_logits+model.scale_context(features)).log_softmax(-1)
        nll=-(terms+logp[:,None]).flatten(1).logsumexp(1).div(counts).mean()
        penalty=.001*sum(p.square().sum() for p in model.scale_context.parameters())
        objective=nll+penalty;optimizer.zero_grad();objective.backward();optimizer.step()
        if step%32==0:curve.append(dict(step=step,nll=float(nll.detach()),penalty=float(penalty.detach())))
    return dict(steps=steps,roots=len(counts),parameters=sum(p.numel() for p in model.scale_context.parameters()),
        objective='Exact training-only root-mean normalized joint NLL + .001 squared conditional-head norm',
        physical_means_and_global_noise_frozen=True,curve=curve,
        features='mean/max observation age, lost fraction, velocity-valid fraction, mean/variance speed, log legal track count, long-lost fraction')


def main():
    original=benchmark.load_model
    def convert(checkpoint,*args,**kwargs):
        model=original(checkpoint,*args,**kwargs);recipe=dict(model.recipe)
        if recipe.pop('variant',None)!='finite_scene_scale_v1':raise ValueError('Use a calibrated finite scene-scale parent')
        converted=ConditionalScaleWorldModel(**recipe)
        mismatch=converted.load_state_dict(model.state_dict(),strict=False)
        if set(mismatch.missing_keys)!={'scale_context.weight','scale_context.bias'} or mismatch.unexpected_keys:
            raise ValueError('Physical parent mismatch')
        return protect_observe(converted)
    benchmark.load_model=convert;benchmark.profile_integrated=profile
    benchmark.EXPERIMENT_SOURCES.update({name:sha256(Path(__file__).parent/name)
        for name in ('model_conditional_scale.py','conditional_calibration.py')})
    benchmark_cached.main()


if __name__=='__main__':main()
