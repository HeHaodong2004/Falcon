"""Training-only likelihood fit of persistent per-agent variance probabilities.

All physical means and Gaussian covariance parameters stay frozen. Exact
component likelihoods are cached once; this fits 7 global probabilities and,
optionally, 63 legal-context parameters, using the same root-origin support as
the scene-scale control. Missing agent targets are analytically marginalized.
"""
import argparse
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from eventbeliefwm.data import sha256
from eventforecastwm import benchmark,benchmark_reuse
from eventforecastwm.model_agent_scale import AgentScaleWorldModel,component_log_densities
from eventforecastwm.train import prepared_batch,encode
from eventforecastwm.observations import protect_observe


def pack_supported(cached):
    """Remove analytically marginalized agent slots from loss computation only.

    Legal contexts were already computed with every legal track. This does not
    change inference inputs or turn an unsupported identity into a negative.
    """
    rows=[]
    for density,count,context,weights in cached:
        for d,c,x,w in zip(density,count,context,weights):
            keep=c>0
            if keep.any():rows.append((d[:,keep],c[keep],x[keep],w))
    if not rows:raise ValueError('No supported calibration roots')
    maximum=max(len(c) for d,c,x,w in rows)
    density=torch.stack([F.pad(d,(0,0,0,maximum-d.shape[1])) for d,c,x,w in rows])
    counts=torch.stack([F.pad(c,(0,maximum-len(c))) for d,c,x,w in rows])
    context=torch.stack([F.pad(x,(0,0,0,maximum-len(x))) for d,c,x,w in rows])
    weights=torch.stack([w for d,c,x,w in rows])
    return density,counts,context,weights


def profile(model,roots,device,guard=None,objective_kind='nll'):
    if objective_kind!='nll':raise ValueError('Discrete probabilities require exact NLL gradients')
    cached=[]
    with torch.no_grad():
        for start in range(0,len(roots),32):
            if guard:guard.check()
            batch=prepared_batch([r['windows'][0] for r in roots[start:start+32]],device)
            f=model.imagine(encode(model,batch),horizon_s=3.)
            raw=batch['targets']['track_valid'][:,:12];mask=raw.all(1)[:,None].expand_as(raw)
            density,count=component_log_densities(f,batch['targets']['track_positions'][:,:12],mask)
            keep=count.sum(-1)>0
            context=f.legal_context if hasattr(f,'legal_context') else density.new_zeros(density.shape[0],density.shape[2],8)
            cached.append((density[keep],count[keep],context[keep],f.weights[keep]))
    density,counts,context,weights=pack_supported(cached)
    total=counts.sum(-1);valid=counts>0
    if not len(total):raise ValueError('No supported calibration roots')
    for p in model.parameters():p.requires_grad_(False)
    model.scale_logits.requires_grad_(True);parameters=[model.scale_logits]
    conditional=getattr(model,'conditional',False);agent_model=isinstance(model,AgentScaleWorldModel)
    if conditional:
        for p in model.scale_context.parameters():p.requires_grad_(True);parameters.append(p)
    optimizer=torch.optim.Adam(parameters,lr=.03);curve=[];steps=512
    for step in range(steps):
        if guard:guard.check()
        if agent_model:
            logp=model.probabilities(context,weights.shape[1]).clamp_min(1e-12).log()
            agent=(density+logp).logsumexp(-1)*valid[:,None]
            component=agent.sum(-1)
        else:
            component=(density.sum(2)+model.scale_logits.log_softmax(-1)).logsumexp(-1)
        nll=-(component+weights.clamp_min(1e-12).log()).logsumexp(-1).div(total).mean()
        penalty=.001*sum(p.square().sum() for p in model.scale_context.parameters()) if conditional else nll*0.
        objective=nll+penalty;optimizer.zero_grad();objective.backward();optimizer.step()
        if step%32==0:curve.append(dict(step=step,nll=float(nll.detach()),penalty=float(penalty.detach())))
    return dict(steps=steps,roots=len(total),parameters=sum(p.numel() for p in parameters),conditional=conditional,
        objective='Exact training-only root-mean normalized joint NLL + .001 squared context-head norm',
        physical_means_and_global_noise_frozen=True,curve=curve,max_supported_agents=int(counts.shape[1]),
        factorization=('shared physical scene mode; independent persistent per-agent variance categories' if agent_model else 'single persistent scene variance category'),
        features='per-agent legal age, lost flags, velocity validity, speed, mode velocity spread, nearest legal belief distance, legal track count')


def main():
    p=argparse.ArgumentParser(add_help=False);group=p.add_mutually_exclusive_group()
    group.add_argument('--unconditional',action='store_true');group.add_argument('--scene-control',action='store_true')
    a,remaining=p.parse_known_args();original=benchmark.load_model
    def convert(checkpoint,*args,**kwargs):
        model=original(checkpoint,*args,**kwargs);recipe=dict(model.recipe)
        if recipe.pop('variant',None)!='finite_scene_scale_v1':raise ValueError('Use a calibrated finite scene-scale parent')
        if a.scene_control:return model
        converted=AgentScaleWorldModel(**recipe,conditional=not a.unconditional)
        mismatch=converted.load_state_dict(model.state_dict(),strict=False)
        if set(mismatch.missing_keys)!={'scale_context.weight','scale_context.bias'} or mismatch.unexpected_keys:
            raise ValueError('Physical parent mismatch')
        return protect_observe(converted)
    benchmark.load_model=convert;benchmark.profile_integrated=profile
    benchmark.EXPERIMENT_SOURCES.update({name:sha256(Path(__file__).parent/name)
        for name in ('model_agent_scale.py','agent_scale_calibration.py')})
    sys.argv=[sys.argv[0],*remaining];benchmark_reuse.main()


if __name__=='__main__':main()
