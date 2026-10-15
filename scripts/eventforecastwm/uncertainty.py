"""Joint likelihood coverage, mixture utilization and physical diagnostics.

Coverage is a Monte Carlo density-level-set check of complete supported joint
trajectories. It is separate from marginal coordinate interval coverage.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json
from eventforecastwm.distribution import trajectory_log_probability
from eventforecastwm.benchmark import load_model,scene_interval
from eventforecastwm.sampling import joint_samples
from eventforecastwm.train import load_roots,prepare_roots,prepared_batch,encode
from eventforecastwm.benchmark_reuse import CooperativeGuard as Guard


def component_log_probability(f,truth,mask):
    if hasattr(f,'branch_probabilities'):
        from eventforecastwm.performance_v2.regime import component_density
        return component_density(f,truth,mask)[0]
    scores=[]
    for s in range(f.weights.shape[1]):
        values={k:getattr(f,k)[:,s:s+1] for k in ('positions','velocities','covariance','initial_covariance','initial_residual_mean')}
        if hasattr(f,'concentration'):
            values.update(concentration=f.concentration[:,s:s+1],rate=f.rate[:,s:s+1])
        if hasattr(f,'agent_scale_probabilities'):values['agent_scale_probabilities']=f.agent_scale_probabilities[:,s:s+1]
        elif hasattr(f,'scale_values'):values['scale_probabilities']=f.scale_probabilities[:,s:s+1]
        one=replace(f,**values,weights=torch.ones_like(f.weights[:,:1]))
        scores.append(trajectory_log_probability(one,truth,mask)[0])
    return torch.stack(scores,1)


@torch.no_grad()
def assess(model,roots,draws=64,guard=None):
    records=[];generator=torch.Generator().manual_seed(81271)
    for start in range(0,len(roots),8):
        if guard:guard.check()
        selected=roots[start:start+8];batch=prepared_batch([r['windows'][0] for r in selected],'cpu')
        f=model.imagine(encode(model,batch),horizon_s=3.);truth=batch['targets']['track_positions'][:,:12]
        raw=batch['targets']['track_valid'][:,:12];mask=raw.all(1)[:,None].expand_as(raw)
        lp,count=trajectory_log_probability(f,truth,mask)
        component=component_log_probability(f,truth,mask)
        posterior=(component+f.weights.clamp_min(1e-12).log()).softmax(-1)
        paths,modes=joint_samples(f,draws,generator)
        sampled_nll=torch.stack([-trajectory_log_probability(f,paths[:,j],mask)[0] for j in range(draws)],1)
        # Rank uses the held-out observation as one additional exchangeable draw.
        rank=(1+(sampled_nll<(-lp[:,None])).sum(1)).float()/(draws+1)
        pair=f.positions[:,:,None]-f.positions[:,None]
        distances=((pair*mask[:,None,None,:,:,None]).square().sum((3,4,5))/count[:,None,None].clamp_min(1)).sqrt()
        separation=(distances*f.weights[:,:,None]*f.weights[:,None]).sum((1,2))
        speed=((paths[:,:,1:]-paths[:,:,:-1])/.25).norm(dim=-1)
        for i,r in enumerate(selected):
            if not count[i]:continue
            supported=mask[i,0];speeds=speed[i,:,:,supported]
            records.append(dict(root_id=r['root_id'],scene=r['scene'],agents=int(supported.sum()),
                joint_density_rank=float(rank[i]),joint_coverage90=bool(rank[i]<=.9),
                posterior_mode_probabilities=posterior[i].tolist(),prior_mode_probabilities=f.weights[i].tolist(),
                posterior_effective_modes=float((-(posterior[i]*posterior[i].clamp_min(1e-12).log()).sum()).exp()),
                prior_effective_modes=float((-(f.weights[i]*f.weights[i].clamp_min(1e-12).log()).sum()).exp()),
                mode_center_separation_rms_m=float(separation[i]),
                sampled_body_center_speed_gt3_fraction=float((speeds>3).float().mean()),
                sampled_body_center_speed_p95=float(torch.quantile(speeds.flatten(),.95)),
                component_mean_speed_gt3_fraction=float((f.velocities[i,:,:,supported].norm(dim=-1)>3.00001).float().mean())))
    keys=('joint_coverage90','joint_density_rank','posterior_effective_modes','prior_effective_modes',
        'mode_center_separation_rms_m','sampled_body_center_speed_gt3_fraction','sampled_body_center_speed_p95',
        'component_mean_speed_gt3_fraction')
    result=dict(roots=len(records),
        draws=draws,nominal_finite_rank_coverage=float(np.floor(.9*(draws+1))/(draws+1)),
        metrics={k:float(np.mean([r[k] for r in records])) if records else None for k in keys},
        scene_intervals={k:scene_interval(records,k) for k in keys},records=records,
        coverage_definition='Randomized-model trajectory density-level-set rank; finite MC resolution1/(draws+1)',
        speed_definition='Finite differences of sampled body centers include readout noise; component mean speed is separate',
        confirmation=False,source_sha256=sha256(__file__))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);p.add_argument('--draws',type=int,default=64)
    p.add_argument('--geometry-manifest');p.add_argument('--prepared-cache');a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);guard=Guard(out.parent)
    model=load_model(a.checkpoint).eval();roots=load_roots(a.manifest,read(a.protocol)['development'],guard)
    if a.geometry_manifest:
        from eventforecastwm.geometry_data import attach_geometry
        attach_geometry(roots,a.geometry_manifest,guard)
    if a.prepared_cache:
        from eventforecastwm.prepared_cache import prepare
        prepare(model,roots,a.manifest,a.prepared_cache,prepare_roots,guard)
    else:prepare_roots(model,roots,guard)
    result=dict(assess(model,roots,a.draws,guard),checkpoint_sha256=sha256(a.checkpoint),manifest_sha256=sha256(a.manifest))
    save_json(out,result);print(json.dumps(result['metrics']))


if __name__=='__main__':main()
