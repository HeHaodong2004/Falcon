"""Intrinsic forecasting benchmarks, calibration and paired clustered intervals.

No query outcomes, selector predictions, ranking or navigation scores are read.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from eventbeliefwm.data import read,sha256
from eventbeliefwm.experiment import save_json,occupancy_targets
from eventbeliefwm.metrics import binary_summary,paired_root_interval
from eventforecastwm.model import ForecastWorldModel
from eventforecastwm.distribution import trajectory_log_probability,energy_score,enable_evaluation_dispatch
from eventforecastwm.readouts import tracked_occupancy,marginal_cdf
from eventforecastwm.train import load_roots,prepare_roots,prepared_batch,encode,evaluate
from eventforecastwm.guard import Guard
from eventforecastwm.sampling import joint_samples
from eventforecastwm.observations import protect_observe

EXPERIMENT_SOURCES={}


def load_model(checkpoint,kind=None,noise='iid',motion_window=.75):
    if not checkpoint:return protect_observe(ForecastWorldModel(kind=kind or 'cv',noise=noise,motion_window=motion_window))
    state=torch.load(checkpoint,map_location='cpu')
    if 'recipe' in state:
        factory=ForecastWorldModel
        if state['recipe'].get('variant')=='age_aware_v1':
            from eventforecastwm.model_age import AgeAwareForecastWorldModel
            factory=AgeAwareForecastWorldModel
        elif state['recipe'].get('variant')=='student_scene_v1':
            from eventforecastwm.model_student import StudentWorldModel
            factory=StudentWorldModel
        elif state['recipe'].get('variant')=='kinematic_v1':
            from eventforecastwm.model_kinematic import KinematicForecastWorldModel
            factory=KinematicForecastWorldModel
        elif state['recipe'].get('variant')=='damped_velocity_v1':
            from eventforecastwm.model_damped import DampedForecastWorldModel
            factory=DampedForecastWorldModel
        elif state['recipe'].get('variant')=='posterior_v1':
            from eventforecastwm.model_posterior import PosteriorForecastWorldModel
            factory=PosteriorForecastWorldModel
        elif state['recipe'].get('variant')=='map_patch_v1':
            from eventforecastwm.model_map_patch import MapPatchForecastWorldModel
            factory=MapPatchForecastWorldModel
        elif state['recipe'].get('variant')=='map_posterior_v1':
            from eventforecastwm.model_map_posterior import MapPosteriorForecastWorldModel
            factory=MapPosteriorForecastWorldModel
        elif state['recipe'].get('variant')=='finite_scene_scale_v1':
            from eventforecastwm.model_scale_mixture import ScaleMixtureWorldModel
            factory=ScaleMixtureWorldModel
        elif state['recipe'].get('variant')=='independent_attention_v1':
            from eventforecastwm.model_independent import IndependentAttentionWorldModel
            factory=IndependentAttentionWorldModel
        elif state['recipe'].get('variant')=='conditional_scene_scale_v1':
            from eventforecastwm.model_conditional_scale import ConditionalScaleWorldModel
            factory=ConditionalScaleWorldModel
        elif state['recipe'].get('variant')=='agent_scene_scale_v1':
            from eventforecastwm.model_agent_scale import AgentScaleWorldModel
            factory=AgentScaleWorldModel
        elif state['recipe'].get('variant')=='stopped_agent_scale_v2':
            from eventforecastwm.performance_v2.regime import RegimeWorldModel
            factory=RegimeWorldModel
        elif state['recipe'].get('variant')=='geometry_occupancy_v2':
            from eventforecastwm.performance_v2.hybrid import OccupancyWorldModel
            factory=OccupancyWorldModel
        elif state['recipe'].get('variant','').endswith('_v2'):
            from eventforecastwm.performance_v2.physical import PhysicalWorldModel,VARIANTS
            if state['recipe']['variant'] not in VARIANTS:raise ValueError('Unknown v2 physical recipe')
            factory=PhysicalWorldModel
        model=factory(**state['recipe']);model.load_state_dict(state['model'])
    else:
        config=state['protocol']['config']
        model=ForecastWorldModel(kind=config['kind'],width=config['width'],modes=config['modes'],noise='iid')
        model.core.load_state_dict(state['model']);model.log_readout.data.copy_(model.core.log_position_scale.detach())
    if hasattr(model,'raw_df') or hasattr(model,'raw_decay_rate'):enable_evaluation_dispatch()
    return protect_observe(model)


@torch.no_grad()
def profile_iid(model,roots,device,guard=None):
    if model.noise!='iid':raise ValueError('This profile only calibrates iid readout scale')
    forecasts=[]
    for start in range(0,len(roots),32):
        if guard:guard.check()
        batch=prepared_batch([r['windows'][0] for r in roots[start:start+32]],device)
        f=model.imagine(encode(model,batch),horizon_s=3.)
        raw=batch['targets']['track_valid'][:,:12];mask=raw.all(1)[:,None].expand_as(raw)
        forecasts.append((f,batch['targets']['track_positions'][:,:12],mask))
    candidates=np.geomspace(.03,4.,81);scores=[]
    for sigma in candidates:
        if guard:guard.check()
        values=[]
        for f,gt,mask in forecasts:
            lp,count=trajectory_log_probability(replace(f,readout_scale=f.readout_scale.new_tensor(sigma)),gt,mask)
            values.extend((-lp[count>0]/count[count>0]).tolist())
        scores.append(float(np.mean(values)))
    best=int(np.argmin(scores));before=float(model.log_readout.exp());model.log_readout.data.fill_(math.log(candidates[best]))
    return dict(before=before,after=float(candidates[best]),roots=sum(int(mask.flatten(1).any(1).sum()) for f,g,mask in forecasts),
                objective='training-only root-mean3s normalized joint NLL',grid=candidates.tolist(),scores=scores)


def profile_integrated(model,roots,device,guard=None,objective_kind='nll'):
    """Fit only four shared noise scales on training roots; physical means fixed."""
    cached=[]
    with torch.no_grad():
        for start in range(0,len(roots),32):
            if guard:guard.check()
            batch=prepared_batch([r['windows'][0] for r in roots[start:start+32]],device)
            f=model.imagine(encode(model,batch),horizon_s=3.)
            raw=batch['targets']['track_valid'][:,:12];mask=raw.all(1)[:,None].expand_as(raw)
            cached.append((f,batch['targets']['track_positions'][:,:12],mask))
    names=['log_initial_position','log_initial_velocity','log_acceleration','log_readout']
    if hasattr(model,'raw_df'):names.append('raw_df')
    if hasattr(model,'raw_decay_rate'):names.append('raw_decay_rate')
    if hasattr(model,'scale_logits'):
        if objective_kind!='nll':raise ValueError('Finite scale probabilities require exact likelihood gradients')
        names.append('scale_logits')
    parameters=[getattr(model,k) for k in names]
    def values():
        result={k:float(getattr(model,k).exp()) for k in names if k.startswith('log_')}
        if hasattr(model,'raw_df'):result['degrees_of_freedom']=float(model.degrees_of_freedom)
        if hasattr(model,'raw_decay_rate'):result['velocity_decay_rate_per_s']=float(model.decay_rate)
        if hasattr(model,'scale_logits'):result['scene_scale_probabilities']=model.scale_logits.softmax(-1).tolist()
        return result
    before=values()
    optimizer=torch.optim.Adam(parameters,lr=.03);curve=[]
    generator=torch.Generator(device=device).manual_seed(20261014)
    # At least eight full passes: the old fixed256 updates would skip some
    # full-pool roots entirely. This leaves the <=1024-root comparisons intact.
    steps=max(256,8*len(cached))
    for step in range(steps):
        if guard:guard.check()
        f,gt,mask=cached[step%len(cached)]
        p0=torch.zeros_like(f.initial_covariance)
        p0[...,0,0]=model.log_initial_position.exp().clamp(.01,3.).square()
        p0[...,1,1]=model.log_initial_velocity.exp().clamp(.01,3.).square()
        if hasattr(model,'initial_covariance_for'):
            initial=replace(f.next_belief.core,age=f.next_belief.core.age-f.positions.shape[2]*f.dt)
            p0=model.initial_covariance_for(initial)
        fc=replace(f,initial_covariance=p0,acceleration_scale=model.log_acceleration.exp().clamp(.01,5.),
                   readout_scale=model.log_readout.exp().clamp(.01,5.))
        if hasattr(model,'raw_df'):
            alpha=(model.degrees_of_freedom/2).expand_as(f.weights);fc=replace(fc,concentration=alpha,rate=alpha)
        if hasattr(model,'raw_decay_rate'):fc=replace(fc,decay_rate=model.decay_rate)
        if hasattr(model,'scale_logits'):fc=replace(fc,scale_probabilities=model.scale_logits.softmax(-1).expand_as(fc.scale_probabilities))
        lp,count=trajectory_log_probability(fc,gt,mask);keep=count>0
        if not keep.any():continue
        nll=(-lp[keep]/count[keep]).mean();objective=nll
        components=dict(nll=float(nll.detach()))
        if objective_kind=='energy_nll':
            es,support=energy_score(fc,gt,mask,draws=max(1,6//fc.weights.shape[1]),generator=generator)
            energy=es[support].mean();objective=energy+.05*nll
            components.update(energy=float(energy.detach()),objective=float(objective.detach()))
        elif objective_kind!='nll':raise ValueError(objective_kind)
        optimizer.zero_grad();objective.backward();optimizer.step()
        if step%32==0:curve.append(dict(step=step,**components))
    return dict(before=before,after=values(),steps=steps,complete_batch_passes=steps//len(cached),
        roots=sum(int(m.flatten(1).any(1).sum()) for f,g,m in cached),curve=curve,
        objective=('training-only root-mean3s normalized joint NLL' if objective_kind=='nll' else
            'training-only root-mean3s Energy Score + .05 normalized joint NLL')+'; physical means/weights frozen',
        objective_kind=objective_kind)


@torch.no_grad()
def anonymous_prior(roots,device,guard=None):
    """Common root-balanced marginal occupancy prior, targets never inputs.

    One root contributes its mean over valid windows. Missing time support never
    adds a negative cell. This is anonymous occupancy, not unseen identities.
    """
    total=torch.zeros(20,32,32,device=device);counts=torch.zeros(20,device=device)
    for root in roots:
        if guard:guard.check()
        batch=prepared_batch(root['windows'],device)
        _,anon=occupancy_targets(batch,20);valid=batch['targets']['physical_exposure']
        supported=valid.any(0)
        mean=(anon*valid[...,None,None]).sum(0)/valid.sum(0).clamp_min(1)[:,None,None]
        total+=mean*supported[:,None,None];counts+=supported
    return (total+.5)/(counts[:,None,None]+1.),counts


def raster(f):
    return tracked_occupancy(f)


@torch.no_grad()
def extra_metrics(model,roots,prior,device,horizons=(1.,3.,5.),guard=None):
    result={}
    for h in horizons:
        l=int(h*4);rows=[]
        generator=torch.Generator(device=device);generator.manual_seed(83199+int(h*10))
        for start in range(0,len(roots),16):
            if guard:guard.check()
            selected=roots[start:start+16];batch=prepared_batch([r['windows'][0] for r in selected],device)
            f=model.imagine(encode(model,batch),horizon_s=h)
            gt=batch['targets']['track_positions'][:,:l];raw=batch['targets']['track_valid'][:,:l]
            complete=raw.all(1);mask=complete[:,None].expand_as(raw)
            target,_=occupancy_targets(batch,l)
            physical=raster(f)
            anonymous=getattr(f,'anonymous_occupancy',prior[:l][None])
            combined=1-(1-physical)*(1-anonymous)
            # Marginal scalar central90% intervals by exact mixture-CDF inversion.
            sigma=f.position_variance.sqrt()[...,None]
            def cdf(x):return (marginal_cdf(f,x[:,None])*f.weights[:,:,None,None,None]).sum(1)
            lower0=(f.positions-7*sigma).amin(1);upper0=(f.positions+7*sigma).amax(1)
            quantiles=[]
            for q in (.05,.95):
                lo=lower0.clone();hi=upper0.clone()
                for _ in range(24):
                    mid=(lo+hi)/2;below=cdf(mid)<q;lo=torch.where(below,mid,lo);hi=torch.where(below,hi,mid)
                quantiles.append((lo+hi)/2)
            covered=(gt>=quantiles[0])&(gt<=quantiles[1]);pit=cdf(gt)
            # FixedK=6 probabilistic draws, common across model scenario counts.
            paths,_=joint_samples(f,k=6,generator=generator)
            distance=(paths-gt[:,None]).norm(dim=-1)
            ade=(distance*mask[:,None]).sum((2,3))/mask.sum((1,2))[:,None].clamp_min(1)
            fde=(distance[:,:,-1]*complete[:,None]).sum(-1)/complete.sum(-1)[:,None].clamp_min(1)
            for i,root in enumerate(selected):
                w=root['windows'][0];p=w['base'].humans
                visible=int((p[:,-1,4]>.5).sum()) if len(p) else 0
                ever=len(p);physical_count=w['metadata']['physical_humans_target_only']
                row=dict(root_id=root['root_id'],scene=root['scene'],visible_tracks=visible,ever_tracks=ever,
                    lost=ever>visible,never_observed=physical_count>0 and ever==0,dense=physical_count>=3)
                if bool(batch['targets']['physical_exposure'][i,:l].all()):
                    row['occupancy']=binary_summary(target[i].cpu().numpy(),combined[i].cpu().numpy())
                    row['tracked_occupancy']=binary_summary(target[i].cpu().numpy(),physical[i].cpu().numpy())
                    row['foreground_present']=bool(target[i].any())
                if mask[i].any():
                    m=mask[i][...,None].expand_as(gt[i]);row.update(
                        marginal_coverage90=float(covered[i][m].float().mean()),
                        marginal_width90=float((quantiles[1][i]-quantiles[0][i])[m].mean()),
                        pit_histogram=torch.histc(pit[i][m],bins=10,min=0,max=1).tolist(),
                        minade_sample6=float(ade[i].min()),minfde_sample6=float(fde[i].min()),
                        max_mean_component_speed=float(f.velocities[i,:,:,complete[i]].norm(dim=-1).max()))
                rows.append(row)
        keys=('marginal_coverage90','marginal_width90','minade_sample6','minfde_sample6','max_mean_component_speed')
        result[str(h)]=dict(records=rows,metrics={k:float(np.mean([r[k] for r in rows if k in r])) if any(k in r for r in rows) else None for k in keys},
            metric_roots={k:sum(k in r for r in rows) for k in keys},
            occupancy_metric_roots={k:sum(r.get('occupancy',{}).get(k) is not None for r in rows)
                for k in ('brier','nll','average_precision','positive_nll','ece10')},
            occupancy={k:float(np.mean([r['occupancy'][k] for r in rows if r.get('occupancy',{}).get(k) is not None]))
                       for k in ('brier','nll','average_precision','positive_nll','ece10')})
    return result


def scene_interval(records,key,other=None,repetitions=1000):
    rows=[r for r in records if key in r and (other is None or key in other[r['root_id']])]
    scenes=sorted({r['scene'] for r in rows});rng=np.random.default_rng(83199)
    groups={s:np.asarray([r[key]-(other[r['root_id']][key] if other is not None else 0.) for r in rows if r['scene']==s]) for s in scenes}
    if not scenes:return dict(roots=0,scenes=0)
    boot=[]
    for _ in range(repetitions):boot.append(np.concatenate([groups[s] for s in rng.choice(scenes,len(scenes),replace=True)]).mean())
    return dict(roots=len(rows),scenes=len(scenes),mean=float(np.concatenate(list(groups.values())).mean()),
        low=float(np.quantile(boot,.025)),high=float(np.quantile(boot,.975)),unit='scene clusters, root-weighted estimand')


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--protocol',required=True)
    p.add_argument('--output',required=True);p.add_argument('--checkpoint');p.add_argument('--kind',default='cv')
    p.add_argument('--motion-window',type=float,default=.75);p.add_argument('--noise',default='iid')
    p.add_argument('--profile',action='store_true');p.add_argument('--fit-scale',default='128');p.add_argument('--device',default='cpu')
    p.add_argument('--profile-objective',choices=('nll','energy_nll'),default='nll')
    p.add_argument('--geometry-manifest')
    p.add_argument('--integrated',action='store_true',help='Calibrate coherent residuals around fixed IID-trained physical means')
    p.add_argument('--student',action='store_true',help='Fit a persistent scene-scale heavy-tailed residual to fixed physical means')
    p.add_argument('--damped',action='store_true',help='Fit exact integrated OU residuals to fixed physical means')
    p.add_argument('--scale-mixture',action='store_true',help='Fit a finite persistent scene-scale law to an existing OU checkpoint')
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    torch.manual_seed(20261014);np.random.seed(20261014);enable_evaluation_dispatch()
    directory=output.parent;directory.mkdir(parents=True,exist_ok=True);guard=Guard(directory)
    started=time.time();protocol=read(a.protocol)
    fit=load_roots(a.manifest,protocol['subsets'][a.fit_scale],guard);dev=load_roots(a.manifest,protocol['development'],guard)
    if a.geometry_manifest:
        from eventforecastwm.geometry_data import attach_geometry
        attach_geometry(fit+dev,a.geometry_manifest,guard)
    model=load_model(a.checkpoint,a.kind,a.noise,a.motion_window)
    if a.integrated:
        if not a.profile or model.noise!='iid':raise ValueError('Integrated conversion requires IID means and training-only calibration')
        model.noise='integrated';model.recipe['noise']='integrated'
    if a.student and a.damped:raise ValueError('Separate controlled residual-law experiments required')
    if a.damped and not hasattr(model,'raw_decay_rate'):
        from eventforecastwm.model_damped import DampedForecastWorldModel
        from eventforecastwm.performance_v2.physical import VARIANTS as v2_variants
        config=dict(model.recipe);variant=config.pop('variant',None)
        if variant not in (None,'age_aware_v1','posterior_v1','map_patch_v1','map_posterior_v1') and variant not in v2_variants:raise ValueError('Unsupported damped mean conversion')
        damped=DampedForecastWorldModel(**config,age_aware=variant=='age_aware_v1',
            mean_variant=variant if variant in ('posterior_v1','map_patch_v1','map_posterior_v1') or variant in v2_variants else None)
        missing=damped.load_state_dict(model.state_dict(),strict=False)
        if missing.missing_keys!=['raw_decay_rate'] or missing.unexpected_keys:raise ValueError('Damped physical checkpoint mismatch')
        model=damped
    if a.student and not hasattr(model,'raw_df'):
        from eventforecastwm.model_student import StudentWorldModel
        config=dict(model.recipe);variant=config.pop('variant',None)
        if variant not in (None,'age_aware_v1','damped_velocity_v1','posterior_v1','map_patch_v1','map_posterior_v1'):raise ValueError('Unsupported Student mean conversion')
        age_aware=config.pop('age_aware',variant=='age_aware_v1')
        mean_variant=config.pop('mean_variant',variant if variant in ('posterior_v1','map_patch_v1','map_posterior_v1') else None)
        student=StudentWorldModel(**config,age_aware=age_aware,damped=variant=='damped_velocity_v1',mean_variant=mean_variant)
        missing=student.load_state_dict(model.state_dict(),strict=False)
        if missing.missing_keys!=['raw_df'] or missing.unexpected_keys:raise ValueError('Student physical checkpoint mismatch')
        model=student
    if a.scale_mixture:
        if a.student or a.damped or not hasattr(model,'raw_decay_rate') or hasattr(model,'raw_df'):
            raise ValueError('Finite-scale comparison requires a fixed Gaussian OU checkpoint')
        from eventforecastwm.model_scale_mixture import ScaleMixtureWorldModel
        config=dict(model.recipe);variant=config.pop('variant',None)
        if variant!='damped_velocity_v1':raise ValueError('Gaussian OU checkpoint required')
        mixture=ScaleMixtureWorldModel(**config);missing=mixture.load_state_dict(model.state_dict(),strict=False)
        if set(missing.missing_keys)!={'scale_values','scale_logits'} or missing.unexpected_keys:raise ValueError('Finite-scale physical checkpoint mismatch')
        model=mixture
    prepare_roots(model,fit+dev,guard);model.to(a.device);model.eval()
    if a.profile_objective!='nll' and model.noise=='iid':raise ValueError('Combined calibration currently requires coherent trajectory residuals')
    calibration=(profile_iid(model,fit,a.device,guard) if model.noise=='iid' else profile_integrated(model,fit,a.device,guard,a.profile_objective)) if a.profile else None
    prior,counts=anonymous_prior(fit,a.device,guard)
    guard.check();scores=evaluate(model,dev,a.device,horizons=(1.,3.,5.),draws=48)
    extras=extra_metrics(model,dev,prior,a.device,guard=guard);guard.check()
    intervals={h:{k:dict(root=paired_root_interval([r[k] for r in v['records'] if k in r]),
                         scene=scene_interval(v['records'],k)) for k in ('energy','nll','mean_ade','mean_fde')} for h,v in scores.items()}
    result=dict(schema_version=3,fixed_k_sampling='six independent scene-mode draws with independent residual paths and persistent scene scale when specified',
        checkpoint=a.checkpoint,checkpoint_sha256=sha256(a.checkpoint) if a.checkpoint else None,
        manifest_sha256=sha256(a.manifest),protocol_sha256=sha256(a.protocol),recipe=model.recipe,
        scores=scores,extras=extras,intervals=intervals,calibration=calibration,anonymous_prior_fit_roots=counts.tolist(),
        runtime_s=time.time()-started,selection_data='train-internal development; not final confirmation',
        horizon5_status='extrapolation beyond3s training; only complete supported targets',
        foreground_definition='physical human-center cell occupancy, complete native scene exposure only',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={**{name:sha256(Path(__file__).parent/name) for name in ('model.py','train.py','benchmark.py','distribution.py','readouts.py','sampling.py','model_student.py','model_age.py','model_kinematic.py','model_damped.py','model_posterior.py','model_map_patch.py','mean_adapters.py','model_scale_mixture.py','model_independent.py','observations.py','model_map_posterior.py','model_agent_scale.py','model_conditional_scale.py')},**EXPERIMENT_SOURCES},
        geometry_manifest_sha256=sha256(a.geometry_manifest) if a.geometry_manifest else None,
        environment=dict(torch=torch.__version__,numpy=np.__version__))
    if calibration:
        calibrated=output.with_suffix('.model.pt')
        if calibrated.exists():raise FileExistsError(calibrated)
        torch.save(dict(recipe=model.recipe,model=model.state_dict(),calibration=calibration,
            parent_checkpoint_sha256=result['checkpoint_sha256'],fit_root_ids=protocol['subsets'][a.fit_scale],
            anonymous_occupancy_prior=prior.cpu(),anonymous_exposure_counts=counts.cpu(),
            geometry_manifest_sha256=result['geometry_manifest_sha256'],source_hashes=result['source_hashes']),calibrated)
        result['calibrated_checkpoint']=str(calibrated);result['calibrated_checkpoint_sha256']=sha256(calibrated)
    save_json(output,result)
    print(json.dumps(dict(output=str(output),seconds=result['runtime_s'],scores={h:v['metrics'] for h,v in scores.items()})),flush=True)


if __name__=='__main__':main()
