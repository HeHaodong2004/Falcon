"""Fixed-membership root-clustered WM comparison, separate from navigation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from eventbeliefwm.data import sha256
from eventbeliefwm.model import BeliefWorldModel, BeliefConfig
from eventbeliefwm.experiment import load_split,select_batch,encode,occupancy_targets,save_json,source_commit
from eventbeliefwm.metrics import trajectory_energy,trajectory_nll,binary_summary,paired_root_interval,candidate_concordance
from eventbeliefwm.temporal import event_summary

EVALUATION_SCHEMA='eventwm-belief-evaluation-v2'
OCCUPANCY_METRICS=('average_precision','positive_nll','brier','nll')


def membership_digest(keys):
    """Order-independent canonical (branch_id, query_id) membership identity."""
    keys=[tuple(key) for key in keys]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate canonical candidate identity')
    return hashlib.sha256(json.dumps(sorted(keys),separators=(',',':')).encode()).hexdigest()


def root_occupancy_intervals(rows):
    return {name:paired_root_interval([
        row.get(name) if row.get(name) is not None else np.nan for row in rows])
        for name in OCCUPANCY_METRICS}


def nullable_mean(values):
    good=[v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(good)) if good else None


def risk_identity_coverage(metadata):
    """Target-only diagnostic; never supplied to encode/imagine or a planner.

    The extractor associates each legal track with at most one physical identity
    and removes duplicate associations. Native collision labels include every
    human, whereas this prototype's risk decoder includes associated tracks.
    """
    physical = metadata.get('physical_humans_target_only')
    associated = int(metadata['association']['associated'])
    return dict(physical_humans_target_only=physical, associated_tracks=associated,
        unassociated_physical_humans=None if physical is None else max(0, physical-associated),
        associated_fraction=None if physical is None else associated/physical if physical else 1.,
        fully_tracked=physical is not None and associated == physical)


@torch.no_grad()
def evaluate_model(model,cache,split,device='cpu',batch_size=8,horizon=5.):
    full=load_split(cache,split,device)
    if hasattr(model,'prepare_history'):
        full['prepared_history']=model.prepare_history(full['base'],
            history_timestamps=full.get('history_timestamps'),
            track_acquisition_times=full.get('track_acquisition_times'))
    model.eval()
    rows=[];occupancy_labels=[];occupancy_predictions=[];detection_y=[];detection_p=[]
    horizon_occupancy_labels=defaultdict(list);horizon_occupancy_predictions=defaultdict(list)
    started=time.time()
    for offset in range(0,len(full['metadata']),batch_size):
        batch=select_batch(full,list(range(offset,min(offset+batch_size,len(full['metadata'])))))
        belief=encode(model,batch)
        forecast=model.imagine(belief,batch['query'],horizon,rasterize=True,allow_query_hold=True)
        temporal=event_summary(forecast,batch,min(round(3./model.config.dt),forecast.human_positions.shape[2]))
        target=batch['targets'];steps=forecast.human_positions.shape[2]
        truth=target['track_positions'][:,:steps];valid=target['track_valid'][:,:steps]
        error=(forecast.human_positions-truth[:,None]).norm(dim=-1)
        weighted=(error*forecast.scenario_weights[:,:,None,None]).sum(1)
        mean=(forecast.human_positions*forecast.scenario_weights[:,:,None,None,None]).sum(1)
        mean_error=(mean-truth).norm(dim=-1)
        all_grid,anon_grid=occupancy_targets(batch,steps,model.config.grid_size,model.config.grid_extent_m)
        physical_probability=1-(1-forecast.physical_occupancy)*(1-forecast.anonymous_occupancy[:,None])
        for i,metadata in enumerate(batch['metadata']):
            identity_coverage=risk_identity_coverage(metadata)
            candidate_keys=[(r['branch_id'],r['query_id']) for r in metadata['rows']]
            if len(candidate_keys)!=batch['query'].shape[1]:
                raise ValueError('Canonical candidate metadata does not match evaluated query bank')
            item=dict(root_id=metadata['root_id'],scene=metadata['rows'][0]['scene'],
                category=metadata['rows'][0].get('category','unknown'),
                physical_humans_diagnostic_only=metadata.get('physical_humans_target_only'),
                associated_tracks=metadata['association']['associated'],
                risk_identity_coverage=identity_coverage,temporal_events_3s=temporal[i],
                candidate_count=len(candidate_keys),candidate_membership_sha256=membership_digest(candidate_keys),horizons={})
            for h in (1.,3.,5.):
                if h>horizon:continue
                length=round(h/model.config.dt)
                prefix_mask=valid[i,:length]
                complete_agents=prefix_mask.all(0)
                mask=prefix_mask & complete_agents[None]
                count=int(mask.sum())
                energy,energy_valid=trajectory_energy(forecast.human_positions[i:i+1,:,:length],truth[i:i+1,:length],
                    mask[None],forecast.scenario_weights[i:i+1])
                nll,_=trajectory_nll(forecast.human_positions[i:i+1,:,:length],truth[i:i+1,:length],
                    mask[None],forecast.scenario_weights[i:i+1],forecast.position_scale[i:i+1,:,:length])
                final=complete_agents
                label=target['collision_interval'][i,:,:length].any(-1)
                known=label | target['collision_valid'][i,:,length-1]
                planned=batch['query'][i,:,-1,2]>=h-1e-6
                known &= planned
                risk=forecast.cumulative_collision[i,:,length-1]
                ey,ep=label[known].cpu().tolist(),risk[known].cpu().tolist()
                eligible_keys=[key for key,keep in zip(candidate_keys,known.cpu().tolist()) if keep]
                supported_y=ey if identity_coverage['fully_tracked'] else []
                supported_p=ep if identity_coverage['fully_tracked'] else []
                occupancy_exposed=target['physical_exposure'][i,:length]
                complete_occupancy=bool(occupancy_exposed.all())
                prefix_oy=all_grid[i,:length][occupancy_exposed].flatten().cpu().numpy()
                prefix_op=physical_probability[i,:length][occupancy_exposed].flatten().cpu().numpy()
                horizon_oy=prefix_oy if complete_occupancy else np.empty(0)
                horizon_op=prefix_op if complete_occupancy else np.empty(0)
                horizon_occupancy_labels[str(h)].append(horizon_oy)
                horizon_occupancy_predictions[str(h)].append(horizon_op)
                div=forecast.uncertainty[i,:length]
                hm=dict(root_physical_exposed=bool(target['physical_exposure'][i,length-1]),
                    associated_agent_time_points=count,associated_agents_at_horizon=int(final.sum()),
                    partial_prefix_agent_time_points=int(prefix_mask.sum()),
                    partial_prefix_expected_ADE=float(weighted[i,:length][prefix_mask].mean()) if prefix_mask.any() else None,
                    expected_ADE=float(weighted[i,:length][mask].mean()) if count else None,
                    mean_path_ADE=float(mean_error[i,:length][mask].mean()) if count else None,
                    expected_FDE=float(weighted[i,length-1][final].mean()) if final.any() else None,
                    min_mode_ADE=float((error[i,:,:length]*mask).sum((1,2)).div(max(count,1)).min()) if count else None,
                    energy=float(energy[0]) if energy_valid[0] else None,
                    joint_nll_per_coordinate=float(nll[0]) if energy_valid[0] else None,
                    joint_nll=float(nll[0]*count*2) if energy_valid[0] else None,
                    scenario_rms_spread_m=float(div[mask].mean()) if count else None,
                    candidate_concordance=candidate_concordance(ey,ep),
                    risk_labels=ey,risk_probabilities=ep,eligible_queries=len(ey),
                    risk_candidate_membership_sha256=membership_digest(eligible_keys),
                    risk_brier=float(np.mean((np.asarray(ey)-np.asarray(ep))**2)) if ey else None,
                    fully_tracked_candidate_concordance=candidate_concordance(supported_y,supported_p),
                    fully_tracked_risk_labels=supported_y,fully_tracked_risk_probabilities=supported_p,
                    fully_tracked_eligible_queries=len(supported_y),
                    fully_tracked_risk_brier=float(np.mean((np.asarray(supported_y)-np.asarray(supported_p))**2)) if supported_y else None,
                    physical_occupancy_complete_horizon=complete_occupancy,
                    physical_occupancy_exposed_steps=int(occupancy_exposed.sum()),
                    physical_occupancy=binary_summary(horizon_oy,horizon_op),
                    partial_prefix_physical_occupancy=binary_summary(prefix_oy,prefix_op),
                    outside_training_horizon=h>model.config.trained_horizon_s)
                if count:
                    # Coverage of mixture mean +/- two total marginal standard deviations.
                    variance=((forecast.human_positions[i,:,:length]-mean[i,None,:length]).square()*
                        forecast.scenario_weights[i,:,None,None,None]).sum(0)
                    variance += (forecast.position_scale[i,:,:length].square()*forecast.scenario_weights[i,:,None,None,None]).sum(0)
                    covered=((mean[i,:length]-truth[i,:length]).abs() <= 2*variance.sqrt()).all(-1)
                    hm['two_sigma_coordinate_box_coverage']=float(covered[mask].float().mean())
                item['horizons'][str(h)]=hm
            weights=forecast.scenario_weights[i]
            item['mode_entropy']=float(-(weights*weights.clamp_min(1e-8).log()).sum())
            item['effective_modes']=float(1/weights.square().sum())
            exposed=target['physical_exposure'][i,:steps]
            oy=all_grid[i][exposed].flatten().cpu().numpy()
            op=physical_probability[i][exposed].flatten().cpu().numpy()
            item['physical_occupancy']=binary_summary(oy,op)
            occupancy_labels.append(oy);occupancy_predictions.append(op)
            anon_y=anon_grid[i][exposed].flatten().cpu().numpy()
            anon_p=forecast.anonymous_occupancy[i][None].expand(steps,-1,-1)[exposed].flatten().cpu().numpy()
            item['anonymous_occupancy']=binary_summary(anon_y,anon_p)
            omask=target['observed_valid'][i,:,:steps] & forecast.validity['query_time'][i,:,:,None]
            observed=target['observed_tracks'][i,:,:steps]
            probability=(forecast.observation_probability[i]*weights[None,:,None,None]).sum(1)
            dy=observed[omask].float().cpu().numpy();dp=probability[omask].cpu().numpy()
            item['observation_availability']=binary_summary(dy,dp)
            detection_y.append(dy);detection_p.append(dp)
            item['observation_scope']='FOV/range only; static occlusion unmodeled and calibration not established'
            rows.append(item)
    summary=dict(split=split,roots=len(rows),seconds=time.time()-started,horizons={},strata={},
        mode_entropy_mean=nullable_mean([r['mode_entropy'] for r in rows]),
        effective_modes_mean=nullable_mean([r['effective_modes'] for r in rows]),
        physical_occupancy=binary_summary(np.concatenate(occupancy_labels),np.concatenate(occupancy_predictions)),
        observation_availability=binary_summary(np.concatenate(detection_y),np.concatenate(detection_p)),
        risk_scope='Tracked identities only; anonymous human collision and static contact not supported',
        action_ranking_scope='Legacy candidate_concordance and collision metrics are all-outcome diagnostics. fully_tracked_* restricts roots to complete target-only physical identity association; both use matched native candidates at a common planned and observed/event-known horizon. Coverage is not an inference input.',
        novelty_status='Offline prototype measurements, not closed-loop utility or calibrated joint belief evidence')
    summary['physical_occupancy_root_metrics']=root_occupancy_intervals([r['physical_occupancy'] for r in rows])
    summary['foreground_roots']=sum(r['physical_occupancy'].get('positives',0)>0 for r in rows)
    summary['temporal_events_3s']={name:paired_root_interval([
        r['temporal_events_3s'].get(name) if r['temporal_events_3s'].get(name) is not None else np.nan for r in rows])
        for name in ('temporal_nll','fully_tracked_temporal_nll','conditional_event_time_MAE_s',
                     'fully_tracked_conditional_event_time_MAE_s')}
    summary['temporal_events_3s']['coverage']={name:sum(r['temporal_events_3s'][name] for r in rows)
        for name in ('queries_exposed','exposed_intervals','observed_event_queries',
                     'native_events_outside_scored_intervals','zero_mass_event_misses','timing_queries')}
    summary['temporal_events_3s']['scored_horizon_s']=min(3.,horizon)
    summary['horizon_scope']='Primary trajectory metrics require each identity exposed at every canonical tick through the horizon; partial prefixes are separate diagnostics'
    summary['occupancy_scope']='Horizon physical_occupancy covers every canonical tick through that horizon on completely exposed roots only; partial_prefix_physical_occupancy is diagnostic. Top-level occupancy mixes exposed times through the requested rollout horizon and is diagnostic only.'
    names=('expected_ADE','mean_path_ADE','expected_FDE','energy','joint_nll_per_coordinate',
           'scenario_rms_spread_m','candidate_concordance','risk_brier',
           'fully_tracked_candidate_concordance','fully_tracked_risk_brier','two_sigma_coordinate_box_coverage')
    for h in (1.,3.,5.):
        key=str(h)
        if key not in rows[0]['horizons']:continue
        hr=[r['horizons'][key] for r in rows]
        summary['horizons'][key]=dict(
            physical_exposed_roots=sum(x['root_physical_exposed'] for x in hr),
            trajectory_supported_roots=sum(x['associated_agent_time_points']>0 for x in hr),
            associated_agents_at_horizon=sum(x['associated_agents_at_horizon'] for x in hr),
            eligible_queries=sum(x['eligible_queries'] for x in hr),
            fully_tracked_roots=sum(r['risk_identity_coverage']['fully_tracked'] for r in rows),
            fully_tracked_eligible_queries=sum(x['fully_tracked_eligible_queries'] for x in hr),
            informative_ranking_roots=sum(x['candidate_concordance'] is not None for x in hr),
            fully_tracked_informative_ranking_roots=sum(x['fully_tracked_candidate_concordance'] is not None for x in hr),
            occupancy_supported_roots=sum(x['physical_occupancy_complete_horizon'] for x in hr),
            occupancy_exposed_steps=sum(x['physical_occupancy_exposed_steps'] for x in hr if x['physical_occupancy_complete_horizon']),
            occupancy_foreground_roots=sum(x['physical_occupancy'].get('positives',0)>0 for x in hr),
            physical_occupancy=binary_summary(np.concatenate(horizon_occupancy_labels[key]),np.concatenate(horizon_occupancy_predictions[key])),
            physical_occupancy_root_metrics=root_occupancy_intervals([x['physical_occupancy'] for x in hr]),
            partial_prefix_physical_occupancy_root_metrics=root_occupancy_intervals([x['partial_prefix_physical_occupancy'] for x in hr]),
            metrics={name:paired_root_interval([x.get(name) if x.get(name) is not None else np.nan for x in hr]) for name in names},
            collision=binary_summary([y for x in hr for y in x['risk_labels']], [p for x in hr for p in x['risk_probabilities']]),
            fully_tracked_collision=binary_summary([y for x in hr for y in x['fully_tracked_risk_labels']],
                [p for x in hr for p in x['fully_tracked_risk_probabilities']]),
            ranking_coverage_census=[dict(root_id=r['root_id'],category=r['category'],
                **r['risk_identity_coverage'],eligible_queries=r['horizons'][key]['eligible_queries'],
                collision_queries=sum(r['horizons'][key]['risk_labels']),
                safe_queries=r['horizons'][key]['eligible_queries']-sum(r['horizons'][key]['risk_labels']))
                for r in rows if r['horizons'][key]['candidate_concordance'] is not None])
    for category in sorted({r['category'] for r in rows}):
        subset=[r for r in rows if r['category']==category]
        summary['strata'][category]=dict(roots=len(subset),
            energy3=nullable_mean([r['horizons'].get('3.0',{}).get('energy') for r in subset]),
            ADE3=nullable_mean([r['horizons'].get('3.0',{}).get('expected_ADE') for r in subset]),
            occupancy_AP=nullable_mean([r['physical_occupancy'].get('average_precision') for r in subset]))
    return dict(evaluation_schema=EVALUATION_SCHEMA,
        evaluation_protocol=dict(dt=model.config.dt,horizon=horizon),summary=summary,roots=rows)


def comparison_membership(report,name):
    if report.get('evaluation_schema')!=EVALUATION_SCHEMA:
        raise ValueError(name+': regenerate evaluation with canonical candidate membership digests')
    cache=report.get('cache_manifest_sha256')
    if not isinstance(cache,str) or len(cache)!=64:
        raise ValueError(name+': missing cache_manifest_sha256')
    if not report.get('summary',{}).get('split') or not report.get('evaluation_protocol'):
        raise ValueError(name+': missing split or evaluation protocol')
    rows=report['roots']
    index={r['root_id']:r for r in rows}
    if len(index)!=len(rows) or len(rows)!=report['summary']['roots']:
        raise ValueError(name+': duplicate or inconsistent root membership')
    if not index:
        raise ValueError(name+': empty root membership')
    for root in rows:
        digest=root.get('candidate_membership_sha256')
        if not isinstance(digest,str) or len(digest)!=64:
            raise ValueError(name+': missing canonical candidate digest for '+root['root_id'])
        if set(root['horizons'])!=set(report['summary']['horizons']):
            raise ValueError(name+': inconsistent horizon membership for '+root['root_id'])
        for h,values in root['horizons'].items():
            digest=values.get('risk_candidate_membership_sha256')
            if not isinstance(digest,str) or len(digest)!=64:
                raise ValueError(name+': missing eligible candidate digest for '+root['root_id']+'/'+h)
    return index


def require_matched_reports(a,b,a_index,b_index):
    for field in ('cache_manifest_sha256','evaluation_protocol'):
        if a[field]!=b[field]:
            raise ValueError('Comparison mismatch: '+field)
    if a['summary']['split']!=b['summary']['split']:
        raise ValueError('Comparison mismatch: split')
    if a_index.keys()!=b_index.keys():
        raise ValueError('Comparison mismatch: exact root membership (intersection is not allowed)')
    for rid in a_index:
        ar,br=a_index[rid],b_index[rid]
        for field in ('scene','candidate_count','candidate_membership_sha256','risk_identity_coverage'):
            if ar[field]!=br[field]:
                raise ValueError('Comparison mismatch: '+rid+'/'+field)
        if ar['horizons'].keys()!=br['horizons'].keys():
            raise ValueError('Comparison mismatch: '+rid+'/horizons')
        for h in ar['horizons']:
            for field in ('risk_candidate_membership_sha256','eligible_queries','fully_tracked_eligible_queries',
                          'associated_agent_time_points','associated_agents_at_horizon',
                          'physical_occupancy_complete_horizon','physical_occupancy_exposed_steps'):
                if ar['horizons'][h][field]!=br['horizons'][h][field]:
                    raise ValueError('Comparison mismatch: '+rid+'/'+h+'/'+field)


def compare(paths,output):
    names=[Path(p).stem for p in paths]
    if len(names)!=len(set(names)):
        raise ValueError('Comparison report names must be unique')
    reports={Path(p).stem:json.loads(Path(p).read_text()) for p in paths}
    indexes={name:comparison_membership(report,name) for name,report in reports.items()}
    comparisons={}
    for i,name in enumerate(names):
        for other in names[:i]:
            a,b=indexes[name],indexes[other]
            require_matched_reports(reports[name],reports[other],a,b)
            common=sorted(a)
            pair={}
            for h in reports[name]['summary']['horizons']:
                for metric in ('energy','expected_ADE','expected_FDE','risk_brier','candidate_concordance',
                               'fully_tracked_risk_brier','fully_tracked_candidate_concordance'):
                    av=[a[r]['horizons'].get(h,{}).get(metric) for r in common]
                    bv=[b[r]['horizons'].get(h,{}).get(metric) for r in common]
                    pair[h+'/'+metric]=paired_root_interval([v if v is not None else np.nan for v in av],
                        [v if v is not None else np.nan for v in bv])
                for metric in OCCUPANCY_METRICS:
                    av=[a[r]['horizons'][h]['physical_occupancy'].get(metric) for r in common]
                    bv=[b[r]['horizons'][h]['physical_occupancy'].get(metric) for r in common]
                    pair[h+'/physical_occupancy/'+metric]=paired_root_interval(
                        [v if v is not None else np.nan for v in av],
                        [v if v is not None else np.nan for v in bv])
            comparisons[name+' minus '+other]=pair
    save_json(output,dict(paired_root_differences=comparisons,evaluation_schema=EVALUATION_SCHEMA,
        membership_validation='Exact cache/split/protocol/root/canonical-candidate and horizon-support equality required; no intersections',
        interpretation='Exploratory bounded pilot; root intervals exclude seed/scene/model-selection uncertainty'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint');p.add_argument('--kind',choices=['cv','deterministic','stochastic'],default='cv')
    p.add_argument('--split',default='validation');p.add_argument('--device',default='cuda:1')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--horizon',type=float,default=5.)
    args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    torch.manual_seed(20261014)
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location=args.device)
        model=BeliefWorldModel(BeliefConfig(**state['protocol']['config'])).to(args.device)
        model.load_state_dict(state['model'])
    else:model=BeliefWorldModel(BeliefConfig(kind=args.kind)).to(args.device)
    report=evaluate_model(model,args.cache,args.split,args.device,args.batch_size,args.horizon)
    report.update(model=model.parameter_counts(),source_commit=source_commit(),
        evaluator_sha256=sha256(Path(__file__)),
        checkpoint=args.checkpoint,checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,
        checkpoint_step=state['step'] if args.checkpoint else None,
        training_source_commit=state['protocol']['source_commit'] if args.checkpoint else None,
        postfit_calibration=state.get('postfit_calibration') if args.checkpoint else None,
        cache_manifest_sha256=sha256(Path(args.cache)/'manifest.json'))
    save_json(args.output,report)
    print(json.dumps(dict(output=args.output,split=args.split,roots=report['summary']['roots'],
        seconds=report['summary']['seconds'],horizons={h:dict(
            trajectory_roots=values['trajectory_supported_roots'],
            energy=values['metrics']['energy']['mean'],ADE=values['metrics']['expected_ADE']['mean'],
            informative_ranking_roots=values['informative_ranking_roots'],
            fully_tracked_informative_ranking_roots=values['fully_tracked_informative_ranking_roots'])
            for h,values in report['summary']['horizons'].items()}),indent=2))


if __name__=='__main__':main()
