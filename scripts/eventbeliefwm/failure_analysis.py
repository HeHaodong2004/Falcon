"""Report-only diagnostics for matched completed WM pilots; never trains a model."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

from eventbeliefwm.data import sha256
from eventbeliefwm.evaluate import comparison_membership, require_matched_reports
from eventbeliefwm.metrics import paired_root_interval


def strata(row,split,dense_min=4):
    """All physical-count fields here are evaluation diagnostics only."""
    category=row.get('category','unknown')
    physical=row.get('physical_humans_diagnostic_only')
    result=['all','category/'+category]
    if category=='unseen':result.append('never_observed')
    if category in ('lost','visible_lost','observed_expired'):result.append('lost_or_expired')
    if physical is not None and physical>=2:result.append('multiple_humans')
    if physical is not None and physical>=dense_min:result.append('dense_count_proxy')
    if split=='unseen_evaluation':result.append('unseen_scene')
    result.append('fully_tracked' if row['risk_identity_coverage']['fully_tracked'] else 'incomplete_identity_coverage')
    return result


def finite(value):
    return value is not None and np.isfinite(value)


def interval(values):
    return paired_root_interval([v if finite(v) else np.nan for v in values])


def difference(a,b):
    return float(a-b) if finite(a) and finite(b) else None


def metric_values(row,horizon):
    h=row['horizons'][horizon]
    return dict(energy=h.get('energy'),expected_ADE=h.get('expected_ADE'),
        candidate_concordance=h.get('candidate_concordance'),
        fully_tracked_candidate_concordance=h.get('fully_tracked_candidate_concordance'),
        risk_brier=h.get('risk_brier'),fully_tracked_risk_brier=h.get('fully_tracked_risk_brier'),
        occupancy_AP=h['physical_occupancy'].get('average_precision'),
        occupancy_positive_nll=h['physical_occupancy'].get('positive_nll'))


def summarize(rows,horizon):
    hs=[r['horizons'][horizon] for r in rows]
    values=[metric_values(r,horizon) for r in rows]
    return dict(roots=len(rows),scenes=len({r['scene'] for r in rows}),
        trajectory_supported_roots=sum(h['associated_agent_time_points']>0 for h in hs),
        occupancy_supported_roots=sum(h['physical_occupancy_complete_horizon'] for h in hs),
        eligible_queries=sum(h['eligible_queries'] for h in hs),
        informative_ranking_roots=sum(h.get('candidate_concordance') is not None for h in hs),
        fully_tracked_informative_ranking_roots=sum(h.get('fully_tracked_candidate_concordance') is not None for h in hs),
        observed_bank_all_collision_roots=sum(bool(h['risk_labels']) and all(h['risk_labels']) for h in hs),
        observed_bank_all_safe_roots=sum(bool(h['risk_labels']) and not any(h['risk_labels']) for h in hs),
        no_eligible_outcome_roots=sum(not h['risk_labels'] for h in hs),
        metrics={key:interval([v[key] for v in values]) for key in values[0]} if values else {})


def pair_diagnostics(model_rows,baseline_rows,horizon,top_k=8,tie_tolerance=1e-8):
    """Signs are descriptive root diagnostics, never a causal attribution."""
    deltas=[];quadrants={'fully_tracked':Counter(),'incomplete_identity_coverage':Counter()}
    examples=[]
    for rid in sorted(model_rows):
        row,base=model_rows[rid],baseline_rows[rid]
        values,bvalues=metric_values(row,horizon),metric_values(base,horizon)
        delta={name:difference(values[name],bvalues[name]) for name in values}
        deltas.append(delta)
        motion,risk=delta['energy'],delta['candidate_concordance']
        if not finite(motion) or not finite(risk):continue
        sign=lambda value: 'improved' if value < -tie_tolerance else 'worsened' if value > tie_tolerance else 'tied'
        motion_sign=sign(motion)
        # Lower energy is better; higher concordance is better.
        risk_sign=sign(-risk)
        support='fully_tracked' if row['risk_identity_coverage']['fully_tracked'] else 'incomplete_identity_coverage'
        quadrants[support]['physics_'+motion_sign+'/ranking_'+risk_sign]+=1
        if motion_sign=='improved' and risk_sign=='worsened':
            h=row['horizons'][horizon]
            examples.append(dict(root_id=rid,scene=row['scene'],category=row['category'],support=support,
                physical_humans_target_only=row.get('physical_humans_diagnostic_only'),
                associated_tracks=row.get('associated_tracks'),energy_delta=motion,concordance_delta=risk,
                expected_ADE_delta=delta['expected_ADE'],
                model_concordance=h['candidate_concordance'],baseline_concordance=base['horizons'][horizon]['candidate_concordance'],
                risk_min=min(h['risk_probabilities']),risk_max=max(h['risk_probabilities']),
                eligible_safe_queries=len(h['risk_labels'])-sum(h['risk_labels']),
                eligible_collision_queries=sum(h['risk_labels'])))
    examples.sort(key=lambda row:row['energy_delta'])
    return dict(paired_root_differences={key:interval([d[key] for d in deltas]) for key in deltas[0]} if deltas else {},
        motion_ranking_quadrants={key:dict(value) for key,value in quadrants.items()},
        discordant_examples=examples[:top_k],discordant_example_selection='Largest energy improvements among roots whose all-outcome ranking worsened; selected for inspection, not independent evidence')


def risk_resolution(rows,horizon,epsilon=1e-3):
    """Report saturation separately for the supported and all-outcome views."""
    result={}
    for name,subset in (('all_outcome',rows),('fully_tracked',[r for r in rows if r['risk_identity_coverage']['fully_tracked']])):
        informative=[r['horizons'][horizon] for r in subset if r['horizons'][horizon].get('candidate_concordance') is not None]
        probabilities=[p for h in informative for p in h['risk_probabilities']]
        result[name]=dict(informative_roots=len(informative),queries=len(probabilities),
            near_zero=sum(p<=epsilon for p in probabilities),near_one=sum(p>=1-epsilon for p in probabilities),
            tied_risk_roots=sum(max(h['risk_probabilities'])-min(h['risk_probabilities'])<=1e-8 for h in informative),
            probability_threshold=epsilon)
    return result


def analyze_group(reports,horizon='3.0',dense_min=4,top_k=8):
    indexes={name:comparison_membership(report,name) for name,report in reports.items()}
    baselines=[name for name,report in reports.items() if report.get('model',{}).get('config',{}).get('kind')=='cv']
    if len(baselines)!=1:raise ValueError('Each split requires exactly one CV baseline report')
    baseline=baselines[0];reference=reports[baseline]
    if horizon not in reference['summary']['horizons']:raise ValueError('Requested horizon is absent')
    split=reference['summary']['split'];result=dict(baseline=baseline,split=split,horizon_s=float(horizon),models={},findings=[])
    for name,report in reports.items():
        require_matched_reports(report,reference,indexes[name],indexes[baseline])
        groups=defaultdict(list)
        for row in report['roots']:
            for label in strata(row,split,dense_min):groups[label].append(row)
        model=dict(kind=report['model']['config']['kind'],strata={label:summarize(rows,horizon) for label,rows in groups.items()},
            risk_resolution=risk_resolution(report['roots'],horizon))
        if name!=baseline:model['versus_cv']=pair_diagnostics(indexes[name],indexes[baseline],horizon,top_k)
        result['models'][name]=model
    deterministic=[name for name,model in result['models'].items() if model['kind']=='deterministic']
    if len(deterministic)==1:
        for name,model in result['models'].items():
            if model['kind']=='stochastic':
                model['versus_deterministic']=dict(reference=deterministic[0],
                    **pair_diagnostics(indexes[name],indexes[deterministic[0]],horizon,top_k))
    coverage=result['models'][baseline]['strata']['all']
    informative=coverage['informative_ranking_roots'];supported=coverage['fully_tracked_informative_ranking_roots']
    if supported<informative:
        result['findings'].append(dict(type='identity_coverage',
            evidence=f'{informative-supported}/{informative} informative roots lack complete physical identity association; {supported} support tracked-dynamics ranking.',
            conclusion='All-outcome risk failures may involve unrepresented humans. This is a measured coverage limitation, not identified contact causality.'))
    if supported==0:
        result['findings'].append(dict(type='ranking_gate_unavailable',evidence='Zero fully tracked informative ranking roots.',
            conclusion='No supported learned-dynamics ranking gate can be established on this split.'))
    if coverage['observed_bank_all_collision_roots']:
        result['findings'].append(dict(type='candidate_bank_coverage',
            evidence=f"{coverage['observed_bank_all_collision_roots']} roots have no safe observed eligible candidate at this horizon.",
            conclusion='A better ranking alone cannot choose a safe action from those observed candidates; this is not an episode-level proposal verdict.'))
    for name,model in result['models'].items():
        resolution=model['risk_resolution']['all_outcome']
        if resolution['tied_risk_roots'] or (resolution['queries'] and
                resolution['near_zero']+resolution['near_one']>=.9*resolution['queries']):
            result['findings'].append(dict(type='risk_resolution',model=name,
                evidence=f"{resolution['tied_risk_roots']}/{resolution['informative_roots']} informative roots have tied candidate risks; {resolution['near_zero']} near-zero and {resolution['near_one']} near-one risks among {resolution['queries']} eligible queries (threshold {resolution['probability_threshold']}).",
                conclusion='Inspect identity coverage and contact calibration before increasing dynamics capacity; saturation alone does not identify a cause.'))
        if name==baseline:continue
        paired=model['versus_cv']['paired_root_differences'];energy=paired['energy']
        if energy['roots'] and energy['low']<=0<=energy['high']:
            result['findings'].append(dict(type='prediction_uncertainty',model=name,
                evidence=f"Energy delta {energy['mean']:.6g}, root 95% interval [{energy['low']:.6g}, {energy['high']:.6g}], n={energy['roots']}.",
                conclusion='This interval does not establish a held-out energy advantage over the supplied CV baseline.'))
        if 'versus_deterministic' in model:
            energy_b=model['versus_deterministic']['paired_root_differences']['energy']
            if energy_b['roots'] and energy_b['low']<=0<=energy_b['high']:
                result['findings'].append(dict(type='stochastic_complexity_unproven',model=name,
                    evidence=f"Stochastic-minus-deterministic energy {energy_b['mean']:.6g}, root 95% interval [{energy_b['low']:.6g}, {energy_b['high']:.6g}].",
                    conclusion='The extra stochastic representation has not established a predictive advantage over the deterministic model in this comparison.'))
        foreground=paired['occupancy_positive_nll']
        if foreground['roots'] and foreground['mean']>0:
            result['findings'].append(dict(type='foreground_probability',model=name,
                evidence=f"Foreground occupancy NLL delta {foreground['mean']:.6g}, root 95% interval [{foreground['low']:.6g}, {foreground['high']:.6g}].",
                conclusion='Foreground probability is worse by point estimate; a background-dominated occupancy score cannot establish improvement.'))
        for support,counts in model['versus_cv']['motion_ranking_quadrants'].items():
            count=counts.get('physics_improved/ranking_worsened',0)
            if count:
                result['findings'].append(dict(type='forecast_risk_discordance',model=name,support=support,
                    evidence=f'{count} informative roots improve energy but worsen candidate concordance.',
                    conclusion='Inspect native contact identity/time and swept clearance before attributing the disagreement to model capacity.'))
    return result


def analyze(paths,horizon=3.,dense_min=4,top_k=8):
    grouped=defaultdict(dict);sources=[]
    for source in paths:
        path=Path(source);report=json.loads(path.read_text());split=report['summary']['split']
        if path.stem in grouped[split]:raise ValueError('Duplicate report name within split: '+path.stem)
        grouped[split][path.stem]=report
        sources.append(dict(path=str(path.resolve()),sha256=sha256(path)))
    return dict(version='eventwm-offline-failure-analysis-v1',sources=sources,
        interpretation='Exploratory offline diagnostics; no navigation, SOTA, causal or calibrated-uncertainty claim.',
        definitions=dict(never_observed='Legal category unseen; distinct from unseen evaluation scenes.',
            lost_or_expired='Legal categories lost, visible_lost or observed_expired.',
            dense_count_proxy=f'At least {dense_min} physical humans, evaluation only; not measured local spatial density.',
            clustering='Equal-root bootstrap; excludes scene/seed/model-selection dependence.',
            overlapping_strata=True),
        splits={split:analyze_group(reports,str(float(horizon)),dense_min,top_k) for split,reports in grouped.items()})


def markdown(report):
    lines=['# Offline WM pilot failure analysis','',report['interpretation'],'']
    for split,result in report['splits'].items():
        lines += [f'## {split}: {result["horizon_s"]:g} seconds','',f'Baseline: `{result["baseline"]}`.','',
            '| Model | Supported trajectory roots | Energy | Fully tracked informative ranking roots | Supported concordance |',
            '| --- | ---: | ---: | ---: | ---: |']
        for name,model in result['models'].items():
            overall=model['strata']['all'];metrics=overall['metrics']
            fmt=lambda x:'—' if x is None else f'{x:.4f}'
            lines.append(f'| {name} | {overall["trajectory_supported_roots"]} | {fmt(metrics["energy"]["mean"])} | {overall["fully_tracked_informative_ranking_roots"]} | {fmt(metrics["fully_tracked_candidate_concordance"]["mean"])} |')
        lines += ['']
        for finding in result['findings']:
            prefix=finding.get('model','split')
            lines.append(f'- **{prefix}: {finding["type"]}.** {finding["evidence"]} {finding["conclusion"]}')
        lines += ['']
    lines += ['Strata overlap. Never-observed humans and unseen scenes are separate conditions. Dense-count strata use privileged counts only for evaluation and do not certify local crowd density. Root intervals exclude scene, training-seed and model-selection uncertainty.','',
        'The companion JSON contains coverage, stratum metrics, risk saturation, root-level discordant examples and exact source hashes.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--markdown',type=Path)
    p.add_argument('--horizon',type=float,choices=(1.,3.,5.),default=3.)
    p.add_argument('--dense-min-humans',type=int,default=4);p.add_argument('--top-k',type=int,default=8)
    args=p.parse_args()
    outputs=[args.output]+([args.markdown] if args.markdown else [])
    if any(path.exists() for path in outputs):raise FileExistsError('Failure reports never overwrite existing artifacts')
    report=analyze(args.reports,args.horizon,args.dense_min_humans,args.top_k)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.markdown:
        args.markdown.parent.mkdir(parents=True,exist_ok=True);args.markdown.write_text(markdown(report))
    print(json.dumps({split:value['findings'] for split,value in report['splits'].items()},indent=2))


if __name__=='__main__':main()
